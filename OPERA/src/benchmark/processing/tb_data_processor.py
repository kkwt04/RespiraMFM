"""Prepare TBscreen or CODA TB records for the existing RespiraMFM HDF5 loader.

This processor preserves supplied official splits. It never creates a random
split, modifies raw audio, or silently drops invalid rows.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf


def _normalise_label(value, positive_values):
    value = str(value).strip().casefold()
    if value in {item.casefold() for item in positive_values}:
        return 1
    if value in {"negative", "non-tb", "nontb", "no", "0", "control"}:
        return 0
    raise ValueError(f"Unrecognised TB label: {value!r}")


def _context(row, columns):
    values = []
    for column in columns:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            values.append(f"{column}: {row[column]}")
    return "; ".join(values) or "No clinical metadata supplied."


def _read_tbscreen(args):
    records = pd.read_csv(args.records_csv)
    required = {args.audio_column, args.participant_column}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"TBscreen records CSV missing columns: {sorted(missing)}")
    if args.clinical_csv:
        clinical = pd.read_csv(args.clinical_csv)
        if args.participant_column not in clinical.columns:
            raise ValueError("TBscreen clinical CSV is missing the participant column")
        if clinical[args.participant_column].duplicated().any():
            raise ValueError("TBscreen clinical CSV has duplicate participant identifiers")
        records = records.merge(clinical, on=args.participant_column, how="left", validate="many_to_one")
    if args.label_column not in records.columns:
        raise ValueError(f"TBscreen records/clinical CSV missing label column: {args.label_column}")
    return records.rename(columns={
        args.audio_column: "filename",
        args.participant_column: "participant",
        args.label_column: "raw_label",
    })


def _read_codatb(args):
    coughs = pd.read_csv(args.records_csv)
    clinical = pd.read_csv(args.clinical_csv)
    required_cough = {"filename", "participant"}
    required_clinical = {"participant", args.label_column}
    if required_cough - set(coughs.columns):
        raise ValueError("CODA cough metadata must contain filename and participant")
    if required_clinical - set(clinical.columns):
        raise ValueError(f"CODA clinical metadata missing: {sorted(required_clinical - set(clinical.columns))}")
    if clinical["participant"].duplicated().any():
        raise ValueError("CODA clinical metadata has duplicate participant identifiers")
    records = coughs.merge(clinical, on="participant", how="left", validate="many_to_one")
    if records[args.label_column].isna().any():
        raise ValueError("Some CODA cough rows did not join to a labelled clinical participant")
    return records.rename(columns={args.label_column: "raw_label"})


def _apply_official_split(records, args):
    if not args.split_csv:
        records["split"] = "train"
        return records
    splits = pd.read_csv(args.split_csv)
    required = {args.split_participant_column, args.split_column}
    if required - set(splits.columns):
        raise ValueError(f"Split CSV missing columns: {sorted(required - set(splits.columns))}")
    splits = splits.rename(columns={args.split_participant_column: "participant", args.split_column: "split"})
    if splits["participant"].duplicated().any():
        raise ValueError("Official split CSV contains duplicate participant identifiers")
    records = records.merge(splits[["participant", "split"]], on="participant", how="left", validate="many_to_one")
    if records["split"].isna().any():
        raise ValueError("Some records have no official split assignment")
    patient_splits = records.groupby("participant")["split"].nunique()
    if (patient_splits > 1).any():
        raise ValueError("Patient-level leakage: a participant appears in multiple supplied splits")
    return records


def prepare_records(args):
    if args.label_column is None:
        args.label_column = "label" if args.dataset == "tbscreen" else "tb_status"
    if args.positive_value is None:
        args.positive_value = ["TB"] if args.dataset == "tbscreen" else ["positive"]
    records = _read_tbscreen(args) if args.dataset == "tbscreen" else _read_codatb(args)
    records = _apply_official_split(records, args)
    records["label"] = records["raw_label"].map(lambda value: _normalise_label(value, args.positive_value))
    audio_root = Path(args.audio_root)
    records["filepath"] = records["filename"].map(lambda value: str(audio_root / str(value)))
    records["metadata"] = records.apply(lambda row: _context(row, args.metadata_columns), axis=1)
    return records[["filepath", "participant", "label", "split", "metadata"]].copy()


def audit_records(records):
    valid, excluded = [], []
    for row in records.to_dict("records"):
        try:
            info = sf.info(row["filepath"])
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError("empty audio stream")
            row["sample_rate"] = info.samplerate
            row["duration_seconds"] = info.duration
            valid.append(row)
        except Exception as error:
            row["reason"] = str(error)
            excluded.append(row)
    return pd.DataFrame(valid), pd.DataFrame(excluded)


def write_hdf5(records, output_path):
    from RespFeatureExtractor.feature_extractor import OperaExtractor

    extractor = OperaExtractor(pretrain="operaCT")
    features, rows = [], []
    for row in records.to_dict("records"):
        feature = np.asarray(extractor.extract_opera_features(row["filepath"], input_sec=8)).reshape(-1)
        if feature.shape != (768,):
            raise ValueError(f"Unexpected OPERA-CT embedding shape {feature.shape} for {row['filepath']}")
        features.append(feature.astype(np.float32))
        rows.append(row)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as store:
        store.create_dataset("filenames", data=[row["filepath"] for row in rows], dtype=string_dtype)
        store.create_dataset("labels", data=np.asarray([row["label"] for row in rows], dtype=np.int64))
        store.create_dataset("audio_embeddings", data=np.stack(features))
        store.create_dataset("metadata", data=[row["metadata"] for row in rows], dtype=string_dtype)
        store.create_dataset("text_embeddings", data=np.empty((len(rows), 0), dtype=np.float32))


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = prepare_records(args)
    valid, excluded = audit_records(records)
    records.to_csv(output_dir / "records_manifest.csv", index=False)
    excluded.to_csv(output_dir / "excluded_audio.csv", index=False)
    report = {
        "dataset": args.dataset,
        "records": len(records),
        "valid_audio": len(valid),
        "excluded_audio": len(excluded),
        "class_distribution": {str(k): int(v) for k, v in records["label"].value_counts().items()},
        "split_distribution": {str(k): int(v) for k, v in records["split"].value_counts().items()},
    }
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not excluded.empty and not args.allow_invalid:
        raise RuntimeError("Invalid/unreadable audio found; inspect excluded_audio.csv before continuing")
    if not args.extract_features:
        return
    for split, split_records in valid.groupby("split"):
        write_hdf5(split_records, output_dir / f"{split}_features.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("tbscreen", "codatb"), required=True)
    parser.add_argument("--records-csv", required=True)
    parser.add_argument("--clinical-csv", help="Required for codatb; optional TBscreen participant metadata")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", help="Defaults to label (TBscreen) or tb_status (CODA)")
    parser.add_argument("--positive-value", action="append", help="Repeat for each positive label spelling")
    parser.add_argument("--audio-column", default="path", help="TBscreen audio-path column")
    parser.add_argument("--participant-column", default="participant", help="TBscreen participant column")
    parser.add_argument("--metadata-columns", nargs="*", default=["age", "sex"])
    parser.add_argument("--split-csv", help="Official participant-level split file; no split is generated")
    parser.add_argument("--split-participant-column", default="participant")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--extract-features", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true")
    arguments = parser.parse_args()
    if arguments.dataset == "codatb" and not arguments.clinical_csv:
        parser.error("--clinical-csv is required for codatb")
    main(arguments)

