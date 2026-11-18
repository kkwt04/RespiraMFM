"""Synthetic metadata contracts for TBscreen and CODA TB preparation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OPERA" / "src" / "benchmark" / "processing"))

from tb_data_processor import prepare_records


def _args(tmp_path, dataset, records_csv, clinical_csv=None):
    return SimpleNamespace(
        dataset=dataset,
        records_csv=str(records_csv),
        clinical_csv=str(clinical_csv) if clinical_csv else None,
        audio_root=str(tmp_path / "audio"),
        output_dir=str(tmp_path / "out"),
        label_column=None,
        positive_value=None,
        audio_column="path",
        participant_column="participant",
        metadata_columns=["age", "sex"],
        split_csv=None,
        split_participant_column="participant",
        split_column="split",
    )


def test_tbscreen_labels_are_normalised_without_creating_a_split(tmp_path):
    records_csv = tmp_path / "passive.csv"
    pd.DataFrame({"path": ["a.wav", "b.wav"], "participant": ["p1", "p2"], "label": ["TB", "control"]}).to_csv(records_csv, index=False)
    records = prepare_records(_args(tmp_path, "tbscreen", records_csv))
    assert records["label"].tolist() == [1, 0]
    assert records["split"].tolist() == ["train", "train"]


def test_codatb_joins_labels_at_participant_level(tmp_path):
    cough_csv = tmp_path / "coughs.csv"
    clinical_csv = tmp_path / "clinical.csv"
    pd.DataFrame({"filename": ["a.wav", "b.wav"], "participant": ["p1", "p1"]}).to_csv(cough_csv, index=False)
    pd.DataFrame({"participant": ["p1"], "tb_status": ["positive"], "age": [42], "sex": ["Female"]}).to_csv(clinical_csv, index=False)
    records = prepare_records(_args(tmp_path, "codatb", cough_csv, clinical_csv))
    assert records["label"].tolist() == [1, 1]
    assert all("age: 42" in context for context in records["metadata"])
