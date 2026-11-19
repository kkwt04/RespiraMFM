# PBS templates

The templates intentionally do not select a queue, module, GPU model, or
filesystem path. Confirm Katana's current PBS resource syntax and substitute
the values below. GPU jobs request one GPU; preprocessing may be submitted as a
CPU job unless the selected feature-extraction command explicitly uses a GPU.

```bash
qsub -N respiramfm-eval \
  -l select=1:ncpus=8:mem=64gb:ngpus=1 \
  -l walltime=02:00:00 \
  -o /path/to/logs -e /path/to/logs \
  -v VENV=/path/to/RespiraMFM/.venv,OUTPUT_DIR=/path/to/results,COMMAND='CUDA_VISIBLE_DEVICES=0 python src/evaluate.py --disease covid --projector contrastive --datasets ukcovid19 --data-root /path/to/data --aligner-checkpoint /path/to/phase1.pth --checkpoint /path/to/phase2.pth --results-dir /path/to/results' \
  pbs/evaluate_pretrained.pbs
```

Use the same pattern with `preprocess_features.pbs`, `phase1_alignment.pbs`,
`phase2_instruction_tuning.pbs`, or `final_evaluation.pbs`. Replace `ncpus`,
`mem`, `walltime`, GPU request, output path, and command for your allocation.
The sample evaluation command will produce `metrics.json`, `predictions.csv`,
and `evaluation.log` after the required released artifacts are staged.

For multi-scale cache extraction, use `multiscale_feature_extraction.pbs` with
`COMMAND='python src/extract_multiscale_features.py --input-h5 INPUT.h5 --output-h5 OUTPUT.h5 --checkpoint-id encoder-operaCT.ckpt'`.

For a single GPU, set `COMMAND` to one of these existing entry points (replace all
uppercase paths):

```bash
CUDA_VISIBLE_DEVICES=0 python src/extract_multiscale_features.py --input-h5 INPUT.h5 --output-h5 MULTISCALE_TRAIN.h5 --checkpoint-id cks/model/encoder-operaCT.ckpt
CUDA_VISIBLE_DEVICES=0 python src/projector_train_text_audio.py --datasets ukcovid19 --disease covid --audio-mode multiscale --feature-path ukcovid19=MULTISCALE_TRAIN.h5 --checkpoint-path PHASE1.pth
CUDA_VISIBLE_DEVICES=0 python src/train.py --disease covid --audio-mode multiscale --multiscale-checkpoint PHASE1.pth --train-feature-path ukcovid19=UK_TRAIN.h5 --train-feature-path coughvid=COUGHVID_TRAIN.h5 --test-feature-path ukcovid19=UK_TEST.h5 --test-feature-path coughvid=COUGHVID_TEST.h5 --test-feature-path coswara=COSWARA_TEST.h5 --output-dir PHASE2_DIR
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py --disease covid --projector contrastive --audio-mode multiscale --multiscale-checkpoint PHASE1.pth --datasets ukcovid19 --data-root DATA_ROOT --aligner-checkpoint PHASE1.pth --checkpoint PHASE2_MODEL.pth --results-dir RESULTS_DIR
```

Resume Phase 1 with `--resume PHASE1.pth.resume.pth`; resume Phase 2 with
`--resume PHASE2_DIR/phase2_last.resume.pth`. The feature extractor resumes its
own output HDF5 when it already exists.
