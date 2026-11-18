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
