# Long-Context Mini Examples

Small practical scripts compare `fp16`, `fp4`, and `thrift` without HELMET or large eval installs.

```bash
pip install -e ".[hf]"
pip install -r examples/long_context_quality/requirements.txt
```

## Forward / NLL

Runs a patched HF forward pass and reports `forward_s`, token/s, mean NLL, and delta vs fp16. It uses repeated synthetic text by default; pass `--text-file` for a local corpus.

```bash
python examples/long_context_quality/run_nll_mini.py --lengths 4096,8192 --methods fp16,fp4,thrift
```

## Generation

Runs real RULER synthetic samples via `ruler_gen.generate_samples()` and `ruler_score.score_sample()`. By default the script looks for those files in `/workspace/nvfp4-experiments/experiments`; override with `--ruler-dir`.

```bash
python examples/long_context_quality/run_ruler_mini.py --lengths 4096,8192 --methods fp16,fp4,thrift
```

Timing is split into initial prefill forward and incremental decode. Rows include `prompt_tokens`, `prefill_s`, `decode_s`, `total_s`, generated tokens, decode token/s, and RULER score.

All scripts write `metrics.jsonl`, `summary.md`, and `environment.json` under `results/long_context_quality/`.
