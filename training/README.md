# Bibley model training

Turns 10 small, Apache-2.0 open models into Bibley models that look up and quote KJV verses exactly.

| Bibley name | Base model | Params | Release format (< 1 GB) |
|---|---|---|---|
| Bibley-Qwen3-0.6B | Qwen/Qwen3-0.6B | 0.75B | Q8_0 GGUF |
| Bibley-Qwen2.5-0.5B | Qwen/Qwen2.5-0.5B-Instruct | 0.49B | Q8_0 GGUF |
| Bibley-Smol-360M | HuggingFaceTB/SmolLM2-360M-Instruct | 0.36B | F16 GGUF |
| Bibley-Smol-135M | HuggingFaceTB/SmolLM2-135M-Instruct | 0.13B | F16 GGUF |
| Bibley-Granite-350M | ibm-granite/granite-4.0-350m | 0.35B | F16 GGUF |
| Bibley-Granite-H-350M | ibm-granite/granite-4.0-h-350m | 0.34B | F16 GGUF |
| Bibley-Granite-H-1B | ibm-granite/granite-4.0-h-1b | 1.46B | Q4_K_M GGUF |
| Bibley-OLMo2-1B | allenai/OLMo-2-0425-1B-Instruct | 1.48B | Q4_K_M GGUF |
| Bibley-TinyLlama-1.1B | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Q4_K_M GGUF |
| Bibley-Danube3-500M | h2oai/h2o-danube3-500m-chat | 0.51B | Q8_0 GGUF |

All are Apache 2.0 (checked on Hugging Face, Nov 26 2026). Excluded: Qwen2.5-1.5B and Granite 4.0 1B (over 1.5B params), and Gemma 3, Llama 3.2 and LFM2 (custom licenses).

## Run it
Easiest: on Kaggle, run `bibley_train_kaggle.ipynb` (7 small models) and `bibley_train_kaggle_part2.ipynb` (the 3 models over 1B) with a T4 GPU and Save & Run All. Each fits in one 12-hour session.

By hand on any GPU:
```bash
pip install -r training/requirements-train.txt
python training/build_dataset.py                  # KJV dataset -> training/data/
python training/train_lora.py --model Bibley-Qwen3-0.6B   # or --all
python training/eval_quotes.py training/out/Bibley-Qwen3-0.6B/merged
bash training/export_gguf.sh Bibley-Qwen3-0.6B   # under 1 GB
```

## What it learns
- **lookup**: "What does John 3:16 say?" → the exact KJV text
- **locate**: a quoted fragment → its reference
- **topic**: the same retrieved-verses prompt `server.py` uses → exact quotes + "In plain words:"
- **honest**: off-topic questions → says the verses don't answer it
- **identity**: says it's Bibley, built on its base model

All quotes come from `verses.json`, so the training data never contains a misquote. Verse lookup at answer time still comes from Bibley's retrieval and reference repair in `server.py`, which is what keeps quotes accurate. Fine-tuning teaches the format and the KJV wording; it doesn't make a small model a reliable source on its own.

## Licensing: what "ours" means
Apache 2.0 lets you rename, modify, sell and redistribute these models. To stay compliant, every release folder includes:
- `LICENSE` (the Apache 2.0 text)
- `NOTICE` (names the base model and says what you changed)
- `README.md` model card naming the base model

Don't remove those, and don't claim a model was trained from scratch. Data: the KJV is public domain in the US. The topic mappings come from the NEUU dataset under CC BY 4.0, which needs credit (already in the model card).

## Notes
- Training time on a free T4 is a rough guess: about 30–60 min for models under 0.6B and 1.5–2 h for the 1–1.5B ones, at 1 epoch.
- The Granite-H models are hybrid Mamba models. They train without extra packages, but slower; `pip install mamba-ssm causal-conv1d` speeds them up.
- `training/data/` and `training/out/` are generated files and are git-ignored.
