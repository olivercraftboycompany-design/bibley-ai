"""Fine-tune one (or all) Bibley base models with LoRA, then merge.

Usage (on a GPU, e.g. Kaggle T4):
  python training/build_dataset.py
  python training/train_lora.py --model Bibley-Qwen3-0.6B
  python training/train_lora.py --all            # trains all 10 in a row

Output: training/out/<bibley_name>/merged   (full merged model, Apache 2.0 base + your changes)
"""
import argparse, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "models.json")))["models"]

def train_one(m, epochs, max_len, batch, grad_accum, max_steps=-1, limit=0):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_dir = os.path.join(HERE, "out", m["bibley_name"])
    data = load_dataset("json", data_files={
        "train": os.path.join(HERE, "data", "train.jsonl"),
        "validation": os.path.join(HERE, "data", "val.jsonl")})

    if limit:
        data["train"] = data["train"].select(range(limit))
        data["validation"] = data["validation"].select(range(min(limit, len(data["validation"]))))
    tok = AutoTokenizer.from_pretrained(m["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(m["base"], dtype=dtype)

    lora = LoraConfig(r=m["lora_r"], lora_alpha=m["lora_r"] * 2, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=os.path.join(out_dir, "ckpt"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=m["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        eval_strategy="steps", eval_steps=250,
        save_strategy="no",
        max_length=max_len,
        assistant_only_loss=False,
        completion_only_loss=None,
        bf16=bf16, fp16=(not bf16 and torch.cuda.is_available()),
        gradient_checkpointing=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=data["train"],
                         eval_dataset=data["validation"], processing_class=tok,
                         peft_config=lora)
    trainer.train()

    merged = trainer.model.merge_and_unload()
    merged_dir = os.path.join(out_dir, "merged")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    shutil.rmtree(os.path.join(out_dir, "ckpt"), ignore_errors=True)
    write_card(m, merged_dir)
    print(f"saved {merged_dir}")
    del trainer, model, merged
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def write_card(m, merged_dir):
    tpl = open(os.path.join(HERE, "MODEL_CARD_TEMPLATE.md")).read()
    card = (tpl.replace("{{BIBLEY_NAME}}", m["bibley_name"])
               .replace("{{BASE}}", m["base"])
               .replace("{{PARAMS}}", m["params"]))
    open(os.path.join(merged_dir, "README.md"), "w").write(card)
    # Apache 2.0 section 4: ship the license and a notice of changes
    shutil.copy(os.path.join(HERE, "LICENSE-APACHE-2.0"), os.path.join(merged_dir, "LICENSE"))
    open(os.path.join(merged_dir, "NOTICE"), "w").write(
        f"{m['bibley_name']}\n"
        f"Modified from {m['base']} (Apache License 2.0) by Oliver Holder / Bibley.\n"
        "Changes: LoRA fine-tune on public-domain KJV verse lookup and Bible Q&A data, merged into the weights.\n"
        "Original model copyright belongs to its authors; see https://huggingface.co/" + m["base"] + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="bibley_name from models.json")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="quick test: stop after N steps")
    ap.add_argument("--limit", type=int, default=0, help="quick test: use only N examples")
    a = ap.parse_args()
    todo = CFG if a.all else [m for m in CFG if m["bibley_name"] == a.model]
    if not todo:
        raise SystemExit("pick --model <name> from models.json, or --all")
    for m in todo:
        print(f"=== {m['bibley_name']} <- {m['base']} ===")
        train_one(m, a.epochs, a.max_len, a.batch, a.grad_accum, a.max_steps, a.limit)

if __name__ == "__main__":
    main()
