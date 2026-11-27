"""Score a trained Bibley model on exact verse lookup.

  python training/eval_quotes.py training/out/Bibley-Qwen3-0.6B/merged --n 100

Asks "What does <ref> say?" for random verses and counts answers whose quote
matches the KJV text exactly. Use it to compare the 10 models.
"""
import argparse, json, os, random, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
VERSES = json.load(open(os.path.join(os.path.dirname(HERE), "verses.json")))

def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    first = json.loads(open(os.path.join(HERE, "data", "val.jsonl")).readline())
    system = first["messages"][0]["content"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.path)
    model = AutoModelForCausalLM.from_pretrained(a.path, dtype="auto").to(dev).eval()
    random.seed(1); hits = 0
    for v in random.sample(VERSES, a.n):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"What does {v['ref']} say?"}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev); ids = enc["input_ids"]
        out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
        reply = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        if " ".join(norm(v["text"])) in " ".join(norm(reply)): hits += 1
    print(f"{a.path}: exact quote {hits}/{a.n} ({100*hits/a.n:.0f}%)")

if __name__ == "__main__":
    main()
