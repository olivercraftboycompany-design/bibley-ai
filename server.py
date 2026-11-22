#!/usr/bin/env python3
"""Bibley 2.1, Oliver's local Bible assistant.

Qwen3-0.6B with KJV hybrid retrieval and a topical index.
First-run model and dependency downloads require internet access.
Generated explanations and reference repair can be wrong. See README.md.
"""
import json
import math
import os
import pickle
import re
import shutil
import sys
import threading
import time
import webbrowser

# Plain, reliable downloads (the newer xet downloader can fail on older Macs).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("BIBLEY_PORT", "8741"))
MAX_NEW = int(os.environ.get("BIBLEY_MAX_NEW", "300"))
TOP_K = 5
MODEL_ID = os.environ.get("BIBLEY_MODEL", "Qwen/Qwen3-0.6B")
EMBED_ID = os.environ.get("BIBLEY_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
# bge models retrieve best when the question carries this prefix.
QUERY_PREFIX = ("Represent this sentence for searching relevant passages: "
                if "bge" in EMBED_ID else "")
# Below this meaning-match score, and with no verse named in the question,
# Bibley says honestly that the Bible does not cover the question.
OFFTOPIC_GATE = float(os.environ.get("BIBLEY_OFFTOPIC_GATE", "0.338"))

STATE = {"stage": "Starting...", "ready": False, "error": None}
HISTORY = []
GEN_LOCK = threading.Lock()

print("=" * 56)
print("  BIBLEY 2.1")
print("  Fresh Qwen3-0.6B + real KJV verse lookup (hybrid search)")
print("=" * 56)
sys.stdout.flush()

# ---- Verse-search data (downloaded once, on first run) ----
# These files are too big to fit in the zip, so Bibley downloads them once
# from the same Google Drive folder the zip came from. They are Bible verse
# search data - nothing personal.
import hashlib
from urllib.request import urlopen

DATA_FILES = [
    # (local file, google drive id, expected sha256, expected size in bytes)
    ("verse_embeddings.npy", "1UfMQtqlDma8xgLbHFqEwjayMwEG7baDZ",
     "cfaa698e6f46d3950d8b6cc60a90b40e3946f6f16e321231a51c2f2caac526df", 23886464),
    ("topics_embeddings.part1", "1Q3-wGZR7-u1XGswfRl298qgaKPfi5kmA",
     "bbc0940c0df24b91e75c01891a57fd142ace25effadfc103b8cee70dd7ca15c0", 14360512),
    ("topics_embeddings.part2", "1FYhYNezwYen8ATzfnb3Z0bvuTwMUkzVX",
     "297869d6db2e40b56c3bacc13112d0a97601257ff4c87930885641b9c3794da7", 14360512),
]
TOPICS_SHA256 = "4d8eaf7d732f814ab21b4768d22929cdced6eef5a48ceb0f0d09d4257087fdbd"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data_files():
    """Download the verse-search data files once, with checksums."""
    topics_path = os.path.join(BASE, "topics_embeddings.npy")
    topics_ok = (os.path.exists(topics_path)
                 and _sha256_file(topics_path) == TOPICS_SHA256)

    def have(name, sha, size):
        p = os.path.join(BASE, name)
        return (os.path.exists(p) and os.path.getsize(p) == size
                and _sha256_file(p) == sha)

    missing = [d for d in DATA_FILES if not have(d[0], d[2], d[3])]
    if not missing and topics_ok:
        return
    print("Bibley needs to download its verse-search data (about 53 MB,")
    print("one time only). Please keep this window open and stay online.")
    sys.stdout.flush()
    for name, fid, sha, size in missing:
        dest = os.path.join(BASE, name)
        url = f"https://drive.google.com/uc?export=download&id={fid}"
        ok = False
        wait = 5
        for attempt in range(4):
            try:
                print(f"  Downloading {name} (attempt {attempt + 1} of 4)...")
                sys.stdout.flush()
                with urlopen(url, timeout=180) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f)
                if os.path.getsize(dest) == size and _sha256_file(dest) == sha:
                    ok = True
                    break
                print("  That copy did not pass its safety check; trying again.")
            except Exception as e:
                print(f"  Problem: {e}")
            sys.stdout.flush()
            time.sleep(wait)
            wait *= 2
        if not ok:
            print()
            print("  Bibley could not download its verse-search data.")
            print("  Check Wi-Fi and double-click START.command again.")
            print("  It picks up where it left off.")
            sys.stdout.flush()
            raise SystemExit(1)
    if not topics_ok:
        with open(topics_path, "wb") as out:
            for part in ("topics_embeddings.part1", "topics_embeddings.part2"):
                with open(os.path.join(BASE, part), "rb") as pf:
                    shutil.copyfileobj(pf, out)
        if _sha256_file(topics_path) != TOPICS_SHA256:
            print()
            print("  The verse-search data did not pass its safety check.")
            print("  Delete the topics_embeddings files in this folder")
            print("  and double-click START.command again.")
            sys.stdout.flush()
            raise SystemExit(1)
    print("  Verse-search data is ready.")
    sys.stdout.flush()


ensure_data_files()


# ---------------- verse library ----------------
import numpy as np

with open(os.path.join(BASE, "verses.json")) as f:
    VERSES = json.load(f)
EMB = np.load(os.path.join(BASE, "verse_embeddings.npy"))  # fp16, normalized
print(f"  Verse library ready: {len(VERSES):,} real KJV verses.")
sys.stdout.flush()

BOOKS = {}
for v in VERSES:
    BOOKS.setdefault(v["ref"].rsplit(" ", 1)[0], []).append(v)
BOOK_RE = "|".join(re.escape(b) for b in sorted(BOOKS, key=len, reverse=True))
REF_RE = re.compile(
    r"\b(" + BOOK_RE + r")s?\.?\s+(\d{1,3})\s*:\s*(\d{1,3})"
    r"(?:\s*[-\u2013\u2014]\s*(?:\d{1,3}\s*:\s*)?(\d{1,3}))?",
    re.IGNORECASE,
)
BOOK_LOOKUP = {b.lower(): b for b in BOOKS}
# common short names people type
ALIASES = {"psalm": "Psalms", "psalms": "Psalms", "rev": "Revelation",
           "gen": "Genesis", "matt": "Matthew", "jn": "John"}
for a, real in ALIASES.items():
    if real in BOOKS:
        BOOK_LOOKUP.setdefault(a, real)


def find_by_reference(prompt):
    """Pull out verses the user named directly, like 'John 3:16-17'."""
    found = []
    for m in REF_RE.finditer(prompt):
        book = BOOK_LOOKUP.get(m.group(1).lower())
        if not book:
            continue
        ch, v1 = int(m.group(2)), int(m.group(3))
        v2 = int(m.group(4)) if m.group(4) else v1
        if v2 < v1:
            v1, v2 = v2, v1
        want = []
        for v in BOOKS[book]:
            c, vs = v["ref"].rsplit(" ", 1)[1].split(":")
            if int(c) == ch and v1 <= int(vs) <= v2:
                want.append(v)
        found.extend(want[:12])
    return found


def _norm_text(s):
    s = " ".join(s.split())
    # the packaged KJV text keeps a space before punctuation ("LORD ,");
    # ignore that when checking quotes so a correctly quoted verse matches
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip().strip('"').lower()

# exact-quote lookup: normalized verse text -> ref (for answer repair)
TEXT2REF = {}
for v in VERSES:
    TEXT2REF.setdefault(_norm_text(v["text"]), v["ref"])

# ---------------- keyword search (BM25, prebuilt index) ----------------
with open(os.path.join(BASE, "bm25_index.pkl"), "rb") as f:
    _BM25 = pickle.load(f)
BM25_INV = _BM25["inv"]        # word -> (doc ids np.array, term freqs np.array)
BM25_DL = _BM25["dl"]          # document lengths np.array
BM25_IDF = _BM25["idf"]        # word -> idf float
BM25_AVGDL = float(_BM25["avgdl"])
_BM25_K1, _BM25_B = 1.2, 0.75
_TOKEN_RE = re.compile(r"[a-z]+")
_QUERY_STOP = set(
    "what does the bible say about is it a an and or of to in on for me my "
    "i do how should can".split())


def bm25_rank(query, topn=100):
    words = [w for w in _TOKEN_RE.findall(query.lower())
             if w not in _QUERY_STOP]
    scores = {}
    for w in words:
        hit = BM25_INV.get(w)
        if hit is None:
            continue
        docs, tfs = hit
        idf = BM25_IDF[w]
        tf = tfs.astype(np.float32)
        denom = tf + _BM25_K1 * (1 - _BM25_B
                                 + _BM25_B * BM25_DL[docs] / BM25_AVGDL)
        contrib = idf * tf * (_BM25_K1 + 1) / denom
        for d, c in zip(docs.tolist(), contrib.tolist()):
            scores[d] = scores.get(d, 0.0) + c
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:topn]
    return [d for d, _ in ranked]


# ---------------- topical index (Nave's + Torrey, CC BY 4.0) ----------------
with open(os.path.join(BASE, "topics_index.json")) as f:
    TOPICS = json.load(f)
TEMB = np.load(os.path.join(BASE, "topics_embeddings.npy"))  # fp16, normalized


def topical_rank(q_vec, top_aspects=8, cap=60):
    sims = TEMB.astype(np.float32) @ q_vec
    take = min(top_aspects, len(sims))
    idx = np.argpartition(-sims, take)[:take]
    idx = idx[np.argsort(-sims[idx])]
    out = []
    seen = set()
    for i in idx:
        for vi in TOPICS[int(i)]["verses"]:
            if vi not in seen:
                seen.add(vi)
                out.append(vi)
        if len(out) >= cap:
            break
    return out[:cap]


def rrf_fuse(lists, k=60):
    score = {}
    for lst in lists:
        for r, d in enumerate(lst):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r + 1)
    return sorted(score.items(), key=lambda kv: -kv[1])


# ---------------- meaning search ----------------
_tok_s = None
_emb_model = None
_torch = None


def _hf_cache_dir(repo_id):
    hf_home = os.environ.get("HF_HOME",
                             os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(hf_home, "hub",
                        "models--" + repo_id.replace("/", "--"))


def _safetensors_ok(path):
    """Read the file's header; a half-downloaded file fails this."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n <= 0 or n > 500_000_000:
                return False
            json.loads(f.read(n).decode("utf-8"))
        return True
    except Exception:
        return False


def _verify_snapshot(path, checks):
    for rel, min_size in checks:
        p = os.path.join(path, rel)
        if not os.path.isfile(p) or os.path.getsize(p) < min_size:
            return rel
        if rel.endswith(".safetensors") and not _safetensors_ok(p):
            return rel
    return None


def ensure_download(repo_id, label, checks, attempts=4):
    """Download a model with retries; verify it is complete before using it.

    snapshot_download continues partial downloads where they left off, so
    re-running START.command after a dropped download just keeps going.
    """
    from huggingface_hub import snapshot_download
    wait = 5
    for attempt in range(1, attempts + 1):
        bad = None
        try:
            STATE["stage"] = f"Downloading {label} (one time only)..."
            path = snapshot_download(repo_id)
            bad = _verify_snapshot(path, checks)
            if bad is None:
                print(f"  {label}: downloaded and checked. OK.")
                sys.stdout.flush()
                return path
            print(f"  The {label} download is incomplete ({bad}).")
        except Exception as e:
            print(f"  {label} download hiccup: {type(e).__name__}: {e}")
        if attempt >= 2 and bad is not None:
            # A file that stayed bad after a full re-download pass: wipe the
            # cached copy of this model and fetch it fresh.
            shutil.rmtree(_hf_cache_dir(repo_id), ignore_errors=True)
            print("  Cleared the half-downloaded copy; fetching it fresh.")
        if attempt < attempts:
            print(f"  Trying again in {wait} seconds "
                  f"(attempt {attempt + 1} of {attempts})...")
            sys.stdout.flush()
            time.sleep(wait)
            wait *= 2
    STATE["error"] = f"Could not finish downloading {label}."
    print()
    print(f"  Bibley could not finish downloading {label}.")
    print("  This is almost always the internet dropping mid-download.")
    print("  What to do:")
    print("    1. Check Wi-Fi.")
    print("    2. Keep the Mac awake (do not close the lid).")
    print("    3. Double-click START.command again - it continues")
    print("       where it left off.")
    print("  If it keeps failing, tell Instinct what this window says.")
    sys.stdout.flush()
    raise SystemExit(1)


EMBED_CHECKS = [
    ("config.json", 100),
    ("tokenizer.json", 100_000),
    ("model.safetensors", 80_000_000),
]

BRAIN_CHECKS = [
    ("config.json", 100),
    ("tokenizer.json", 1_000_000),
    ("tokenizer_config.json", 100),
    ("model.safetensors", 1_500_000_000),
]


def load_embedder():
    global _tok_s, _emb_model, _torch
    import torch
    from transformers import AutoModel, AutoTokenizer
    _torch = torch
    path = ensure_download(EMBED_ID, "the meaning search (90 MB)",
                           EMBED_CHECKS)
    _tok_s = AutoTokenizer.from_pretrained(path)
    _emb_model = AutoModel.from_pretrained(path)
    _emb_model.eval()


def embed_query(text):
    import torch
    with torch.no_grad():
        batch = _tok_s([QUERY_PREFIX + text], padding=True, truncation=True,
                       max_length=256, return_tensors="pt")
        out = _emb_model(**batch)
        tok_emb = out.last_hidden_state[0]
        mask = batch["attention_mask"][0].unsqueeze(-1).float()
        vec = (tok_emb * mask).sum(0) / mask.sum().clamp(min=1e-9)
        vec = vec / vec.norm().clamp(min=1e-9)
    return vec.numpy().astype(np.float32)


def search_meaning(query, k=TOP_K, skip=None):
    q = embed_query(query)
    sims = EMB.astype(np.float32) @ q
    take = k + (len(skip) if skip else 0) + 2
    idx = np.argpartition(-sims, take)[:take]
    idx = idx[np.argsort(-sims[idx])]
    out = []
    for i in idx:
        if skip and VERSES[i]["ref"] in skip:
            continue
        out.append((int(i), float(sims[i])))
        if len(out) >= k:
            break
    return out


def hybrid_search(query, k=TOP_K, skip=None):
    """Meaning + exact words + topical index, fused. Returns (idx list,
    top-1 meaning similarity)."""
    q = embed_query(query)
    sims = EMB.astype(np.float32) @ q
    top_emb = np.argsort(-sims)[:100].tolist()
    top1 = float(sims[top_emb[0]])
    fused = rrf_fuse([top_emb, bm25_rank(query, 100), topical_rank(q)])
    out = []
    for d, _ in fused:
        if skip and VERSES[d]["ref"] in skip:
            continue
        out.append(d)
        if len(out) >= k:
            break
    return out, top1


# ---------------- the brain ----------------
_tok = None
_model = None


def load_brain():
    global _tok, _model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_name = os.environ.get("BIBLEY_DTYPE", "float32")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]
    path = ensure_download(MODEL_ID, "Bibley's brain (about 1.5 GB)",
                           BRAIN_CHECKS)
    _tok = AutoTokenizer.from_pretrained(path)
    _model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    _model.eval()


SYSTEM = (
    "You are Bibley, a warm and honest Bible helper running on Oliver's own "
    "computer. You answer questions about the Bible using the real King James "
    "verses provided. You always quote the verses word for word before "
    "explaining them, and you explain them in plain words. You are gentle: "
    "you never condemn the person asking."
)

INSTRUCTIONS = (
    "Write your answer like this:\n"
    "1. Quote the 2 or 3 most important verses, each on its own line, in "
    "exactly this shape: the reference copied letter for letter from the "
    "list above, then a colon, then the exact words inside quote marks. "
    "Never change a single word or a single letter of a quote or of a "
    "reference.\n"
    "2. Then write 'In plain words:' and explain in 2 or 3 simple sentences "
    "what these verses say about the question, gently and honestly. Only "
    "claim what these verses actually say - do not add big claims they do "
    "not make. If the verses do not really answer the question, say so "
    "honestly instead of pretending they do."
)


def verse_block(verses):
    return "\n".join(f"{v['ref']}: {v['text']}" for v in verses)


def user_prompt(question, picked):
    return (
        f"Question: {question}\n\n"
        "Real King James Bible verses found for this question:\n"
        f"{verse_block(picked)}\n\n"
        f"{INSTRUCTIONS}"
    )


# One worked example, so the small brain can copy the shape of a good answer.
# The verses in it are pulled from the real verse library at startup, so the
# example quotes are always exact.
def build_example():
    ex_q = "what does the Bible say about worry?"
    ex_v = find_by_reference("Philippians 4:6") + find_by_reference("Matthew 6:34")
    p = ex_v[0]["text"]
    m = ex_v[1]["text"]
    ex_a = (
        f"Philippians 4:6: \"{p}\"\n"
        f"Matthew 6:34: \"{m}\"\n\n"
        "In plain words: God does not want you to live worried. Instead of "
        "carrying every fear by yourself, bring each one to Him in prayer "
        "with thanks, and take life one day at a time. That is His way out "
        "of worry: pray, and trust Him daily."
    )
    return ({"role": "user", "content": user_prompt(ex_q, ex_v)},
            {"role": "assistant", "content": ex_a})


EXAMPLE = build_example()


def build_messages(question, picked):
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs.extend(EXAMPLE)
    msgs.append({"role": "user", "content": user_prompt(question, picked)})
    return msgs


# ---------------- answer repair ----------------
_QUOTED_LINE_RE = re.compile(r"^([A-Za-z0-9 .:]+?):\s*[\"'\u201c*](.+?)[\"'\u201d*]\s*$")


def repair_references(reply):
    """Check every quoted verse line against the real library.

    The small brain sometimes writes the right words under the wrong
    reference. Quotes are matched to the verse library word for word; a
    wrong reference is replaced with the true one. Quotes that match no
    real verse are dropped. Never alters the quoted words themselves.
    """
    lines = reply.split("\n")
    fixed = 0
    dropped = 0
    out_lines = []
    for line in lines:
        m = _QUOTED_LINE_RE.match(line.strip())
        if not m:
            out_lines.append(line)
            continue
        claimed_ref, quote = m.group(1).strip(), m.group(2).strip()
        true_ref = TEXT2REF.get(_norm_text(quote))
        if true_ref is None:
            # maybe the brain cut the quote short: unique verse containing it
            probe = _norm_text(quote)
            if len(probe) >= 25:
                matches = [v["ref"] for v in VERSES
                           if probe in _norm_text(v["text"])]
                if len(matches) == 1:
                    true_ref = matches[0]
        if true_ref is None:
            dropped += 1
            continue  # not a real verse - do not show it
        if true_ref.lower() != claimed_ref.lower():
            fixed += 1
        out_lines.append(f"{true_ref}: \"{quote}\"")
    return "\n".join(out_lines), fixed, dropped


def honest_offtopic_reply(question, ranked_idx):
    closest = VERSES[ranked_idx[0]] if ranked_idx else None
    msg = ("I searched the whole Bible for this, and honestly, the Bible "
           "does not really talk about that.\n\n")
    if closest is not None:
        msg += (f"The closest verse I found is {closest['ref']}: "
                f"\"{closest['text']}\" - but it is only loosely related.\n\n")
    msg += ("In plain words: this one is outside what the Bible covers. "
            "Ask me anything about God, faith, or life and I will find the "
            "real verses for it.")
    return msg


def answer(question):
    direct = find_by_reference(question)
    skip = {v["ref"] for v in direct}
    ranked_idx, top1 = hybrid_search(question, k=TOP_K, skip=skip)
    # Honesty gate: nothing named, and the meaning match is weak -> the
    # Bible likely does not cover this. Say so instead of rambling.
    if not direct and top1 < OFFTOPIC_GATE:
        reply = honest_offtopic_reply(question, ranked_idx)
        refs = [VERSES[i]["ref"] for i in ranked_idx[:1]]
        return reply, refs
    picked = direct + [VERSES[i] for i in ranked_idx]
    msgs = build_messages(question, picked)
    text = _tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False)
    ids = _tok(text, return_tensors="pt")
    with GEN_LOCK, _torch.no_grad():
        out = _model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False)
    reply = _tok.decode(out[0][ids["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip()
    reply, _, _ = repair_references(reply)
    return reply, [v["ref"] for v in picked]


# ---------------- web server ----------------
def load_all():
    try:
        STATE["stage"] = "Loading the meaning search (quick)..."
        print("  " + STATE["stage"])
        sys.stdout.flush()
        load_embedder()

        STATE["stage"] = ("Loading Bibley's brain. First time downloads "
                          "about 1.5 GB, one time only...")
        print("  " + STATE["stage"])
        sys.stdout.flush()
        load_brain()

        STATE["stage"] = "Ready"
        STATE["ready"] = True
        print()
        print("  READY. Bibley 2.1 is listening at:")
        print(f"    http://127.0.0.1:{PORT}")
        print()
        print("  Keep this window open while you chat.")
        print("  Replies can take a minute or two on this Mac - that is normal.")
        sys.stdout.flush()
    except BaseException as e:  # show the real error, honest failure
        if STATE["error"] is None:
            STATE["error"] = f"{type(e).__name__}: {e}"
        STATE["stage"] = "Something went wrong."
        print("  ERROR:", STATE["error"])
        sys.stdout.flush()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(BASE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(200, json.dumps(STATE))
        elif self.path == "/api/history":
            self._send(200, json.dumps({"history": HISTORY}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/chat":
            self._send(404, json.dumps({"error": "not found"}))
            return
        if not STATE["ready"]:
            self._send(503, json.dumps({"error": "Bibley is still waking up. "
                                        "Watch the status line."}))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                self._send(400, json.dumps({"error": "empty question"}))
                return
            reply, refs = answer(prompt)
            HISTORY.append({"q": prompt, "a": reply, "verses": refs})
            self._send(200, json.dumps({"reply": reply, "verses": refs,
                                        "model": MODEL_ID}))
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
    t = threading.Thread(target=load_all, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{PORT}/"
    print(f"  Opening {url} in your browser...")
    sys.stdout.flush()
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n  Bye!")
