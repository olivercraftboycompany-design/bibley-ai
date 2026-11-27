"""Build Bibley's training set from the bundled KJV + topical index.

Every verse quote in the output is copied from verses.json, so the model only
ever learns exact KJV wording. Output: training/data/train.jsonl and val.jsonl,
each line {"messages": [...]} in chat format, matching the prompt server.py uses.

Task mix:
  lookup   - "What does John 3:16 say?"  -> exact verse text with its reference
  locate   - "Where does the Bible say '...'?" -> the reference
  topic    - server.py-style RAG prompt with real retrieved verses -> quotes + plain words
  honest   - RAG prompt whose verses don't fit the question -> says so honestly
  identity - who Bibley is and what it's built on (license honesty)
"""
import json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data")
random.seed(7)

# Keep in sync with server.py
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

def verse_block(vs):
    return "\n".join(f"{v['ref']}: {v['text']}" for v in vs)

def rag_prompt(q, vs):
    return (f"Question: {q}\n\nReal King James Bible verses found for this question:\n"
            f"{verse_block(vs)}\n\n{INSTRUCTIONS}")

def quote_lines(vs):
    return "\n".join(f"{v['ref']}: \"{v['text']}\"" for v in vs)

def msg(user, assistant, system=SYSTEM):
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}

VERSES = json.load(open(os.path.join(ROOT, "verses.json")))
TOPICS = json.load(open(os.path.join(ROOT, "topics_index.json")))

LOOKUP_Q = ["What does {r} say?", "Quote {r}.", "Can you look up {r} for me?",
            "Show me {r} in the KJV.", "{r}", "what's {r}"]
LOCATE_Q = ["Where in the Bible does it say \"{t}\"?", "What verse says \"{t}\"?",
            "Find the reference for: {t}"]
TOPIC_Q = ["What does the Bible say about {t}?", "Bible verses about {t}?",
           "What does Scripture teach on {t}?", "I'm struggling with {t}. What does the Bible say?"]

def topic_name(label):
    # "Vanity: A consequence of the fall" -> ("vanity", "a consequence of the fall")
    head, _, aspect = label.partition(":")
    return head.strip().lower(), aspect.strip().lower()

def lookup_examples(n):
    out = []
    for v in random.sample(VERSES, n):
        q = random.choice(LOOKUP_Q).format(r=v["ref"])
        a = f"{v['ref']} (KJV): \"{v['text']}\""
        out.append(msg(q, a))
    return out

def locate_examples(n):
    out = []
    cands = [v for v in VERSES if 6 <= len(v["text"].split()) <= 40]
    for v in random.sample(cands, n):
        words = v["text"].split()
        # use a fragment so the model learns to recognise partial quotes
        start = random.randint(0, max(0, len(words) - 8))
        frag = " ".join(words[start:start + random.randint(6, 12)]).strip(",;:.")
        q = random.choice(LOCATE_Q).format(t=frag)
        a = f"That's {v['ref']}. In the King James Version it reads: \"{v['text']}\""
        out.append(msg(q, a))
    return out

def topic_examples(n):
    out = []
    usable = [t for t in TOPICS if 2 <= len(t["verses"]) <= 12 and ":" in t["label"]]
    for t in random.sample(usable, min(n, len(usable))):
        idx = random.sample(t["verses"], min(3, len(t["verses"])))
        vs = [VERSES[i] for i in sorted(idx)]
        head, aspect = topic_name(t["label"])
        q = random.choice(TOPIC_Q).format(t=head)
        # retrieved list = the relevant verses + 1-2 distractors, shuffled
        shown = vs + random.sample(VERSES, random.randint(1, 2))
        random.shuffle(shown)
        refs = ", ".join(v["ref"] for v in vs)
        plain = (f"In plain words: {refs} {'speak' if len(vs) > 1 else 'speaks'} to {head}"
                 + (f", especially {aspect}" if aspect else "")
                 + ". Read them slowly in context, and let them guide how you think about this.")
        out.append(msg(rag_prompt(q, shown), quote_lines(vs) + "\n\n" + plain))
    return out

OFFTOPIC_Q = ["Who won the Super Bowl last year?", "How do I fix my wifi?",
              "What's the best phone to buy?", "Can you do my math homework: 34*17?",
              "What's the weather tomorrow?", "Write me a rap about cars."]

def honest_examples(n):
    out = []
    for _ in range(n):
        q = random.choice(OFFTOPIC_Q)
        shown = random.sample(VERSES, 3)
        a = ("The verses I found don't really answer that question, so I won't pretend they do. "
             "I'm a Bible helper, so I'm best at questions about Scripture. "
             "Try asking me what the Bible says about a topic, or give me a verse reference to look up.")
        out.append(msg(rag_prompt(q, shown), a))
    return out

def identity_examples(base_name):
    qs = ["Who are you?", "What model are you?", "Who made you?", "Are you ChatGPT?"]
    a = (f"I'm Bibley, a Bible helper built by Oliver. I'm a fine-tuned version of the open "
         f"{base_name} model (Apache 2.0), trained on the public-domain King James Bible. "
         "I quote verses from the real KJV text so you can check them yourself.")
    return [msg(q, a) for q in qs]

def main():
    base_name = sys.argv[1] if len(sys.argv) > 1 else "small open-source"
    rows = (lookup_examples(6000) + locate_examples(3000) + topic_examples(6000)
            + honest_examples(400) + identity_examples(base_name) * 25)
    random.shuffle(rows)
    os.makedirs(OUT, exist_ok=True)
    n_val = 300
    with open(os.path.join(OUT, "val.jsonl"), "w") as f:
        for r in rows[:n_val]: f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "train.jsonl"), "w") as f:
        for r in rows[n_val:]: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows) - n_val} train / {n_val} val examples to {OUT}")

if __name__ == "__main__":
    main()
