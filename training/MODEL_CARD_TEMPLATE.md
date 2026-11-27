---
license: apache-2.0
base_model: {{BASE}}
language: en
tags: [bible, kjv, bibley]
---
# {{BIBLEY_NAME}}

A Bibley model by Oliver Holder: a small Bible helper ({{PARAMS}} parameters) that looks up and quotes King James Version verses word for word and explains them in plain words.

## Built on
Fine-tuned from [{{BASE}}](https://huggingface.co/{{BASE}}), released under the Apache License 2.0. This model is a modified version of that work (LoRA fine-tune, merged). See `LICENSE` and `NOTICE`.

## Training data
- King James Version text (public domain in the United States), 31,102 verses from Bibley's `verses.json`.
- Topic-to-verse mappings derived from Nave's Topical Bible and Torrey's New Topical Textbook (public domain), via the NEUU Bible Topics Dataset (CC BY 4.0).
- Task types: verse lookup by reference, finding a reference from a quote, topic answers that quote retrieved verses, and honest "these verses don't answer that" replies.

## Use it with verse lookup
Run it inside Bibley's server, which retrieves real verses first and repairs any misquoted reference. On its own, a small model can still misquote or invent; always check references.

## Limits
Small model: can be wrong, can misread questions, and is not a pastor or an authority. English only. KJV only.
