# Bibley 2.1

Oliver's Bibley AI project: a local Bible assistant with a chat page, KJV passage lookup and a topical index.

## Start on a Mac

1. Download this repository and extract it into a regular folder.
2. Install Python 3.10 if it is not already available. The retained dependency versions target older Python environments; this is not tested on every Python or macOS release.
3. Right-click `START.command` and choose Open, or run `bash START.command` from Terminal in this folder.
4. Keep Terminal open while using the local chat page at `http://127.0.0.1:8741`.

First setup installs the packages in `requirements.txt` and downloads Qwen3-0.6B and the MiniLM search model from Hugging Face. Allow approximately 2 GB of downloads plus package installation and additional disk space. The KJV and search-index data are included here. The topic embedding parts are joined locally on first launch. No API key or paid inference service is required. Your device, internet connection and upstream services have their own terms and costs.

The launcher uses `python3`; if that is not Python 3.10, create a Python 3.10 virtual environment manually and install `requirements.txt` before running `server.py`. The original launcher is retained apart from its help text.

## What is included

- Local Python inference server and browser chat UI.
- All 31,102 KJV verses in `verses.json`.
- Meaning-based retrieval, BM25 keyword retrieval and a topical index.
- Reference-repair logic that tries to match generated quotations to the verse library.
- Bundled verse embeddings and two topical-embedding parts, with checksums validated at startup.

The base model is **Qwen/Qwen3-0.6B**, by the Qwen team. It is downloaded from its official model repository, not rebranded or redistributed here as a new set of trained weights. The search encoder is **sentence-transformers/all-MiniLM-L6-v2**. This Bibley release has no extra selected Bible-training adapter. Experimental training outputs are not included.

## Limits and privacy

Bibley can select an irrelevant passage, misread a question, produce an incorrect explanation or fail to repair a quotation. Do not treat its output as verified Scripture or authoritative advice. Check the supplied references yourself. Biblical Hebrew and Greek translation quality have not been tested. Reference parsing currently recognizes `Psalms 23:1` but misses singular `Psalm 23:1`; use the exact Bible book name from the dataset.

This release packages the retained 2.1 project. Publication does not establish that the complete model setup runs successfully on your particular Mac. Model generation has not been rerun as part of this packaging check. The browser/WebLLM experiment and the full Bible-reader website are separate projects and are not included here.

Setup contacts Python package servers and Hugging Face. The original server can also re-download missing search data from its existing Drive links. After all dependencies and model files are cached, inference is intended to run locally. The server binds only to `127.0.0.1`. Chat history is kept in process memory, not synced between devices, and is lost when the server stops. This is not a hosted chat service or a GitHub Pages inference server.

## Credits and rights

- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- MiniLM search encoder: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Bible Topics Dataset by NEUU, used under CC BY 4.0: https://github.com/neuu-org/bible-topics-dataset
- Topical source works: Nave's Topical Bible (1896), by Orville J. Nave, and Torrey's New Topical Textbook (1897), by R. A. Torrey. Digitisation credit: Christian Classics Ethereal Library, https://www.ccel.org/.

The topic data have been transformed into Bibley labels, verse-row mappings and search embeddings. The modern dataset contributions retain their CC BY 4.0 terms; the original source works are public domain. KJV text is treated as public domain in the United States; rights may differ elsewhere. See `THIRD_PARTY_NOTICES.md` and `third_party/` for notices and license material. No blanket license is asserted over components owned by others. Publication of Bibley's own code does not add a new general license grant.
