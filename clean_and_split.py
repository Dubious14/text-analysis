from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import List, Union

# ------------------------------------------------------------------
# 1. BASIC UNICODE-LEVEL CLEANING
# ------------------------------------------------------------------
_CTRL_CHARS = dict.fromkeys(c for c in range(32) if c not in (9, 10, 13))  # keep TAB/LF/CR


def _basic_clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)  # compose code-points
    text = text.translate(_CTRL_CHARS)  # strip ASCII control chars
    text = text.replace("\u00A0", " ")  # non-breaking space → normal space
    text = text.replace("\u200B", "")  # zero-width space
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()


# ------------------------------------------------------------------
# 2. SENTENCE SPLITTING
# ------------------------------------------------------------------
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")

# ------------------------------------------------------------------
# 3. TOKENIZER – letters + digits + apostrophes
# ------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:'[A-Za-z0-9]+)?", re.UNICODE)


# ------------------------------------------------------------------
# 4. PREPARE FROM FILE
# ------------------------------------------------------------------
def prepare_text_from_file(path: Union[str, Path], to_lower=True) -> List[List[str]]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    return prepare_text_from_string(raw, to_lower=to_lower)


# ------------------------------------------------------------------
# 5. PREPARE FROM RAW TEXT
# ------------------------------------------------------------------
def prepare_text_from_string(text: str, to_lower=True) -> List[List[str]]:
    text = _basic_clean(text)
    raw_sents = _SENT_RE.split(text)

    sentences: List[List[str]] = []
    for s in raw_sents:
        s = s.strip()
        if not s:
            continue
        if to_lower:
            s = s.lower()
        tokens = _TOKEN_RE.findall(s)
        if tokens:
            sentences.append(tokens)
    return sentences


# ------------------------------------------------------------------
# 6. Optional interactive file-picker
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import tkinter as tk
    from tkinter.filedialog import askopenfilename

    tk.Tk().withdraw()
    path_str = askopenfilename(
        title="Select a plain-text file",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if not path_str:
        sys.exit("⇨ No file selected, exiting.")

    sents = prepare_text_from_file(Path(path_str))
    print(f"\nLoaded: {path_str}")
    print(f"Sentences : {len(sents)}")
    print(f"Tokens    : {sum(len(s) for s in sents)}")
    print("Preview   :", " ".join(sents[0][:20]), "…")
