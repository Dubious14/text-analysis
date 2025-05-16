from __future__ import annotations
import numpy as np
import os

# Tkinter for file browsing (works on Windows if installed)
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog
except ImportError:
    tk = None  # If Tkinter isn't available, fallback or raise an error.

# Global variables
WORDS = None
VECTORS = None
WORD2IDX = None


def browse_and_convert_to_npz():
    if tk is None:
        raise RuntimeError("Tkinter not available. Install or switch to load_model(path) approach.")

    root = tk.Tk()
    root.withdraw()

    # 1) Browse for the text-based model
    input_file = filedialog.askopenfilename(
        title="Select text-based embedding file",
        filetypes=[("Text files", "*.txt *.dep *.*"), ("All files", "*.*")]
    )
    if not input_file:
        print("No file selected. Aborting.")
        return

    # 2) Ask for output .npz filename
    output_file = filedialog.asksaveasfilename(
        title="Save converted .npz file as",
        defaultextension=".npz",
        filetypes=[("NumPy Zip", "*.npz"), ("All files", "*.*")]
    )
    if not output_file:
        print("No output file chosen. Aborting.")
        return

    # 3) Convert text -> .npz
    #    We'll assume 1 word + 200 dims, but we'll ask user just in case:
    dim = simpledialog.askinteger("Embedding Dimension",
                                  "Number of floats per word? (default 200)",
                                  initialvalue=200, minvalue=1)
    if dim is None:
        dim = 200

    convert_text_to_npz(input_file, output_file, dim)
    print(f"Converted {input_file} -> {output_file}")

    # 4) Load that .npz globally
    load_model(output_file)


def convert_text_to_npz(input_file, output_file, embedding_dim=None):
    words_list = []
    vectors_list = []

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        # Read the first line as header: num_words dim
        first_line = f.readline().strip()
        parts = first_line.split()

        try:
            num_words = int(parts[0])
            dim = int(parts[1])
            if embedding_dim is None:
                embedding_dim = dim
        except (ValueError, IndexError):
            raise ValueError("Header line must be in format: <num_words> <embedding_dim>")

        # Read the rest of the file: word vec[0] vec[1] ... vec[dim-1]
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                continue  # skip malformed lines
            word = parts[0]
            try:
                floats = [float(x) for x in parts[1:]]
            except ValueError:
                continue  # skip if cannot convert
            words_list.append(word)
            vectors_list.append(floats)

    words_array = np.array(words_list, dtype=object)
    vectors_array = np.array(vectors_list, dtype=np.float32)

    np.savez_compressed(output_file, words=words_array, vectors=vectors_array)
    print(f"Converted {len(words_array)} embeddings with dim = {vectors_array.shape[1]}")


def load_model(model_path):
    global WORDS, VECTORS, WORD2IDX

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    data = np.load(model_path, allow_pickle=True)
    WORDS = data["words"]  # shape: (num_words,)
    VECTORS = data["vectors"]  # shape: (num_words, embedding_dim)
    WORD2IDX = {w: i for i, w in enumerate(WORDS)}

    print(f"Loaded {len(WORDS)} words, dimension {VECTORS.shape[1]}.")


def get_vector(word):
    """
    Return the embedding vector for 'word', or None if not in vocab.
    """
    if WORD2IDX is None:
        print("Model not loaded. Call browse_and_convert_to_npz() or load_model(path).")
        return None
    idx = WORD2IDX.get(word)
    if idx is None:
        return None
    return VECTORS[idx]


def most_similar(query_word, top_n=10):
    """
    Return list of (similarity, word) for top N words by cosine similarity.
    """
    import math
    if WORD2IDX is None:
        print("Model not loaded. Call browse_and_convert_to_npz() or load_model(path).")
        return []

    qvec = get_vector(query_word)
    if qvec is None:
        return []

    qnorm = np.linalg.norm(qvec)
    sims = []
    for w, i in WORD2IDX.items():
        if w == query_word:
            continue
        wvec = VECTORS[i]
        sim = np.dot(qvec, wvec) / (qnorm * np.linalg.norm(wvec))
        sims.append((sim, w))
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:top_n]


if __name__ == "__main__":

    if tk is None:
        print("Tkinter not available. Please use load_model(path) manually.")
    else:
        browse_and_convert_to_npz()
        if WORD2IDX is not None:
            test_word = input("Enter a test word: ").strip()
            top_sims = most_similar(test_word, top_n=5)
            if top_sims:
                print(f"\nTop 5 words similar to '{test_word}':")
                for score, w in top_sims:
                    print(f"{w:20s}  similarity: {score:.4f}")
            else:
                print(f"Word '{test_word}' not found or no model loaded.")
