#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import random
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import textstat
import tkinter as tk
from tkinter import filedialog

# Initialize Tkinter and hide root window
root = tk.Tk()
root.withdraw()

# File selection dialog (Ubuntu & PyCharm compatible)
PROCESSED_TEXT_FILE = filedialog.askopenfilename(title="Select Processed Sentences File",
                                                 filetypes=[("Text files", "*.txt")])

# Check if the user selected a file
if not PROCESSED_TEXT_FILE:
    print("Error: No file selected. Exiting.")
    exit()

SAMPLE_SIZE = 500000  # Number of sentences to randomly sample for training


def load_sampled_texts():
    """
    Randomly samples sentences from the selected text file.
    Returns: List of tokenized sentences.
    """
    if not os.path.exists(PROCESSED_TEXT_FILE):
        print(f"Error: '{PROCESSED_TEXT_FILE}' not found.")
        exit()

    # Read and randomly sample lines from the large dataset
    print(f"Sampling {SAMPLE_SIZE} sentences from {PROCESSED_TEXT_FILE}...")
    with open(PROCESSED_TEXT_FILE, "r", encoding="utf-8") as f:
        all_sentences = f.readlines()

    sampled_sentences = random.sample(all_sentences, min(SAMPLE_SIZE, len(all_sentences)))
    tokenized_sentences = [sentence.strip().split() for sentence in sampled_sentences]

    return tokenized_sentences


# Compute FleschComplex score (G(t))
def compute_flesch_complexity(text):
    """
    Computes FleschComplex score: 100 - FRE (higher = more complex).
    """
    fre_score = textstat.flesch_reading_ease(text)
    return 100 - fre_score  # Higher score = more complex


# Compute E_θ(t) using the mean vector for OOV words
def compute_embedding_complexity(text, model):
    """
    Computes E_θ(t) = Var[cosine similarity] * Mean[cosine similarity]
    for consecutive word pairs, handling OOV words using mean vector.
    """
    words = [word.lower() for word in text.split()]

    # Get valid words (ones that exist in Word2Vec)
    valid_words = [word for word in words if word in model.wv]

    # If too few valid words, return NaN
    if len(valid_words) < 2:
        return np.nan

        # Compute the mean vector of all known words
    mean_vector = np.mean([model.wv[word] for word in valid_words], axis=0)

    # Convert words to vectors, replacing OOV words with mean vector
    vectors = np.array([model.wv[word] if word in model.wv else mean_vector for word in words])

    # Compute pairwise cosine similarities for consecutive words
    cos_sims = [cosine_similarity(vectors[i].reshape(1, -1), vectors[i + 1].reshape(1, -1))[0][0]
                for i in range(len(vectors) - 1)]

    # Compute variance and mean of cosine similarities
    var_cosine = np.var(cos_sims)
    mean_cosine = np.mean(cos_sims)

    # Compute E_θ(t)
    embedding_complexity = var_cosine * mean_cosine

    return embedding_complexity


# Train Word2Vec with given parameters
def train_word2vec(sentences, vector_size, window, min_count, negative):
    """
    Trains a Word2Vec model with given parameters and returns it.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        workers=4
    )
    return model


# Objective function for Bayesian Optimization
@use_named_args([
    Integer(200, 300, name='vector_size'),
    Integer(3, 10, name='window'),
    Integer(5, 10, name='min_count'),
    Integer(5, 10, name='negative')
])
def objective(vector_size, window, min_count, negative):
    """
    Objective function for Bayesian optimization:
    - Trains Word2Vec with given parameters.
    - Computes E_θ(t).
    - Maximizes correlation with G(t).
    """
    print(
        f"\nTraining Word2Vec with vector_size={vector_size}, window={window}, min_count={min_count}, negative={negative}...")

    # Load sampled data for training
    sampled_texts = load_sampled_texts()

    # Train Word2Vec with selected parameters
    model = train_word2vec(sampled_texts, vector_size, window, min_count, negative)

    # Compute complexity scores
    complexity_data = []
    for text in sampled_texts[:5000]:  # Compute on a subset for efficiency
        raw_text = " ".join(text)
        flesch_score = compute_flesch_complexity(raw_text)
        embedding_score = compute_embedding_complexity(raw_text, model)
        complexity_data.append([flesch_score, embedding_score])

    # Convert to DataFrame
    df = pd.DataFrame(complexity_data, columns=["FleschComplex", "E_theta"])

    # Drop NaN values
    df.dropna(inplace=True)

    # Compute Spearman Correlation
    correlation, _ = spearmanr(df["FleschComplex"], df["E_theta"])

    # Store best model if correlation improves
    global best_correlation, best_model, best_params
    if correlation > best_correlation:
        best_correlation = correlation
        best_model = model
        best_params = (vector_size, window, min_count, negative)
        model.save("best_word2vec.model")

    print(f"Correlation: {correlation:.4f}")
    return -correlation  # We minimize negative correlation to maximize it


# Main Optimization Loop
if __name__ == "__main__":
    # Initialize best correlation tracking
    best_correlation = -1.0
    best_model = None
    best_params = None

    # Define Bayesian Optimization search space
    search_space = [
        Integer(200, 300, name='vector_size'),
        Integer(3, 10, name='window'),
        Integer(5, 10, name='min_count'),
        Integer(5, 10, name='negative')
    ]

    # Run Bayesian Optimization
    print("\nStarting Bayesian Optimization...")
    result = gp_minimize(objective, search_space, n_calls=15, random_state=14)

    # Print Best Found Parameters
    print("\nBest parameters found:")
    print(
        f"Vector Size: {best_params[0]}, Window: {best_params[1]}, Min Count: {best_params[2]}, Negative: {best_params[3]}")
    print(f"Best Correlation Achieved: {best_correlation:.4f}")

    print("\nOptimization complete! Best model saved as 'best_word2vec.model'.")
