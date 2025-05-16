#!/usr/bin/env python3
"""
umap_visualize.py

A script to visualize word embeddings using UMAP.
It loads a .npz model (with arrays "words" and "vectors"),
reduces the embedding dimensions to 2D or 3D,
and plots the projection with annotated words.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import umap.umap_ as umap


def load_npz_model(model_path):
    """
    Load the .npz model file containing arrays 'words' and 'vectors'.
    """
    data = np.load(model_path, allow_pickle=True)
    words = data["words"]
    vectors = data["vectors"]
    return words, vectors


def reduce_dimensions(vectors, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Use UMAP to reduce high-dimensional vectors to n_components dimensions.
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='cosine')
    embedding = reducer.fit_transform(vectors)
    return embedding


def plot_2d(embedding, words, annotate_indices):
    """
    Plot a 2D scatter plot of the UMAP projection with annotations.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.7)
    # Annotate selected points if any
    for i in annotate_indices:
        plt.annotate(words[i], (embedding[i, 0], embedding[i, 1]), fontsize=8)
    plt.title("2D UMAP Projection of Word Embeddings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.show()


def plot_3d(embedding, words, annotate_indices):
    """
    Plot a 3D scatter plot of the UMAP projection with annotations.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5, alpha=0.7)
    for i in annotate_indices:
        ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2], words[i], fontsize=8)
    ax.set_title("3D UMAP Projection of Word Embeddings")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize word embeddings from a .npz model using UMAP."
    )
    parser.add_argument("model_path", help="Path to the .npz model file (with words and vectors).")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2,
                        help="Projection dimensions: 2 for 2D (default) or 3 for 3D.")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP parameter: number of neighbors (default: 15).")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP parameter: minimum distance between points (default: 0.1).")
    parser.add_argument("--annotate", type=int, default=100,
                        help="Default number of points to annotate if no words are specified (default: 100).")
    args = parser.parse_args()

    # Load model data
    words, vectors = load_npz_model(args.model_path)
    print(f"Loaded model with {len(words)} words and vector dimension {vectors.shape[1]}.")

    # Ask the user which words to annotate
    user_input = input("Enter comma-separated words to annotate (or leave blank for default sampling): ").strip()

    # Convert words array to a list for easier indexing/search
    words_list = list(words)
    annotate_indices = []

    if user_input:
        chosen_words = [word.strip() for word in user_input.split(",") if word.strip()]
        missing_words = []
        for word in chosen_words:
            try:
                idx = words_list.index(word)
                annotate_indices.append(idx)
            except ValueError:
                missing_words.append(word)
        if missing_words:
            print("The following words were not found in the model and will be skipped:", ", ".join(missing_words))
        if not annotate_indices:
            print("No valid words provided for annotation. Proceeding without annotations.")
    else:
        # Default behavior: evenly sample indices for annotation.
        num_points = len(words)
        if num_points > args.annotate:
            annotate_indices = np.linspace(0, num_points - 1, args.annotate, dtype=int).tolist()
        else:
            annotate_indices = list(range(num_points))

    # Run UMAP dimensionality reduction
    embedding = reduce_dimensions(vectors,
                                  n_components=args.dim,
                                  n_neighbors=args.n_neighbors,
                                  min_dist=args.min_dist)

    # Plot based on the chosen dimension
    if args.dim == 2:
        plot_2d(embedding, words_list, annotate_indices)
    else:
        plot_3d(embedding, words_list, annotate_indices)


if __name__ == "__main__":
    main()
