import numpy as np
from sklearn.decomposition import PCA


def pca_entropy(embeddings, n_components=None):
    d = embeddings.shape[1]
    k = n_components or d
    k = min(k, d, embeddings.shape[0])

    if k < 2:
        return np.nan

    pca = PCA(n_components=k, svd_solver="full")
    pca.fit(embeddings)
    lambdas = pca.explained_variance_ratio_
    entropy = -np.sum(lambdas * np.log2(lambdas))
    return entropy


def weighted_mahalanobis(embeddings, alpha=2, eps=1e-5):
    mu = embeddings.mean(axis=0)
    cov = np.cov(embeddings.T)
    cov += eps * np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov)

    diffs = embeddings - mu
    dists = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))
    return np.mean(dists ** alpha)


def mahalanobis_spike_entropy(embeddings, num_bins=10, eps=1e-5):
    mu = embeddings.mean(axis=0)
    cov = np.cov(embeddings.T)
    cov += eps * np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov)

    diffs = embeddings - mu
    dists = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))

    bins = min(num_bins, max(2, embeddings.shape[0]))
    hist, _ = np.histogram(dists, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return np.nan
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def rarity_score(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    rarities = 1.0 / (norms + 1e-8)
    return np.mean(rarities)


def covariance_trace(embeddings):
    cov = np.cov(embeddings.T)
    trace_val = np.trace(cov)
    return trace_val


def compute_all_metrics(embeddings):
    return {
        "pca_entropy": pca_entropy(embeddings),
        "weighted_mahalanobis": weighted_mahalanobis(embeddings, alpha=2),
        "covariance_trace": covariance_trace(embeddings),
        "mahalanobis_spike_entropy": mahalanobis_spike_entropy(embeddings),
        "rarity_mean": rarity_score(embeddings)
    }
