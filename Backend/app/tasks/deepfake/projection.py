"""Visualization-only PCA / t-SNE / UMAP reduction of an embedding matrix.

Deliberately its own module rather than a reuse of Speaker Verification's
`project_batch_embeddings`: that one validates against a speaker model's fixed
embedding dimension, and this task's three detectors each have their own.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_REDUCTION_METHODS = {"pca", "tsne", "umap"}
SUPPORTED_PROJECTION_COMPONENTS = {2, 3}


def reduce_embedding_matrix(
    matrix: np.ndarray,
    reduction_method: str,
    n_components: int,
) -> tuple[np.ndarray, int, str]:
    """Reduce an (n_samples, n_features) matrix to `n_components` columns.

    Always returns exactly `n_components` columns: PCA (and PCA-as-UMAP-fallback)
    cannot manufacture more components than a very small batch allows (e.g. 2
    recordings requesting 3D), so the shortfall is deterministically
    zero-padded and the achievable count is reported separately. t-SNE and
    UMAP never need padding here — for tiny batches they run with
    `init="random"` instead of their density/spectral-based default init,
    which removes their sample-count ceiling entirely (verified empirically
    against sklearn 1.9 / umap-learn 0.5.12: the default `init="pca"` for
    t-SNE and `init="spectral"` for UMAP both fail below a measured sample
    threshold, independent of any `perplexity`/`n_neighbors` clamping).

    Returns (coordinates, effective_components, reduction_method_used).
    """

    n_samples, n_features = matrix.shape

    method_used = reduction_method
    if reduction_method == "umap" and n_samples < 3:
        # UMAP requires n_neighbors >= 2, which requires >= 3 samples under
        # any initialization strategy — deterministic, documented fallback.
        method_used = "pca"
    elif reduction_method == "tsne" and n_samples < 2:
        # t-SNE cannot define a positive perplexity (n_samples - 1) for one sample.
        method_used = "pca"

    try:
        if method_used == "pca":
            from sklearn.decomposition import PCA

            achievable = min(n_samples, n_features, n_components)
            coords = PCA(n_components=achievable, random_state=42).fit_transform(matrix)
        elif method_used == "tsne":
            from sklearn.manifold import TSNE

            perplexity = min(30, n_samples - 1)
            init = "random" if n_samples < n_components else "pca"
            coords = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=perplexity,
                init=init,
            ).fit_transform(matrix)
            achievable = n_components
        else:  # umap, n_samples >= 3
            import umap

            n_neighbors = max(2, min(15, n_samples - 1))
            init = "random" if n_samples < n_components + 2 else "spectral"
            coords = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=n_neighbors,
                init=init,
            ).fit_transform(matrix)
            achievable = n_components
    except Exception as error:
        # Normalize any library-internal exception (e.g. UMAP's TypeError on
        # spectral init for small batches) into a ValueError, so callers get
        # one consistent failure type regardless of which reducer ran.
        raise ValueError(f"Projection failed: {error}") from error

    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] < n_components:
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])

    return coords, min(achievable, n_components), method_used
