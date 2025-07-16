from Data.Embedded import embedded_data
from sklearn.manifold import trustworthiness
import heapq, itertools, umap
import numpy as np, matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
final_embeddings = embedded_data.load_tensor("/Users/ivan/Desktop/final_embeddings.pt")


def umap_grid_search(embeddings, top_n):
    umap_params = {
        'n_components': [6],
        'n_neighbors': list(range(3, 22)),
        'min_dist': np.arange(0.05, 0.75, 0.05)
    }

    results = []
    for nc, nn, md in itertools.product(
            umap_params["n_components"],
            umap_params["n_neighbors"],
            umap_params["min_dist"]
    ):
        reducer = umap.UMAP(
            n_neighbors=nn,
            min_dist=md,
            n_components=nc,
            random_state=111,
            metric="cosine"
        )
        reduced = reducer.fit_transform(embeddings)
        score = round(trustworthiness(X=embeddings, X_embedded=reduced, n_neighbors=nn), 4)
        results.append((score, {
            'n_components': nc,
            'n_neighbors': nn,
            'min_dist': md
        }))

    top_results = heapq.nlargest(top_n, results, key=lambda x: x[0])
    return top_results


def reduce_dimensionality(embeddings: np.ndarray, n_components = 5, n_neighbors = 6, min_dist=0.05) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='cosine', random_state=111)
    return reducer.fit_transform(embeddings).astype(np.float64)


def plot_reduced_embeddings(reduced_embeddings):
    plt.figure(figsize=(10,7))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=15, c='blue')
    plt.title(f"UMAP проекция")
    plt.show()


# reduced = reduce_dimensionality(final_embeddings)
# embedded_data.save_tensor("/Users/ivan/Desktop/final_reduced_embeddings.pt", reduced)
# plot_reduced_embeddings(reduced)
