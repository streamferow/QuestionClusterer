from Data.Raw import raw_data
from Data.Processed import processed_data
from Data.Embedded import embedded_data
from Models.Umap import umap_reducing
from Models.Hdbscan import hdbscan_clustering
from sklearn.metrics.pairwise import cosine_distances
import numpy as np


path = "/Users/ivan/PycharmProjects/PostsFilter/Test/test.json"
texts = raw_data.load_text_from_json(path)
test_df = processed_data.texts_to_df(texts)
test_embeddings = embedded_data.texts_to_tensor(test_df)
test_reduced = umap_reducing.reduce_dimensionality(test_embeddings)
reduced = embedded_data.load_tensor("/Users/ivan/Desktop/final_reduced_embeddings.pt")
labels = hdbscan_clustering.cluster(reduced, 49, 9)


def assign_test_points_to_nearest_clusters(train_reduced, train_labels, test_reduced):
    assigned_labels = []
    for point in test_reduced:
        distances = cosine_distances(point.reshape(1, -1), train_reduced).flatten()
        nearest_index = np.argmin(distances)
        nearest_label = train_labels[nearest_index]
        assigned_labels.append(nearest_label)
    return assigned_labels

test_cluster_labels = assign_test_points_to_nearest_clusters(reduced, labels, test_reduced)
print(test_cluster_labels)