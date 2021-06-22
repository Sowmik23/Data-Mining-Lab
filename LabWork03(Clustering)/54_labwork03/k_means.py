import timeit
import numpy as np
from tabulate import tabulate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Cluster_viz:
    def __init__(self, data: np.ndarray):
        pca = PCA(n_components=2).fit(data)
        self.pca_data = pca.transform(data)

    def visualize_iteration(self, iteration, cluster_assignment):
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1], c=cluster_assignment)
        plt.title('Iteration ' + str(iteration))
        plt.show()


def load_dataset(file: str, exclude_cols: list, exclude_rows: list, sep=','):
    data_pt = np.genfromtxt(file, delimiter=sep)

    y_true = data_pt[:, exclude_cols[0]]

    data_pt = np.delete(data_pt, obj=exclude_rows, axis=0)
    data_pt = np.delete(data_pt, obj=exclude_cols, axis=0)
    data_pt = np.where(np.isnan(data_pt), np.ma.array(data_pt, mask=np.isnan(data_pt)).mean(axis=0), data_pt)


    print("Dataset Summary: ", file)
    print("Dataset Size: ", data_pt.shape[0])
    print("Instance Dimension: ", data_pt.shape[1])

    return data_pt, y_true.tolist()



def compute_dist(pt1: np.ndarray, pt2: np.ndarray):
    return np.sqrt(np.sum((pt1 - pt2) ** 2))


def init_centroid(data_pt: np.ndarray, k: int):  
    centroid_idx = np.random.randint(low=0, high=data_pt.shape[0], size=k)

    return centroid_idx


def cluster_assignment(data_pt: np.ndarray, data_idx: int, centroids: np.ndarray, clusters: dict):
    distance = dict()
    for centroid_idx in clusters.keys():
        distance[centroid_idx] = compute_dist(data_pt[data_idx], centroids[centroid_idx])

    min_idx = min(distance, key=distance.get)

    return min_idx


def update_centroid(data_pt: np.ndarray, clusters: dict):
    centroids_updated = list()

    for data_idx in clusters.values():
        cl_data = data_pt[data_idx]
        mean = np.mean(cl_data, axis=0)
        centroids_updated.append(mean)

    return centroids_updated



def k_means(data_pt: np.ndarray, k: int, visualize):
    centroids = data_pt[init_centroid(data_pt, k)]
    i = 0

    if visualize:
        viz = Cluster_viz(data_pt)

    global clstr

    while True:
        i = i + 1
        y_pred = list()

        _clusters = np.zeros(data_pt.shape[0])
        clusters = {label: [] for label in range(k)}
        for data_idx in range(data_pt.shape[0]):
            min_idx = cluster_assignment(data_pt, data_idx, centroids, clusters)

            y_pred.append(min_idx)

            clusters[min_idx].append(data_idx)
            _clusters[data_idx] = min_idx

        if np.array_equal(centroids, update_centroid(data_pt, clusters)):
            break

        centroids = update_centroid(data_pt, clusters)
        table = []

        for title, indices in clusters.items():
            table.append([title, len(indices)])

        print("\nIteration: ", i)
        print("[Cluster: Number of Members]")
        print(table)

        if visualize and i==1:
            viz.visualize_iteration(i, _clusters)
        clstr = _clusters

    print("Final cluster visualization: ")
    viz.visualize_iteration(i, clstr)

    return centroids, clusters, y_pred



if __name__ == '__main__':

    data_pt, _ = load_dataset('buddymove_holidayiq.csv', exclude_cols=[0, 21], exclude_rows=[0])
    k = 5

    start = timeit.default_timer()
    centroids, clusters, _ = k_means(data_pt, k, visualize=True)
    stop = timeit.default_timer()

    print('\nTotal Time Elapsed (Sec) :', stop - start)
    
