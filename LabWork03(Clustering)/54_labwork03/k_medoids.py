import timeit
from copy import deepcopy

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
    data_pt = np.delete(data_pt, obj=exclude_cols, axis=1)
    data_pt = np.where(np.isnan(data_pt), np.ma.array(data_pt, mask=np.isnan(data_pt)).mean(axis=0), data_pt)

    print("Dataset Summary: ", file)
    print("Dataset Size: ", data_pt.shape[0])
    print("Instance Dimension: ", data_pt.shape[1])

    return data_pt, y_true.tolist()


def get_distance(pt1: np.ndarray, pt2: np.ndarray, manhattan=True):
    if manhattan:
        return np.sum(np.abs(pt1 - pt2), axis=-1)
    return np.sqrt(np.sum((pt1 - pt2) ** 2, axis=-1))


def init_meloid(data_pt: np.ndarray, k: int):
    centroid_idx = np.random.randint(low=0, high=data_pt.shape[0] + 1, size=k)

    return centroid_idx


def get_cluster_assignment_with_cost(data: np.ndarray, k: int, medoid_idx, use_abs_error):
    medoids = data[medoid_idx]
    dist = np.array([get_distance(data, m, use_abs_error) for m in medoids])
    cluster_assignment = np.argmin(dist, axis=0)
    min_dist = dist[cluster_assignment, np.arange(dist.shape[1])]
    cost = np.sum(min_dist)
    return cluster_assignment, cost


def k_medoids(data: np.ndarray, k: int, max_iter=20, clara=True, sampling=10, use_abs_error=True, visualize=False):
    n_sample, n_feat = data.shape
    medoid_idx = init_meloid(data, k)

    if visualize:
        viz = Cluster_viz(data)

    cluster_assignment, old_cost = get_cluster_assignment_with_cost(data, k, medoid_idx, use_abs_error)
    print("\nInitial: ", "\nmedoids: ", medoid_idx, "\ncost: ", old_cost)
    if visualize:
        viz.visualize_iteration(1, cluster_assignment)

    for _it in range(max_iter):
        swap_flag = False
        if clara:
            samples = np.random.randint(low=0, high=n_sample, size=n_sample // sampling)
        else:
            samples = range(n_sample)
        for n in samples:
            if n in medoid_idx:
                continue
            for swap_pos in range(k):
                new_medoid_idx = deepcopy(medoid_idx)
                new_medoid_idx[swap_pos] = n
                _new_cluster_assignment, new_cost = get_cluster_assignment_with_cost(data, k, new_medoid_idx, use_abs_error)

                if new_cost < old_cost:
                    swap_flag = True
                    medoid_idx = new_medoid_idx
                    old_cost = new_cost
                    cluster_assignment = _new_cluster_assignment
        if swap_flag is False:
            cl, member_count = np.unique(cluster_assignment, return_counts=True)
            table = [[cl[i], member_count[i]] for i in range(len(cl))]
            print(tabulate(table, headers=["Cluster", "Number of Members"], tablefmt="fancy_grid"))
            print(_it)
            viz.visualize_iteration(_it+1, cluster_assignment)
            
            return old_cost, medoid_idx
        
        print("\nIteration: ", _it + 1, "\nmedoids: ", medoid_idx, "\ncost: ", old_cost)


if __name__ == '__main__':
    data_pt, _ = load_dataset('weather_madrid_LEMD_1997_2015.csv', exclude_cols=[0, 21], exclude_rows=[0])

    k = 5
    start = timeit.default_timer()
    k_medoids(data_pt, k, visualize=True, sampling=10)
    stop = timeit.default_timer()

    print('\nTotal Time Elapsed (Sec) :', stop - start)
