import timeit
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score

from k_means import k_means, compute_dist
from k_medoids import k_medoids, load_dataset


def sum_squared_error(data_pt: np.ndarray, centroids: list, clusters: dict):
    dist_sq_list = list()

    for cluster_idx, data_idx in clusters.items():
        cluster_data = data_pt[data_idx]

        for data in cluster_data:
            dist = compute_dist(data, centroids[cluster_idx])
            dist_sq_list.append(np.square(dist))

    return sum(dist_sq_list)


def elbow_method(data_pt: np.ndarray):
    n = data_pt.shape[0]

    x_val = list()
    y_val = list()

    for k in range(2, 11):
        centroids, clusters = k_means(data_pt, k)
        within_cluster_variance = sum_squared_error(data_pt, centroids, clusters)

        x_val.append(k)
        y_val.append(within_cluster_variance)

    plt.plot(x_val, y_val, color='g')
    plt.plot(x_val, y_val, 'or')
    plt.title('Dataset : Weather Madrid (1997 - 2015)')
    plt.xlabel('Value of K')
    plt.ylabel('Sum Squared within cluster variance')
    plt.show()

    return dict(zip(x_val, y_val))


def elbow_method_kmedoid(data_pt: np.ndarray):
    n = data_pt.shape[0]

    x_val = list()
    y_val = list()

    for k in range(2, 11):
        within_cluster_variance, _ = k_medoids(data_pt, k, use_abs_error=False)

        x_val.append(k)
        y_val.append(within_cluster_variance)

    plt.plot(x_val, y_val, color='b')
    plt.plot(x_val, y_val, '^r')
    plt.title('Elbow Method : K - Medoids')

    plt.xlabel('Value of K')
    plt.ylabel('Sum Squared within cluster variance')

    plt.show()

    return dict(zip(x_val, y_val))


def time_comparison_graph(data_pt: np.ndarray, y_true: list):
    kmeans_time = list()
    kmedoids_time = list()
    x_val = list()

    eval = list()

    for k in range(2, 11):
        x_val.append(k)

        start_kmeans = timeit.default_timer()
        # _, _, _ = k_means(data_pt, k=k, visualize=False)
        stop_kmeans = timeit.default_timer()

        elapsed_time_kmeans = stop_kmeans - start_kmeans
        kmeans_time.append(elapsed_time_kmeans)

        start_kmedoids = timeit.default_timer()
        # _, _ = k_medoids(data_pt, k=k, use_abs_error=False)
        stop_kmedoids = timeit.default_timer()

        elapsed_time_kmedoids = stop_kmedoids - start_kmedoids
        kmedoids_time.append(elapsed_time_kmedoids)

        eval.append(cluster_eval(data_pt, k, y_true, purity=True))

    # print(eval)

    # print(kmeans_time)
    # print(kmedoids_time)

    plt.plot(x_val, eval, color='g', label='Silhouette Score')
    plt.plot(x_val, eval, 'or')

    # plt.plot(x_val, kmedoids_time, color='b', label='K - Medoids')
    # plt.plot(x_val, kmedoids_time, '^r')

    plt.xlabel('Value of K')
    plt.ylabel('Purity')

    # plt.legend()
    plt.title('Dataset : Wine')

    plt.savefig('plots/eval_wine.png')
    plt.show()


def cluster_eval(data_pt: np.ndarray, k: int, y_true: list, purity=False):
    centroids, clusters, y_pred = k_means(data_pt, k, visualize=True)

    X = []
    labels = []

    for label, clusters in clusters.items():
        labels += ([label] * len(clusters))
        X.append(data_pt[clusters])

    X = np.vstack(X)

    s = silhouette_score(X, labels)
    print('\nSilhouette Score :', s)

    if purity:
        p = purity_score(y_true, y_pred)
        print('Purity :', purity_score(y_true, y_pred))

        return p

    return s


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == '__main__':
    data_pt, y_true = load_dataset('wine.data', exclude_cols=[0], exclude_rows=[])

    _elbow_data = elbow_method(data_pt)
    # elbow_method_kmedoid(data_pt)

    # time_comparison_graph(data_pt, y_true)

    # k = 3
    # print(y_true)
    # _ = cluster_eval(data_pt, k, y_true, purity=True)