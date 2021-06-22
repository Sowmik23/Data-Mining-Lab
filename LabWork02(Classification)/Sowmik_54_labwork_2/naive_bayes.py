import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


file1 = open("naive_output.txt", "a") 

def load_dataset(file: str, class_label_column):
    dataset = pd.read_csv(file, header=None)

    
    print("\nDataset Summary: ", file, "\nDataset Size: ", dataset.shape[0], "\nNumber of Attributes: ", dataset.shape[1]-1)
    
    file1.write("Dataset Summary: ")
    file1.write(file)
    file1.write("\n")

    file1.write("Dataset Size: ")
    file1.write(str(dataset.shape[0]))
    file1.write("\n") 

    file1.write("Number of Attributes: ")
    file1.write(str(dataset.shape[1]-1))
    file1.write("\n")


    feature_columns = list(dataset.columns.values)


    feature_columns = [x for x in feature_columns if x != class_label_column]
    
    X = dataset.iloc[:, feature_columns].values
    y = dataset.iloc[:, class_label_column].values

    return X, y


def clf_report(y_true, y_pred):
    support = {}
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            if y_true[i] not in support:
                support[y_true[i]] = 0
            else:
                support[y_true[i]] += 1

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    for label, sup_count in support.items():
        precision = sup_count / len(y_pred[y_pred == label])
        recall = sup_count / len(y_true[y_true == label])
        f1 = 2 * precision * recall / (precision + recall)
        print(label, 'precision:', precision, 'recall:', recall, 'f1: ', f1)


def compute_probability_distribution(y: np.ndarray):
    labels, cnt = np.unique(y, return_counts=True)
    cls_prob = cnt / len(y)

    prob_dict = dict(zip(labels, cls_prob))

    return prob_dict


def groupby_class(X: np.ndarray, y: np.ndarray):
    cls_split = dict()
    cls_tuples = dict()

    for label in np.unique(y):
        cls_split[label] = np.where(y == label)[0]
        cls_tuples[label] = X[cls_split[label]]

    return cls_tuples


def feature_extraction(cls_tups: dict, categorical: list):
    attr_dict = dict()
    for label in cls_tups.keys():
        attr_dict[label] = {}

    for label, features in cls_tups.items():
        for i in range(len(features[0])):
            if categorical[i]:
                attr_dict[label][i] = compute_probability_distribution(features[:, i])
            else:
                attr_dict[label][i] = {'mean': np.mean(features[:, i]), 'std': np.std(features[:, i])}

    return attr_dict


def train(X: np.ndarray, y: np.ndarray, categorical: list):
    cls_probs = compute_probability_distribution(y)
    cls_tups = groupby_class(X, y)

    attr_dick = feature_extraction(cls_tups, categorical)

    return cls_probs, attr_dick


def gaussian_distribution(x, mean, std):
    probability = (1 / (np.sqrt(2 * np.pi) * std)) * (np.exp((-(x - mean) ** 2) / (2 * (std ** 2))))

    return probability


def predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, categorical: list):
    cls_prob, cond_prob = train(X_train, y_train, categorical)

    tab = []
    for label, prob in cls_prob.items():
        tab.append([label, prob])



    y_pred = list()

    for test_data_feature in X_test:


        max_prob = -float('inf')
        pred_class = None

        for label in cls_prob.keys():

            prior_prob = cls_prob[label]
            posterior_prob = 1.0

            for idx in range(len(test_data_feature)):
                if categorical[idx]:
                    if test_data_feature[idx] in cond_prob[label][idx].keys():
                        posterior_prob *= cond_prob[label][idx][test_data_feature[idx]]
                    else:
                        cls_cnt = np.count_nonzero(y_train == label)
                        uniq_cnt = len(cond_prob[label][idx].keys())
                        posterior_prob *= (1 / (cls_cnt + uniq_cnt + 1))
                else:
                    posterior_prob *= gaussian_distribution(test_data_feature[idx], cond_prob[label][idx]['mean'],
                                                            cond_prob[label][idx]['std'])


            bayes_prob = prior_prob * posterior_prob

            if bayes_prob > max_prob:
                max_prob = bayes_prob
                pred_class = label

        y_pred.append(pred_class)

    return np.array(y_pred)


if __name__ == '__main__':
    X, y = load_dataset('agaricus-lepiota.data',class_label_column=22)

    num_idx = []
    categorical = [True] * len(X[0])

    for idx in range(len(categorical)):
        if idx in num_idx:
            categorical[idx] = False


    cls_prob, cond_prob = train(X, y, categorical)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    skf.get_n_splits(X, y)


    k = 0

    for train_index, test_index in skf.split(X, y):
        k = k + 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        y_pred = predict(X_train, y_train, X_test, categorical)

        cls_label = list(np.unique(y_test))
    print(classification_report(y_test, y_pred, target_names=cls_label))

    file1.write(classification_report(y_test, y_pred, target_names=cls_label)) 
    file1.write("\n\n++++++++++++++++++++++++++++++++@++++++++++++++++++++++++++++\n\n")
    file1.close()
        