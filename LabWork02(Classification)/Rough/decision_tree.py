# import math
# from itertools import combinations
# import timeit
# from pprint import pprint
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
# from tabulate import tabulate


file1 = open("decisionTree_output.txt", "a")  # append mode 

def load_dataset(filename: str):
    dataset = pd.read_csv(filename, header=None)

    global dataset_size
    dataset_size = dataset.shape[0]

    print("\nDataset Summary: ", filename, "\nDataset Size: ", dataset.shape[0], "\nNumber of Attributes: ", dataset.shape[1]-1)
    
    file1.write("Dataset Summary: ")
    file1.write(filename)
    file1.write("\n")

    file1.write("Dataset Size: ")
    file1.write(str(dataset.shape[0]))
    file1.write("\n") 

    file1.write("Number of Attributes: ")
    file1.write(str(dataset.shape[1]-1))
    file1.write("\n")

    return dataset


def entropy(dataset: pd.DataFrame, class_label_column):
    entropy_node = 0

    values = dataset[class_label_column].unique()

    for val in values:
        # print(val, dataset[class_label_column].value_counts()[val])

        p_i = dataset[class_label_column].value_counts()[val] / len(dataset[class_label_column])
        entropy_node += (-p_i * np.log2(p_i))

    return entropy_node


def entropy_attribute(dataset: pd.DataFrame, class_label_column, attribute):
    entropy_attr = 0

    attr_vars = dataset[attribute].unique()

    for val in attr_vars:
        # print('\nAj =', val)

        df_attr = dataset[dataset[attribute] == val]
        # print(df_attr)

        info = entropy(df_attr, class_label_column)
        # print('Info(Dj) =', info)

        # print('Dj :', df_attr.shape[0], 'D :', dataset_size)
        fraction = df_attr.shape[0] / dataset_size
        # print('Dj / D = ', fraction)

        entropy_attr += (fraction * info)

    return entropy_attr


def entropy_attribute_cont(dataset: pd.DataFrame, class_label_column, attribute):
    attr_col = dataset[attribute].sort_values()

    min_entropy = float('inf')
    split_pt = 0
    # print(len(attr_col.unique()))

    for i in range(len(attr_col) - 1):
        if attr_col.iloc[i] == attr_col.iloc[i + 1]:
            continue
        mid_pt = (attr_col.iloc[i] + attr_col.iloc[i + 1]) / 2

        d1 = dataset[dataset[attribute] <= mid_pt]
        d2 = dataset[dataset[attribute] > mid_pt]

        e1 = entropy(d1, class_label_column)
        e2 = entropy(d2, class_label_column)

        _entropy = ((d1.shape[0] / dataset_size) * e1) + ((d2.shape[0] / dataset_size) * e2)

        if _entropy < min_entropy:
            min_entropy = _entropy
            split_pt = mid_pt

    return min_entropy, split_pt


def gain(dataset: pd.DataFrame, class_label_column, attribute):
    _gain = entropy(dataset, class_label_column) - entropy_attribute(dataset, class_label_column, attribute)

    return _gain


def gini(dataset: pd.DataFrame, class_label_column):
    labels = dataset[class_label_column].unique()

    list_pi = list()

    for val in labels:
        # print(val, dataset[class_label_column].value_counts()[val])

        p_i = dataset[class_label_column].value_counts()[val] / len(dataset[class_label_column])
        list_pi.append(p_i ** 2)

    _gini = 1 - sum(list_pi)

    return _gini


def gini_attribute(dataset: pd.DataFrame, class_label_column, attribute):
    attr_vals = dataset[attribute].unique()
    # print(attr_vals)

    min_gini = float('inf')
    splitting_attr = list()

    for r in range(1, math.floor(len(attr_vals) / 2) + 1):
        comb_list = list(combinations(attr_vals, r))

        for subset in comb_list:
            d1 = set(attr_vals) - set(subset)
            d2 = set(subset)

            # print('D1 :', d1, 'D2 :', d2)

            g1 = dataset[dataset[attribute].isin(d1)]
            g2 = dataset[dataset[attribute].isin(d2)]

            # print('G1 :', g1.shape[0], 'g2 :', g2.shape[0])

            G1 = gini(g1, class_label_column)
            G2 = gini(g2, class_label_column)

            # print('GINI - 1 :', G1, 'GINI - 2 :', G2)

            _gini_attr = ((g1.shape[0] / dataset_size) * G1) + ((g2.shape[0] / dataset_size) * G2)

            # print('GINI_ATTR :', _gini_attr)

            if _gini_attr <= min_gini:
                min_gini = _gini_attr
                splitting_attr = [d1, d2]

            # print('MAX GINI:', mx_gini)
            # print(splitting_attr, '\n')

    return min_gini, splitting_attr


def gini_cont(dataset: pd.DataFrame, class_label_column, attribute):
    attr_col = dataset[attribute].sort_values()

    min_gini = float('inf')
    split_pt = 0
    # print(attr_col)

    for i in range(len(attr_col) - 1):
        if attr_col.iloc[i] == attr_col.iloc[i + 1]:
            continue
        mid_pt = (attr_col.iloc[i] + attr_col.iloc[i + 1]) / 2

        d1 = dataset[dataset[attribute] <= mid_pt]
        d2 = dataset[dataset[attribute] > mid_pt]

        g1 = gini(d1, class_label_column)
        g2 = gini(d2, class_label_column)

        _gini = ((d1.shape[0] / dataset_size) * g1) + ((d2.shape[0] / dataset_size) * g2)

        if _gini < min_gini:
            min_gini = _gini
            split_pt = mid_pt

    return min_gini, split_pt



class DecisionTree:
    def __init__(self, attr_selection_measure='entropy', is_binary=False):
        self.decision_tree = None
        self.majority = None
        self.attr_selection_measure = attr_selection_measure
        self.is_binary = is_binary
        self.numeric_cols = list()
        self.prune_threshold_count = 1

    def fit(self, dataset: pd.DataFrame, class_label_column, numeric_col_list: list = [], prune_threshold=None):
        self.majority = dataset[class_label_column].mode()[0]
        self.numeric_cols = numeric_col_list
        if prune_threshold is not None:
            self.prune_threshold_count = math.floor(
                dataset.shape[0] * prune_threshold)
        if len(numeric_col_list) == dataset.shape[1] - 1 or self.attr_selection_measure == 'ginni':
            self.is_binary = True
        self.decision_tree = self._train_recursive(dataset, class_label_column)

    def _train_recursive(self, dataset: pd.DataFrame, class_label_column, tree=None):
        # print(dataset.shape)
        spl_pt, spl_attr = self._find_splitting_attribute(
            dataset, class_label_column)
        # spl_attr,spl_pt = selct_attr_gini_cont(dataset, class_label_column)
        # print(spl_attr,spl_pt)
        # print(spl_attr, attr_values)

        if tree is None:
            tree = {}
            if self.attr_selection_measure == 'ginni':
                if spl_attr is None:
                    return self.majority
                if type(spl_pt) is list:
                    tree[(spl_attr, tuple(spl_pt[0]))] = dict()
                else:
                    tree[(spl_attr, spl_pt)] = dict()
            else:
                tree[(spl_attr, spl_pt)] = dict()

        if spl_attr in self.numeric_cols:
            d1 = dataset[dataset[spl_attr] < spl_pt]
            d2 = dataset[dataset[spl_attr] >= spl_pt]
            # print(dataset.shape,d1.shape,d2.shape)
            if d1.shape[0] == 0 or d2.shape[0] == 0:
                tree[(spl_attr, spl_pt)]['_'] = dataset[class_label_column].mode()[0]
            else:
                self._assign_node(d1, spl_attr, 'l', tree, spl_pt=spl_pt)
                self._assign_node(d2, spl_attr, 'ge', tree, spl_pt=spl_pt)
            return tree

        elif self.attr_selection_measure == 'ginni':
            g1 = list(spl_pt[0])
            # print(spl_attr,g1)
            d1 = pd.DataFrame()
            for value in g1:
                dt = dataset[dataset[spl_attr] == value]
                d1 = pd.concat([d1, dt])
            d2 = dataset.drop(d1.index)
            # print(dataset.shape,d1.shape,d2.shape)
            if d1.shape[0] == 0 or d2.shape[0] == 0:
                tree[(spl_attr, spl_pt)] = dataset[class_label_column].mode()[0]
            else:
                self._assign_node(d1, spl_attr, 'in', tree,
                                  spl_pt=tuple(spl_pt[0]))
                self._assign_node(d2, spl_attr, 'out', tree,
                                  spl_pt=tuple(spl_pt[0]))
            return tree

        else:
            attr_values = dataset[spl_attr].unique()
            for val in attr_values:
                data_partition = dataset[dataset[spl_attr] == val]
                data_partition = data_partition.drop(
                    columns=[spl_attr])  # remove the splitting attribute
                if data_partition.shape[0] == 0:
                    tree[(spl_attr, spl_pt)
                    ][val] = dataset[class_label_column].mode()[0]
                elif data_partition.shape[1] == 1:
                    tree[(spl_attr, spl_pt)
                    ][val] = data_partition[class_label_column].mode()[0]
                else:
                    self._assign_node(data_partition, spl_attr, val, tree)
            return tree

    def _assign_node(self, data_partition, spl_attr, attr_val, tree, spl_pt='_'):
        class_values = data_partition[class_label_column].unique()
        if len(class_values) == 1:
            tree[(spl_attr, spl_pt)][attr_val] = class_values[0]
            # print('pure class')
        elif data_partition.shape[0] < self.prune_threshold_count:
            tree[(spl_attr, spl_pt)
            ][attr_val] = data_partition[class_label_column].mode()[0]
        else:
            # recursion
            tree[(spl_attr, spl_pt)][attr_val] = self._train_recursive(
                data_partition, class_label_column)

    def predict(self, data: pd.DataFrame):
        preds = []
        if self.decision_tree is None:
            print('call fit first')
            return
        for index, r in data.iterrows():
            preds.append(self.predict_single(r, self.decision_tree))
        return preds

    def predict_single(self, row, dt: dict = None):
        for (key, pt) in dt:
            val = row[key]
            if key in self.numeric_cols:
                dt = dt[(key, pt)]
                if ('l' not in dt) and ('ge' not in dt):
                    dt = dt['_']
                elif val > pt:
                    dt = dt['ge']
                else:
                    dt = dt['l']
                if type(dt) is dict:
                    pred = self.predict_single(row, dt)
                else:
                    pred = dt
            else:
                if self.attr_selection_measure == 'ginni':
                    l1 = pt
                    if val in l1:
                        dt = dt[(key, pt)]['in']
                    else:
                        dt = dt[(key, pt)]['out']
                else:
                    if val not in dt[(key, pt)]:
                        pred = self.majority
                        continue
                    dt = dt[(key, pt)][val]
                if type(dt) is dict:
                    pred = self.predict_single(row, dt)
                else:
                    pred = dt
            return pred

    def _find_splitting_attribute(self, dataset: pd.DataFrame, class_label_column):
        attr_cols = list(dataset.columns)
        attr_cols.remove(class_label_column)
        # print(attr_cols)
        min_measure_val = float('inf')  # entropy or ginni
        splitting_attr = None
        splitting_pt = None
        for col in attr_cols:
            if self.attr_selection_measure == 'entropy':
                if col in self.numeric_cols:
                    attr_measure_val, pt = entropy_attribute_cont(
                        dataset, class_label_column, col)
                else:
                    attr_measure_val, pt = entropy_attribute(
                        dataset, class_label_column, col), '_'
            else:
                if col in self.numeric_cols:
                    attr_measure_val, pt = gini_cont(
                        dataset, class_label_column, col)
                else:
                    attr_measure_val, pt = gini_attribute(
                        dataset, class_label_column, col)

            if attr_measure_val < min_measure_val:
                splitting_attr = col
                min_measure_val = attr_measure_val
                splitting_pt = pt

        return splitting_pt, splitting_attr

    def print_tree(self):
        pprint(self.decision_tree)


def calculate_accuracy(predictions: list, class_labels: list):
    t_len = len(predictions)
    correct = 0.0
    for i in range(t_len):
        if predictions[i] == class_labels[i]:
            correct += 1
    return 100 * (correct / t_len)


def run_k_fold(dataset, attr_sl, class_label_column, numeric_cols, prune_threshold, k=10):
    data = load_dataset(dataset)
    # shuffle
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.reset_index(drop=True)

    fold_size = data.shape[0] // k
    begin_index = 0
    end_index = begin_index + fold_size
    acc_list = []
    tr_time = []
    tt_time = []

    for i in range(k):
        # print(begin_index, end_index)
        test_frame = data[begin_index:end_index + 1].reset_index(drop=True)
        test_labels = test_frame[class_label_column]
        test_frame.drop(columns=class_label_column, inplace=True)
        train_frame = data.drop(
            data.iloc[begin_index:end_index + 1].index).reset_index(drop=True)
        # print(len(train_frame))
        dt = DecisionTree(attr_sl)
        # start = timeit.default_timer()
        dt.fit(train_frame, class_label_column=class_label_column, numeric_col_list=numeric_cols,
               prune_threshold=prune_threshold)
        # stop = timeit.default_timer()
        # tr_time.append(stop - start)
        # print('k=', i+1, ',training time: ', stop-start, 'second')
        # dt.print_tree()

        # start = timeit.default_timer()
        preds = dt.predict(test_frame)
        # stop = timeit.default_timer()
        # print('Average inference time: ', 1000*((stop-start)/test_frame.shape[0]),'millisecond')
        # tt_time.append(1000 * ((stop - start) / test_frame.shape[0]))

        acc = calculate_accuracy(preds, list(test_labels))
        acc_list.append(acc)

        cls_label = list(test_labels.unique())
        # print('fold: ', i + 1)
        # print(classification_report(test_labels, preds, target_names=cls_label))

        begin_index = end_index + 1
        if i == k - 2:
            end_index = data.shape[0] - 1
        else:
            end_index = end_index + fold_size
    print(classification_report(test_labels, preds, target_names=cls_label))
    file1.write(classification_report(test_labels, preds, target_names=cls_label))
    file1.write("\n\n++++++++++++++++++++++++++++++++++++++@++++++++++++++++++++++++++++++++++++++++\n\n")
    # accs = [[i + 1, acc_list[i]] for i in range(k)]
    # accs.append(['Average', pd.Series(acc_list).mean()])

    # print('\nk-fold cross validation (Accuracy Measure)')
    # print(tabulate(accs, headers=['k', 'Accuracy'], tablefmt='grid'))
    # print('average training time: ', pd.Series(tr_time).mean(), 'second')
    # print('Average inference time: (per tuple)', pd.Series(tt_time).mean(), 'millisecond')






if __name__ == '__main__':
    # data = load_dataset('agaricus-lepiota.data')
    # print(gini(data, class_label_column=4))
   
    # print()
    # spl_attr, spl_pt = selct_attr_gini_cont(data, class_label_column=4)
    # print(spl_attr, spl_pt)


    class_label_column = 0
    # iris_numeric_cols = [0, 1, 2, 3]
    # anneal_numeric_cols = [3, 4, 8, 32, 33, 34]

    k = 10

    attr_sl = 'entropy'
    prune_threshold = None

    run_k_fold('agaricus-lepiota.data', attr_sl, class_label_column, [], prune_threshold, k)
