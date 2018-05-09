import os
import time
import math
import pickle
from pprint import pprint
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import ParameterGrid


def read_mnist_with_train_valid_test(path):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(path, one_hot=False)

    X_train_all = mnist.train.images
    Y_train_all = mnist.train.labels

    train_ratio = 0.8
    train_len = int(X_train_all.shape[0] * train_ratio)

    X_train = X_train_all[:train_len]
    Y_train = Y_train_all[:train_len]

    X_valid = X_train_all[train_len:]
    Y_valid = Y_train_all[train_len:]

    label_dict = get_label_dict(Y_train)
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    label_dict = get_label_dict(Y_train_all)
    Y_train_one_hot = label_to_one_hot(Y_train, label_dict)
    Y_valid_one_hot = label_to_one_hot(Y_valid, label_dict)
    Y_test_one_hot = label_to_one_hot(Y_test, label_dict)

    input_dim = X_train.shape[1]
    output_dim = Y_train_one_hot.shape[1]

    print("input dimension : {}, output dimension : {} ".format(
        input_dim, output_dim))
    print("X_train shape : {}\nY_train shape : {}".format(
        np.shape(X_train), np.shape(Y_train_one_hot)))
    print("X_valid shape : {}\nY_valid shape : {}".format(
        np.shape(X_valid), np.shape(Y_valid_one_hot)))
    print("X_test shape : {}\nY_test shape : {}".format(
        np.shape(X_test), np.shape(Y_test_one_hot)))

    return (X_train,
            Y_train,
            Y_train_one_hot,
            X_valid,
            Y_valid,
            Y_valid_one_hot,
            X_test,
            Y_test,
            Y_test_one_hot)

# ======================================
# Manage Dataset
def random_sequences_batch_major(length_from, length_to,
                                 vocab_lower, vocab_upper,
                                 n_data, EOS=0, PAD=1):
    """
    based on : https://github.com/ematvey/tensorflow-seq2seq-tutorials/ 
    """

    if length_from > length_to:
        raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    X_list = []
    length_list = []
    Y_list = []
    for _ in range(n_data):
        length = random_length()
        data = np.random.randint(low=vocab_lower,
                                 high=vocab_upper,
                                 size=length).tolist()
        X_list.append(data + [0] * (length_to - length))
        length_list.append(length)
        Y_list.append(data + [EOS] + [PAD] * 2 + [0] * (length_to - length))

    return np.array(X_list, dtype=np.int32), np.array(length_list, dtype=np.int32), np.array(Y_list, dtype=np.int32)


"""
Split datasets
"""


def split_train_valid_test(X_all,
                           Y_all,
                           train_ratio,
                           valid_ratio,
                           test_ratio=None,
                           flag_random=True):
    if flag_random:
        rand_idx = np.random.permutation(range(len(X_all)))
        X_all = X_all[rand_idx]
        Y_all = Y_all[rand_idx]

    # test_ratio = rest

    data_num = len(X_all)

    train_num = round(data_num * train_ratio)
    valid_num = round(data_num * valid_ratio)

    X_train = X_all[:train_num]
    Y_train = Y_all[:train_num]

    X_valid = X_all[train_num: train_num + valid_num]
    Y_valid = Y_all[train_num: train_num + valid_num]

    if test_ratio:
        test_num = round(data_num * test_ratio)
        X_test = X_all[train_num + valid_num:train_num + valid_num + test_num]
        Y_test = Y_all[train_num + valid_num:train_num + valid_num + test_num]
    else:
        X_test = X_all[train_num + valid_num:]
        Y_test = Y_all[train_num + valid_num:]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def split_train_valid_test_rnn(X_all,
                               X_eos_pad_all,
                               Y_all,
                               train_ratio,
                               valid_ratio,
                               test_ratio=None,
                               flag_random=True):
    if flag_random:
        rand_idx = np.random.permutation(range(len(X_all)))
        X_all = X_all[rand_idx]
        X_eos_pad_all = X_eos_pad_all[rand_idx]
        Y_all = Y_all[rand_idx]

    # test_ratio = rest

    data_num = len(X_all)

    train_num = round(data_num * train_ratio)
    valid_num = round(data_num * valid_ratio)

    X_train = X_all[:train_num]
    X_eos_pad_train = X_eos_pad_all[:train_num]
    Y_train = Y_all[:train_num]

    X_valid = X_all[train_num: train_num + valid_num]
    X_eos_pad_valid = X_eos_pad_all[train_num: train_num + valid_num]
    Y_valid = Y_all[train_num: train_num + valid_num]

    if test_ratio:
        test_num = round(data_num * test_ratio)
        X_test = X_all[train_num + valid_num:train_num + valid_num + test_num]
        X_eos_pad_test = X_eos_pad_all[train_num +
                                       valid_num:train_num + valid_num + test_num]
        Y_test = Y_all[train_num + valid_num:train_num + valid_num + test_num]
    else:
        X_test = X_all[train_num + valid_num:]
        X_eos_pad_test = X_eos_pad_all[train_num + valid_num:]
        Y_test = Y_all[train_num + valid_num:]

    return X_train, X_eos_pad_train, Y_train, X_valid, X_eos_pad_valid, Y_valid, X_test, X_eos_pad_test, Y_test


"""
Dataset utils
"""


def get_label_dict(data_label):
    """
    labels are sorted by key
    """
    data_label = data_label.copy()
    key = np.unique(data_label)
    label_dict = dict(zip(key, range(len(key))))

    return label_dict


def label_with_label_dict(data_label, label_dict):
    data_label = data_label.copy()
    for key, new_key in label_dict.items():
        data_label[data_label == key] = new_key

    return data_label


def label_to_one_hot(data_label, label_dict):
    data_label = data_label.copy()
    for key, new_key in label_dict.items():
        data_label[data_label == key] = new_key

    n_sample = len(data_label)
    data_one_hot = np.zeros((n_sample, len(label_dict)))
    data_one_hot[np.arange(n_sample), data_label.astype(np.int32)] = 1
    return data_one_hot


"""
Numerical preprocess
"""


def preprocess(data, mean, std):
    return (data - mean) / (std + 1e-3)


def calc_moments_of_data(X_target=None):
    mean = np.mean(X_target, axis=0)
    std = np.std(X_target, axis=0)
    return mean, std


"""
"""


def find_k_for_pca(X, k_start, k_end, k_step=1,  thresh_variance_retained=0.95):
    print("calc cov X start")
    sigma = np.cov(X)
    print("calc cov X done")
    print("calc svd X start")
    U, s, V = np.linalg.svd(sigma)
    print("calc svd X done")

    sum_s_k = 0.
    sum_s_n = 0.

    for k in range(k_start, k_end + 1, k_step):
        for i in range(k):
            sum_s_k += s[i]

        for i in range(s.shape[0]):
            sum_s_n += s[i]

        variance_retained = sum_s_k / sum_s_n
        if variance_retained == thresh_variance_retained:
            break

    return k, variance_retained


def calc_rate_of_convergence(x, x_exact):
    """
    reference : http://hplgit.github.io/prog6comp/doc/pub/._p4c-solarized-Python030.html
    """
    e = [abs(x_ - x_exact) for x_ in x]
    q = [math.log(e[n + 1] / e[n]) / math.log(e[n] / e[n - 1])
         for n in range(1, len(e) - 1, 1)]
    return q


"""
Train model
"""
GRID_MODEL_TYPE_SCIKIT = 0
GRID_MODEL_TYPE_TF = 1


def grid_search(model_class, init_param_dict, param_grid, X_train, Y_train, X_valid, Y_valid, model_type, verbose=1, working_dir='./tmp/grid_search'):
    """
    model_type 
    0 : Scikit learn classifier
    1 : ScikitTf classifier
    """

    try:
        os.makedirs(working_dir)
    except Exception as e:
        pass

    best_model_base_path = "{}/tmp_best_model".format(working_dir)

    failed_grid = []
    report_list = []

    def make_report(grid, valid_score, train_score): return {"grid": grid,
                                                             "valid_score": valid_score,
                                                             "train_score": train_score}

    best_grid = None
    best_score = 0
    for g_i, g in enumerate(ParameterGrid(param_grid)):
        while True:
            try:
                model = model_class(**init_param_dict)
                model.set_params(**g)
                model.fit(X_train, Y_train)
                curr_score = model.score(X_valid, Y_valid)
                train_score = model.score(X_train, Y_train)
                report_list.append(make_report(g, curr_score, train_score))
            except Exception as e:
                print(str(e))
                if model_type == 1:
                    print("-" * 30)
                    print("reduce batch size")
                    del model
                    time.sleep(3)
                    init_param_dict['batch_size'] = init_param_dict['batch_size'] // 2
                    if init_param_dict['batch_size'] < 4:
                        failed_grid.append(g)
                        break
                    print("batch_size : {}".format(
                        init_param_dict['batch_size']))
                    continue
            break

        if curr_score > best_score:
            best_score = curr_score
            best_grid = g

            if model_type == GRID_MODEL_TYPE_SCIKIT:
                with open("{}.pickle".format(best_model_base_path), "wb") as f:
                    pickle.dump(model, f)
            elif model_type == GRID_MODEL_TYPE_TF:
                model.save(best_model_base_path)
        del model

        if verbose >= 1:
            print("-" * 30)
            pprint("parameters")
            pprint(g)
            print("valid score : {}".format(curr_score))

    try:
        del model
        time.sleep(3)
    except:
        pass

    if model_type == GRID_MODEL_TYPE_SCIKIT:
        with open("{}.pickle".format(best_model_base_path), "rb") as f:
            model = pickle.load(f)
    elif model_type == GRID_MODEL_TYPE_TF:
        model = model_class(**init_param_dict)
        model.set_params(**best_grid)
        model.load(best_model_base_path)

    model.best_grid = best_grid

    if verbose >= 1:
        train_score = model.score(X_train, Y_train)
        print("*" * 30)
        print(str(model_class))
        print("-" * 5)
        print("best paramemters")
        pprint(best_grid)
        print("-" * 5)
        print("train score : {}".format(train_score))
        print("valid score : {}".format(best_score))
        print("*" * 30)

    return model, report_list, failed_grid


def save_model(model, base_path, model_type, X_train=None, Y_train=None, X_valid=None, Y_valid=None, X_test=None, Y_test=None):
    base_dir_path = os.path.dirname(base_path)

    try:
        os.makedirs(base_dir_path)
    except Exception as e:
        pass

    model_path = "{}.pickle".format(base_path)
    meta_path = "{}.txt".format(base_path)

    if model_type == GRID_MODEL_TYPE_SCIKIT:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    elif model_type == GRID_MODEL_TYPE_TF:
        model.save(base_path)

    if X_train is not None:
        with open(meta_path, "wt") as f:
            f.write("*" * 30)
            f.write("\n")
            f.write(str(model.best_grid))
            f.write("\n")
            f.write("*" * 30)
            f.write("\n")
            f.write("train score\n")
            f.write(str(model.score(X_train, Y_train)))
            f.write("\n")
            f.write("valid score\n")
            f.write(str(model.score(X_valid, Y_valid)))
            f.write("\n")
            f.write("test score\n")
            f.write(str(model.score(X_test, Y_test)))


"""
NLP utilities
"""


def get_X_length(X):
    """
    If an element of X is shorter than max length of the X, then the element should be padded with 0 to fill empty sequence
    """
    if type(X) != np.ndarray:
        X = np.array(X)
    X_padded = np.concatenate([X, np.zeros(shape=(X.shape[0], 1))], axis=1)
    X_length = np.argmin(X_padded, axis=1)
    return X_length


def token_list_to_idx_list(token_list, token_idx_dict):
    return [token_idx_dict[word] for word in token_list if word in token_idx_dict]


def token_list_list_to_idx_list_list(token_list_list, token_idx_dict):

    return list(map(token_list_to_idx_list, token_list_list))


def idx_list_to_token_list(idx_list, idx_token_dict):
    return [idx_token_dict[idx] for idx in idx_list]


def idx_list_list_to_token_list_list(idx_list_list, idx_token_dict):

    return list(map(idx_list_to_token_list, idx_list_list))


def cut_and_pad_to_token_list(token_list, max_len=40, pad_token='PAD'):
    return token_list[:max_len] + [pad_token] * (max_len - len(token_list))

"""
ML utilities
"""


def get_top_n(pred, top_n, idx_label_dict=None):
    if type(pred) != np.ndarray:
        pred = np.array(pred)

    asc_idx_order = np.argsort(pred, axis=1)
    top_n_idx_reversed = asc_idx_order[:, -top_n:]
    top_n_idx = np.apply_along_axis(
        lambda row: row[::-1], 1, top_n_idx_reversed)

    top_n_prob_list = []

    for idx, idx_row in enumerate(top_n_idx):
        top_n_prob_list.append(pred[idx][idx_row].tolist())

    top_n_label_list = None
    if idx_label_dict:
        top_n_label_list = list(
            map(lambda row: [idx_label_dict[idx] for idx in row], top_n_idx.tolist()))

    return top_n_idx.tolist(), top_n_prob_list, top_n_label_list


def count_label(Y):
    return dict(zip(*np.unique(Y, return_counts=True)))


"""
hierarchy utilities
"""


def name_of_distribution(tree,
                         target_depth=-1,
                         key=None,
                         curr_depth=0,
                         ):
    if tree and curr_depth != target_depth:
        name_list = []
        for key, child in tree.items():
            child_name_list = name_of_distribution(tree=child,
                                                   key=key,
                                                   curr_depth=curr_depth+1,
                                                   target_depth=target_depth)
            name_list.extend(child_name_list)

        return name_list
    else:
        return [key]


def make_hierarchy_tree(data_dict_list,
                        hierarchy_dict=None,
                        label_depth_start_idx=1,
                        label_depth_prefix='label_depth',
                        flag_concat_prev_label=True):
    if not hierarchy_dict:
        hierarchy_dict = OrderedDict()

    for data_dict in data_dict_list:
        curr_node = hierarchy_dict
        prev_label = None

        curr_label_depth_start_idx = label_depth_start_idx

        while True:
            label_depth_name = '{}_{}'.format(label_depth_prefix,
                                              curr_label_depth_start_idx)
            next_label_depth_name = '{}_{}'.format(label_depth_prefix,
                                                   curr_label_depth_start_idx+1)
            label = data_dict[label_depth_name]

            if flag_concat_prev_label:
                label = "{}_{}".format(prev_label, label) if prev_label else label
                prev_label = label

            if next_label_depth_name in data_dict:
                if label not in curr_node:
                    curr_node[label] = OrderedDict()
                curr_node = curr_node[label]
            else:
                curr_node[label] = None
                break

            curr_label_depth_start_idx += 1
            
    return hierarchy_dict


def get_name_of_distribution_by_depth(hierarchy_dict):
    name_of_distribution_by_depth = []
    
    i = 0
    prev_name_dist = None
    while True:
        name_dist = name_of_distribution(hierarchy_dict,
                                         target_depth=i)
        if name_dist == prev_name_dist:
            break
        name_of_distribution_by_depth.append(name_dist)
        prev_name_dist = name_dist
        i += 1
    return name_of_distribution_by_depth


def get_idx_label_dict(label_name_list):
    idx_label_dict = {idx: label for idx,
                      label in enumerate(label_name_list)}
    label_idx_dict = {label: idx for idx,
                      label in idx_label_dict.items()}
    return idx_label_dict, label_idx_dict


def get_idx_label_dict_by_depth(nod_by_depth):
    idx_label_dict_by_depth = []
            
    for label_name_list in nod_by_depth:
        (idx_label_dict,
         label_idx_dict) = get_idx_label_dict(label_name_list)
        
        idx_label_dict_by_depth.append({
                            'idx_label': idx_label_dict,
                            'label_idx': label_idx_dict, })

    return idx_label_dict_by_depth

