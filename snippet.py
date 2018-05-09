import os
import pickle
from importlib import reload

import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from models.tf_template import BaseTfClassifier
import models.classifier.cnn as c
import models.classifier.scikit_wrapper.scikit_cnn as sc
import models.classifier.mlp as m
import models.classifier.scikit_wrapper.scikit_mlp as sm
import models.classifier.lstm as ls
import models.classifier.scikit_wrapper.scikit_lstm as scilstm
import models.classifier.oselm as o
import models.classifier.scikit_wrapper.scikit_oselm as se
import models.classifier.ensemblenn as sbm
import models.classifier.scikit_wrapper.scikit_ensemblenn as scisbm
import models.classifier.lstm as l
from libs.utils import (grid_search,
                        read_mnist_with_train_valid_test,
                        save_model,
                        find_k_for_pca,
                        calc_rate_of_convergence)
from libs.utils import GRID_MODEL_TYPE_SCIKIT, GRID_MODEL_TYPE_TF
from libs.plots import plot_loss_vs_epoch_and_save
from configs.project_config import project_path


"""
Read Mnist data and split
"""
(X_train,
 Y_train,
 Y_train_one_hot,
 X_valid,
 Y_valid,
 Y_valid_one_hot,
 X_test,
 Y_test,
 Y_test_one_hot) = read_mnist_with_train_valid_test(
                        "{}/data/mnist".format(project_path))
input_dim = X_train.shape[1]
output_dim = Y_train_one_hot.shape[1]


"""
PCA
"""
pca_compnenets = 250
pca_path = "{}/checkpoints/pca/pca_{}.pickle".format(
                project_path, pca_compnenets)
if os.path.exists(pca_path):
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
else:
    pca = PCA(n_components=pca_compnenets)
    pca.fit(X_train)
    try:
        os.makedirs(os.path.dirname(pca_path))
    except:
        pass
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
X_train = pca.transform(X_train)
X_valid = pca.transform(X_valid)
X_test = pca.transform(X_test)
input_dim = X_train.shape[1]
output_dim = Y_train_one_hot.shape[1]


"""
Optimize Parameter
"""
"""
SVM
"""
init_param_dict = {}
param_grid = {
                'kernel': ['linear'],
                'C': [1.0],
             },
svm = grid_search(SVC, init_param_dict, param_grid, X_train, Y_train, X_valid, Y_valid,
                  model_type=GRID_MODEL_TYPE_SCIKIT, verbose=1)
print("test score : {}".format(svm.score(X_test, Y_test)))

save_dir_path = "{}/tmp/svm".format(project_path)
save_model(svm, "{}/best_model".format(save_dir_path), GRID_MODEL_TYPE_SCIKIT,
           X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

del svm


"""
RandomForestClassifier
"""
init_param_dict = {}
param_grid = {
                'n_estimators': [500, 1000],
                'max_features': ['auto'],
                'criterion': ['entropy'],
                'max_depth': [10],
                'min_samples_split': [5],
                'min_samples_leaf': [5],
             }
rdf, report_list, failed_list = grid_search(RandomForestClassifier, init_param_dict, param_grid,
                                            X_train, Y_train, X_valid, Y_valid, model_type=GRID_MODEL_TYPE_SCIKIT, verbose=1)
print("test score : {}".format(rdf.score(X_test, Y_test)))

save_dir_path = "{}/tmp/rdf".format(project_path)
save_model(rdf, "{}/best_model".format(save_dir_path), GRID_MODEL_TYPE_SCIKIT,
           X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

del rdf


"""
SciKitMLP classifier
"""
init_param_dict = {
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "flag_preprocess": True,
                    "X_valid": X_valid,
                    "Y_valid": Y_valid_one_hot,
                    "batch_size": 256,
                    "n_epoch": 100,
                    "patience": 50,
                    "verbose_interval": 5,
                }
param_grid = {
                'fc_layer_info_dict_list': [
                    [
                        {
                            'type': 'fc',
                            "n_output": 1024,
                            "is_batch_norm": False,
                            "is_dropout": False,
                            "keep_prob": 0.5,
                            "activation": tf.nn.relu,
                            "batch_norm_decay": 0.99,
                            "name": "mlp_1",
                        },
                        {
                            'type': 'fc',
                            "n_output": 1024,
                            "is_batch_norm": False,
                            "is_dropout": False,
                            "keep_prob": 0.5,
                            "activation": tf.nn.relu,
                            "batch_norm_decay": 0.99,
                            "name": "mlp_2",
                        },
                    ],
                ],
                'output_layer_info_dict': [
                    {
                        'type': 'fc',
                        "n_output": output_dim,
                        "is_batch_norm": False,
                        "is_dropout": False,
                        "keep_prob": 0.5,
                        "batch_norm_decay": 0.99,
                        "name": "output",
                    },
                ],
                'learning_rate': [0.001],
                'reg_lambda': [0.00001],
            }
scikit_mlp, scikit_mlp_report_list, failed_list = grid_search(
    sm.SciKitMLP, init_param_dict, param_grid, X_train, Y_train_one_hot, X_valid, Y_valid_one_hot, model_type=GRID_MODEL_TYPE_TF, verbose=1)

print("test score : {}".format(scikit_mlp.score(X_test, Y_test_one_hot)))

save_dir_path = "{}/tmp/mlp".format(project_path)
fig, axs = plot_loss_vs_epoch_and_save(
    scikit_mlp.clf.report_dict, save_dir_path)
fig.show()

save_model(scikit_mlp, '{}/best_model'.format(save_dir_path), GRID_MODEL_TYPE_TF,
           X_train, Y_train_one_hot, X_valid, Y_valid_one_hot, X_test, Y_test_one_hot)

del scikit_mlp


"""
SciKitCNN classifier
"""
init_param_dict = {
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "flag_preprocess": True,
                    "X_valid": X_valid,
                    "Y_valid": Y_valid_one_hot,
                    "batch_size": 256,
                    "n_epoch": 100,
                    "patience": 10,
                    "verbose_interval": 1,
        }
param_grid = {
                'cnn_layer_info_dict_list': [
                    [
                        {
                            'type': 'transform',
                            'transform_function': lambda X: tf.reshape(X, shape=[-1, input_dim, 1])
                        },
                        {
                            'type': 'res_start',
                            'id': 0,
                        },
                        {
                            'type': 'conv',
                            'n_output': 16,
                            'filter_size': 3,
                            'activation': None,
                            'stride': 2,
                            'padding': 'SAME',
                            'is_batch_norm': True,
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type': 'conv',
                            'n_output': 32,
                            'filter_size': 3,
                            'activation': None,
                            'stride': 2,
                            'padding': 'SAME',
                            'is_batch_norm': True,
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type': 'res_end',
                            'id_list': [0],
                        },
                        {
                            'type': 'res_start',
                            'id': 2,
                        },
                        {
                            'type': 'conv',
                            'n_output': 64,
                            'filter_size': 3,
                            'activation': None,
                            'stride': 2,
                            'is_batch_norm': True,
                            'padding': 'SAME',
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type': 'conv',
                            'n_output': 128,
                            'filter_size': 3,
                            'activation': None,
                            'stride': 2,
                            'is_batch_norm': True,
                            'padding': 'SAME',
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type': 'res_end',
                            'id_list': [2],
                        },
                        {
                            'type': 'res_start',
                            'id': 4,
                        },
                        {
                            'type': 'conv',
                            'n_output': 256,
                            'filter_size': 3,
                            'activation': None,
                            'stride': 2,
                            'is_batch_norm': True,
                            'padding': 'SAME',
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type': 'conv',
                            'n_output': 512,
                            'filter_size': 3,
                            'activation': None,
                            'stride': 2,
                            'is_batch_norm': True,
                            'padding': 'SAME',
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type': 'res_end',
                            'id_list': [4],
                        },
                        {
                            'type': 'pool',
                            'window_shape': -1,
                            'pooling_type': 'AVG',
                            'padding': 'VALID',
                            'strides': [1],
                        },
                    ],
                ],
                'fc_layer_info_dict_list': [
                    [
                        {
                            'type': 'fc',
                            "n_output": 1024,
                            "is_batch_norm": True,
                            "is_dropout": False,
                            "keep_prob": 0.8,
                            "activation": tf.nn.relu,
                            "batch_norm_decay": 0.99,
                            "name": "fc_1",
                        },
                    ]
                ],
                'output_layer_info_dict': [
                    {
                        "type": "fc",
                        "n_output": output_dim,
                        "is_batch_norm": True,
                        "is_dropout": False,
                        "keep_prob": 0.8,
                        "batch_norm_decay": 0.99,  # when use batch norm
                        "name": "output"
                    },
                ],
                'learning_rate': [1e-3],
                'reg_lambda': [0.0],
            }
scikit_cnn, scikit_cnn_report_list, scikit_cnn_failed_list = grid_search(
    sc.SciKitCNN, init_param_dict, param_grid, X_train, Y_train_one_hot, X_valid, Y_valid_one_hot, model_type=GRID_MODEL_TYPE_TF, verbose=1)
print("test score : {}".format(scikit_cnn.score(X_test, Y_test_one_hot)))

save_dir_path = "{}/tmp/cnn".format(project_path)
fig, axs = plot_error_vs_model_size_and_save(
    scikit_cnn, scikit_cnn_report_list, save_dir_path)

save_model(scikit_cnn, '{}/best_model'.format(save_dir_path), GRID_MODEL_TYPE_TF,
           X_train, Y_train_one_hot, X_valid, Y_valid_one_hot, X_test, Y_test_one_hot)

fig.show()

del scikit_cnn.clf

del scikit_cnn


"""
SciKitLSTM classifier
"""
init_param_dict = {
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "flag_preprocess": True,
                    "X_valid": X_valid,
                    "Y_valid": Y_valid_one_hot,
                    "batch_size": 256,
                    "n_epoch": 20,
                    "patience": 5,
                    "verbose_interval": 5,
        }
param_grid = {
                'lstm_info_dict': [
                    {
                        'timesteps': 10,
                        'hidden_num': 128,
                    },
                    {
                        'timesteps': 250,
                        'hidden_num': 128,
                    },
                ],
                'fc_layer_info_dict_list': [
                    [
                        {
                            'type': 'fc',
                            "n_output": 256,
                            "is_batch_norm": True,
                            "is_dropout": False,
                            "activation": tf.nn.relu,
                            "batch_norm_decay": 0.99,
                            "name": "fc_1",
                        }
                     ]
                ],
                'output_layer_info_dict': [
                    {
                        'type': 'fc',
                        "n_output": output_dim,
                        "is_batch_norm": True,
                        "is_dropout": False,
                        "activation": tf.nn.softmax,
                        "batch_norm_decay": 0.99,
                        "name": "output",
                    }
                ],
                'learning_rate': [0.001],
                'reg_lambda': [0.0],
            }
scikit_lstm, scikit_lstm_report_list, failed_list = grid_search(
    scilstm.SciKitLSTM, init_param_dict, param_grid, X_train, Y_train_one_hot, X_valid, Y_valid_one_hot, model_type=GRID_MODEL_TYPE_TF, verbose=1)
print("test score : {}".format(scikit_lstm.score(X_test, Y_test_one_hot)))

save_dir_path = "{}/tmp/lstm".format(project_path)
save_model(scikit_lstm, '{}/best_model'.format(save_dir_path), GRID_MODEL_TYPE_TF,
           X_train, Y_train_one_hot, X_valid, Y_valid_one_hot, X_test, Y_test_one_hot)

del scikit_lstm

reload(sm)


"""
SciKitEnsembleNN classifier
"""
init_param_dict = {
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "flag_preprocess": True,
                    "X_valid": X_valid,
                    "Y_valid": Y_valid_one_hot,
                    "batch_size": 256,
                    "n_epoch": 100,
                    "patience": 20,
                    "verbose_interval": 1,
                    "mode": sbm.MODE_TRAIN_GLOBAL,
        }
param_grid = {
                'lstm_info_dict': [
                    {
                        'timesteps': 25,
                        'hidden_num': 256,
                    },
                    {
                        'timesteps': 25,
                        'hidden_num': 512,
                    },
                    {
                        'timesteps': 25,
                        'hidden_num': 1024,
                    },
                    {
                        'timesteps': 125,
                        'hidden_num': 256,
                    },
                    {
                        'timesteps': 125,
                        'hidden_num': 512,
                    },
                    {
                        'timesteps': 125,
                        'hidden_num': 1024,
                    },
                    {
                        'timesteps': 500,
                        'hidden_num': 256,
                    },
                    {
                        'timesteps': 500,
                        'hidden_num': 512,
                    },
                    {
                        'timesteps': 500,
                        'hidden_num': 1024,
                    },
                ],
                'learning_rate': [0.001],
                'reg_lambda': [0.0],
            }
scikit_ensemblenn, scikit_ensemblenn_report_list, scikit_ensemblenn_failed_list = grid_search(
    scisbm.SciKitEnsembleNN, init_param_dict, param_grid, X_train, Y_train_one_hot, X_valid, Y_valid_one_hot, model_type=GRID_MODEL_TYPE_TF, verbose=1)
print("test score : {}".format(scikit_ensemblenn.score(X_test, Y_test_one_hot)))

save_dir_path = "{}/tmp/ensemblenn".format(project_path)
save_model(scikit_ensemblenn, "{}/best_model".format(save_dir_path), GRID_MODEL_TYPE_TF,
           X_train, Y_train_one_hot, X_valid, Y_valid_one_hot, X_test, Y_test_one_hot)

fig.show()

del scikit_ensemblenn

"""
Origin model
"""

"""
MLP classifier
"""
mlp = m.MLP(input_dim, output_dim,
            fc_layer_info_dict_list=[
                    {
                        "type": 'fc',
                        "n_output": 32,
                        "is_batch_norm": True,
                        "activation": tf.nn.relu,
                        "batch_norm_decay": 0.9,  # when use batch norm
                    },
                ],
            flag_preprocess=False,
            )
mlp.train(X_train, Y_train_one_hot,
          X_valid=X_valid,
          Y_valid=Y_valid_one_hot,
          batch_size=2048,
          n_epoch=10,
          learning_rate=0.01,
          reg_lambda=0.0,
          patience=5,
          verbose_interval=10)

"""
CNN classifier
"""
cnn = c.CNN(input_dim, output_dim, flag_preprocess=True)
cnn.train(X_train, Y_train_one_hot,
          X_valid=X_valid,
          Y_valid=Y_valid_one_hot,
          batch_size=2048,
          n_epoch=100,
          learning_rate=0.001,
          reg_lambda=0.0,
          patience=5,
          verbose_interval=1)

"""
LSTM classifier
"""
lstm = l.LSTM(input_dim, output_dim, flag_preprocess=True)
lstm.train(X_train, Y_train_one_hot,
           X_valid=X_valid,
           Y_valid=Y_valid_one_hot,
           batch_size=2048,
           n_epoch=100,
           learning_rate=0.001,
           reg_lambda=0.0,
           patience=5,
           verbose_interval=1)


"""
OSELM
"""
oselm = o.OSELM(input_dim,
                output_dim,
                hidden_num=1000,
                batch_size=1100,
                flag_preprocess=True,
                gpu_memory_fraction=0.3,)
oselm.train(X_train, Y_train_one_hot, X_valid, Y_valid_one_hot)


"""
EnsembleNN Model
"""
ensemblenn = sbm.EnsembleNN(input_dim, output_dim, flag_preprocess=True)

ensemblenn.min_loss = None
ensemblenn.train(X_train, Y_train_one_hot,
                 X_valid=X_valid,
                 Y_valid=Y_valid_one_hot,
                 batch_size=256,
                 n_epoch=20,
                 learning_rate=0.001,
                 reg_lambda=0.0,
                 patience=10,
                 verbose_interval=1,
                 mode=sbm.MODE_TRAIN_MLP)

ensemblenn.min_loss = None
ensemblenn.train(X_train, Y_train_one_hot,
                 X_valid=X_valid,
                 Y_valid=Y_valid_one_hot,
                 batch_size=256,
                 n_epoch=20,
                 learning_rate=0.001,
                 reg_lambda=0.0,
                 patience=10,
                 verbose_interval=1,
                 mode=sbm.MODE_TRAIN_CONV)

ensemblenn.min_loss = None
ensemblenn.train(X_train, Y_train_one_hot,
                 X_valid=X_valid,
                 Y_valid=Y_valid_one_hot,
                 batch_size=256,
                 n_epoch=20,
                 learning_rate=0.001,
                 reg_lambda=0.0,
                 patience=10,
                 verbose_interval=1,
                 mode=sbm.MODE_TRAIN_LSTM)

ensemblenn.min_loss = None
ensemblenn.train(X_train, Y_train_one_hot,
                 X_valid=X_valid,
                 Y_valid=Y_valid_one_hot,
                 batch_size=256,
                 n_epoch=20,
                 learning_rate=0.001,
                 reg_lambda=0.0,
                 patience=10,
                 verbose_interval=1,
                 mode=sbm.MODE_TRAIN_CLF)

ensemblenn.min_loss = None
ensemblenn.train(X_train, Y_train_one_hot,
                 X_valid=X_valid,
                 Y_valid=Y_valid_one_hot,
                 batch_size=256,
                 n_epoch=100,
                 learning_rate=0.001,
                 reg_lambda=0.0,
                 patience=20,
                 verbose_interval=1,
                 mode=sbm.MODE_TRAIN_GLOBAL)

del ensemblenn


"""
fit loss
"""
with open("{}/checkpoints/already_trained/best_model.meta".format(project_path), "rb") as f:
    report_dict = pickle.load(f)['report_dict']

train_loss = report_dict['train_loss']
valid_loss = report_dict['valid_loss']

degree = 5
sample_unit = 1
train_loss_sample_y = [l for i, l in enumerate(
    train_loss) if i % sample_unit == 0]
train_loss_sample_x = [i for i in range(
    len(train_loss)) if i % sample_unit == 0]

z = np.polyfit(train_loss_sample_x, train_loss_sample_y, degree)
f = lambda x:  np.sum([z_val * np.power(x, degree - z_i)
                      for z_i, z_val in enumerate(z)])

y = [f(x) for x in range(len(train_loss))][:int(0.95 * (len(train_loss))]
q=calc_rate_of_convergence(y, 0.)

fig, axs=plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(train_loss, label='train_loss', color='blue')
axs[0].plot(y, label='train_loss polyfit(degree of 5)', color='green')
axs[0].legend(loc='best')
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('loss')

axs[1].plot(train_loss, label='train_loss', color='blue')
axs[1].plot(y, label='train_loss polyfit(degree of 5)', color='green')
axs[1].legend(loc='best')
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('loss (log scale)')
axs[1].set_yscale('log')
