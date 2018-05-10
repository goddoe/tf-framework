#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np

from models.tf_template import BaseTfClassifier
from models.tf_layer import batch_norm_wrapper, flatten, linear, conv1d, bilstm, build_layer_with_info_dict_list

from libs.various_utils import generate_id_with_date, get_date_time_prefix


"""
CONSTANT
"""
MODE_TRAIN_GLOBAL = 'MODE_TRAIN_GLOBAL'
MODE_TRAIN_CONV = 'MODE_TRAIN_CONV'
MODE_TRAIN_LSTM = 'MODE_TRAIN_LSTM'
MODE_TRAIN_MLP = 'MODE_TRAIN_MLP'
MODE_TRAIN_CLF = 'MODE_TRAIN_CLF'

class EnsembleNN(BaseTfClassifier):
    def __init__(self, input_dim, output_dim,
                conv_layer_info_dict_list=None,
                lstm_info_dict=None,
                fc_layer_info_dict_list=None,
                mlp_layer_info_dict_list=None,
                clf_layer_info_dict_list=None,
                flag_preprocess=False,
                mode=MODE_TRAIN_GLOBAL,
                tensorboard_path=None,
                model_name='submit_model',
                **args
                ):
        super().__init__()
        if conv_layer_info_dict_list is None:
            conv_layer_info_dict_list = [
                        {
                            'type': 'pad',
                            'paddings' : [[0,0,],[6,6]],
                            'mode':'CONSTANT',
                        },
                        { 
                            'type': 'transform',
                            'transform_function': lambda X: tf.reshape(X, shape=[-1, input_dim+12, 1])
                        },
                        {
                            'type':'res_start',
                            'id':0,
                        },
                        {
                            'type':'conv',
                            'n_output': 16,
                            'filter_size': 3,
                            'activation': tf.nn.relu,
                            'stride':2,
                            'padding':'SAME',
                            'is_batch_norm': True,
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type':'res_start',
                            'id':1,
                        },
                        {
                            'type':'conv',
                            'n_output': 32,
                            'filter_size': 3,
                            'activation': tf.nn.relu,
                            'stride':2,
                            'padding':'SAME',
                            'is_batch_norm': True,
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type':'res_start',
                            'id':2,
                        },
                        {
                            'type':'conv',
                            'n_output': 64,
                            'filter_size': 3,
                            'activation': None,
                            'stride':2,
                            'is_batch_norm': True,
                            'padding':'SAME',
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type':'conv',
                            'n_output': 32,
                            'filter_size': 1,
                            'activation': None,
                            'stride':1,
                            'is_batch_norm': True,
                            'padding':'SAME',
                            'batch_norm_decay': 0.99,
                        },
                        {
                            'type':'res_end',
                            'id_list':[0,1,2],
                        },
                    ]
            
        if lstm_info_dict is None:
            lstm_info_dict = {
                        'timesteps': 10,
                        'hidden_num': 128,
                    }

        if mlp_layer_info_dict_list is None:
            mlp_layer_info_dict_list = [
                    {
                        'type': 'fc',
                        "n_output": 1024,
                        "is_batch_norm": True,
                        "is_dropout": True,
                        "keep_prob": 0.8,
                        "activation": tf.nn.relu,
                        "batch_norm_decay": 0.99,
                        "name": "mlp_1",
                    },
                    {
                        'type': 'fc',
                        "n_output": 1024,
                        "is_batch_norm": True,
                        "is_dropout": True,
                        "keep_prob": 0.8,
                        "activation": tf.nn.relu,
                        "batch_norm_decay": 0.99,
                        "name": "mlp_2",
                    },
                ]

        if fc_layer_info_dict_list is None:
            fc_layer_info_dict_list = [
                        {
                            'type': 'fc',
                            "n_output": 1024,
                            "is_batch_norm": True,
                            "is_dropout": True,
                            "keep_prob": 0.8,
                            "activation": tf.nn.relu,
                            "batch_norm_decay": 0.99,
                            "name": "fc_1",
                        },
                        {
                            'type': 'fc',
                            "n_output": output_dim,
                            "is_batch_norm": True,
                            "is_dropout": True,
                            "keep_prob": 0.8,
                            "activation": None,
                            "batch_norm_decay": 0.99,
                            "name": "fc_2",
                        },
                    ]

        if clf_layer_info_dict_list is None:
            clf_layer_info_dict_list = [
                        {
                            'type': 'fc',
                            "n_output": 64,
                            "is_batch_norm": True,
                            "is_dropout": True,
                            "keep_prob": 0.8,
                            "activation": tf.nn.relu,
                            "batch_norm_decay": 0.99,
                            "name": "clf_1",
                        },
                        {
                            'type': 'fc',
                            "n_output": output_dim,
                            "is_batch_norm": True,
                            "is_dropout": True,
                            "keep_prob": 0.8,
                            "activation": None,
                            "batch_norm_decay": 0.99,
                            "name": "clf_output",
                        },
                    ]


        self.model_name = model_name

        self.mean = None
        self.std = None
        self.min_loss = None
        self.best_accuracy = None

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_layer_info_dict_list = conv_layer_info_dict_list
        self.lstm_info_dict = lstm_info_dict
        self.fc_layer_info_dict_list = fc_layer_info_dict_list
        self.mlp_layer_info_dict_list = mlp_layer_info_dict_list
        self.clf_layer_info_dict_list= clf_layer_info_dict_list

        self.mode = mode
        self.flag_preprocess = flag_preprocess
        self.tensorboard_path = tensorboard_path

        for key, val in args.items():
            setattr(self, key, val)

        self.g = tf.Graph()
        with self.g.as_default():
            self.build_model()
            self.saver = tf.train.Saver()

        self.var_list = self.g.get_collection('variables')

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.variables_initializer(self.var_list))


    def build_model(self):
        with tf.variable_scope('variable'):
            X = tf.placeholder(tf.float32, shape=[
                               None, self.input_dim], name='X')
            Y = tf.placeholder(tf.float32, shape=[
                               None, self.output_dim], name='Y')

            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            reg_lambda = tf.placeholder(tf.float32, name='reg_lambda')
            global_step = tf.Variable(0, trainable=False)

        with tf.variable_scope("conv"):
            """
            conv phase
            """
            convolved = build_layer_with_info_dict_list(X, 
                                    self.conv_layer_info_dict_list, 
                                    is_training, 
                                    reg_lambda) 
            conv_feature = flatten(convolved)


        with tf.variable_scope("lstm"):
            """
            LSTM Module 
            """
            timesteps = self.lstm_info_dict['timesteps']
            hidden_num = self.lstm_info_dict['hidden_num']
        
            conv_feature_list = tf.split(convolved, timesteps, axis=1)
            conv_feature_list = list(map(flatten, conv_feature_list))
            lstm_input = tf.concat(conv_feature_list, axis=1)

            lstm_input = tf.reshape(lstm_input, (-1, timesteps, lstm_input.get_shape().as_list()[-1] // timesteps))

            print("="*30)
            print("lstm_input shape")
            print(lstm_input.get_shape().as_list())

            outputs, _, _ = bilstm(lstm_input, timesteps, hidden_num)
            lstm_feature = outputs[-1]

            print("="*30)
            print("lstm_feature shape")
            print(lstm_feature.get_shape().as_list())

        with tf.variable_scope("mlp"):
            """
            MLP Module
            """
            mlp_feature = build_layer_with_info_dict_list(X, self.mlp_layer_info_dict_list, 
                                                            is_training, 
                                                            reg_lambda)


        """
        convolved pretrain fc
        """
        with tf.variable_scope("conv"):
            h = conv_feature
            logits_conv_pretrain = build_layer_with_info_dict_list(conv_feature, self.fc_layer_info_dict_list,
                                                                    is_training,
                                                                    reg_lambda)
            Y_pred_conv_pretrain = tf.nn.softmax(logits_conv_pretrain)

        """
        lstm pretrain fc
        """
        with tf.variable_scope("lstm"):
            logits_lstm_pretrain = build_layer_with_info_dict_list(lstm_feature, self.fc_layer_info_dict_list[-1:],
                                                                    is_training,
                                                                    reg_lambda)
            Y_pred_lstm_pretrain = tf.nn.softmax(logits_lstm_pretrain)
        
        """
        mlp pretrain fc
        """
        with tf.variable_scope("mlp"):
            logits_mlp_pretrain = build_layer_with_info_dict_list(mlp_feature, self.fc_layer_info_dict_list[-1:],
                                                                    is_training,
                                                                    reg_lambda)
            Y_pred_mlp_pretrain = tf.nn.softmax(logits_mlp_pretrain)

        """
        clf pretrain fc & global
        """
        with tf.variable_scope("clf"):
            """
            Feature concat
            """
            #feature_concat = tf.concat([conv_feature, mlp_feature, lstm_feature], axis=1)
            feature_concat = tf.concat([Y_pred_mlp_pretrain, Y_pred_conv_pretrain, Y_pred_lstm_pretrain], axis=1)
            print("-"*30)
            print("global feature shape")
            print(h.get_shape().as_list())

            logits = build_layer_with_info_dict_list(feature_concat, self.clf_layer_info_dict_list,
                                                                    is_training,
                                                                    reg_lambda)
            Y_pred = tf.nn.softmax(logits)


        """
        optimization 
        """
        updates_conv_pretrain, accuracy_conv_pretrain, cost_conv_pretrain = self.calc_loss(Y, 
                                                Y_pred_conv_pretrain, 
                                                logits_conv_pretrain, 
                                                learning_rate, 
                                                target_scope=['conv'], 
                                                name='conv')

        updates_lstm_pretrain, accuracy_lstm_pretrain, cost_lstm_pretrain = self.calc_loss(Y, 
                                                Y_pred_lstm_pretrain, 
                                                logits_lstm_pretrain, 
                                                learning_rate, 
                                                target_scope=['lstm'], 
                                                name='lstm')

        updates_mlp_pretrain, accuracy_mlp_pretrain, cost_mlp_pretrain = self.calc_loss(Y, 
                                                Y_pred_mlp_pretrain, 
                                                logits_mlp_pretrain, 
                                                learning_rate, 
                                                target_scope=['mlp'], 
                                                name='mlp')

        updates_clf_pretrain, accuracy_clf_pretrain, cost_clf_pretrain = self.calc_loss(Y, 
                                                Y_pred, 
                                                logits, 
                                                learning_rate, 
                                                target_scope=['clf'], 
                                                name='clf')

        updates, accuracy, cost = self.calc_loss(Y, 
                                                Y_pred, 
                                                logits, 
                                                learning_rate, 
                                                target_scope=None, 
                                                name='global')
                
        """
        tensorboard
        """
        if self.tensorboard_path:
            self.valid_error = tf.summary.scalar('valid_error', cost)   
            self.train_error = tf.summary.scalar('train_error', cost)

            self.valid_accuracy = tf.summary.scalar('valid_accuracy', accuracy)   
            self.train_accuracy = tf.summary.scalar('train_accuracy', accuracy)

            self.summary_writer = tf.summary.FileWriter(self.tensorboard_path, graph=self.g)

        self.X = X
        self.Y = Y
        self.Y_pred = Y_pred
        self.accuracy = accuracy
        self.cost = cost
        self.updates = updates
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.gloabal_step = global_step
        self.reg_lambda = reg_lambda


        self.g.add_to_collection('Y_pred', Y_pred)
        self.g.add_to_collection('X', X)
        self.g.add_to_collection('Y', Y)
        self.g.add_to_collection('accuracy', accuracy)
        self.g.add_to_collection('cost', cost)
        self.g.add_to_collection('is_training', is_training)
        
        """
        conv pretrain
        """
        self.Y_pred_conv_pretrain = Y_pred_conv_pretrain
        self.updates_conv_pretrain = updates_conv_pretrain
        self.cost_conv_pretrain = cost_conv_pretrain
        self.accuracy_conv_pretrain = accuracy_conv_pretrain

        """
        lstm pretrain
        """
        self.Y_pred_lstm_pretrain = Y_pred_lstm_pretrain
        self.updates_lstm_pretrain = updates_lstm_pretrain
        self.cost_lstm_pretrain = cost_lstm_pretrain
        self.accuracy_lstm_pretrain = accuracy_lstm_pretrain

        """
        mlp pretrain
        """
        self.Y_pred_mlp_pretrain = Y_pred_mlp_pretrain
        self.updates_mlp_pretrain = updates_mlp_pretrain
        self.cost_mlp_pretrain = cost_mlp_pretrain
        self.accuracy_mlp_pretrain = accuracy_mlp_pretrain

        """
        clf pretrain
        """
        self.updates_clf_pretrain = updates_clf_pretrain
        self.cost_clf_pretrain = cost_clf_pretrain
        self.accuracy_clf_pretrain = accuracy_clf_pretrain

        self.mode_dict = {
                MODE_TRAIN_GLOBAL: {
                        'updates': self.updates,
                        'Y_pred': self.Y_pred,
                        'accuracy': self.accuracy,
                        'cost': self.cost,
                },
                MODE_TRAIN_CONV: {
                        'updates': self.updates_conv_pretrain,
                        'Y_pred': self.Y_pred_conv_pretrain,
                        'accuracy': self.accuracy_conv_pretrain,
                        'cost': self.cost_conv_pretrain,
                },
                MODE_TRAIN_LSTM: {
                        'updates': self.updates_lstm_pretrain,
                        'Y_pred': self.Y_pred_lstm_pretrain,
                        'accuracy': self.accuracy_lstm_pretrain,
                        'cost': self.cost_lstm_pretrain,
                },
                MODE_TRAIN_MLP: {
                        'updates': self.updates_mlp_pretrain,
                        'Y_pred': self.Y_pred_mlp_pretrain,
                        'accuracy': self.accuracy_mlp_pretrain,
                        'cost': self.cost_mlp_pretrain,
                }, 
                MODE_TRAIN_CLF: {
                        'updates': self.updates_clf_pretrain,
                        'Y_pred': self.Y_pred,
                        'accuracy': self.accuracy_clf_pretrain,
                        'cost': self.cost_clf_pretrain,
                }, }


    def calc_loss(self, Y, Y_pred, logits, learning_rate, target_scope=None, name=None):
        with tf.variable_scope(name or 'calc_loss'):
            with tf.variable_scope('loss'):
                # optimization
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
                if target_scope: 
                    losses = [loss for loss in self.g.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if loss.name.startswith(tuple(target_scope))]
                else:
                    losses = self.g.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

                l2_regularizer = tf.add_n(losses)

                optimizer = tf.train.AdamOptimizer(learning_rate)
                grad_var_tuple_list = []
                clip = tf.constant(5.0, name='clip')

                for grad, var in optimizer.compute_gradients(cost + l2_regularizer):
                    if grad is None:
                        continue
                    if target_scope:
                        if not var.name.startswith(tuple(target_scope)):
                            continue
                    grad_var_tuple_list.append(
                        (tf.clip_by_value(grad, -clip, clip), var))
                updates = optimizer.apply_gradients(grad_var_tuple_list)

            with tf.variable_scope('metric'):
                correct_prediction = tf.equal(
                    tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

        return updates, accuracy, cost

    def evaluate(self, X_target, Y_target, batch_size=32, mode=None):
        if mode is None:
            mode = self.mode

        if self.flag_preprocess:
            X_target = X_target.copy()
            X_target = self.preprocess(X_target)

        Y_pred_list = []
        accuracy = 0.
        loss = 0.

        n_batch = len(X_target) // batch_size
        n_batch += 0 if len(X_target) % batch_size == 0 else 1
        for batch_i in range(n_batch):
            batch_x = X_target[batch_i *
                               batch_size: (batch_i + 1) * batch_size]
            batch_y = Y_target[batch_i *
                               batch_size: (batch_i + 1) * batch_size]

            batch_Y_pred, batch_accuracy, batch_loss = self.sess.run([
                                                                        self.mode_dict[mode]['Y_pred'], 
                                                                        self.mode_dict[mode]['accuracy'], 
                                                                        self.mode_dict[mode]['cost']
                                                                        ],
                                                                        feed_dict={self.X: batch_x, 
                                                                                       self.Y: batch_y,
                                                                                       self.is_training: False})
            accuracy += len(batch_x) * batch_accuracy
            loss += len(batch_y) * batch_loss

            Y_pred_list.append(batch_Y_pred)
        accuracy /= len(X_target)
        loss /= len(X_target)
        Y_pred = np.concatenate(Y_pred_list, axis=0)

        return Y_pred, accuracy, loss


    def train(self, X_train, Y_train, X_valid, Y_valid, batch_size, n_epoch, learning_rate, reg_lambda=0., patience=100, verbose_interval=20, save_dir_path=None, mode=MODE_TRAIN_GLOBAL, **kwargs):
        print("mode : {}".format(mode))
        
        try:
            if self.save_dir_path is None and save_dir_path is None:
                self.save_dir_path = "./tmp/{}".format(generate_id_with_date())

            if save_dir_path:
                self.save_dir_path = save_dir_path

            os.makedirs(self.save_dir_path)
        except Exception as e:
            print("*" * 30)
            print("Make directory with save_dir_path is failed")
            print("Maybe, there is directory already or error because of \"{}\"".format(str(e)))

        X_train_org = X_train
        if self.flag_preprocess:
            print("-" * 30)
            print("preprocess start")
            self.prepare_preprocess(X_train)
            X_train = self.preprocess(X_train)
            print("preprocess done")

        print("-" * 30)
        print("train start")
        patience_origin = patience
        if self.min_loss is None:
            self.min_loss = 9999999.
        for epoch_i in range(n_epoch):
            rand_idx_list = np.random.permutation(range(len(X_train)))
            n_batch = len(rand_idx_list) // batch_size
            for batch_i in range(n_batch):
                rand_idx = rand_idx_list[batch_i *
                                         batch_size: (batch_i + 1) * batch_size]
                batch_x = X_train[rand_idx]
                batch_y = Y_train[rand_idx]
                                   
                self.sess.run(self.mode_dict[mode]['updates'],
                                        feed_dict={self.X: batch_x,
                                        self.Y: batch_y,
                                        self.learning_rate: learning_rate,
                                        self.reg_lambda: reg_lambda,
                                        self.is_training: True})

            _, valid_accuracy, valid_loss = self.evaluate(
                X_valid, Y_valid, batch_size, mode)
            _, train_accuracy, train_loss = self.evaluate(
                X_train_org, Y_train,  batch_size, mode)

            if verbose_interval:
                if epoch_i % verbose_interval == 0:
                    print("-" * 30)
                    print("epoch_i : {}".format(epoch_i))
                    print("train loss: {}, train accuracy: {}".format(
                        train_loss, train_accuracy))
                    print("valid loss: {}, valid accuracy: {}".format(
                        valid_loss, valid_accuracy))
                    print("best valid loss: {}, best valid accuracy : {}".format(
                        self.min_loss, self.best_accuracy))

            if valid_loss < self.min_loss:
                patience = patience_origin + 1

                self.min_loss = valid_loss
                self.best_accuracy = valid_accuracy

                self.meta={
                            'input_dim':self.input_dim,
                            'output_dim':self.output_dim,
                            'min_loss':self.min_loss,
                            'best_accuracy':self.best_accuracy,
                            'mean':self.mean,
                            'std':self.std,
                            'flag_preprocess':self.flag_preprocess,
                        }
                self.save_path = "{}/{}".format(self.save_dir_path, self.model_name)
                self.best_ckpt_path = self.save(self.save_path)

                print("*" * 30)
                print("epoh_i : {}".format(epoch_i))
                print("train loss: {}, train accuracy: {}".format(
                    train_loss, train_accuracy))
                print("valid loss: {}, valid accuracy: {}".format(
                    valid_loss, valid_accuracy))
                print("best valid loss: {}, best valid accuracy : {}".format(
                    self.min_loss, self.best_accuracy))
                print("save current model : {}".format(self.best_ckpt_path))

            patience -= 1
            if patience <= 0:
                break

        self.load(self.best_ckpt_path)

        _, valid_accuracy, valid_loss = self.evaluate(
            X_valid, Y_valid, batch_size, mode)
        _, train_accuracy, train_loss = self.evaluate(
            X_train_org, Y_train,  batch_size, mode)

        date_time_prefix = get_date_time_prefix()
        self.final_model_path = "{}/{}_final_{}".format(self.save_dir_path, date_time_prefix, self.model_name) 
        self.save(self.final_model_path)
        print("*"*30)
        print("final trained performance")
        print("train loss: {}, train accuracy: {}".format(
                    train_loss, train_accuracy))
        print("valid loss: {}, valid accuracy: {}".format(
                    valid_loss, valid_accuracy))
        print("best valid loss: {}, best valid accuracy : {}".format(
                    self.min_loss, self.best_accuracy))
        print("final_model_path: {}".format(self.final_model_path))
        print("train done")
        print("*"*30)

        return self.sess
