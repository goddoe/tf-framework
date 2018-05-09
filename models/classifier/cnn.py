#!/usr/bin/env python3
from pprint import pprint
import tensorflow as tf

from models.tf_template import BaseTfClassifier
from models.tf_layer import (build_layer_with_info_dict_list,
                             flatten)


class CNN(BaseTfClassifier):
    def __init__(self, input_dim, output_dim,
                 cnn_layer_info_dict_list=None,
                 fc_layer_info_dict_list=None,
                 output_layer_info_dict=None,
                 optimizer=None,
                 cost_function=None,
                 flag_preprocess=False,
                 tensorboard_path=None,
                 model_name='cnn',
                 **args
                 ):
        super().__init__()

        if cnn_layer_info_dict_list is None:
            cnn_layer_info_dict_list = [
                {
                    'type': 'conv',
                    'n_input': 1,
                    'n_output': 32,
                    'filter_size': 3,
                    'activation': tf.nn.relu,
                    'is_batch_norm': True,
                    'batch_norm_decay': 0.99,
                },
                {
                    'type': 'conv',
                    'n_input': 32,
                    'n_output': 32,
                    'filter_size': 3,
                    'activation': tf.nn.relu,
                    'is_batch_norm': True,
                    'batch_norm_decay': 0.99,
                }, ]

        if fc_layer_info_dict_list is None:
            fc_layer_info_dict_list = [
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
        if output_layer_info_dict is None:
            output_layer_info_dict = {
                'type': 'fc',
                        "n_output": output_dim,
                        "is_batch_norm": True,
                        "is_dropout": False,
                        "keep_prob": 0.8,
                        "batch_norm_decay": 0.99,
                        "name": "output",
            },

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        if cost_function is None:
            def cost_function(Y_pred, Y): return - \
                tf.reduce_mean(Y * tf.log(Y_pred + 1e-12))

        self.model_name = model_name

        self.mean = None
        self.std = None
        self.min_loss = None
        self.best_accuracy = None

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cnn_layer_info_dict_list = cnn_layer_info_dict_list
        self.fc_layer_info_dict_list = fc_layer_info_dict_list
        self.output_layer_info_dict = output_layer_info_dict

        self.cost_function = cost_function
        self.optimizer = optimizer
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
        self.sess = tf.Session(
            graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.variables_initializer(self.var_list))

        print('')
        print("=" * 30)
        pprint(cnn_layer_info_dict_list)
        print("-" * 30)
        pprint(fc_layer_info_dict_list)
        print("=" * 30)

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

        with tf.variable_scope("cnn_model"):
            cnn_output = build_layer_with_info_dict_list(
                X, self.cnn_layer_info_dict_list, is_training, reg_lambda)

            cnn_output_flattened = flatten(cnn_output)
            print("-" * 30)
            print("conv last shape")
            print(cnn_output.get_shape().as_list())
            print("-" * 30)
            print("conv feature shape")
            print(cnn_output_flattened.get_shape().as_list())

            h = build_layer_with_info_dict_list(
                cnn_output_flattened, self.fc_layer_info_dict_list, is_training, reg_lambda)

            logits = build_layer_with_info_dict_list(
                h, [self.output_layer_info_dict], is_training, reg_lambda)
            Y_pred = tf.nn.softmax(logits)

        with tf.variable_scope('loss'):
            # optimization
            cost = self.cost_function(Y_pred, Y)
            l2_regularizer = tf.add_n(self.g.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES))

            optimizer = self.optimizer(learning_rate)

            grad_var_tuple_list = []
            clip = tf.constant(5.0, name='clip')
            for grad, var in optimizer.compute_gradients(cost + l2_regularizer):
                grad_var_tuple_list.append(
                    (tf.clip_by_value(grad, -clip, clip), var))
            updates = optimizer.apply_gradients(grad_var_tuple_list)

        with tf.variable_scope('metric'):
            correct_prediction = tf.equal(
                tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

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
