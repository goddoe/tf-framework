#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

from models.tf_template import BaseTfClassifier
from models.tf_layer import batch_norm_wrapper, flatten, linear, conv1d, bn_bilstm, bilstm, build_layer_with_info_dict_list


class LSTM(BaseTfClassifier):
    def __init__(self, input_dim, output_dim,       
                 lstm_info_dict=None,
                 fc_layer_info_dict_list=None,
                 output_layer_info_dict=None,
                 optimizer=None,
                 cost_function=None,
                 flag_preprocess=False,
                 tensorboard_path=None,
                 model_name='lstm',
                 **args
                 ):
        super().__init__()

        self.model_name = model_name

        if lstm_info_dict is None:
            lstm_info_dict = {
                        'timesteps': 250,
                        'hidden_num': 128,
                    }

        if output_layer_info_dict is None:
            output_layer_info_dict = {
                            'type': 'fc',
                            "n_output": output_dim,
                            "is_batch_norm": True,
                            "activation": tf.nn.softmax,
                            "batch_norm_decay": 0.99,
                            "name": "output",
                        }

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        if cost_function is None:
            cost_function=lambda Y_pred, Y: -tf.reduce_mean(Y * tf.log(Y_pred + 1e-12))


        self.mean = None
        self.std = None
        self.min_loss = None
        self.best_accuracy = None

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_info_dict = lstm_info_dict
        self.fc_layer_info_dict_list = fc_layer_info_dict_list
        self.output_layer_info_dict = output_layer_info_dict

        self.optimizer = optimizer
        self.cost_function = cost_function
        
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

            timesteps = self.lstm_info_dict['timesteps']
            hidden_num = self.lstm_info_dict['hidden_num']
            X_reshaped = tf.reshape(X, (-1, timesteps,int(self.input_dim/timesteps ) ))

            outputs, _, _ = bilstm(X_reshaped, timesteps, hidden_num)
            h = outputs[-1]

            print(outputs)
            print("*"*30)
            print("h")
            
            print(h.get_shape().as_list())
        
        with tf.variable_scope("fc"):

            if self.fc_layer_info_dict_list is not None:
                h = build_layer_with_info_dict_list(h, self.fc_layer_info_dict_list, is_training, reg_lambda)

            h = build_layer_with_info_dict_list(h, [self.output_layer_info_dict], is_training, reg_lambda)

            logits = h
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
