import pprint
from collections import OrderedDict
from functools import reduce
from operator import mul

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np


def build_layer_with_info_dict_list(X, info_dict_list, is_training, reg_lambda):
    current_input = X
    res_dict = {}
    for layer_i, info in enumerate(info_dict_list):
        if 'name' not in info:
            info['name'] = "{}_{}".format(info['type'], layer_i)

        if info['type'] == 'conv':
            current_input, _, _ = conv1d(X=current_input,
                                         is_training=is_training,
                                         reg_lambda=reg_lambda,
                                         **info)

        elif info['type'] == 'res_start':
            res_dict[info['id']] = current_input

        elif info['type'] == 'res_end':
            add_list = []
            add_list.append(current_input)

            for res_id in info['id_list']:
                res = res_dict[res_id]

                add_list.append(conv1d(X=res,
                                       n_output=current_input.get_shape(
                                       ).as_list()[-1],
                                       filter_size=1,
                                       activation=None,
                                       stride=int(round(res.get_shape().as_list(
                                       )[-2] / current_input.get_shape().as_list()[-2])),
                                       padding='SAME',
                                       is_batch_norm=True,
                                       is_training=is_training,
                                       batch_norm_decay=0.99,
                                       reg_lambda=reg_lambda,
                                       name="{}_{}".format(info['name'], res_id))[0])

            current_input = tf.add_n(add_list)
            current_input = tf.nn.relu(current_input)

        elif info['type'] == 'pool':
            if 'window_shape' in info:
                info['window_shape'] = info['window_shape'] if info['window_shape'] != - \
                    1 else [current_input.get_shape().as_list()[-2]]

            current_input = tf.nn.pool(current_input,
                                       window_shape=info['window_shape'],
                                       pooling_type=info['pooling_type'],
                                       padding=info['padding'],
                                       strides=info['strides'],)

        elif info['type'] == 'transform':
            current_input = info['transform_function'](current_input)

        elif info['type'] == 'fc':
            current_input, _ = linear(X=current_input,
                                      is_training=is_training,
                                      reg_lambda=reg_lambda,
                                      **info)
        elif info['type'] == 'pad':
            current_input = tf.pad(
                current_input, info['paddings'], info['mode'])

        print("-" * 30)
        print("{}_{} output shape".format(info['type'], layer_i))
        print(current_input.get_shape().as_list())
    output = current_input
    return output


def batch_norm_wrapper(X, is_training, decay=0.9, name=None, reuse=None,
                       affine=True):
    with tf.variable_scope(name or 'batch_normalization', reuse=reuse):
        shape = X.get_shape().as_list()
        beta = tf.get_variable(name='beta', shape=[shape[-1]],
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[shape[-1]],
                                initializer=tf.constant_initializer(1.0),
                                trainable=affine)

        pop_mean = tf.Variable(tf.zeros([shape[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([shape[-1]]), trainable=False)

        def train_phase():
            batch_mean, batch_var = tf.nn.moments(X,
                                                  [dim for dim in range(len(shape) - 1)], name='batch_moments')
            train_mean = tf.assign(
                pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(
                pop_var, pop_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(X, batch_mean, batch_var, beta, gamma, 1e-6)

        def test_phase():
            return tf.nn.batch_normalization(X, pop_mean, pop_var, beta, gamma, 1e-6)

    return tf.cond(is_training, train_phase, test_phase)


def flatten(x, name=None):
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) >= 3:
            dim = reduce(mul, dims[1:])
            flattened = tf.reshape(
                x,
                shape=[-1, dim])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2,  4.  Found:',
                             len(dims))

        return flattened


def linear(X, n_output,
           is_batch_norm=False,
           is_training=False,
           name=None,
           activation=None,
           batch_norm_decay=0.9,
           reg_lambda=0.,
           is_dropout=True,
           keep_prob=0.5,
           **kwargs):

    if len(X.get_shape()) != 2:
        X = flatten(X)

    n_input = X.get_shape().as_list()[1]
    with tf.variable_scope(name or "fc"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(reg_lambda))

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(X, W),
            bias=b)

        if is_batch_norm:
            h = batch_norm_wrapper(h, is_training, decay=batch_norm_decay)

        if is_dropout:
            kp = tf.cond(is_training,
                         lambda: keep_prob,
                         lambda: 1.)
            h = tf.nn.dropout(h, kp)

        if activation:
            h = activation(h)

        return h, W


def conv1d(X,
           n_output,
           filter_size,
           activation=None,
           stride=2,
           padding='VALID',
           is_batch_norm=False,
           is_training=False,
           
           name=None,
           W=None,
           b=None,
           batch_norm_decay=0.9,
           reg_lambda=0.,
           **kwargs):

    n_input = X.get_shape().as_list()[-1]
    with tf.variable_scope(name or "conv"):
        if W is None:
            W = tf.get_variable(
                name='W',
                shape=[filter_size, n_input, n_output],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(reg_lambda))

        if b is None:
            b = tf.get_variable(
                name='b',
                shape=[n_output],
                initializer=tf.constant_initializer(0.))

        h = tf.nn.conv1d(X,
                         W,
                         stride=stride,
                         padding=padding
                         )

        h = tf.nn.bias_add(h, b)
        if is_batch_norm:
            h = batch_norm_wrapper(h, is_training, decay=batch_norm_decay)

        if activation:
            h = activation(h)

    return h, W, b


def bigru(X,
          timesteps,
          hidden_num,
          initial_state_fw=None,
          initial_state_bw=None,
          name=None):
    with tf.variable_scope(name or 'bigru'):
        gru_fw_cell = tf.contrib.rnn.GRUCell(hidden_num)
        gru_bw_cell = tf.contrib.rnn.GRUCell(hidden_num)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell,
                                                          gru_bw_cell,
                                                          X,
                                                          initial_state_fw=initial_state_fw,
                                                          initial_state_bw=initial_state_bw,
                                                          dtype=tf.float32)
    return outputs, states


def bilstm(X,
           timesteps,
           hidden_num,
           initial_state_fw=None,
           initial_state_bw=None,
           name=None):
    with tf.variable_scope(name or 'bilstm'):
        X = tf.unstack(X, timesteps, 1)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
            hidden_num, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
            hidden_num, forget_bias=1.0)
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                                            lstm_bw_cell,
                                                                                            X,
                                                                                            dtype=tf.float32)
    return outputs, output_state_fw, output_state_bw


def bn_bilstm(X,
              timesteps,
              hidden_num,
              is_training,
              batch_norm_decay=0.9,
              name=None):
    with tf.variable_scope(name or 'bilstm'):
        X = tf.unstack(X, timesteps, 1)
        lstm_fw_cell = BNLSTMCell(hidden_num, is_training, batch_norm_decay)
        lstm_bw_cell = BNLSTMCell(hidden_num, is_training, batch_norm_decay)
        outputs, output_state_fw, output_state_bw = tf.nn.static_bidirectional_rnn(lstm_fw_cell,
                                                                                   lstm_bw_cell,
                                                                                   X,
                                                                                   dtype=tf.float32)
    return outputs, output_state_fw, output_state_bw


class BNLSTMCell(tf.contrib.rnn.RNNCell):
    """
    Batch normalized LSTM as described in arxiv.org/abs/1603.09025
    Based on :https://github.com/OlavHN/bnlstm  
    """

    def __init__(self, num_units, is_training, batch_norm_decay=0.9):
        self.num_units = num_units
        self.is_training = is_training
        self.batch_norm_decay = batch_norm_decay

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                                   [x_size, 4 * self.num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                                   [self.num_units, 4 * self.num_units],
                                   initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm_wrapper(
                xh, self.is_training, self.batch_norm_decay, name='xh')
            bn_hh = batch_norm_wrapper(
                hh, self.is_training, self.batch_norm_decay, name='hh')

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(hidden, 4, 1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = batch_norm_wrapper(
                new_c, self.is_training, self.batch_norm_decay, name='c')

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)


def orthogonal(shape):
    """
    reference :https://github.com/OlavHN/bnlstm
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def bn_lstm_identity_initializer(scale):
    """
    reference :https://github.com/OlavHN/bnlstm  
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype)

    return _initializer


def orthogonal_initializer():
    """
    reference :https://github.com/OlavHN/bnlstm  
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer


class HNode(object):

    def __init__(self,
                 X,
                 name,
                 child_ordered_dict=None,
                 prob=1.,
                 level=0,
                 is_training=None,
                 reg_lambda=0.,
                 param_dict=None):
        self.name = name
        self.prob = prob  # root : scalar, child : 2d tensor
        self.child_ordered_dict = child_ordered_dict
        self.level = level

        self.W = None
        self.h = None
        self.child_ordered_node_dict = None

        if child_ordered_dict:
            self.h, self.W = linear(X,
                                    n_output=len(child_ordered_dict),
                                    is_training=is_training,
                                    activation=tf.nn.softmax,
                                    reg_lambda=reg_lambda,

                                    is_batch_norm=param_dict['is_batch_norm'],
                                    batch_norm_decay=param_dict['batch_norm_decay'],
                                    is_dropout=param_dict['is_dropout'],
                                    keep_prob=param_dict['keep_prob'],

                                    name="prob_{}".format(self.name))

            self.child_prob = self.h * self.prob

            self.child_ordered_node_dict = OrderedDict()
            for idx, (ch_name, ch_dict) in enumerate(child_ordered_dict.items()):
                self.child_ordered_node_dict[ch_name] = HNode(X,
                                                              name=ch_name,
                                                              child_ordered_dict=ch_dict,
                                                              prob=self.child_prob[:,
                                                                                   idx:idx + 1],
                                                              level=level + 1,
                                                              is_training=is_training,
                                                              reg_lambda=reg_lambda,
                                                              param_dict=param_dict)

        # shortcut
        self.c = self.child_ordered_node_dict

    def __str__(self):
        return "{} : {}".format(self.name, pprint.pformat(self.child_ordered_dict))

    def calc_distribution(self):
        if self.child_ordered_dict:
            prob_tensor_list = []
            name_list = []
            for _, child in self.child_ordered_node_dict.items():
                prob_tensor, child_name_list = child.calc_distribution()
                prob_tensor_list.append(prob_tensor)
                name_list.extend(child_name_list)
            return tf.concat(prob_tensor_list, axis=1), name_list
        else:
            return self.prob, [self.name]


"""
tf.reset_default_graph()

X = tf.placeholder(name='X', shape=[None, 4], dtype=tf.float32)
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

d = OrderedDict
child_ordered_dict = d( [
                            ('loan', d([
                                            ('label_1_1', None), 
                                            ('label_1_2', None)
                                        ])),

                            ('deposit', d([
                                            ('label_2_1', None), 
                                            ('label_2_2', None)
                                        ])),

                            ('foreign_exchange', d([
                                            ('label_3_1', None), 
                                            ('label_3_2', None)
                                        ])),

                            ('fund', d([
                                            ('label_4_1', None), 
                                            ('label_4_2', None)
                                        ])),
                        ]
                    ) 

root = HNode(X, 'root', child_ordered_dict,is_training=is_training)

prob, name_list = root.calc_distribution()
print(prob)

print(root)
"""
