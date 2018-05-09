import os
import tensorflow as tf
import numpy as np

from models.tf_template import BaseTfClassifier
from models.tf_layer import batch_norm_wrapper, flatten, linear

from libs.various_utils import generate_id_with_date, get_date_time_prefix

class OSELM(BaseTfClassifier):
    """
    Based on : https://github.com/iwyoo/OSELM-tensorflow
    """

    def __init__(self, 
                    input_dim, 
                    output_dim,
                    hidden_num, 
                    batch_size, 
                    flag_preprocess=True,
                    model_name='oselm',
                    **kwargs):
        super().__init__()

        # general variables
        self.model_name = model_name
        self.mean = None
        self.std = None
        self.min_loss = None
        self.best_accuracy = None

        self.flag_preprocess = flag_preprocess

        # model specific variables
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_num = hidden_num

        for key, val in kwargs.items():
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
        with tf.variable_scope("train_input"):
            X_batch = tf.placeholder(dtype=tf.float32,
                                shape=[self.batch_size, self.input_dim])
            T_batch = tf.placeholder(dtype=tf.float32,
                                shape=[self.batch_size, self.output_dim])

        with tf.variable_scope("variable"):
            W = tf.get_variable(name="W",
                                shape=[self.input_dim, self.hidden_num],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0., stddev=1.),
                                trainable=False)
            b = tf.get_variable(name="b",
                                shape=[self.hidden_num],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0, stddev=1.),
                                trainable=False)
            
            beta = tf.get_variable(name="beta",
                                shape=[self.hidden_num, self.output_dim],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0),
                                trainable=False)


        with tf.variable_scope("train"):
            P = tf.get_variable(name="P",
                                shape=[self.hidden_num, self.hidden_num],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0),
                                trainable=False)
            H = tf.get_variable(name="H",
                                shape=[self.batch_size, self.hidden_num],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0),
                                trainable=False)

            set_H = H.assign(tf.nn.sigmoid(tf.matmul(X_batch, W) + b))
            H_T = tf.transpose(H)

            t_P0 = tf.matrix_inverse(tf.matmul(H_T, H), name='matrix_inverse_t_P0')
            init_P0 = P.assign(t_P0)

            t_beta0 = tf.matmul(tf.matmul(P, H_T), T_batch)
            init_beta = beta.assign(t_beta0)

            flag_init = False

        with tf.variable_scope("sequential_training"):
            t_beta = tf.get_variable(name="t_beta",
                                    shape=[self.hidden_num, self.output_dim], 
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)
            t_P = tf.get_variable(name="t_P",
                                    shape=[self.hidden_num, self.hidden_num],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)

            swap_P = t_P.assign(P)
            swap_beta = t_beta.assign(beta)

            eye = tf.constant(np.identity(self.batch_size), dtype=tf.float32) 
            t_P1 = t_P - tf.matmul(tf.matmul(tf.matmul(tf.matmul(t_P, H_T),
                                tf.matrix_inverse(eye+tf.matmul(tf.matmul(H, t_P), H_T), name='matrix_inv_t_P1')), H), t_P)

            t_beta1 = t_beta + tf.matmul(tf.matmul(t_P1, H_T), (T_batch - tf.matmul(H, t_beta)) )

            update_P = P.assign(t_P1)
            update_beta = beta.assign(t_beta1)
            
        with tf.variable_scope("predict"):
            X = tf.placeholder(tf.float32,
                                shape=[None, self.input_dim])
            Y = tf.placeholder(tf.float32,
                                shape=[None, self.output_dim])
            
            # this is dummy
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            h = tf.nn.sigmoid(tf.matmul(X, W) +b)
            h = tf.matmul(h, beta)
            Y_pred = tf.nn.softmax(h)

        with tf.variable_scope("metric"):
            correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(
                            tf.cast(correct_prediction, tf.float32))
            cost = -tf.reduce_mean(Y * tf.log(Y_pred + 1e-12))

        # dummy 
        self.is_training = tf.placeholder(dtype=tf.bool, name='dummy_is_training')

        self.flag_init = flag_init
        self.X_batch = X_batch
        self.T_batch = T_batch

        self.X = X
        self.Y = Y
        self.Y_pred = Y_pred
        self.accuracy = accuracy
        self.cost = cost

        self.set_H = set_H
        self.init_P0 = init_P0
        self.init_beta = init_beta
        self.swap_P = swap_P
        self.swap_beta = swap_beta
        self.update_P = update_P
        self.update_beta = update_beta

        self.g.add_to_collection('Y_pred', Y_pred)
        self.g.add_to_collection('X', X)
        self.g.add_to_collection('Y', Y)
        self.g.add_to_collection('accuracy', accuracy)
        self.g.add_to_collection('cost', cost)
        self.g.add_to_collection('is_training', is_training)
    
    def train(self, X_train, Y_train, X_valid, Y_valid, save_dir_path='./tmp', **kwargs):

        try:
            if self.save_dir_path  is None and save_dir_path is None:
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
            self.prepare_preprocess(X_train)
            X_train = self.preprocess(X_train)

        if self.min_loss is None:
            self.min_loss = 999999999
            
        rand_idx_list = np.random.permutation(range(len(X_train)))
        n_batch = len(rand_idx_list) // self.batch_size
        for batch_i in range(n_batch):
            rand_idx = rand_idx_list[batch_i *
                                     self.batch_size: (batch_i + 1) * self.batch_size]
            batch_x = X_train[rand_idx]
            batch_y = Y_train[rand_idx]

            h = self.sess.run(self.set_H, feed_dict={self.X_batch:batch_x})
            
            if not self.flag_init:
                self.sess.run(self.init_P0)
                self.sess.run(self.init_beta, feed_dict= {self.T_batch:batch_y})
                self.flag_init=True
            else:
                self.sess.run(self.swap_P)
                self.sess.run(self.swap_beta)
                self.sess.run(self.update_P)
                self.sess.run(self.update_beta, {self.T_batch:batch_y})
            
        _, valid_accuracy, valid_loss = self.evaluate(
            X_valid, Y_valid, self.batch_size)
        _, train_accuracy, train_loss = self.evaluate(
            X_train_org, Y_train, self.batch_size)

        print("*"*30)
        self.min_loss = valid_loss
        self.best_accuracy = valid_accuracy
    
        print("*" * 30)
        print("train loss: {}, train accuracy: {}".format(
            train_loss, train_accuracy))
        print("valid loss: {}, valid accuracy: {}".format(
            valid_loss, valid_accuracy))
        print("best valid loss: {}, best valid accuracy : {}".format(
            self.min_loss, self.best_accuracy))

        self.meta={
                    'input_dim':self.input_dim,
                    'output_dim':self.output_dim,
                    'min_loss':self.min_loss,
                    'best_accuracy':self.best_accuracy,
                    'mean':self.mean,
                    'std':self.std,
                    'flag_preprocess':self.flag_preprocess,
                    }

        date_time_prefix = get_date_time_prefix()
        self.final_model_path = "{}/{}_final_{}".format(self.save_dir_path, date_time_prefix, self.model_name) 
        self.save(self.final_model_path) 
        print("final_model_path: {}".format(self.final_model_path))
        print("*"*30)

        return self.sess

