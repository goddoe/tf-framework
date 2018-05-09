import os
import tensorflow as tf
import numpy as np

from models.tf_template import BaseTfClassifier
from libs.utils import (get_X_length,
                        token_list_list_to_idx_list_list,
                        idx_list_list_to_token_list_list)
from libs.various_utils import (generate_id_with_date,
                                get_date_time_prefix,
                                print_metric,
                                makedirs)


class Char2CharEncoder(BaseTfClassifier):
    def __init__(self,
                 hidden_num=None,
                 vocab_size=None,
                 embed_dim=None,

                 tensorboard_path=None,
                 model_name='Char2CharEncoder',
                 **kwargs
                 ):
        super().__init__()

        self.model_name = model_name

        self.min_loss = None
        self.best_sa_accuracy = None

        self.vocab_size = vocab_size
        self.hidden_num = hidden_num
        self.embed_dim = embed_dim

        self.tensorboard_path = tensorboard_path

        self.report_dict.update({'valid_sa_accuracy': [],
                                'train_sa_accuracy': []})

        for key, val in kwargs.items():
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

        # For compatibility, will remove soon

    def encoder(self, X, X_length, hidden_num):
        with tf.variable_scope("encoder"):
            fw_cell = tf.contrib.rnn.GRUCell(hidden_num)
            bw_cell = tf.contrib.rnn.GRUCell(hidden_num)

            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_final_state,
              encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                                            fw_cell,
                                            bw_cell,
                                            inputs=X,
                                            sequence_length=X_length,
                                            dtype=tf.float32,
                                            time_major=False)

            encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], 2)
            encoder_final_state = tf.concat([encoder_fw_final_state, encoder_bw_final_state], 1)
           
        return encoder_outputs, encoder_final_state

    def decoder(self, encoder_final_state, X_length, X, embeddings, hidden_num, vocab_size, name=None):
        with tf.variable_scope(name or "decoder"):
            with tf.variable_scope("projection"):
                W = tf.Variable(tf.random_uniform([hidden_num, vocab_size], -1, 1), dtype=tf.float32)
                b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32) 

            with tf.variable_scope("decoding"):
                decoder_cell = tf.contrib.rnn.GRUCell(hidden_num)
                batch_size, max_timesteps = tf.unstack(tf.shape(X)) 
        
                decoder_lengths = max_timesteps + 3  # padding

                eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
                pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')
                
                eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
                pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

                def loop_fn_initial():
                    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
                    initial_input = eos_step_embedded
                    initial_cell_state = encoder_final_state
                    initial_cell_output = None
                    initial_loop_state = None
                    return (initial_elements_finished,
                            initial_input,
                            initial_cell_state,
                            initial_cell_output,
                            initial_loop_state)

                def loop_fn_transition(time,
                                       previous_output,
                                       previous_state,
                                       previous_loop_state):
                    def get_next_input():
                        output_logits = tf.add(tf.matmul(previous_output, W), b)
                        prediction = tf.argmax(output_logits, axis=1)
                        next_input = tf.nn.embedding_lookup(embeddings, prediction)
                        return next_input

                    # this operation produces boolean tensor of [batch_size]
                    # defining if corresponding sequence has ended
                    elements_finished = (time >= decoder_lengths)

                    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
                    next_input = tf.cond(finished, lambda:pad_step_embedded, get_next_input)
                    state = previous_state
                    output = previous_output
                    loop_state = None

                    return (elements_finished,
                            next_input,
                            state,
                            output,
                            loop_state)

                def loop_fn(time,
                            previous_output,
                            previous_state,
                            previous_loop_state):
                    if previous_state is None:  # time == 0
                        assert previous_output is None and previous_state is None, "previous is None"
                        return loop_fn_initial()
                    else:
                        return loop_fn_transition(time,
                                                  previous_output,
                                                  previous_state,
                                                  previous_loop_state)
                
                (decoder_outputs_ta,
                 decoder_final_state,
                 _) = tf.nn.raw_rnn(decoder_cell, loop_fn)
                decoder_outputs_time_major = decoder_outputs_ta.stack()

                decoder_outputs = tf.transpose(
                    decoder_outputs_time_major, [1, 0, 2]) 

                (decoder_batch_size,
                 decoder_max_steps,
                 decoder_dim) = tf.unstack(tf.shape(decoder_outputs))
                decoder_outputs_flat = tf.reshape(
                    decoder_outputs, (-1, decoder_dim))
                decoder_logits_flat = tf.add(
                    tf.matmul(decoder_outputs_flat, W), b)
                decoder_logits = tf.reshape(
                    decoder_logits_flat,
                    (decoder_batch_size, decoder_max_steps, vocab_size))
                decoder_prediction = tf.argmax(decoder_logits, 2)

        return decoder_prediction, decoder_logits

    def build_model(self):
        with tf.variable_scope('model_variables'):
            X = tf.placeholder(tf.int32, shape=[
                               None, None], name='X')

            Y = tf.placeholder(tf.int32, shape=[
                               None, None], name='Y')

            X_length = tf.placeholder(tf.int32,
                                      shape=[None, ],
                                      name='X_length')

            embeddings = tf.get_variable(name="embeddings",
                                         shape=[self.vocab_size, self.embed_dim],
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-1., 1.))

            X_embedded = tf.nn.embedding_lookup(embeddings, X)

            # learning params
            is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
            reg_lambda = tf.placeholder_with_default(0., shape=None, name='reg_lambda')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            global_step = tf.Variable(0, trainable=False)
   
        with tf.variable_scope("seq2seq"):
            (encoder_outputs,
             encoder_final_state) = self.encoder(X_embedded,
                                                 X_length,
                                                 self.hidden_num)
            (decoder_prediction,
             decoder_logits) = self.decoder(encoder_final_state,
                                            X_length,
                                            X,
                                            embeddings,
                                            self.hidden_num*2,
                                            self.vocab_size)

        """
        loss
        """
        
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=decoder_logits,
                labels=tf.one_hot(
                    Y, depth=self.vocab_size, dtype=tf.float32)))

        optimizer = tf.train.AdamOptimizer(learning_rate)

        seq2seq_scope_list = ['seq2seq', 'model_variables']
        seq2seq_var_list = [seq2seq_var
                            for scope in seq2seq_scope_list
                            for seq2seq_var in tf.contrib.framework.get_variables(scope)]

        updates = self.optimize(cost=cost,
                                optimizer=optimizer,
                                target_scope=None,
                                var_list=seq2seq_var_list,
                                name='optimize')
        
        """
        tensorboard
        """
        
        self.X = X
        self.X_length = X_length
        self.Y = Y

        self.encoder_outputs = encoder_outputs
        self.encoder_final_state = encoder_final_state

        self.decoder_logits = decoder_logits
        self.decoder_prediction = decoder_prediction
        self.embeddings = embeddings

        self.is_training = is_training
        self.learning_rate = learning_rate
        self.gloabal_step = global_step
        self.reg_lambda = reg_lambda

        self.updates = updates
        self.cost = cost

        self.g.add_to_collection('X', X)
        self.g.add_to_collection('Y', Y)
        self.g.add_to_collection('X_length', X_length)
        self.g.add_to_collection('encoder_outputs', encoder_outputs)
        self.g.add_to_collection('encoder_final_state', encoder_final_state)

        self.g.add_to_collection('decoder_logits', decoder_logits)
        self.g.add_to_collection('decoder_prediction', decoder_prediction)
        self.g.add_to_collection('embeddings', embeddings)

        self.g.add_to_collection('is_training', is_training)

        self.g.add_to_collection('updates', updates)
        self.g.add_to_collection('cost', cost)

    def optimize(self,
                 cost,
                 optimizer,
                 target_scope=None,
                 var_list=None,
                 clip_grad_val=1.0,
                 name=None):
        with tf.variable_scope(name or 'calc_loss'):
            with tf.variable_scope('loss'):
                losses = []
                l2_regularizer = 0.
                if target_scope:
                    losses = [loss
                              for loss in self.g.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                              if loss.name.startswith(tuple(target_scope))]
                else:
                    losses = self.g.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES)

                if losses:
                    l2_regularizer = tf.add_n(losses)

                grad_var_tuple_list = []
                clip = tf.constant(clip_grad_val, name='clip')

                for grad, var in optimizer.compute_gradients(cost + l2_regularizer, var_list=var_list):
                    if grad is None:
                        continue
                    if target_scope:
                        if not var.name.startswith(tuple(target_scope)):
                            continue
                    grad_var_tuple_list.append(
                        (tf.clip_by_value(grad, -clip, clip), var))
                updates = optimizer.apply_gradients(grad_var_tuple_list)

        return updates

    def calc_metric(self, Y, Y_pred, name=None):
        with tf.variable_scope(name or 'metric'):
            correct_prediction = tf.equal(
                tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        return accuracy, correct_prediction

    def token_to_idx(self, token_list_list):
        if 'word_idx_dict' not in self.meta:
            raise Exception("There is no word_idx_dict in meta")
        return token_list_list_to_idx_list_list(
            token_list_list, self.meta['word_idx_dict'])

    def idx_to_token(self, idx_list_list):
        if 'idx_word_dict' not in self.meta:
            raise Exception("There is no idx_word_dict in meta")
        return idx_list_list_to_token_list_list(
            idx_list_list, self.meta['idx_word_dict'])

    def evaluate(self, X, Y, X_length=None, batch_size=32):
        if X_length is None:
            X_length = get_X_length(X)

        loss = 0.

        n_batch = len(X) // batch_size
        n_batch += 0 if len(X) % batch_size == 0 else 1
        for batch_i in range(n_batch):
            batch_x = X[batch_i *
                        batch_size: (batch_i + 1) * batch_size]
            batch_y = Y[batch_i *
                        batch_size: (batch_i + 1) * batch_size]
            batch_x_len = X_length[batch_i *
                                   batch_size: (batch_i + 1) * batch_size]

            batch_loss = self.sess.run(self.cost,
                                       feed_dict={self.X: batch_x,
                                                  self.Y: batch_y,
                                                  self.X_length: batch_x_len,
                                                  self.is_training: False})
            loss += len(batch_y) * batch_loss
        loss /= len(X)

        return loss

    def evaluate_with_da(self, X, X_length, Y, dataset_init_op):
        total_num = 0
        loss = 0.

        self.sess.run(dataset_init_op)
        while True:
            try:
                X_batch, X_length_batch, Y_batch = self.sess.run([X, X_length, Y])
                batch_loss = self.sess.run(self.cost,
                                           feed_dict={
                                               self.X: X_batch,
                                               self.X_length: X_length_batch,
                                               self.Y: Y_batch,
                                               self.is_training: False})

                total_num += X_batch.shape[0]
                loss += X_batch.shape[0] * batch_loss
            except tf.errors.OutOfRangeError:
                break

        loss /= total_num
        return loss, total_num

    def encode(self, X, X_length=None, batch_size=32):
        if X_length is None:
            X_length = get_X_length(X)

        encoder_outputs_list = []
        encoder_final_state_list = []

        n_batch = len(X) // batch_size
        n_batch += 0 if len(X) % batch_size == 0 else 1
        for batch_i in range(n_batch):
            batch_x = X[batch_i *
                        batch_size: (batch_i + 1) * batch_size]
            batch_x_len = X_length[batch_i *
                                   batch_size: (batch_i + 1) * batch_size]

            (batch_encoder_outputs,
             batch_encoder_final_state) = self.sess.run(
                [self.encoder_outputs, self.encoder_final_state],
                feed_dict={self.X: batch_x,
                           self.X_length: batch_x_len,
                           self.is_training: False})

            encoder_outputs_list.append(batch_encoder_outputs)
            encoder_final_state_list.append(batch_encoder_final_state)

        encoder_outputs = np.concatenate(encoder_outputs_list, axis=0)
        encoder_final_state = np.concatenate(encoder_final_state_list, axis=0)

        return encoder_outputs, encoder_final_state

    def train(self,
              X_train,
              Y_train,
              X_valid,
              Y_valid,
              batch_size,
              n_epoch,
              learning_rate,
              reg_lambda=0.,
              patience=10,
              verbose_interval=1,
              save_dir_path=None,
              **kwargs):

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

        print("-" * 30)
        print("train start")
        patience_origin = patience
        if self.min_loss is None:
            self.min_loss = 999999999.

        X_length_train = get_X_length(X_train)
        X_length_valid = get_X_length(X_valid)

        for epoch_i in range(n_epoch):
            rand_idx_list = np.random.permutation(range(len(X_train)))
            n_batch = len(rand_idx_list) // batch_size
            for batch_i in range(n_batch):
                rand_idx = rand_idx_list[batch_i *
                                         batch_size: (batch_i + 1) * batch_size]
                batch_x = X_train[rand_idx]
                batch_y = Y_train[rand_idx]
                batch_x_len = X_length_train[rand_idx]
                
                self.sess.run(self.updates,
                              feed_dict={self.X: batch_x,
                                         self.Y: batch_y,
                                         self.X_length: batch_x_len,
                                         self.learning_rate: learning_rate,
                                         self.reg_lambda: reg_lambda,
                                         self.is_training: True})

            train_loss = self.evaluate(
                X_train,  Y_train, X_length=X_length_train, batch_size=batch_size)
            valid_loss = self.evaluate(
                X_valid,  Y_valid, X_length=X_length_valid, batch_size=batch_size)

            self.report_dict['train_loss'].append(train_loss)
            self.report_dict['valid_loss'].append(valid_loss)

            flag_print = epoch_i % verbose_interval == 0
            flag_better = valid_loss < self.min_loss
            if flag_print or flag_better:
                print("*"*30) if flag_better else print("-"*30)
                print_metric(epoch_i=epoch_i,
                             train_loss=train_loss,
                             valid_loss=valid_loss,
                             min_loss=self.min_loss)

                pred = self.sess.run(self.decoder_prediction,
                                     feed_dict={self.X: X_train[:1],
                                                self.X_length: X_length_train[:1],
                                                self.is_training: False})
                print("-" * 30)
                print("sample")
                print("orig : {}".format(X_train[:1]))
                print("pred : {}".format(pred))

            if flag_better:
                patience = patience_origin + 1
                self.min_loss = valid_loss
                meta = {
                            'min_loss': self.min_loss,
                        }
                self.meta.update(meta)
                self.save_path = "{}/{}".format(self.save_dir_path, self.model_name)
                self.best_ckpt_path = self.save(self.save_path)
            patience -= 1
            if patience <= 0:
                break

        self.load(self.best_ckpt_path)

        train_loss = self.evaluate(
            X_train,  Y_train, X_length=X_length_train, batch_size=batch_size)
        valid_loss = self.evaluate(
            X_valid,  Y_valid, X_length=X_length_valid, batch_size=batch_size)

        pred = self.sess.run(self.decoder_prediction,
                             feed_dict={self.X: X_train[:10],
                                        self.X_length: X_length_train[:10],
                                        self.is_training: False})

        self.meta['report_dict'] = self.report_dict

        date_time_prefix = get_date_time_prefix()
        self.final_model_path = "{}/{}_final_{}".format(
            self.save_dir_path, date_time_prefix, self.model_name) 
        self.save(self.final_model_path)
        print("*"*30)
        print("final trained performance")
        print("-" * 30)
        print_metric(epoch_i=epoch_i,
                     train_loss=train_loss,
                     valid_loss=valid_loss,
                     min_loss=self.min_loss)
        print("-" * 30)
        print("sample")
        print("orig : {}".format(X_train[:10]))
        print("pred : {}".format(pred))
        print("final_model_path: {}".format(self.final_model_path))
        print("train done")
        print("*"*30)

        return self

    def train_with_dataset_api(self,
                               X,
                               X_length,
                               Y,
                               init_dataset_train,
                               init_dataset_valid,
                               batch_size,
                               n_epoch,
                               learning_rate,
                               reg_lambda=0.,
                               patience=10,
                               verbose_interval=5,
                               save_dir_path=None,
                               **kwargs):

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

        print("-" * 30)
        print("train start")
        patience_origin = patience

        # make sample
        self.sess.run(init_dataset_train)
        (sample_x,
         sample_x_len,
         sample_y) = self.sess.run([X, X_length, Y])

        if self.min_loss is None:
            self.min_loss = 999999999.

        for epoch_i in range(n_epoch):
            self.sess.run(init_dataset_train)
    
            batch_i = 0 
            while True:
                try:
                    (batch_x,
                     batch_x_len,
                     batch_y) = self.sess.run([X, X_length, Y])
                            
                    self.sess.run(self.updates,
                                  feed_dict={self.X: batch_x,
                                             self.Y: batch_y,
                                             self.X_length: batch_x_len,
                                             self.learning_rate: learning_rate,
                                             self.reg_lambda: reg_lambda,
                                             self.is_training: True})
                    batch_i += 1
                except tf.errors.OutOfRangeError:
                    break

            (train_loss,
             _) = self.evaluate_with_da(X,
                                        X_length,
                                        Y,
                                        init_dataset_train)
            (valid_loss,
             _) = self.evaluate_with_da(X,
                                        X_length,
                                        Y,
                                        init_dataset_valid)

            self.report_dict['train_loss'].append(train_loss)
            self.report_dict['valid_loss'].append(valid_loss)

            flag_print = epoch_i % verbose_interval == 0
            flag_better = valid_loss < self.min_loss
            if flag_print or flag_better:
                print("*"*30) if flag_better else print("-"*30)
                print_metric(epoch_i=epoch_i,
                             train_loss=train_loss,
                             valid_loss=valid_loss,
                             min_loss=self.min_loss)

                pred = self.sess.run(self.decoder_prediction,
                                     feed_dict={self.X: sample_x[:1],
                                                self.X_length: sample_x_len[:1],
                                                self.is_training: False})
                print("-" * 30)
                print("sample")
                print("orig : {}".format(sample_x[:1]))
                print("pred : {}".format(pred))

            if flag_better:
                patience = patience_origin + 1
                self.min_loss = valid_loss
                meta = {
                            'min_loss': self.min_loss,
                        }
                self.meta.update(meta)
                self.save_path = "{}/{}".format(self.save_dir_path, self.model_name)
                self.best_ckpt_path = self.save(self.save_path)
            patience -= 1
            if patience <= 0:
                break

        self.load(self.best_ckpt_path)

        (train_loss,
         _) = self.evaluate_with_da(X,
                                    X_length,
                                    Y,
                                    init_dataset_train)
        (valid_loss,
         _) = self.evaluate_with_da(X,
                                    X_length,
                                    Y,
                                    init_dataset_valid)

        pred = self.sess.run(self.decoder_prediction,
                             feed_dict={self.X: sample_x[:10],
                                        self.X_length: sample_x_len[:10],
                                        self.is_training: False})

        self.meta['report_dict'] = self.report_dict

        date_time_prefix = get_date_time_prefix()
        self.final_model_path = "{}/{}_final_{}".format(
            self.save_dir_path, date_time_prefix, self.model_name) 
        self.save(self.final_model_path)
        print("*"*30)
        print("final trained performance")
        print("-" * 30)
        print_metric(epoch_i=epoch_i,
                     train_loss=train_loss,
                     valid_loss=valid_loss,
                     min_loss=self.min_loss)
        print("-" * 30)
        print("sample")
        print("orig : {}".format(sample_x[:10]))
        print("pred : {}".format(pred))
        print("final_model_path: {}".format(self.final_model_path))
        print("train done")
        print("*"*30)

        return self

    def save_with_saved_model(self, path):
        """Description

        Args:
            path: save path
        """
        if not os.path.exists(os.path.dirname(path)):
            makedirs(os.path.dirname(path))
        try:
            with self.g.as_default():
                builder = tf.saved_model.builder.SavedModelBuilder(path)
                tensor_info_X = tf.saved_model.utils.build_tensor_info(self.X)
                tensor_info_E = tf.saved_model.utils.build_tensor_info(self.encoder_final_state)

                prediction_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                        inputs={'X': tensor_info_X},
                        outputs={'encoded': tensor_info_E},
                        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

                init = tf.tables_initializer()
                legacy_init_op = tf.group(
                    init, name='legacy_init_op')

                builder.add_meta_graph_and_variables(
                    self.sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={'predict': prediction_signature},
                    legacy_init_op=legacy_init_op)
                builder.save()
        except Exception as e:
            raise Exception("Error in save_with_saved_model: {}".format(str(e)))

        return self
