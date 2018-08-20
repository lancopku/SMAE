
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import data
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

FLAGS = tf.app.flags.FLAGS

class Classification(object):
    """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps
        # encoder part
        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='enc_batch')
        self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size,hps.max_dec_steps], name='enc_padding_mask')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
        self._target_batch = tf.placeholder(tf.int32,[hps.batch_size],name='target_batch')

    def _make_feed_dict(self, batch):
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        #feed_dict[self._enc_padding_mask] = batch.enc_padding_mask

        feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        feed_dict[self._target_batch] = batch.labels
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len, hps):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_sen_number <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size*max_sen_num].
          seq_num: [batch]
        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            with tf.variable_scope('word'):
                cell = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init,state_is_tuple=True)
            with tf.variable_scope('word-rnn'):
                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell,cell_bw=cell, inputs=encoder_inputs,
                                                                      dtype=tf.float32, sequence_length=seq_len,
                                                                      swap_memory=True)
                output_fw, output_bw = outputs
                states_fw, states_bw = states
        return states_fw, output_fw + output_bw

    def attention(self,decoder_state, encoder_states, attention_vec_size,enc_padding_mask,hps):
        """Calculate the context vector and attention distribution from the decoder state.

        Args:
          decoder_state: state of the decoder

        Returns:
          context_vector: weighted sum of encoder_states
          attn_dist: attention distribution
        """
        with tf.variable_scope('attention'):
            w_dec = tf.get_variable('w_dec', [attention_vec_size,hps.hidden_dim], dtype=tf.float32,
                                initializer=self.trunc_norm_init)
            v_dec = tf.get_variable('v_dec', [attention_vec_size], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_m = tf.get_variable('w_m', [hps.hidden_dim, 100], dtype=tf.float32, initializer=self.trunc_norm_init)
            w_r = tf.get_variable('w_r', [100, 1], dtype=tf.float32, initializer=self.trunc_norm_init)

            # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
            #decoder_features = tf.nn.xw_plus_b(decoder_state, w_dec,v_dec)  # shape (batch_size, attention_vec_size)
            #decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1),1)  # reshape to (batch_size, 1, 1, attention_vec_size)
            lstm_outputs_flat = tf.reshape(encoder_states, [-1, hps.hidden_dim])
            logits_flat = tf.matmul(lstm_outputs_flat, w_m)
            sigmoid_input = tf.matmul(logits_flat, w_r)
            sigmoid_weight = tf.sigmoid(tf.reshape(sigmoid_input, [hps.batch_size, hps.max_dec_steps]))
            masked_weight = sigmoid_weight * enc_padding_mask

            square_weight_pena = 0.25 - tf.square(sigmoid_weight - 0.5)
            masked_weight_pena = square_weight_pena * enc_padding_mask

            #context_vector = masked_weight * encoder_states
            #context_vector = array_ops.reshape(context_vector, [-1, hps.hidden_dim])
            encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)
            context_vector = math_ops.reduce_sum(
                array_ops.reshape(masked_weight, [hps.batch_size, -1, 1, 1]) * encoder_states,
                [1, 2])  # shape (batch_size, attn_size).
            context_vector = array_ops.reshape(context_vector, [-1, hps.hidden_dim])


            '''
            def masked_attention(e):
                """Take softmax of e then apply enc_padding_mask and re-normalize"""
                attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                attn_dist *= enc_padding_mask  # apply mask
                masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

            encoder_states = tf.expand_dims(encoder_states, axis=2)  # now is shape (batch_size, attn_len, 1, attn_size)
            W_h = tf.get_variable("W_h", [1, 1, hps.hidden_dim, attention_vec_size])
            encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1],"SAME")  # shape (batch_size,attn_length,1,attention_vec_size)

            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            v = tf.get_variable("v_h", [attention_vec_size])
            e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3])  # calculate e

            # Calculate attention distribution
            attn_dist = masked_attention(e)

            # Calculate the context vector from attn_dist and encoder_states
            context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [hps.batch_size, -1, 1, 1]) * encoder_states,
                                                 [1, 2])  # shape (batch_size, attn_size).
            context_vector = array_ops.reshape(context_vector, [-1, hps.hidden_dim])
            '''

        return context_vector, masked_weight, masked_weight_pena

    def _build_model(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self._hps
        vsize = self._vocab.size()  # size of the vocabulary
        with tf.variable_scope('classification'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag,seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)
            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,initializer=self.trunc_norm_init)
                emb_enc_inputs = tf.nn.embedding_lookup(embedding,self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
                self.emb_enc_inputs = emb_enc_inputs
            # Add the encoder.
            fw_st, encoder_vector = self._add_encoder(emb_enc_inputs, self._enc_lens, hps)
            context, self.atten_weight, self.masked_weight_pena = self.attention(fw_st.h, encoder_vector, hps.hidden_dim, self._enc_padding_mask, hps)
            self.atten_weight = self.atten_weight*(-1)+1

            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w_output', [hps.hidden_dim, 2], dtype=tf.float32, initializer=self.trunc_norm_init)
                v = tf.get_variable('v_output', [2], dtype=tf.float32, initializer=self.trunc_norm_init)

                logits = tf.nn.xw_plus_b(context, w, v)
                self.y_pred_auc = tf.nn.softmax(logits)

                batch_nums = tf.range(0, hps.batch_size)
                indices = tf.stack([batch_nums, self._target_batch], axis=1)  # batch 2  # 长度一点要-1
                self.y_pred_auc = tf.gather_nd(self.y_pred_auc, indices)  # batch dim

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self._target_batch, logits = logits)
            penalty_weight = tf.reduce_sum(self.masked_weight_pena)
            self.loss = tf.reduce_mean(loss)
            #self.loss = tf.reduce_mean(loss + 0.001 * penalty_weight)
            self.best_output = tf.argmax(tf.nn.softmax(logits),1)

    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize =  self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
        # Add a summary
        tf.summary.scalar('global_norm', global_norm)
        # Apply adagrad optimizer
        #optimizer = tf.train.AdamOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)
        optimizer = tf.train.AdamOptimizer(0.001)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        with tf.device("/gpu:" + str(FLAGS.gpuid)):
            tf.logging.info('Building graph...')
            t0 = time.time()
            self._add_placeholders()
            self._build_model()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self._add_train_op()
            t1 = time.time()
            tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch, decay=False):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'loss': self.loss,
            "predictions":self.best_output,
            'global_step': self.global_step,
        }

        return sess.run(to_return, feed_dict)
    def run_pre_train_step(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'loss': self.loss,
            'predictions': self.best_output,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def run_ypred_auc(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {'y_pred_auc': self.y_pred_auc}
        return sess.run(to_return, feed_dict)

    def run_attention_weight_ypred_auc(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'y_pred_auc': self.y_pred_auc,
            'weight':self.atten_weight
        }
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        error_list =[]
        error_label = []
        #right_label = []
        to_return = {
            'predictions': self.best_output
        }
        results = sess.run(to_return, feed_dict)
        right =0
        for i in range(len(batch.labels)):
            if results['predictions'][i] == batch.labels[i]:
                right +=1

            error_label.append(results['predictions'][i])
            error_list.append(batch.original_reviews[i])
            #right_label.append(batch.labels[i])
        return right, len(batch.labels),error_list,error_label
