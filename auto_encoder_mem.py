import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import random
FLAGS = tf.app.flags.FLAGS

def sample_output(embedding, embedding_dec, output_projection=None,
                               given_number=None):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev,_):

    prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = tf.cast(tf.reshape(tf.multinomial(tf.log(prev), 1), [FLAGS.batch_size]), tf.int32)
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    return emb_prev

  def loop_function_max(prev,_):
      """function that feed previous model output rather than ground truth."""
      if output_projection is not None:
          prev = tf.nn.xw_plus_b(
              prev, output_projection[0], output_projection[1])
      prev_symbol = tf.argmax(prev, 1)
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      #emb_prev = tf.stop_gradient(emb_prev)
      return emb_prev

  def loop_given_function(prev, i):
      return tf.cond(tf.less(i,2), lambda :loop_function(prev,i), lambda:loop_function_max(prev,i))

  return loop_function,loop_function_max,loop_given_function

class Seq2seq_AE(object):
    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab
        self.train_or_test = 'train'
        self.epoch = 0

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_enc_steps],name='enc_batch')
        self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_sen_lens')
        self._weight = tf.placeholder(tf.float32, [hps.batch_size,hps.max_enc_steps], name = "weight")
        self.score = tf.placeholder(tf.int32, name = 'score')

        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
        self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')
        self.reward = tf.placeholder(tf.float32, [hps.batch_size], name='reward')

    def _make_feed_dict(self, batch, just_enc=False):
        feed_dict = {}
        if FLAGS.run_method == 'auto-encoder':
            feed_dict[self._enc_batch] = batch.enc_batch
            feed_dict[self._enc_lens] = batch.enc_lens

        feed_dict[self._dec_batch] = batch.dec_batch
        feed_dict[self.score] = batch.score
        feed_dict[self._weight] = batch.weight
        feed_dict[self.reward]= batch.reward
        feed_dict[self._target_batch] = batch.target_batch
        feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st)) = tf.nn.dynamic_rnn(cell_fw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
        return fw_st

    def _add_decoder(self, loop_function, loop_function_max, input):  # input batch sequence dim
        hps = self._hps
        #input = tf.unstack(input, axis=1)
        cell = tf.contrib.rnn.LSTMCell(
          hps.hidden_dim,
          initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=1),
          state_is_tuple=True)

        decoder_outputs_pretrain,_ = tf.contrib.legacy_seq2seq.rnn_decoder(
          input, self._dec_in_state,
          cell, loop_function=None
        )

        decoder_outputs_max_generator, _ = tf.contrib.legacy_seq2seq.rnn_decoder(
            input, self._dec_in_state,
            cell, loop_function=loop_function_max
        )

        decoder_outputs_pretrain = tf.stack(decoder_outputs_pretrain, axis=1)
        decoder_outputs_max_generator = tf.stack(decoder_outputs_max_generator, axis=1)
        return decoder_outputs_pretrain,decoder_outputs_max_generator

    def add_mem_decoder(self, embedding, emb_dec_inputs, vsize, hps):
        with tf.variable_scope('mem_decoder'):
            with tf.variable_scope('output_projection'):
                w = tf.get_variable(
                    'w', [hps.hidden_dim, vsize], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                v = tf.get_variable(
                    'v', [vsize], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
            # Add the decoder.

            with tf.variable_scope('decoder'):
                loop_function, loop_function_max, loop_given_function = sample_output(embedding, emb_dec_inputs, (w, v))

            decoder_outputs_pretrain, decoder_outputs_max_generator = self._add_decoder(loop_function = loop_function,
                loop_function_max=loop_function_max, input=emb_dec_inputs )

            decoder_outputs_pretrain = tf.reshape(decoder_outputs_pretrain,[hps.batch_size * hps.max_dec_steps, hps.hidden_dim])
            decoder_outputs_pretrain = tf.nn.xw_plus_b(decoder_outputs_pretrain, w, v)
            self.decoder_outputs_pretrain = tf.reshape(decoder_outputs_pretrain,[hps.batch_size, hps.max_dec_steps, vsize])
            decoder_outputs_max_generator = tf.reshape(decoder_outputs_max_generator,[hps.batch_size * hps.max_dec_steps, hps.hidden_dim])
            decoder_outputs_max_generator = tf.nn.xw_plus_b(decoder_outputs_max_generator, w, v)
            self._max_best_output = tf.reshape(tf.argmax(decoder_outputs_max_generator, 1),[hps.batch_size, hps.max_dec_steps])
        return self.decoder_outputs_pretrain, self._max_best_output

    def _build_model(self):
        """Add the whole generator model to the graph."""
        hps = self._hps
        vsize = self._vocab.size() # size of the vocabulary

        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

            with tf.variable_scope('memory'):
                self.pos_mem = tf.get_variable('pos_mem', [128, 60], dtype=tf.float32, initializer=self.trunc_norm_init)
                self.neg_mem = tf.get_variable('neg_mem', [128, 60], dtype=tf.float32, initializer=self.trunc_norm_init)
                self.neg_matrix = tf.get_variable('neg_matrix', [128, hps.hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                self.pos_matrix = tf.get_variable('pos_matrix', [128, hps.hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                #embedding_score = tf.get_variable('embedding_score', [5, hps.hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

                emb_dec_inputs = tf.nn.embedding_lookup(embedding, self._dec_batch) # list length max_dec_steps containing shape (batch_size, emb_size)
                emb_dec_inputs = tf.unstack(emb_dec_inputs, axis=1)
                if FLAGS.run_method == 'auto-encoder':
                    emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
                    n_emb_enc_inputs = emb_enc_inputs * tf.expand_dims(self._weight, axis = -1)

                    self.polar_weight = 1 - self._weight
                    emb_polar = emb_enc_inputs * tf.expand_dims(self.polar_weight, axis = -1)
                    self.senti_vector = tf.reduce_sum(emb_polar,1)  # shape (batch_size, attn_size).
                    self.neutral_vector = tf.reduce_sum(n_emb_enc_inputs, 1)
                    #context_vector = tf.reshape(context_vector, [-1, hps.hidden_dim])


                    fw_st = self._add_encoder(n_emb_enc_inputs, self._enc_lens)
                    self.c, self.h = fw_st
                    self.return_hidden = fw_st.h

                    if self.train_or_test == 'train':
                        tf.cond(tf.less(self.score, 1), lambda: self.update_neg_mem(), lambda :self.update_pos_mem())
                        choiced_mem = tf.cond(tf.less(self.score, 1), lambda: self.add_neg_mem(), lambda: self.add_pos_mem())
                    elif self.train_or_test == 'test':
                        choiced_mem = tf.cond(tf.less(self.score, 1), lambda: self.add_neg_mem(), lambda: self.add_pos_mem())

                    self.c += choiced_mem
                    #self.h += choiced_mem
                    self._dec_in_state = tf.contrib.rnn.LSTMStateTuple(self.c, self.h)  # self._reduce_states(fw_st, bw_st)

            self.decoder_outputs_pretrain, self._max_best_output = self.add_mem_decoder(embedding, emb_dec_inputs, vsize, hps)
            loss = tf.contrib.seq2seq.sequence_loss(
               self.decoder_outputs_pretrain,
               self._target_batch,
               self._dec_padding_mask,
               average_across_timesteps=True,
               average_across_batch=False)

            reward_loss = tf.contrib.seq2seq.sequence_loss(
               self.decoder_outputs_pretrain,
               self._target_batch,
               self._dec_padding_mask,
               average_across_timesteps=True,
               average_across_batch=False) * self.reward

            # Update the cost
            self._cost = tf.reduce_mean(loss)
            self._reward_cost = tf.reduce_mean(reward_loss)
            self.optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)

    def update_pos_mem(self):
        weight = tf.matmul(self.senti_vector, self.pos_mem)
        weight_distri = tf.nn.softmax(weight)
        weight_distri = tf.tile(tf.expand_dims(weight_distri, -1), [1, 1, 128])  # 64 30 128
        weight_distri = tf.transpose(weight_distri, [0, 2, 1])  # 64 128 30
        #self.pos_mem *= (1 - tf.reduce_sum(weight_distri, 0))
        update = weight_distri * tf.expand_dims(self.senti_vector, -1)
        update = tf.reduce_sum(update, 0)
        self.pos_mem += update
        return self.pos_mem

    def update_neg_mem(self):
        weight = tf.matmul(self.senti_vector, self.neg_mem)
        weight_distri = tf.nn.softmax(weight)
        weight_distri = tf.tile(tf.expand_dims(weight_distri, -1), [1, 1, 128])  # 64 30 128
        weight_distri = tf.transpose(weight_distri, [0, 2, 1])  # 64 128 30
        #self.neg_mem *= (1 - tf.reduce_sum(weight_distri, 0))
        update = weight_distri * tf.expand_dims(self.senti_vector, -1)
        update = tf.reduce_sum(update, 0)
        self.neg_mem += update
        return self.neg_mem

    def add_pos_mem(self):
        #neu_pos_trans = tf.matmul(self.neutral_vector, self.pos_matrix)
        weight = tf.matmul(self.neutral_vector, self.pos_mem)
        weight_distri = tf.nn.softmax(weight)
        weight_distri = tf.tile(tf.expand_dims(weight_distri, -1), [1, 1, 128])
        choiced_mem = tf.transpose(weight_distri, [0, 2, 1]) * tf.expand_dims(self.pos_mem, 0)
        choiced_mem = tf.reduce_sum(choiced_mem, -1)
        choiced_mem = tf.matmul(choiced_mem, self.pos_matrix)
        return choiced_mem

    def add_neg_mem(self):
        #neu_neg_trans = tf.matmul(self.neutral_vector, self.neg_matrix)
        weight = tf.matmul(self.neutral_vector, self.neg_mem)
        weight_distri = tf.nn.softmax(weight)
        weight_distri = tf.tile(tf.expand_dims(weight_distri, -1), [1, 1, 128])
        choiced_mem = tf.transpose(weight_distri, [0,2,1]) * tf.expand_dims(self.neg_mem, 0)
        choiced_mem = tf.reduce_sum(choiced_mem, -1)
        choiced_mem = tf.matmul(choiced_mem, self.neg_matrix)
        return choiced_mem

    def _add_train_op(self):
        loss_to_minimize = self._cost
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        # Clip the gradients
        grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
        # Add a summary
        tf.summary.scalar('global_norm', global_norm)
        # Apply adagrad optimizer
        self._train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def _add_reward_train_op(self):
        loss_to_minimize = self._reward_cost
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        # Clip the gradients
        grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
        self._train_reward_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        with tf.device("/gpu:"+str(FLAGS.gpuid)):
            tf.logging.info('Building generator graph...')
            t0 = time.time()
            self._add_placeholders()
            self._build_model()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self._add_train_op()
            self._add_reward_train_op()
            t1 = time.time()
            tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_pre_train_step(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'loss': self._cost,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def run_hidden_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
           'hidden': self.return_hidden,
        }
        return sess.run(to_return, feed_dict)

    def run_train_step(self, sess, batch, epoch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        self.epoch = epoch
        to_return = {
            'train_op': self._train_reward_op,
            'generated': self._max_best_output,
            'loss': self._reward_cost,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def max_generator(self,sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
           'generated': self._max_best_output,
        }
        return sess.run(to_return, feed_dict)
