import time
import codecs
import os
from collections import namedtuple
from data import Vocab
from Batch_iter import Example
from Batch_iter import Batch
from Batch_iter import GenBatcher
from auto_encoder_mem import Seq2seq_AE
import json
from generated_sample import  Generated_sample
from batcher_classification import ClaBatcher, AttenBatcher
from my_classifier import  Classification
import util
from generate_new_training_data import Generate_training_sample, generate_confident_examples
from cnn_classifier import *

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('train_data_path', './dataset/train/*', 'training files.')
tf.app.flags.DEFINE_string('valid_data_path', './dataset/valid/*', 'validation files.')
tf.app.flags.DEFINE_string('vocab_path', './dataset/vocab.txt', 'Path expression to text vocabulary file.')
# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
# Where to save output
tf.app.flags.DEFINE_string('log_root', 'log_seq2seq_ae', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'my_ae', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_integer('gpuid', 0, 'for gradient clipping')
tf.app.flags.DEFINE_string('run_method', 'auto-encoder', 'must be one of auto-encoder/language_model')

tf.app.flags.DEFINE_integer('max_enc_sen_num', 1, 'max timesteps of encoder (max source text tokens)')   # for discriminator
tf.app.flags.DEFINE_integer('max_enc_seq_len', 50, 'max timesteps of encoder (max source text tokens)')   # for discriminator
# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states') # for discriminator and generator
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings') # for discriminator and generator
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size') # for discriminator and generator
tf.app.flags.DEFINE_integer('max_enc_steps', 50, 'max timesteps of encoder (max source text tokens)') # for generator
tf.app.flags.DEFINE_integer('max_dec_steps', 50, 'max timesteps of decoder (max summary tokens)') # for generator
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode') # for generator
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.6, 'learning rate') # for discriminator and generator
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.0005, 'initial accumulator value for Adagrad') # for discriminator and generator
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization') # for discriminator and generator
tf.app.flags.DEFINE_float('trunc_norm_init_std', 0.1, 'std of trunc norm init, used for initializing everything else') # for discriminator and generator
tf.app.flags.DEFINE_float('max_grad_norm', 5.0, 'for gradient clipping') # for discriminator and generator

config = {
    'n_epochs' : 5,
    'kernel_sizes' : [3, 4, 5],
    'dropout_rate' : 0.5,
    'val_split' : 0.4,
    'edim' : 300,
    'n_words' : None,   #Leave as none
    'std_dev' : 0.05,
    'sentence_len' : 50,
    'n_filters'  : 100,
    'batch_size' : 50}
config['n_words'] = 50000

def setup_training_classifier(model):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train-classifier")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    model.build_graph()  # build the graph
    saver = tf.train.Saver(max_to_keep=5)  # we use this to load checkpoints for decoding
    sess = tf.Session(config=util.get_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    #util.load_ckpt(saver, sess, ckpt_dir="train-classifier")
    return sess, saver,train_dir

def run_train_cnn_classifier(model, batcher, max_run_epoch,  sess,saver, train_dir):
    tf.logging.info("starting train_cnn_discriminator")
    epoch = 0
    best_result = 0.0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            step += 1
            results = model.run_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")
            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 1000 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training classifier step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0
            if train_step % 5000 == 0:
                acc = run_test_classification(model, batcher, sess, saver, str(train_step))
                tf.logging.info('Accuracy of cnn classifier on valid dataset is {:.3f}'.format(acc))  # print the loss to screen
                if acc > best_result:
                    saver.save(sess, train_dir + "/model", global_step=train_step)
                    best_result = acc
        epoch += 1
        tf.logging.info("finished %d epoches", epoch)
    acc = run_test_classification(model, batcher, sess, saver, str('final'))
    tf.logging.info('Final accuracy of cnn classifier on valid dataset is {:.3f}'.format(acc))  # print the loss to screen


def setup_training_generator(model):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train-generator")
  if not os.path.exists(train_dir): os.makedirs(train_dir)
  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=5)  # we use this to load checkpoints for decoding
  sess = tf.Session(config=util.get_config())
  init = tf.global_variables_initializer()
  sess.run(init)
  #util.load_ckpt(saver, sess, ckpt_dir="train-generator")

  return sess, saver,train_dir

def setup_training_attention_classification(model):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train-classification")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph()  # build the graph

    saver = tf.train.Saver(max_to_keep=5)  # we use this to load checkpoints for decoding
    sess = tf.Session(config=util.get_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    #util.load_ckpt(saver, sess, ckpt_dir="train-classification")

    return sess, saver,train_dir

def run_test_classification(model, batcher, sess, saver, train_step):
    tf.logging.info("starting run testing emotional words detection model...")
    batches = batcher.get_batches("valid")
    step = 0
    right =0.0
    all = 0.0
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        right_s,number,error_list, error_label = model.run_eval_step(sess, current_batch)
        error_list = error_list
        error_label = error_label
        all += number
        right += right_s
    return right/all

def run_train_attention_classification(model, bachter, max_run_epoch, sess,saver, train_dir):
    tf.logging.info("starting run training emotional words detection model...")
    epoch = 0
    best_result = 0.0
    while epoch < max_run_epoch:
        batches = bachter.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            step += 1
            #print_batch(current_batch)
            results = model.run_pre_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 1000 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training classification step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0

            if train_step % 5000 == 0:
                acc = run_test_classification(model, bachter, sess, saver, str(train_step))
                tf.logging.info('evaluate valid acc: %.6f', acc)  # print the loss to screen
                if acc > best_result:
                    saver.save(sess, train_dir + "/model", global_step=train_step)
                    best_result = acc
        epoch +=1
        tf.logging.info("finished %d epoches", epoch)
    acc = run_test_classification(model, bachter, sess, saver, str("final acc"))
    tf.logging.info('final sigmoid attention valid acc: %.6f', acc)  # print the loss to screen

def run_train_auto_encoder(model, batcher, max_run_epoch, sess, saver, train_dir, generatored,model_class,sess_cls,cla_batcher):
    tf.logging.info("starting run training generator...")
    epoch = 0
    best_overall = 0
    best_bleu = 0.0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            step += 1
            results = model.run_train_step(sess, current_batch, epoch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 1000 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0
            if train_step % 5000 == 0:
                #bleu_score = generatored.compute_BLEU(str(train_step))
                #tf.logging.info('bleu: %f', bleu_score)  # print the loss to screen
                model.train_or_test = 'test'
                tranfer_acc, bleu = generatored.generator_validation_transfer_example("valid-generated-transfer/" + str(epoch) + "epoch_step" + str(step) + "_transfer",
                                                                                      batcher, model_class,sess_cls,cla_batcher,
                                                                                      'valid-transfer')
                generatored.generator_validation_original_example("valid-generated/" + str(epoch) + "epoch_step" + str(step) + "_original",
                                                                  batcher, model_class,sess_cls,cla_batcher)
                model.train_or_test = 'train'
                current_overall = 2 * tranfer_acc * bleu / (tranfer_acc + bleu)
                if bleu > 23.3 and bleu > best_bleu:
                    saver.save(sess, train_dir + "/model", global_step=train_step)
                    best_overall = current_overall
                    best_bleu = bleu
        epoch += 1
        tf.logging.info("finished %d epoches", epoch)
    print ("Testing auto-encoder on valid set...")
    generatored.generator_validation_transfer_example("valid-generated-transfer/" + str(epoch) + "epoch_step" + str("final") + "_temp_positive", batcher, model_class,
        sess_cls, cla_batcher,  'valid-transfer')

    generatored.generator_validation_original_example(
        "valid-generated/" + str(epoch) + "epoch_step" + str("final") + "_original", batcher, model_class, sess_cls,
        cla_batcher)

    print ("Testing auto-encoder on test set...")
    generatored.generator_validation_transfer_example(
        "test-generated-transfer/" + str(epoch) + "epoch_step" + str("final") + "_temp_positive", batcher, model_class,
        sess_cls, cla_batcher, 'test-transfer')

def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting running in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

    tf.set_random_seed(6)  # a seed value for randomness

    cnn_classifier = CNN(config)
    #cnn_batcher = ClaBatcher(hps_discriminator, vocab)
    cnn_batcher = ClaBatcher(FLAGS, vocab)
    sess_cnn, saver_cnn, train_dir_cnn = setup_training_classifier(cnn_classifier)
    run_train_cnn_classifier(cnn_classifier, cnn_batcher, 15, sess_cnn, saver_cnn, train_dir_cnn)
    #util.load_ckpt(saver_cnn, sess_cnn, ckpt_dir="train-classifier")
    acc = run_test_classification(cnn_classifier, cnn_batcher, sess_cnn, saver_cnn, str('last'))
    print("The accuracy of sentiment classifier is {:.3f}".format(acc))
    generate_confident_examples(cnn_classifier, cnn_batcher, sess_cnn) ## train_conf

    print("Start training emotional words detection model...")
    model_class = Classification(FLAGS, vocab)
    cla_batcher = AttenBatcher(FLAGS, vocab) # read from train_conf
    sess_cls, saver_cls, train_dir_cls = setup_training_attention_classification(model_class)
    run_train_attention_classification(model_class, cla_batcher, 15, sess_cls, saver_cls, train_dir_cls)
    #util.load_ckpt(saver_cls, sess_cls, ckpt_dir="train-classification")
    acc = run_test_classification(model_class, cla_batcher, sess_cls, saver_cls, str("final_acc"))
    print("The sentiment classification accuracy of the emotional words detection model is {:.3f}".format(acc))
    generated = Generate_training_sample(model_class, vocab, cla_batcher, sess_cls)

    print("Generating training examples......")
    generated.generate_training_example("train_filtered")  #wirte train
    generated.generator_valid_test_example("valid_test_filtered")

    model = Seq2seq_AE(FLAGS, vocab)
    # Create a batcher object that will create minibatches of data
    batcher = GenBatcher(vocab, FLAGS) ##read from train

    sess_ge, saver_ge, train_dir_ge = setup_training_generator(model)

    generated = Generated_sample(model, vocab, batcher, sess_ge)
    print("Start training generator......")
    run_train_auto_encoder(model, batcher, 15, sess_ge, saver_ge, train_dir_ge, generated, cnn_classifier, sess_cnn, cla_batcher)

if __name__ == '__main__':
  tf.app.run()
