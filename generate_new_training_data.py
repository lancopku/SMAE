import os
import json
import time
import codecs
import tensorflow as tf
import data
import shutil
from  result_evaluate import Evaluate

FLAGS = tf.app.flags.FLAGS

def generate_confident_examples(model, batcher, sess):
    def write_negtive_to_json(conf_dir, original_review, score, reward, counter):
        positive_dir = conf_dir
        #print(reward)
        positive_file = os.path.join(conf_dir, "%06d.txt" % (counter // 1000))
        write_positive_file = codecs.open(positive_file, "a", "utf-8")
        dict = {"review": str(original_review),"score": int(score), "confidence": reward, }
        string_ = json.dumps(dict)
        write_positive_file.write(string_ + "\n")
        write_positive_file.close()

    batches = batcher.get_batches(mode='train')
    if not os.path.exists('train_conf'): os.mkdir('train_conf')
    shutil.rmtree('train_conf')
    if not os.path.exists('train_conf'): os.mkdir('train_conf')
    step = 0
    counter = 0
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        results = model.run_eval_conf(sess, current_batch)
        for i in range(FLAGS.batch_size):
            original_review = current_batch.original_reviews[i]  # string
            score = current_batch.labels[i]
            conf = results['pred_conf'][i]
            write_negtive_to_json('./train_conf', original_review, score, str(conf), counter)
            counter += 1  # this is how many examples we've decoded
    print("we processed %d examples before confidence filter in training data" %counter)

    batches = batcher.get_batches(mode='valid')
    step = 0
    counter = 0
    if not os.path.exists('valid_conf'): os.mkdir('valid_conf')
    shutil.rmtree('valid_conf')
    if not os.path.exists('valid_conf'): os.mkdir('valid_conf')
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        results = model.run_eval_conf(sess, current_batch)
        for i in range(FLAGS.batch_size):
            original_review = current_batch.original_reviews[i]  # string
            score = current_batch.labels[i]
            conf = results['pred_conf'][i]
            write_negtive_to_json('./valid_conf', original_review, score, str(conf), counter)
            counter += 1  # this is how many examples we've decoded
    print("we processed %d examples before confidence filter in dev data" %counter)

class Generate_training_sample(object):
    def __init__(self, model, vocab, batcher, sess):
        self._model = model
        self._vocab = vocab
        self._sess = sess

        self.batches = batcher.get_batches(mode='train')
        self.valid_batches = batcher.get_batches(mode='valid')
        print("len valid,", len(self.valid_batches ))

        self.current_batch = 0



    def generate_training_example(self, training_dir):

        self.temp_positive_dir = training_dir
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        counter = 0
        for step in range(len(self.batches)):
            decode_result = self._model.run_attention_weight_ypred_auc(self._sess, self.batches[step])
            #print (decode_result)
            decode_result['y_pred_auc'] = decode_result['y_pred_auc'].tolist()
            #print(decode_result)
            for i in range(FLAGS.batch_size):

                original_review = self.batches[step].original_reviews[i]  # string
                score = self.batches[step].labels[i]

                self.write_negative_to_json(training_dir, original_review, score, decode_result['y_pred_auc'][i], decode_result['weight'][i], counter)
                counter += 1  # this is how many examples we've decoded


    def generator_validation_example(self, validation_positive_dir):
        self.temp_positive_dir = validation_positive_dir
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        counter = 0
        for step in range(len(self.valid_batches)):
            decode_result =  self._model.run_attention_weight_ypred_auc(self._sess, self.valid_batches[step])
            decode_result['y_pred_auc'] = decode_result['y_pred_auc'].tolist()

            for i in range(FLAGS.batch_size):
                original_review = self.valid_batches[step].original_reviews[i]  # string
                score = self.valid_batches[step].labels[i]

                self.write_negative_to_json(validation_positive_dir, original_review, score, decode_result['y_pred_auc'][i],
                                           decode_result['weight'][i], counter)
                counter += 1  # this is how many examples we've decoded

    '''def generator_validation_negetive_example(self, validation_negetive_dir):
        self.temp_positive_dir = validation_negetive_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)

        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)

        counter = 0

        for step in range(len(self.valid_batches)):
            decode_result = self._model.run_attention_weight_ypred_auc(self._sess, self.valid_batches[step])

            for i in range(FLAGS.batch_size):
                original_review = self.valid_batches[step].original_reviews[i]  # string
                score = 1-int(self.valid_batches[step].labels[i])

                self.write_negtive_to_json(validation_negetive_dir, original_review, score,
                                           decode_result['y_pred_auc'][i],
                                           decode_result['weight'][i], counter)
                counter += 1  # this is how many examples we've decoded'''

    def write_negative_to_json(self, positive_dir, original_review, score, reward,weight, counter):
        #print(reward)
        positive_file = os.path.join(positive_dir, "%06d.txt" % (counter // 1000))
        write_positive_file = codecs.open(positive_file, "a", "utf-8")
        dict = {"review": str(original_review),
                "score": int(score),
                "reward": reward,
                "weight": weight.tolist(),
                }

        string_ = json.dumps(dict)
        write_positive_file.write(string_ + "\n")
        write_positive_file.close()

