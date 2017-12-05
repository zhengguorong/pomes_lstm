# -*- coding: utf-8 -*-

import os
import collections
import numpy as np
import tensorflow as tf
from utils.model import rnn_model
from utils.helper import data_process, generate_batch

tf.app.flags.DEFINE_string('file_path', os.path.abspath('./data/poetry.txt'), 'file name of poems.')
FLAGS = tf.app.flags.FLAGS
batch_size = 1

def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]

def get_pomes(heads, type):
    if type!=5 and type!=7:
        print '第二个参数为5或者7，请填写对应值'
        return 
    poems_vector, word_int_map, vocabularies = data_process(FLAGS.file_path)
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    learning_rate = tf.Variable(0.0, trainable=False)
    
    end_points = rnn_model(model='lstm', input_data=input_data,batch_size = batch_size, output_data=None,vocab_size=len(vocabularies))
    
    Session_config = tf.ConfigProto(allow_soft_placement = True)
    Session_config.gpu_options.allow_growth=True
    with tf.Session(config=Session_config) as sess: 
        sess.run(tf.initialize_all_variables())  
        saver = tf.train.Saver(tf.all_variables())  
        saver.restore(sess, './model/poems-2')
        poem = ''
        for head in heads:
            flag = True
            while flag:
                x = np.array([list(map(word_int_map.get, u'['))])  
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x})
                sentence = head
                x = np.zeros((1,1))
                x[0,0] = word_int_map[sentence]  
                [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})
                word = to_word(predict, vocabularies)
                sentence += word
                
                while word != u'。':  
                    x = np.zeros((1,1))  
                    x[0,0] = word_int_map[word]  
                    [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                                 feed_dict={input_data: x, end_points['initial_state']: last_state})
                    word = to_word(predict, vocabularies)
                    sentence += word  

                if len(sentence) == 2 + 2 * type:
                    sentence += u'\n'
                    poem += sentence
                    flag = False
        return poem
    
print(get_pomes(u'郑国榕',5))
    