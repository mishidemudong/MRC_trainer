#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:49:29 2020

@author: liang
"""


import json, os, re
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda
from keras.models import Model
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"                                                                             
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##set gpu memory
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, train_data, tokenizer, batch_size, max_a_len,max_q_len,max_p_len,  buffer_size=None):

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data = train_data
        self.batch_size = batch_size
        self.max_p_len = max_p_len
        self.max_a_len = max_a_len
        self.max_q_len = max_q_len
        
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000
        
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []
        for is_end, D in self.sample(random):
            question = D['question']
            final_answer = D['answer']
            passage = D['passage']
            passage = re.sub(u' |、|；|，', ',', passage)
            
            a_token_ids, _ = self.tokenizer.encode(
                final_answer, max_length=self.max_a_len + 1
            )
            q_token_ids, _ = self.tokenizer.encode(question, max_length=self.max_q_len + 1)
            p_token_ids, _ = self.tokenizer.encode(passage, max_length=self.max_p_len + 1)
            token_ids = [self.tokenizer._token_start_id]
            token_ids += ([self.tokenizer._token_mask_id] * self.max_a_len)
            token_ids += [self.tokenizer._token_end_id]
            token_ids += (q_token_ids[1:] + p_token_ids[1:])
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_a_token_ids.append(a_token_ids[1:])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_a_token_ids = sequence_padding(
                    batch_a_token_ids, self.max_a_len
                )
                yield [batch_token_ids, batch_segment_ids], batch_a_token_ids
                batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []

class Evaluator(keras.callbacks.Callback):
    def __init__(self, model, model_save_path):
        self.lowest = 1e10
        self.model = model
        self.model_save_path = model_save_path

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            self.model.save_weights(os.path.join(self.model_save_path,'best_model.weights'))

class MRCTrainer():
    def __init__(self, train_param, model_save_path):
        self.lr = train_param['learning_rate']
        self.max_p_len = train_param['max_p_len']
        self.max_q_len = train_param['max_q_len']
        self.max_a_len = train_param['max_a_len']
        self.epochs = train_param['epochs']
        self.pretrain_type = train_param['pretrain_type']
        self.batch_size = train_param['batch_size']
        
        self.config_path = train_param['config_path']
        self.checkpoint_path = train_param['checkpoint_path']
        self.dict_path = train_param['dict_path']
        self.model_config = train_param
        self.model_config['model_save_path'] = model_save_path 
        self.model_save_path = model_save_path
        
        self.buildmodel()
    
    def masked_cross_entropy(self, y_true, y_pred):
        y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
        cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
        
        return cross_entropy
        
    def buildmodel(self):
        self.token_dict, self.keep_tokens = load_vocab(dict_path=self.dict_path,
                                             simplified=True,
                                             startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
                                            )
        self.tokenizer = Tokenizer(self.token_dict, do_lower_case=True)
        
        if self.pretrain_type == 'albert':
            model = build_transformer_model(
                config_path,
                checkpoint_path,
                model='albert',
                with_mlm=True,
                keep_tokens=self.keep_tokens,  
            )
        elif self.pretrain_type == 'bert':
            model = build_transformer_model(
                config_path,
                checkpoint_path,
                model='bert',
                with_mlm=True,
                keep_tokens=self.keep_tokens, 
            )
        output = Lambda(lambda x: x[:, 1:self.max_a_len + 1])(model.output)
        #print(output.shape)
        self.model = Model(model.input, output)
        self.model.compile(loss=self.masked_cross_entropy, optimizer=Adam(self.lr))
        self.model.summary()
        
    def fit(self, train_data):
        
        params_file = os.path.join(self.model_save_path,'config.json')
        with open(params_file,'w',encoding='utf-8') as json_file:
            json.dump(self.model_config, json_file, indent=4 ,ensure_ascii=False)
        
        evaluator = Evaluator(self.model, self.model_save_path)
        train_generator = data_generator(train_data, self.tokenizer, self.batch_size, self.max_a_len, self.max_q_len, self.max_p_len)
    
        self.model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
        
        
    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result
    
    
    def gen_answer(self, question, passage):

        token_ids, segment_ids = [], []
        passage = re.sub(u' |、|；|，', ',', passage)
        p_token_ids, _ = self.tokenizer.encode(passage, max_length=self.max_p_len + 1)
        q_token_ids, _ = self.tokenizer.encode(question, max_length=self.max_q_len + 1)
        token_ids = [self.tokenizer._token_start_id]
        token_ids += [self.tokenizer._token_mask_id] * max_a_len
        token_ids += [self.tokenizer._token_end_id]
        token_ids += q_token_ids[1:] + p_token_ids[1:]
        segment_ids= [0] * len(token_ids[-1])
        token_ids = sequence_padding(token_ids)
        segment_ids = sequence_padding(segment_ids)
        probas = self.model.predict([token_ids, segment_ids])
        results = {}
        a, score = tuple(), 0.
        for i in range(max_a_len):
            idxs = list(self.get_ngram_set(token_ids, i + 1)[a])
            print("idxs",idxs)
            if self.tokenizer._token_end_id not in idxs:
                idxs.append(self.tokenizer._token_end_id)
            pi = np.zeros_like(probas[i])
            pi[idxs] = probas[i, idxs]
            a = a + (pi.argmax(),)
            score += pi.max()
            if a[-1] == self.tokenizer._token_end_id:
                break
        score = score / (i + 1)
        a = self.tokenizer.decode(a)
        if a:
            results[a] = results.get(a, []) + [score]
        results = {
            k: (np.array(v)**2).sum() / (sum(v) + 1)
            for k, v in results.items()
        }
        return results

    def evalue(self):
        result=[]
        return result
    
    



if __name__ == '__main__':
    
    max_p_len = 256
    max_q_len = 64
    max_a_len = 32
    batch_size = 32
    learning_rate = 5e-5
    epochs = 10
    pretrain_type = 'albert'
    
    # bert配置
    #modeldir = '/media/liang/Nas/PreTrainModel/hit-pretrain_model/'
    #config_path = modeldir + 'bert_config.json'
    #checkpoint_path = modeldir + 'bert_model.ckpt'
    #dict_path = modeldir + 'vocab.txt'
    
    config = {}
    
    bert_path = '/media/liang/Nas/PreTrainModel'
    config_path = bert_path + '/albert/albert_tiny_zh_google/albert_config_tiny_g.json' #albert_xlarge_google_zh_183k
    checkpoint_path = bert_path + '/albert/albert_tiny_zh_google/albert_model.ckpt'
    dict_path = bert_path + '/albert/albert_tiny_zh_google/vocab.txt'
    
    config['max_p_len'] = max_p_len
    config['max_q_len'] = max_q_len
    config['max_a_len'] = max_a_len
    config['batch_size'] = batch_size
    config['epochs'] = epochs
    config['pretrain_type'] = pretrain_type
    config['learning_rate'] = learning_rate
    
    config['config_path'] = config_path
    config['checkpoint_path'] = checkpoint_path
    config['dict_path'] = dict_path
    
    model_save_path = './model'
    
    model_train = MRCTrainer(config, model_save_path)
    
    train_data = []
    with open('./datasets/train.json') as f:
        for item in f:
#            print(item)
            train_data.append(eval(item))
    dev_data = []        
    with open('./datasets/dev.json') as f:
        for item in f:
            dev_data.append(eval(item))

        
    model_train.fit(train_data)
    

    

