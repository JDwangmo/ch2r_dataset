#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-16'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function


import numpy as np
import pandas as pd
import logging
import timeit

from data_processing_util.jiebanlp.jieba_util import Jieba_Util

jutil = Jieba_Util(verbose=0)
remove_sentence_punctuation = lambda x: jutil.seg(x, sep='', remove_url=False)

# 统计 进入协处理的对话段数

ch2r_dialogue_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/ch2r_test_dataset/start-20150613测试集/data/dialogue_usersentence_ge_1.csv'

ch2r_dialogue = pd.read_csv(
    ch2r_dialogue_file_path,
    sep='\t',
    encoding='utf8',
    header= 0,
)

user_sentence = ch2r_dialogue[ch2r_dialogue['Name'] != 'Ch2R']

print(user_sentence.head())
print(user_sentence.shape)

aiml_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160712/data/v2.2_aiml_file_2378.csv'


aiml_data = pd.read_csv(
    aiml_file_path,
    sep='\t',
    encoding='utf8',
    header= 0,
)
print(aiml_data.head())
print(aiml_data.shape)

# print(data_aiml.shape)
aiml_data[u'SEG'] = aiml_data['SENTENCE'].apply(remove_sentence_punctuation)

ch2r_dialogue.loc[ch2r_dialogue['Name'] != u'Ch2R',u'SEG'] = ch2r_dialogue.loc[ch2r_dialogue['Name'] != 'Ch2R','Record'].fillna('').apply(remove_sentence_punctuation)

user_sentence = ch2r_dialogue[ch2r_dialogue['Name'] != 'Ch2R']

result = []
for counter, sentence in enumerate(aiml_data['SEG'].values):
    # print('第%d个...'%(counter+1))
    in_dataB = user_sentence['SEG'] == sentence
    # print(sum(in_dataB))
    # quit()
    if sum(in_dataB) > 0:
        result.extend(user_sentence.loc[in_dataB, 'SessionID'].values)

        print(user_sentence.loc[in_dataB])

# data['AIML'] = aiml_result
print(len(result))
print(len(set(result)))