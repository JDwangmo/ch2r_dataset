#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-12'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import logging
import timeit

from data_processing_util.jiebanlp.jieba_util import Jieba_Util


class DataUtil(object):

    def __init__(self):
        jutil = Jieba_Util(verbose=0)
        self.remove_sentence_punctuation = lambda x:jutil.seg(x,sep='',remove_url=False)


    def load_data(self, path, header=0):
        """
            加载 csv格式的数据

        :param header: pd.read_csv() 设置选项，是否有表头
        :type header:
        :param path: 数据文件的路径
        :type path: str
        :return: pd.DataFrame格式的对象数据
        :rtype: pd.DataFrame()
        """
        data = pd.read_csv(path,
                           sep='\t',
                           header=header,
                           encoding='utf8')
        return data

    def remove_repet_data(self, data):
        '''
            去除重复的的句子（去除标点符号后一样的句子则算一样）
                1. 初始化jieba分词，并用分词去除标点符号
                2. 去重处理

        :param data:
        :return:
        '''

        jutil = Jieba_Util(verbose=0)
        # 去除标点符号
        remove_sentence_punctuation = lambda x: jutil.seg(x, sep='', remove_url=False)

        labels = []
        sentences = []
        for label, group in data.groupby(by=[u'LABEL']):
            # print(label,len(group),len(group[u'SENTENCE'].unique()))
            # 去除该类别之后的句子和句子数
            # print(group[u'SENTENCE'])
            # print(group[u'SENTENCE'].apply(remove_sentence_punctuation))
            norepet_sentcence_set = set()
            sentences_after_rm_rep = []
            for item in group[u'SENTENCE'].as_matrix():
                seged_sentence = remove_sentence_punctuation(item)
                if seged_sentence not in norepet_sentcence_set:
                    norepet_sentcence_set.add(seged_sentence)
                    sentences_after_rm_rep.append(item)
                    # print(seged_sentence)
                else:
                    pass
                    # print(item)
            num_after_rm_rep = len(sentences_after_rm_rep)
            sentences.extend(sentences_after_rm_rep)
            labels.extend([label] * num_after_rm_rep)

        # print(len(labels))
        # print(len(sentences))
        return pd.DataFrame(data={'LABEL': labels, 'SENTENCE': sentences})

    def save_data(self, data, path):
        """
            保存数据成 csv格式的文件

        :param data: 待保存的数据
        :type data: pd.DataFrame()
        :param path: 数据文件的路径
        :type path: str
        :return: None
        :rtype: None
        """
        data.to_csv(path,
                    sep='\t',
                    encoding='utf8',
                    index=True,
                    )
    def clear_data(self,data):
        '''
            处理数据，最后数据只有一列， SENTENCE

        :param data:
        :return:
        '''

        transform_sentence = lambda x: x.split('|')[-1]
        data[u'SENTENCE'] = data[u'PAST1_SENTENCE'].apply(transform_sentence)

        data = data[[u'SENTENCE']]

        return data


    def sentence_is_equal(self,sentence1,sentence2):
        # print(self.remove_sentence_punctuation(sentence1))
        # print(self.remove_sentence_punctuation(sentence2))
        # if self.remove_sentence_punctuation(sentence1) == self.remove_sentence_punctuation(sentence2):
        #     print(sentence1,sentence2)


        return self.remove_sentence_punctuation(sentence1) == self.remove_sentence_punctuation(sentence2)



    def check_dataA_in_dataB(self, dataA,dataB_file_path):
        '''
            检查数据dataA是不是在dataB中出现过

        :param data:
        :return:
        '''
        # dataB_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160712/data/v2.2_train_S_1518.csv'


        dataB = dutil.load_data(dataB_file_path)
        # print(data_aiml.head())
        # print(data_aiml.shape)
        dataA[u'SEG'] = dataA['SENTENCE'].apply(self.remove_sentence_punctuation)
        dataB[u'SEG'] = dataB['SENTENCE'].apply(self.remove_sentence_punctuation)
        result = []
        for counter,sentence in enumerate(dataA['SEG'].values):
            # print('第%d个...'%(counter+1))
            in_dataB = dataB['SEG']==sentence
            # print(sum(in_aiml))
            # quit()
            if sum(in_dataB) >0:
                result.append(dataB.loc[in_dataB,'LABEL'].values[0])
            else:
                result.append('NONE')

        # data['AIML'] = aiml_result
        return np.asarray(result)


    def check_data_in_train_data(self,data):
        '''
            检查数据是不是在AIML和3045句的全部训练数据中出现过

            1.获取训练库句子

        :param data:
        :return:
        '''
        aiml_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160712/data/v2.2_train_L_2728.csv'


        data_aiml = dutil.load_data(aiml_file_path)
        # print(data_aiml.head())
        # print(data_aiml.shape)
        aiml_result = []
        data[u'SEG'] = data['SENTENCE'].apply(self.remove_sentence_punctuation)
        data_aiml[u'SEG'] = data_aiml['SENTENCE'].apply(self.remove_sentence_punctuation)
        # print(data['SEG'])
        for counter,sentence in enumerate(data['SEG'].values):
            # print('第%d个...'%(counter+1))
            in_aiml = data_aiml['SEG']==sentence
            # print(sum(in_aiml))
            # quit()
            if sum(in_aiml) >0:
                aiml_result.append(data_aiml.loc[in_aiml,'LABEL'].values[0])
            else:
                aiml_result.append('NONE')
        # quit()
        # data['AIML'] = aiml_result
        return np.asarray(aiml_result)
        # print(data.head())


def label_ood_sentence():
    '''
        处理文件： /data/ood_sentence.csv,366句协处理句子
        输出文件：

        主要是想对366句协处理句子进行标注，如果在AIML中出现过，则取现有的标注，剩下没匹配上的设置为NONE，之后进行人工标注

        1. 读取数据
        2. 检查是否在AIML中出现过，注意这里的AIML是现有的所有协处理句子。
        3. 检查是否在训练

    :return:
    '''

    ood_sentence_file_path = 'data/ood_sentence_366.csv'

    ood_sentence_labeled_file_path = './result/ood_sentence_%d_labeled.csv'
    # ood_sentence_no_aiml_file_path = './result/ood_sentence_no_aiml_%d.csv'


    data_ood = dutil.load_data(ood_sentence_file_path)
    data_ood = dutil.clear_data(data_ood)



    print(data_ood.shape)

    print('与AIML训练集比较')
    result = dutil.check_dataA_in_dataB(data_ood,'/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160712/data/v2.2_aiml_file_2378.csv')

    data_ood['LABEL'] = result
    data_aiml = data_ood.loc[data_ood['LABEL']!='NONE']
    data_no_aiml = data_ood.loc[data_ood['LABEL']=='NONE']

    data_aiml = data_aiml[[u'LABEL',u'SENTENCE']]
    data_no_aiml = data_no_aiml[[u'LABEL',u'SENTENCE']]
    print('匹配上AIML：%d'%len(data_aiml))
    print('匹配不上AIML：%d'%len(data_no_aiml))

    dutil.save_data(data_ood, ood_sentence_labeled_file_path % len(data_ood))
    # dutil.save_data(data_no_aiml, ood_sentence_no_aiml_file_path % len(data_no_aiml))

def process_ood_sentence():

    '''
        丢弃

    :return:
    '''

    quit()
    #
    # print('与全部训练集比较')
    # result = dutil.check_data_in_train_data(data_ood)
    # data_in_train_data = data_ood.loc[result != 'NONE']
    # data_no_in_train_data = data_ood.loc[result == 'NONE']
    # print(data_no_in_train_data.head())

    # print(len(data_in_train_data))
    # print(len(data_no_in_train_data))
    print('去重后...')

    data_in_train_data = dutil.remove_repet_data(data_in_train_data)

    print(len(data_in_train_data['SENTENCE'].unique()))
    print(len(data_no_in_train_data['SENTENCE'].unique()))


    print('与AIML训练集比较')
    result = dutil.check_dataA_in_dataB(data_ood)
    data_aiml = data_ood.loc[result != 'NONE']
    data_no_aiml = data_ood.loc[result == 'NONE']

    print(len(data_aiml))
    print(len(data_no_aiml))
    print('去重后...')
    print(len(data_aiml['SENTENCE'].unique()))
    print(len(data_no_aiml['SENTENCE'].unique()))


    data_aiml = data_aiml.sort_values(by=['SENTENCE'])
    data_no_aiml = data_no_aiml.sort_values(by=['SENTENCE'])

    dutil.save_data(data_aiml,ood_sentence_aiml_file_path%len(data_aiml))
    dutil.save_data(data_no_aiml,ood_sentence_no_aiml_file_path%len(data_no_aiml))


def process_ood_sentence_no_aiml():
    '''
        处理文件： data/ood_sentence_no_aiml_235_labeled.csv
        1. 去重

    :return:
    '''

    ood_sentence_no_aiml_file_path = './data/ood_sentence_no_aiml_235_labeled.csv'
    output_file_path = './result/ood_sentence_no_aiml_norepet_%d.csv'

    data = dutil.load_data(ood_sentence_no_aiml_file_path)
    data = data[[u'LABEL',u'SENTENCE']]
    print(data.head())

    print(len(data))

    data = dutil.remove_repet_data(data)
    data = data.sort_values(by=[u'LABEL',u'SENTENCE'])

    dutil.save_data(data,output_file_path%len(data))

    print(len(data))
    print(data['LABEL'].value_counts().sort_index())


if __name__ == '__main__':
    dutil = DataUtil()
    label_ood_sentence()
