#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-09'
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
        super(DataUtil, self).__init__()
        pass

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

    def save_data(self,data, path):
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



    def remove_repet_data(self,data):
        '''
            去除重复的的句子（去除标点符号后一样的句子则算一样）
                1. 初始化jieba分词，并用分词去除标点符号
                2. 去重处理

        :param data:
        :return:
        '''

        jutil = Jieba_Util(verbose=0)
        # 去除标点符号
        remove_sentence_punctuation = lambda x:jutil.seg(x,sep='',remove_url=False)


        labels = []
        sentences = []
        for label,group in data.groupby(by=[u'LABEL']):
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
            labels.extend([label]*num_after_rm_rep)

        # print(len(labels))
        # print(len(sentences))
        return pd.DataFrame(data={'LABEL':labels,'SENTENCE':sentences})

def data_1_2_process():
    '''
        ./data/1.2.csv 文件的处理
    :return:
    '''
    data_1_2_file_path = './data/1.2.csv'
    output_file = 'result/v2.1_test_%d.csv'
    data_1_2 = dutil.load_data(data_1_2_file_path)

    data_1_2[u'LABEL'] = data_1_2[u'20160628标注']
    data_1_2 = data_1_2[[u'SENTENCE',u'LABEL']]
    data_1_2.sort_values(by=[u'LABEL',u'SENTENCE'])
    print(data_1_2.shape)
    print(data_1_2[u'LABEL'].value_counts().sort_index())
    print('|'.join(data_1_2[u'LABEL'].value_counts().sort_index().index))
    print('|'.join([str(item) for item in data_1_2[u'LABEL'].value_counts().sort_index().values]))
    print(len(data_1_2[u'LABEL'].value_counts()))

    data_1_2 = dutil.remove_repet_data(data_1_2)
    print(data_1_2.shape)
    output_file = output_file%len(data_1_2)
    print(output_file)
    dutil.save_data(data_1_2,output_file)




def data_1_3_process():
    '''
        ./data/1.3.csv 文件的处理
    :return:
    '''
    data_1_3_file_path = './data/1.3.csv'
    output_file = 'result/v2.1_train_%d.csv'

    data_1_3 = dutil.load_data(data_1_3_file_path)
    # 取出用户话语'USER'
    data_1_3 = data_1_3[data_1_3[u'Who']=='User']
    data_1_3[u'LABEL'] = data_1_3[u'原始语义']
    data_1_3[u'SENTENCE'] = data_1_3[u'话语']
    data_1_3 = data_1_3[[u'LABEL',u'SENTENCE']]
    data_1_3.sort_values(by=[u'LABEL',u'SENTENCE'])
    print(data_1_3[u'LABEL'].value_counts().sort_index())
    # print(data_1_3.head())
    data_1_3 = dutil.remove_repet_data(data_1_3)
    print(data_1_3.shape)
    print('|'.join(data_1_3[u'LABEL'].value_counts().sort_index().index))
    print('|'.join([str(item) for item in data_1_3[u'LABEL'].value_counts().sort_index().values]))
    # print(data_1_3[u'LABEL'].value_counts().sort_index())
    # print(sum(data_1_3[u'LABEL']!='ID'))
    # print(data_1_3[u'LABEL'].value_counts().sort_index())
    # print(data_1_3.head())
    print(data_1_3.shape)
    output_file = output_file%len(data_1_3)
    print(output_file)
    dutil.save_data(data_1_3,output_file)

    # 检验在v2.1_test_64.csv出现过的句子和数量
    data_1_2 = dutil.load_data('data/v2.1_test_64.csv')
    in_data_1_2 = lambda x: x in data_1_2['SENTENCE'].as_matrix()

    is_in = data_1_3['SENTENCE'].apply(in_data_1_2)
    print(data_1_3[is_in])
    print(sum(is_in))



def data_1_4_process():
    '''
        ./data/1.4.csv 文件的处理
    :return:
    '''
    data_1_4_file_path = './data/1.4.csv'
    output_file = 'result/v2.1_train_%d.csv'

    data_1_4 = dutil.load_data(data_1_4_file_path)
    data_1_4[u'LABEL'] = data_1_4[u'20160628标注']
    data_1_4 = data_1_4[[u'LABEL',u'SENTENCE']]
    data_1_4.sort_values(by=[u'LABEL',u'SENTENCE'])
    # 去重
    data_1_4 = dutil.remove_repet_data(data_1_4)
    print('|'.join(data_1_4[u'LABEL'].value_counts().sort_index().index))
    print('|'.join([str(item) for item in data_1_4[u'LABEL'].value_counts().sort_index().values]))
    print(len(data_1_4[u'LABEL'].value_counts().sort_index()))


    print(data_1_4.head())
    print(data_1_4.shape)
    output_file = output_file%len(data_1_4)
    print(output_file)

    dutil.save_data(data_1_4,output_file)

    # 检验在v2.1_test_64.csv出现过的句子和数量
    data_1_2 = dutil.load_data('data/v2.1_test_64.csv')
    in_data_1_2 = lambda x : x in data_1_2['SENTENCE'].as_matrix()

    is_in = data_1_4['SENTENCE'].apply(in_data_1_2)
    print(data_1_4[is_in])
    print(sum(is_in))

def merge_data_1_3_and_1_4_process():


    data_1_3 = dutil.load_data('data/v2.1_train_1104.csv')
    data_1_4 = dutil.load_data('data/v2.1_train_S_1786.csv')



    data = pd.concat((data_1_3,data_1_4),axis=0)
    print(data.shape)
    data = dutil.remove_repet_data(data)


    print(data.shape)
    output_file = 'result/v2.1_train_%d.csv'
    output_file = output_file%len(data)
    print(output_file)
    # 保存结果
    dutil.save_data(data,output_file)

    # 检验在v2.1_test_64.csv出现过的句子和数量
    data_1_2 = dutil.load_data('result/v2.1_test_64.csv')
    in_data_1_2 = lambda x: x in data_1_2['SENTENCE'].as_matrix()

    print('---|'*len(data[u'LABEL'].value_counts()))
    print('|'.join(data[u'LABEL'].value_counts().sort_index().index))
    print('|'.join([str(item) for item in data[u'LABEL'].value_counts().sort_index().values]))
    is_in = data['SENTENCE'].apply(in_data_1_2)
    print(data[is_in])
    print(sum(is_in))
    data = data[-is_in]
    # 将64句中出现过的句子去除，并保存一个结果
    print(data.shape)

    output_file = 'result/v2.1_train_%d.csv'
    output_file = output_file%len(data)
    print(output_file)
    # 保存结果
    dutil.save_data(data,output_file)










if __name__ == '__main__':
    dutil = DataUtil()
    # data_1_2_process()
    # data_1_3_process()
    data_1_4_process()
    # merge_data_1_3_and_1_4_process()



