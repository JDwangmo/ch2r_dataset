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
        jutil = Jieba_Util(verbose=0)
        self.remove_sentence_punctuation = lambda x: jutil.seg(x, sep='', remove_url=False)
        self.get_sentence_length = lambda x: len(jutil.seg(x,
                                                    sep=' ',
                                                    full_mode=False,
                                                    remove_stopword=False,
                                                    replace_number=False,
                                                    lowercase=True,
                                                    zhs2zht=True,
                                                    remove_url=True,
                                                    HMM=False,
                                                    ).split())


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

    def print_data_detail(self,data):
        '''
            打印模型详情

        :param data:
        :return:
        '''
        print(data.shape)
        print(data.head())
        print(data[u'LABEL'].value_counts().sort_index())
        print(len(data[u'LABEL'].value_counts().sort_index()))



    def check_data_in_test(self, data):
        '''
            检查数据是不是在测试集中出现过

            1.获取AIML训练库句子

        :param data:
        :return:
        '''
        aiml_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160712/data/v2.2_test_L_76.csv'

        data_aiml = dutil.load_data(aiml_file_path)
        # print(data_aiml.head())
        # print(data_aiml.shape)
        data[u'SEG'] = data['SENTENCE'].apply(self.remove_sentence_punctuation)
        data_aiml[u'SEG'] = data_aiml['SENTENCE'].apply(self.remove_sentence_punctuation)
        aiml_result = []
        for counter, sentence in enumerate(data['SEG'].values):
            # print('第%d个...'%(counter+1))
            in_aiml = data_aiml['SEG'] == sentence
            # print(sum(in_aiml))
            # quit()
            if sum(in_aiml) > 0:
                aiml_result.append(data_aiml.loc[in_aiml, 'LABEL'].values[0])
                print(sentence)
            else:
                aiml_result.append('NONE')

        # data['AIML'] = aiml_result
        return np.asarray(aiml_result)

    def check_dataA_in_dataB(self, dataA, dataB_file_path):
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
        for counter, sentence in enumerate(dataA['SEG'].values):
            # print('第%d个...'%(counter+1))
            in_dataB = dataB['SEG'] == sentence
            # print(sum(in_aiml))
            # quit()
            if sum(in_dataB) > 0:
                result.append(dataB.loc[in_dataB, 'LABEL'].values[0])
            else:
                result.append('NONE')

        # data['AIML'] = aiml_result
        return np.asarray(result)

    def check_data_in_3045(self, data):
        '''
            检查数据是不是在3045中出现过

            1.获取AIML训练库句子

        :param data:
        :return:
        '''
        aiml_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160712/data/v2.1_train_1058.csv'

        data_aiml = dutil.load_data(aiml_file_path)
        data_aiml = data_aiml[data_aiml['LABEL']!='ID']
        print(data_aiml.shape)

        # print(data_aiml.head())
        # print(data_aiml.shape)
        data[u'SEG'] = data['SENTENCE'].apply(self.remove_sentence_punctuation)
        data_aiml[u'SEG'] = data_aiml['SENTENCE'].apply(self.remove_sentence_punctuation)
        aiml_result = []
        for counter, sentence in enumerate(data['SEG'].values):
            # print('第%d个...'%(counter+1))
            in_aiml = data_aiml['SEG'] == sentence
            # print(sum(in_aiml))
            # quit()
            if sum(in_aiml) > 0:
                aiml_result.append(data_aiml.loc[in_aiml, 'LABEL'].values[0])
                print(data_aiml.loc[in_aiml])
            else:
                aiml_result.append('NONE')

        # data['AIML'] = aiml_result
        return np.asarray(aiml_result)

    def show_sentence_length_detail(self,data):
        data['LENGTH']= data['SENTENCE'].apply(self.get_sentence_length)
        print(sum(data['LENGTH']==0))

        print(data.head())
        print(data['LENGTH'].value_counts().sort_index())



def process_train_data_S():
    '''
        生成新版 小数据集 v2.2(S)

        处理文件：dev_vesion/20160712/data/v2.2_train_S_1793_raw.csv
        输出文件：
            1. dev_vesion/20160712/result/v2.2_train_S_1518.csv
            2. dev_vesion/20160712/result/v2.2_train_S_1423.csv

        1. 排序
        2. 去除 ID类 句子，剩下1518句
        3. 去除3045句中出现的（95句），剩下1423句
        4. 排序

    :return:
    '''

    input_file_path = './data/v2.2_train_1793_raw.csv'
    output_file_path = './result/v2.2_train_S_%d.csv'

    raw_data_S = dutil.load_data(input_file_path)
    raw_data_S = raw_data_S.sort_values(by=[u'LABEL',u'SENTENCE'])
    # 2. 去除 ID类 句子，剩下1518句
    data_S = raw_data_S[[u'LABEL',u'SENTENCE']]
    data_S = data_S[data_S['LABEL']!='ID']
    print(data_S.shape)

    dutil.save_data(data_S,output_file_path%len(data_S))

    # 3. 去除3045句中出现的（95句），剩下1423句
    result = dutil.check_dataA_in_dataB(data_S,'data/v2.2_train_Sa_879.csv')

    is_in = result != 'NONE'
    is_not_in = result == 'NONE'
    print(sum(is_in))
    print(sum(is_not_in))
    data_S = data_S[is_not_in]

    data_S = data_S[[u'LABEL',u'SENTENCE']]
    data_S = data_S.sort_values(by=[u'LABEL',u'SENTENCE'])

    # print(data_S.shape)
    # print(data_S['LABEL'].value_counts().sort_index())

    # 保存
    dutil.save_data(data_S,output_file_path%len(data_S))

def process_train_data_L():
    '''
        生成新版 大数据集

        处理文件：dev_vesion/20160712/data/v2.2_train_1058.csv 和 dev_vesion/20160712/data/v2.2_train_S_1423.csv
        输出文件：dev_vesion/20160712/result/v2.2_train_L_2302.csv

        1. 将1059句（从3045去重而来）中去除 ID类 句子,剩下880句
        2. 将880句与小数据集（1417句）合并，有2297句，构成大数据集
        3. 排序
        4. 保存

    :return:
    '''

    data_3045_file_path = './data/v2.2_train_1058.csv'
    data_S_file_path = './data/v2.2_train_S_1423.csv'
    output_file_path = './result/v2.2_train_L_%d.csv'

    data_3045 = dutil.load_data(data_3045_file_path)
    data_S = dutil.load_data(data_S_file_path)
    # 1. 将1059句（从3045去重而来）中去除 ID类 句子,剩下880句
    data_3045 = data_3045[data_3045['LABEL'] != 'ID']
    print(data_3045.shape)


    data = pd.concat((data_3045,data_S),axis=0)
    print(data.shape)
    data = dutil.remove_repet_data(data)
    print(data.shape)

    data = data[[u'LABEL',u'SENTENCE']]
    data = data.sort_values(by=[u'LABEL',u'SENTENCE'])


    # 保存
    print(output_file_path%len(data))
    dutil.save_data(data,output_file_path%len(data))



def process_test_data():
    '''
        处理文件： ./data/v2.2_test_366.csv，366句,已经使用v2.2标注
        1. 去除 ID 类的句子，剩下362句, 输入各类别的分布情况
        2. 去重，
        检查是否在
        去除在3045句中的句子,剩下 76句
        保存作为v2.2(S)版本测试数据集
        3. 保存作为v2.2(L)版本测试数据集

    :return:
    '''
    # 1. 去除 ID 类的句子，剩下362句
    output_file_path = 'result/v2.2_test_%d.csv'
    data = dutil.load_data('./data/v2.2_test_366.csv')
    data = data[data['LABEL'] != 'ID']
    # data = data[data['LABEL'] != u'其它#其它']
    print(len(data))
    # 2、去重
    data = dutil.remove_repet_data(data)
    dutil.save_data(data,'result/data.tmp')

    print(len(data))
    # dutil.print_data_detail(data)


    # 3、检查是否在AIML（小数据集 v2.2 S）中，并去除其句子，剩下131句
    result_aiml = dutil.check_dataA_in_dataB(data,'data/v2.2_train_S_1518.csv')

    data_not_match_aiml = data[result_aiml=='NONE']
    data_match_aiml = data[result_aiml!='NONE']
    data_not_match_aiml = dutil.remove_repet_data(data_not_match_aiml)
    data_match_aiml = dutil.remove_repet_data(data_match_aiml)
    # dutil.save_data(data_not_match_aiml,'result/data_not_match.tmp')
    dutil.save_data(data_not_match_aiml[[u'LABEL',u'SENTENCE']],output_file_path%len(data_not_match_aiml))

    # print(len(data_not_match_aiml))
    # print(len(data_match_aiml))
    # dutil.print_data_detail(data_not_match_aiml)
    # 4、检查是否在3045句（小数据集 v2.2 Sa）中，并去除其句子，剩下95句
    result_3045 = dutil.check_dataA_in_dataB(data, 'data/v2.2_train_Sa_879.csv')

    data_not_match_3045 = data[result_3045 == 'NONE']
    data_match_3045 = data[result_3045 != 'NONE']
    data_not_match_3045 = dutil.remove_repet_data(data_not_match_3045)
    data_match_3045 = dutil.remove_repet_data(data_match_3045)
    print(len(data_not_match_3045))
    print(len(data_match_3045))
    dutil.save_data(data_not_match_3045[[u'LABEL',u'SENTENCE']],output_file_path%len(data_not_match_3045))

    # 5、 合并不在AIML和不在3045句，并去重，剩下132句
    result_merge = (result_3045=='NONE') * (result_aiml=='NONE')

    data_not_match_all = data[result_merge]
    dutil.print_data_detail(data_not_match_all)

    dutil.save_data(data_not_match_all[[u'LABEL',u'SENTENCE']],output_file_path%len(data_not_match_all))



def genernate_aiml_file():
    '''
        生成新的aiml训练库文件

        输入文件：
            1. ./data/v2.2_train_L_2302.csv,2302句
            2. ./data/v2.2_test_L_76.csv,76句
        输出文件：
            1. ./result/v2.2_aiml_file_2378.csv，2378句

        步骤：
            1. 合并数据1和2

    :return:
    '''

    output_file_path = './result/v2.2_aiml_file_%d.csv'

    train_data_L = dutil.load_data('./data/v2.2_train_L_2302.csv')
    test_data_L = dutil.load_data('./data/v2.2_test_L_76.csv')
    # data = dutil.remove_repet_data(train_data_L)

    dutil.check_data_in_test(train_data_L)
    # quit()

    print(train_data_L.shape)
    print(test_data_L.shape)

    data = pd.concat((train_data_L,test_data_L),axis=0)

    print(data.shape)
    data = dutil.remove_repet_data(data)
    print(data.shape)

    dutil.save_data(data,output_file_path%len(data))


def process_train_data_Sa():
    """
        生成新版 小数据集 v2.2(Sa)

        处理文件：dev_vesion/20160712/data/v2.2_train_1058.csv,1058句
        输出文件：dev_vesion/20160712/result/v2.2_train_Sa_879.csv

        1. 去除 ID类 句子，剩下879句
        2. 排序

    :return:
    """

    data = dutil.load_data('./data/v2.2_train_1058.csv')

    data = data[data['LABEL'] != u'ID']
    data = dutil.remove_repet_data(data)
    data = data.sort_values(by=[u'LABEL',u'SENTENCE'])

    print(data.shape)

    dutil.save_data(data,'./result/v2.2_train_Sa_879.csv')


def process_train_data_sentence_length():
    '''
        打印数据中句子的长度分布情况

        处理文件： data/v2.2_aiml_file_2378.csv

    :return:
    '''

    data_file_path = 'data/v2.2_train_L_2302.csv'
    data = dutil.load_data(data_file_path)
    data = data[data['LABEL']!=u'其它#其它']
    data = data[data['LABEL']!=u'其它#捣乱']
    print(data.shape)
    dutil.show_sentence_length_detail(data)





if __name__ == '__main__':
    dutil = DataUtil()
    # process_train_data_Sa()
    # process_train_data_S()
    # process_train_data_L()
    # process_test_data()
    #
    # genernate_aiml_file()

    # process_train_data_sentence_length()

    quit()

    # dutil.print_data_detail(dutil.load_data('data/v2.2_test_138.csv'))
    dutil.print_data_detail(dutil.load_data('data/v2.2_train_S_1423.csv'))
    quit()
    # dutil.print_data_detail(dutil.load_data('data/v2.2_train_S_1787.csv'))
    # dutil.print_data_detail(dutil.load_data('data/v2.1_train_1059.csv'))
    data = dutil.load_data('data/v2.2_train_S_1787.csv')
    print(data.shape)
    result = dutil.check_data_in_3045(data)
    data = data.loc[result=='NONE']
    print(data.shape)
    dutil.print_data_detail(data)
