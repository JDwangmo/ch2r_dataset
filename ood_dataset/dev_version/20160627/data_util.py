# encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-28'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
import io


class DataUtil(object):
    def __init__(self):
        pass

    def load_data(self, path,header=0):
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


    def merge_sentence_label(self,aiml_sentence_file_path,label_file_path):
        # 读取OOD标注中3045句的原始句子,已经按用户句子排序
        aiml_sentence_data = data_util_object.load_data(aiml_sentence_file_path, header=None)
        aiml_sentence_data.columns = [u'SENTENCE']
        aiml_sentence_data = aiml_sentence_data[[u'SENTENCE']].sort_values(by=[u'SENTENCE'])

        # -------------- region start : 加载人工标注的标签 -------------
        logging.debug('-' * 20)
        logging.debug('加载人工标注的标签')
        print '加载人工标注的标签'

        labels = []
        with io.open(label_file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                labels.append(line.strip())
        logging.debug('-' * 20)
        # -------------- region end : 加载人工标注的标签 ---------------
        # -------------- region start : 添加到ood数据中 -------------
        logging.debug('-' * 20)
        print('-' * 20)
        logging.debug('添加到ood数据中')
        print '添加到ood数据中'

        aiml_sentence_data['LABEL'] = labels

        logging.debug('-' * 20)
        # -------------- region end : 添加到ood数据中 ---------------
        return aiml_sentence_data


if __name__ == '__main__':

    data_0516_train_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160627/data/20160526数据集重新标注-train.csv'
    data_0516_test_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160627/data/20160526数据集重新标注-test.csv'

    # 读取OOD标注中3045句的原始句子,已经按用户句子排序
    aiml_sentence_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160616/data/aiml_merge_sortedByuser.csv'
    label_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160627/data/OOD标注-label.csv'


    data_util_object = DataUtil()
    data_0516_train = data_util_object.load_data(data_0516_train_file_path, header=0)
    # 使用 20160628标注 的标注数据
    data_0516_train[u'LABEL'] = data_0516_train[u'20160628标注']
    data_0516_train = data_0516_train[[u'LABEL', u'SENTENCE']].sort_values(by = ['LABEL','SENTENCE'])

    data_0516_train.to_csv('output/20160526-train.csv',
                           encoding='utf8',
                           sep='\t',
                           index=False)
    # print data.head()

    data_0516_test = data_util_object.load_data(data_0516_test_file_path, header=0)
    data_0516_test[u'LABEL'] = data_0516_test[u'20160628标注']
    data_0516_test = data_0516_test[[u'LABEL',u'SENTENCE']].sort_values(by = ['LABEL','SENTENCE'])
    data_0516_test.to_csv('output/20160526-test.csv',
                          encoding='utf8',
                          sep='\t',
                          index=False)
    # print data_0516_test[u'LABEL'].value_counts()
    # quit()

    aiml_data = data_util_object.merge_sentence_label(aiml_sentence_file_path,label_file_path)
    aiml_data.to_csv('output/OOD标注-all.csv',
                     encoding='utf8',
                     sep='\t',
                     index=False)
    # 将两份数据合并
    data = pd.concat((data_0516_train, aiml_data), axis=0)
    # 按 句子和 类别排序，保存一份 全部数据集的数据
    data = data.sort_values(by=['LABEL','SENTENCE'])
    data.to_csv('output/新数据集标注-train-all.csv',
                encoding='utf8',
                sep='\t',
                index=False)
    print len(set(data['LABEL']))
    temp = []
    no_repet_label = []
    no_repeat_sentence = []
    for label, group in data.groupby(by=[u'LABEL']):

        no_repet_label.extend([label]*len(group[u'SENTENCE'].unique()))
        no_repeat_sentence.extend(group[u'SENTENCE'].unique())
        print label, len(group[u'SENTENCE']),len(group[u'SENTENCE'].unique())
        temp.append((label, len(group[u'SENTENCE']),len(group[u'SENTENCE'].unique())))

    no_repeat_data = pd.DataFrame(data={'LABEL':no_repet_label,
                                        'SENTENCE':no_repeat_sentence})
    print no_repeat_data.head()
    no_repeat_data.to_csv('output/新数据集标注-train-all-norepetition.csv',
                          encoding='utf8',
                          sep='\t',
                          index=False)
    print u'|'.join([unicode('---') for i, j,k in temp])
    print u'|'.join([unicode(i) for i, j,k in temp ])
    print u'|'.join([unicode(j) for i, j,k in temp ])
    print u'|'.join([unicode(k) for i, j,k in temp ])
    print '去重前句子数：%d'%sum([j for i, j,k in temp ])
    print '去重后句子数：%d'%sum([k for i, j,k in temp ])
    # print aiml_sentences.head()
    # aiml_sentences[]
    # data = data.sort_values(by=[u'SENTENCE'])
    # print data.head()
    # print data[u'LABEL'].value_counts()


    quit()

    # -------------- region start : 首先读取20160526版本的train_all.csv数据集,然后生成新文件,去进行重新类别检查,使用的OOD分类标准是:OOD标注标准-20160627.xlsx -------------
    logging.debug('-' * 20)
    print '-' * 20
    logging.debug('首先读取20160526版本的train_all.csv数据集,然后生成新文件,去进行重新类别检查,使用的OOD分类标准是:OOD标注标准-20160627.xlsx')
    print '首先读取20160526版本的train_all.csv数据集,然后生成新文件,去进行重新类别检查,使用的OOD分类标准是:OOD标注标准-20160627.xlsx'

    train_all_20160526_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160526/train_all.csv'

    train_all_20160526 = pd.read_csv(train_all_20160526_file_path,
                                     sep='\t',
                                     encoding='utf8',
                                     header=0)
    train_all_20160526 = train_all_20160526[['SENTENCE', 'LABEL']].sort_values(by=['LABEL', 'SENTENCE'])

    train_all_20160526.to_csv('output/train_all_simple.csv',
                              sep='\t',
                              encoding='utf8',
                              header=True,
                              index=False)

    print  train_all_20160526.head()

    logging.debug('-' * 20)
    print '-' * 20
    # -------------- region end : 首先读取20160526版本的train_all.csv数据集,然后生成新文件,去进行重新类别检查,使用的OOD分类标准是:OOD标注标准-20160627.xlsx ---------------

    # ------------------------------------------------------------------------------
    # -------------- region start : 接下来对测试集同样处理 -------------
    # ------------------------------------------------------------------------------
    logging.debug('=' * 20)
    logging.debug('接下来对测试集同样处理')

    test_all_20160526_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160526/ood_labeled.csv'

    test_all_20160526 = pd.read_csv(test_all_20160526_file_path,
                                    sep='\t',
                                    encoding='utf8',
                                    header=0)
    test_all_20160526 = test_all_20160526[['SENTENCE', 'LABEL']].sort_values(by=['LABEL', 'SENTENCE'])

    test_all_20160526.to_csv('output/test_all_simple.csv',
                             sep='\t',
                             encoding='utf8',
                             header=True,
                             index=False)

    logging.debug('=' * 20)
    # ------------------------------------------------------------------------------
    # -------------- region end : 接下来对测试集同样处理 ---------------
    # ------------------------------------------------------------------------------
