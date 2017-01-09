#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-22'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
from jiebanlp.toolSet import seg

class DataUtil(object):
    def __init__(self,full_mode=True,remove_stopword=True,verbose=0):
        self.__full_mode__ = full_mode
        self.__remove_stopword__ = remove_stopword
        self.__verbose__ = verbose

    def __seg__(self, sentence):
        '''
        对句子进行分词,使用jieba分词
        :param sentence: 句子
        :type sentence: str
        :return:
        '''
        sentence_to_seg = lambda x: seg(x,
                                        sep=' ',
                                        full_mode=self.__full_mode__,
                                        remove_stopword=self.__remove_stopword__,
                                        verbose=self.__verbose__
                                        )
        return sentence_to_seg(sentence)

    def count_sentences_length(self,sentences):
        '''
            统计句子的长度分布情况,按词统计
        :type sentences: numpy.array()
        :param sentences: 句子集合
        :return:
        :rtype: numpy.array()
        '''
        segmented_sentences = map(self.__seg__,sentences)
        sentences_length = [len(item.split()) for item in segmented_sentences]
        # print sentences_length

        return sentences_length


if __name__ == '__main__':
    train_data_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160526/train_all.csv'
    test_data_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160526/ood_labeled.csv'
    train_data = pd.read_csv(
        train_data_file_path,
        sep='\t',
        encoding='utf8',
        header=0
    )

    test_data = pd.read_csv(
        test_data_file_path,
        sep='\t',
        encoding='utf8',
        header=0
    )

    logging.debug('fit data shape is :%s' % (str(train_data.shape)))
    print('fit data shape is :%s' % (str(train_data.shape)))

    logging.debug('test data shape is :%s' % (str(test_data.shape)))
    print('test data shape is :%s' % (str(test_data.shape)))
    logging.debug('-' * 20)
    # 去除类别 其他#其他
    logging.debug('去除类别 其他#其他')
    print('去除类别 其他#其他')
    train_data = train_data[train_data['LABEL'] != u'其他#其他']
    test_data = test_data[test_data['LABEL'] != u'其他#其他']
    logging.debug('fit data shape is :%s' % (str(train_data.shape)))
    print('fit data shape is :%s' % (str(train_data.shape)))

    logging.debug('test data shape is :%s' % (str(test_data.shape)))
    print('test data shape is :%s' % (str(train_data.shape)))
    logging.debug('-' * 20)

    train_data = train_data[['LABEL', 'SENTENCE']]
    test_data = test_data[['LABEL', 'SENTENCE']]

    index_to_label = list(train_data['LABEL'].unique())
    logging.debug(u'总共类别数:%d,分别为:%s' % (len(index_to_label), ','.join(index_to_label)))

    print('总共类别数:%d' % (len(index_to_label)))

    data_util = DataUtil()
    # sentences = train_data['SENTENCE'].as_matrix()
    # sentences = [u'你好吗',u'买手机吗',u'我要买手机']
    train_sentences_length = data_util.count_sentences_length(train_data['SENTENCE'].as_matrix())
    print train_sentences_length
    test_sentences_length = data_util.count_sentences_length(test_data['SENTENCE'].as_matrix())
    print test_sentences_length
    # 长度情况
    from collections import Counter
    print sorted(Counter(test_sentences_length).items(),key=lambda x:x[0])
    print '|'.join([str(item[0]) for item in sorted(Counter(train_sentences_length).items(),key=lambda x:x[0])])
    print '|'.join(['%d'%item[1] for item in sorted(Counter(train_sentences_length).items(),key=lambda x:x[0])])
    print '|'.join(['%d'%item[0] for item in sorted(Counter(test_sentences_length).items(),key=lambda x:x[0])])
    print '|'.join(['%d'%item[1] for item in sorted(Counter(test_sentences_length).items(),key=lambda x:x[0])])

    print sorted(Counter(train_sentences_length).items(),key=lambda x:x[0])

    # train_sentences_length = [1,2,3,4,5,2,5,5,1,3,4,2,4,1,2,3,4,1,1,2,2,3,4,7]
    quit()
    from matplotlib import pyplot as plt
    from pylab import mpl
    mpl.rcParams['axes.labelsize']=25
    mpl.rcParams['ytick.labelsize']=25
    mpl.rcParams['xtick.labelsize']=25
    plt.figure(figsize=(500,100))
    plt.title(u'训练集句子长度分布(按词统计)')
    plt.xlim(min(train_sentences_length)-0.5,max(train_sentences_length)+0.5)
    train_count = plt.hist(train_sentences_length,normed=False,bins=len(set(train_sentences_length)))
    print sorted(train_sentences_length)
    print train_count[0]
    print sorted(set(train_sentences_length))
    print len(train_count[0])
    print sum(train_count[0])

    plt.show()
    plt.close()

    plt.figure(figsize=(500,100))
    plt.title(u'测试集句子长度分布(按词统计)')
    plt.xlim(min(test_sentences_length)-0.5,max(test_sentences_length)+0.5)
    test_count = plt.hist(test_sentences_length,normed=False,bins=len(set(test_sentences_length)))
    print test_count[0]
    print sorted(set(test_sentences_length))
    print len(test_count[0])
    print sum(test_count[0])
    plt.show()
    plt.close()


