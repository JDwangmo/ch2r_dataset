# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-01-09'; 'last updated date: 2017-01-09'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import logging
import timeit
import re


class DataUtil(object):
    def __init__(self):
        pass

    @staticmethod
    def load_ch2r_log_files(file_path):
        data = pd.read_csv(
            file_path,
            sep=',',
            encoding='utf8',
            quoting=0,
        )
        return data

    @staticmethod
    def save_pd_data_to_file(data, file_path):
        data.to_csv(
            file_path,
            encoding='utf8',
            sep='\t',
            quoting=1,
            index=None,
        )
    @staticmethod
    def output_id_sentence_with_label():
        """
        输出 ID 话语和 句型模式的 标签

        :return:
        """
        data = DataUtil.load_ch2r_log_files('origin/ch2r_log_from20160601_to20170109.csv')
        # 找到 ID 句子
        is_id_sentence = np.asarray(data[u'Remark'].fillna('').apply(lambda x: x.__contains__(u'所用句型')).values)

        func_find_sentence_label = lambda x: re.findall(u'所用句型：(.*?)</p>', x)[0].replace(u' + 值未定', '').replace(
            u' + 值已定', '')
        sentences_label = map(func_find_sentence_label, data.loc[is_id_sentence, u'Remark'].values)
        # 总共有多少句
        print(len(sentences_label))
        # print(sum(is_id_sentence))
        # 找出真正的ID 句子 -- 上一句才是 识别的句子
        id_sentences_index = np.arange(len(data))[is_id_sentence] - 1

        new_data = pd.DataFrame(data={
            'SessionID': data[u'SessionID'].iloc[id_sentences_index].values,
            'Name': data[u'Name'].iloc[id_sentences_index].values,
            'Record': data[u'Record'].iloc[id_sentences_index].values,
            'Label': sentences_label
        })
        print(new_data.head())
        # 输出
        DataUtil.save_pd_data_to_file(new_data, 'after_clean/ch2r_log_from20160601_to20170109_id_sentences.csv')


def main():
    data = DataUtil.load_ch2r_log_files('origin/ch2r_log_from20160601_to20170109.csv')
    is_id_sentence = np.asarray(data[u'Remark'].fillna('').apply(lambda x: x.__contains__(u'所用句型')).values)
    func_find_sentence_label = lambda x: re.findall(u'所用句型：(.*?)</p>', x)[0].replace(u' + 值未定','').replace(u' + 值已定','')
    sentences_label = map(func_find_sentence_label, data.loc[is_id_sentence, u'Remark'].values)
    print(len(sentences_label))
    # print(sum(is_id_sentence))
    # 上一句才是 识别的句子
    id_sentences_index = np.arange(len(data))[is_id_sentence] - 1
    new_data = pd.DataFrame(data={
        'SessionID': data[u'SessionID'].iloc[id_sentences_index].values,
        'Name': data[u'Name'].iloc[id_sentences_index].values,
        'Record': data[u'Record'].iloc[id_sentences_index].values,
        'Label': sentences_label
    })
    print(new_data.head())
    DataUtil.save_pd_data_to_file(new_data, 'after_clean/ch2r_log_from20160601_to20170109_id_sentences.csv')


if __name__ == '__main__':
    main()
