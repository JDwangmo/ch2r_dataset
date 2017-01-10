# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-01-09'; 'last updated date: 2017-01-09'
    Email:   '383287471@qq.com'
    Describe: OOD 数据集 - stable version 数据工具类
                - 数据的部分设置 在文件 setting.py 中
"""

from __future__ import print_function
import logging
import pandas as pd
# 导入设置
from setting import DATA_ROOT_PATH,WORD2VEC_MODEL_ROOT_PATH
from data_processing_util.cross_validation_util import data_split_k_fold
from data_processing_util.jiebanlp.jieba_util import Jieba_Util


class DataUtil(object):
    """
    OOD 数据集 - stable version 数据工具类
        - 数据的部分设置 在文件 setting.py 中
    """
    def __init__(self):
        # 训练数据的根目录
        self.dataset_root_path = DATA_ROOT_PATH
        self.word2vec_model_root_path = WORD2VEC_MODEL_ROOT_PATH
        self.jieba_util = None

    def get_label_index(self, version='v2.0'):
        """
            获取 DA 分类类别的列表，总共有24类

        :return: label_to_index,index_to_label
        """
        if version == 'v1.0':
            # 16分类标签
            index_to_label = [
                u'捣乱#骂人',

                u'导购#开始', u'导购#成交',
                u'导购#更换', u'导购#详情',

                u'表态#附和', u'表态#否定',
                u'表态#犹豫', u'表态#肯定', u'表态#否定#不满', u'表态#随便',

                u'闲聊#身份信息', u'闲聊#天气',
                u'闲聊#问候', u'闲聊#时间', u'闲聊#结束语',
            ]
        elif version == 'v2.0':
            # 24分类标签
            index_to_label = [
                u'其它#骂人',

                u'导购#不成交', u'导购#不理解', u'导购#开始',
                u'导购#成交', u'导购#更换', u'导购#结束', u'导购#详情',

                u'表态#不满', u'表态#否定', u'表态#满意',
                u'表态#犹豫', u'表态#疑问', u'表态#肯定', u'表态#附和', u'表态#随便',

                u'社交义务#不用谢', u'社交义务#接受道歉', u'社交义务#致谢',
                u'社交义务#道歉', u'社交义务#问候',

                u'闲聊#天气', u'闲聊#时间', u'闲聊#身份信息'
            ]
        elif version == 'v2.0++':
            # 24分类标签 ++所用版本，临时用
            index_to_label = [
                u'其它#骂人',

                u'导购#不成交', u'导购#不理解', u'导购#开始',
                u'导购#成交', u'导购#更换', u'导购#结束', u'导购#详情',

                u'社交义务#不用谢', u'社交义务#接受道歉', u'社交义务#致谢',
                u'社交义务#道歉', u'社交义务#问候',

                u'表态#不满', u'表态#否定', u'表态#满意',
                u'表态#犹豫', u'表态#疑问', u'表态#肯定', u'表态#附和', u'表态#随便',

                u'闲聊#天气', u'闲聊#时间', u'闲聊#身份信息'
            ]
        elif version == 'v2.0_small':
            # 17分类标签
            index_to_label = [
                u'其它#骂人',

                u'导购#开始',
                u'导购#成交', u'导购#更换', u'导购#结束', u'导购#详情',

                u'表态#否定', u'表态#不满',
                u'表态#犹豫', u'表态#肯定', u'表态#附和', u'表态#随便',

                u'社交义务#不用谢',
                u'社交义务#问候',

                u'闲聊#天气', u'闲聊#时间', u'闲聊#身份信息'
            ]

        # print('类别数为：%d'%len(index_to_label))
        label_to_index = {label: idx for idx, label in enumerate(index_to_label)}
        return label_to_index, index_to_label

    def transform_word2vec_model_name(self, flag):
        """
            根据 flag 转换成完整的 word2vec 模型文件名

        :param flag:
        :return:
        """

        from data_processing_util.word2vec_util.word2vec_util import Word2vecUtil
        return Word2vecUtil().transform_word2vec_model_name(flag)

    def transform_dataset_name(self, flag):
        """
            将数据集标记转为真正的训练集和测试集文件名

        :param flag: 数据集标记 v1.0(S),v2.2(S),v2.2(Sa),v2.2(L),v2.3(S)
        :type flag: str
        :return: train_data_file_path,test_data_file_path
        """

        if flag == 'v1.0(S)':
            # 使用v2.2 L版本的数据集
            train_data_file_path = self.dataset_root_path + '20160526/train_all.csv'
            test_data_file_path = self.dataset_root_path + '20160526/ood_labeled.csv'
        elif flag == 'v2.2(L)':
            # 使用v2.2 L版本的数据集
            train_data_file_path = self.dataset_root_path + 'v2.2/v2.2_train_L_2302.csv'
            test_data_file_path = self.dataset_root_path + 'v2.2/v2.2_test_L_76.csv'
        elif flag == 'v2.2(S)':
            # 使用v2.2 S版本的数据集
            train_data_file_path = self.dataset_root_path + 'v2.2/v2.2_train_S_1518.csv'
            test_data_file_path = self.dataset_root_path + 'v2.2/v2.2_test_S_131.csv'
        elif flag == 'v2.2(Sa)':
            # 使用v2.2 Sa版本的数据集
            train_data_file_path = self.dataset_root_path + 'v2.2/v2.2_train_Sa_893.csv'
            test_data_file_path = self.dataset_root_path + 'v2.2/v2.2_test_Sa_79.csv'
            # else:
            #     如果匹配不上，则使用v2.2 Sa版本的数据集
            # train_data_file_path = self.dataset_root_path + 'v2.2/v2.2_train_L_2302.csv'
            # test_data_file_path = self.dataset_root_path + 'v2.2/v2.2_test_L_76.csv'
        elif flag == 'v2.3(L)':
            # 使用v2.2 L版本的数据集
            train_data_file_path = self.dataset_root_path + 'v2.3/v2.3_train_L_2300.csv'
            test_data_file_path = self.dataset_root_path + 'v2.3/v2.3_test_L_76.csv'
        elif flag == 'v2.3(S)':
            # 使用v2.2 S版本的数据集
            train_data_file_path = self.dataset_root_path + 'v2.3/v2.3_train_S_1518.csv'
            test_data_file_path = self.dataset_root_path + 'v2.3/v2.3_test_S_131.csv'
        elif flag == 'v2.3(Sa)':
            # 使用v2.2 Sa版本的数据集
            train_data_file_path = self.dataset_root_path + 'v2.3/v2.3_train_Sa_891.csv'
            test_data_file_path = self.dataset_root_path + 'v2.3/v2.3_test_Sa_79.csv'
        else:
            # 如果匹配不上，则使用v2.2 Sa版本的数据集
            train_data_file_path = self.dataset_root_path + 'v2.3/v2.3_train_L_2300.csv'
            test_data_file_path = self.dataset_root_path + 'v2.3/v2.3_test_L_76.csv'

        return train_data_file_path, test_data_file_path

    def merge_to_17class(self, data):
        '''
            将新版数据集合并成17个类别

        :param data:
        :return:
        '''

        data.loc[data['LABEL'] == u'导购#不理解', 'LABEL'] = u'其它#其它'
        data.loc[data['LABEL'] == u'表态#疑问', 'LABEL'] = u'其它#其它'
        data.loc[data['LABEL'] == u'表态#满意', 'LABEL'] = u'表态#肯定'
        data.loc[data['LABEL'] == u'导购#不成交', 'LABEL'] = u'导购#结束'
        data.loc[data['LABEL'] == u'社交义务#接受道歉', 'LABEL'] = u'导购#结束'
        data.loc[data['LABEL'] == u'社交义务#致谢', 'LABEL'] = u'导购#结束'
        data.loc[data['LABEL'] == u'社交义务#道歉', 'LABEL'] = u'导购#结束'
        # print(','.join(data['LABEL'].unique()))
        # print(len(data['LABEL'].unique()))
        # quit()
        return data

    def load_train_test_data(self, config):
        """
            加载训练数据和测试数据，根据配置选择
            加载的文件中一定要有 LABEL 和 SENTENCE 字段

        :param config:
        :return:
        """

        logging.debug('=' * 20)

        train_data_file_path, test_data_file_path = self.transform_dataset_name(config['dataset_type'])

        # -------------- print start : just print info -------------
        if config['verbose'] > 0:
            logging.debug('加载%s版本数据集的训练数据和测试数据\n标注版本：%s' % (config['dataset_type'], config['label_version']))
            print('加载%s版本数据集的训练数据和测试数据\n标注版本：%s' % (config['dataset_type'], config['label_version']))
            logging.debug('train_data_file_path:%s' % train_data_file_path)
            logging.debug('test_data_file_path:%s' % test_data_file_path)
            print('train_data_file_path:%s' % train_data_file_path)
            print('test_data_file_path:%s' % test_data_file_path)
        # -------------- print end : just print info -------------

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

        if config['label_version'] == 'v2.0_small':
            train_data = self.merge_to_17class(train_data)
            test_data = self.merge_to_17class(test_data)

        if config['verbose'] > 0:
            logging.debug('fit data shape is :%s' % (str(train_data.shape)))
            print('fit data shape is :%s' % (str(train_data.shape)))

            logging.debug('test data shape is :%s' % (str(test_data.shape)))
            print('test data shape is :%s' % (str(test_data.shape)))
            logging.debug('-' * 20)
            # 去除类别 其他#其他
            logging.debug('去除类别 其他#其他 ID,其他#捣乱')
            print('去除类别 其它#其它 ID 其他#捣乱')

        filter_row = lambda x: x not in [u'其它#其它', u'其他#其他', u'ID', u'其它#捣乱']

        train_data['IS_FILTER'] = train_data['LABEL'].apply(filter_row)
        test_data['IS_FILTER'] = test_data['LABEL'].apply(filter_row)

        train_data = train_data[train_data['IS_FILTER'] == True]
        test_data = test_data[test_data['IS_FILTER'] == True]

        if config['verbose'] > 0:
            logging.debug('fit data shape is :%s' % (str(train_data.shape)))
            print('fit data shape is :%s' % (str(train_data.shape)))

            logging.debug('test data shape is :%s' % (str(test_data.shape)))
            print('test data shape is :%s' % (str(test_data.shape)))
            logging.debug('-' * 20)

        train_data = train_data[['LABEL', 'SENTENCE']]
        test_data = test_data[['LABEL', 'SENTENCE']]

        label_to_index, index_to_label = self.get_label_index(version=config['label_version'])
        if config['verbose'] > 0:
            logging.debug(u'总共类别数:%d,分别为:%s' % (len(index_to_label), ','.join(index_to_label)))
            print(u'总共类别数:%d,分别为:%s' % (len(index_to_label), ','.join(index_to_label)))

        train_data['LABEL_INDEX'] = train_data['LABEL'].map(label_to_index)

        test_data['LABEL_INDEX'] = test_data['LABEL'].map(label_to_index)

        return train_data, test_data

    def batch_segment_sentences(self, sentences):
        '''
            对多个句子批量分词

        :param sentences: array-like
        :return:
        '''
        self.jieba_util = Jieba_Util()
        segmented_sentences = map(self.segment_sentence, sentences)
        return segmented_sentences

    def segment_sentence(self, sentence):
        '''
            将句子进行分词

        :param sentence:
        :return:
        '''
        segmented_sentence = self.jieba_util.seg(sentence=sentence,
                                                 sep=' ',
                                                 full_mode=True,
                                                 remove_stopword=False,
                                                 replace_number=True,
                                                 lowercase=True,
                                                 zhs2zht=True,
                                                 remove_url=True,
                                                 )
        return segmented_sentence

    def save_data(self, data, path):
        '''
            保存DataFrame格式的数据

        :param data: 数据
        :param path: 数据文件的路径
        :return: None
        '''
        data.to_csv(path,
                    sep='\t',
                    header=True,
                    index=False,
                    encoding='utf8',
                    )

    def save_result(self, data, predict, is_correct, path):
        '''
            将预测结果进行保存

        :param data: 数据，DataFrame
        :param predict: 预测结果
        :type predict: array-like
        :param is_correct: 是否正确
        :param path: 路径
        :return: None
        '''
        label_to_index, index_to_label = self.get_label_index()
        data['PREDICT'] = [index_to_label[item] for item in predict]
        data['is_correct'] = is_correct
        self.save_data(data, path)

    def load_data(self, path):
        '''
            加载DataFrame格式的数据

        :param data: 数据
        :param path: 数据文件的路径
        :return: None
        '''
        data = pd.read_csv(path,
                           sep='\t',
                           header=0,
                           encoding='utf8',
                           index_col=0,
                           )
        return data

    def get_k_fold_data(self,
                        k=5,
                        data=None,
                        rand_seed=0,
                        ):
        '''
            将数据分为K-fold

        :param k:
        :param data:
        :type data: pd.DataFrame()
        :return:
        '''

        train_X = data['SENTENCE'].as_matrix()
        train_y = data['LABEL_INDEX'].as_matrix()

        cv_x = []
        cv_y = []
        for x, y in data_split_k_fold(k=k, data=(train_X, train_y), rand_seed=rand_seed):
            cv_x.append(x)
            cv_y.append(y)
        return cv_x, cv_y


if __name__ == '__main__':
    data_util = DataUtil()
    # quit()

    data = pd.read_csv(
        '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/stable_vesion/v2.2/v2.2_train_Sa_893.csv',
        sep='\t')
    # data = data_util.load_data('/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/stable_vesion/20160708/v2.1_train_S_1786.csv')

    print(data.head())
    print(data.columns)
    print(data.shape)
    print(data[u'LABEL'].value_counts().sort_index())
    print(len(data[u'LABEL'].value_counts().sort_index()))
