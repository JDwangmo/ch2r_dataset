#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-11'
__email__ = '383287471@qq.com'

import numpy as np
import pandas as pd
import logging
import timeit
logging.basicConfig(filename='log/count.log',filemode = 'w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()
print('='*30)
print '统计20150616-20160615测试集的数据'
print('='*30)
print 'start running!'
logging.debug('='*30)
logging.debug('统计20150616-20160615测试集的数据')
logging.debug('='*30)
logging.debug('start running!')
logging.debug('='*20)
import re
import io
from jiebanlp.toolSet import seg

log_data_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/20150616-20160615测试集/output/dialogue_usersentence_ge_1.csv'

aiml_usersentence_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/20150616-20160615测试集/output/aiml_usersentence.csv'

train_data_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160526/train_all.csv'

# 20150614-20150615 测试集中 OOD话语(已去除检测错误(ID)),总共366句
log_data_20150614To0615_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/20150614-20150615测试集/20160125/data/ood_sentence.csv'

# 20150614-20150615 测试集中 进入分类OOD话语,已人工标注好,总共64句
log_labeled_data_20150614To0615_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/20150614-20150615测试集/20160125/data/ood_labeled.csv'


# ------------------------------------------------------------------------------
# -------------- region start : 取出所有进入协处理(AIML)的句子 -------------
# -------------- 并检测是否出现在AIML训练库和20150614-20150615日志中 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('取出所有进入协处理(AIML)的句子')
# 读取AIML训练库中所有句子
train_data = pd.read_csv(train_data_file_path,sep='\t',encoding='utf8')
log_data_20150614To0615 = pd.read_csv(log_data_20150614To0615_file_path,sep='\t',encoding='utf8',index_col=0)
log_labled_data_20150614To0615 = pd.read_csv(log_labeled_data_20150614To0615_file_path,sep='\t',encoding='utf8',index_col=0)
verbose = 1


train_data = train_data[train_data['LABEL']!=u'其他#其他']
sentence_to_seg = lambda x: seg(x, sep=' ',
                                full_mode=False,
                                remove_stopword=False,
                                replace_number=False,
                                verbose=verbose)
# 分词
train_data['WORDS'] = train_data['SENTENCE'].apply(sentence_to_seg)
log_labled_data_20150614To0615['WORDS'] = log_labled_data_20150614To0615['SENTENCE'].apply(sentence_to_seg)

# 分词
log_data_20150614To0615['PAST1_SENTENCE'] = log_data_20150614To0615['PAST1_SENTENCE'].apply(lambda x: (x.split('|')[-1]).replace(' ',''))
log_data_20150614To0615['CUR_SENTENCE'] = log_data_20150614To0615['CUR_SENTENCE'].apply(lambda x: ''.join(x.split('|')[1:]).replace(' ',''))

log_data_20150614To0615['WORDS'] = log_data_20150614To0615['PAST1_SENTENCE'].apply(sentence_to_seg)

# print log_data_20150614To0615.head()

log_data = pd.read_csv(log_data_file_path,sep='\t',encoding='utf8')

# 因为只有 协处理回答的句子才有 做标记 ,所以只能
# 通过标记 '【进入状态机' 找出 协处理的回答,并通过 回答的句子 往上找出协处理句子和上一句Ch2R话语
corprocessor_response = log_data['Record'].apply(lambda x:unicode(x).__contains__(u'【'))
corprocessor_response = corprocessor_response.as_matrix()

corprocessor_sentence = np.asarray(corprocessor_response.tolist()[1:]+[False]*1)
pre_sentence = np.asarray(corprocessor_response.tolist()[2:]+[False]*2)
# print log_data[pre_sentence]
# print log_data[corprocessor_response]
# 全部输出
count = 0
aiml_sentences = pd.DataFrame(columns = ['CH2R','USER','USER_WORDS','IN_AIML','IN_20150614-0615'])

for ch2r_sentence,user_sentence,res in zip(log_data[pre_sentence].as_matrix(),
                                           log_data[corprocessor_sentence].as_matrix(),
                                           log_data[corprocessor_response].as_matrix()):

    ch2r_sentence = ch2r_sentence[2:]
    user_sentence = user_sentence[2:]
    # print sentence_to_seg(user_sentence[-1])
    # 是否在AIML训练库中
    sentence_seg = sentence_to_seg(user_sentence[-1])

    if sum(train_data['WORDS'] == sentence_seg) > 0:
        aiml_result = train_data[train_data['WORDS'] == sentence_seg].as_matrix()[0][0]
    else:
        aiml_result = 'None'

    # 是否在20150614-20150615对话段中出现
    if sum(log_data_20150614To0615['WORDS'] == sentence_seg) > 0:
        # 在20150614-20150615对话段中出现
        if sum(log_labled_data_20150614To0615['WORDS'] == sentence_seg) > 0:
            # 在标注好的分类集中,可提供分类参考
            in_log14To15_result = log_labled_data_20150614To0615[log_labled_data_20150614To0615['WORDS'] == sentence_seg].as_matrix()[0][1]
            # print in_log14To15_result

        elif sum(train_data['WORDS'] == sentence_seg) > 0:
            # 未出现在标注好的分类集但出现AIML库中,可提供分类参考
            # print log_data_20150614To0615[log_data_20150614To0615['WORDS'] == sentence_seg]
            in_log14To15_result = train_data[train_data['WORDS'] == sentence_seg].as_matrix()[0][0]
            # print in_log14To15_result
        else:
            # 未出现,无法提供参考,暂归为 '其他#其他'
            in_log14To15_result = '其他#其他'
        # print user_sentence[-1]
    else:
        in_log14To15_result = 'None'
    # print aiml_result
    count += 1
    # fout.write(u'---------第%d句\n'%(count))
    # fout.write(u'%d\n'%(count))
    # fout.write('%s\n' % ('\t'.join([unicode(i) for i in ch2r_sentence]+['AIML','in 20150614-0615'])))
    # print ch2r_sentence[-1], user_sentence[-1].strip(), sentence_seg,aiml_result,in_log14To15_result

    aiml_sentences.loc[len(aiml_sentences)] = [ch2r_sentence[-1], user_sentence[-1].strip(), sentence_seg,aiml_result,in_log14To15_result]
    # fout.write('%s\n' % ('\t'.join([unicode(i) for i in user_sentence]+[aiml_result,in_log14To15_result])))
    # print user_sentence[-1]
# 保存成 DataFrame格式,好进行处理,比如排序等
aiml_sentences.to_csv(aiml_usersentence_file_path,sep='\t',encoding='utf8',header=None,index=False)

quit()

# 按用户话语进行归类,目的在于使得同样的用户话语聚集在一起,方便对比.
aiml_sentences = aiml_sentences.sort_values(by=['USER'])
print aiml_sentences.head()

patten = re.compile(u'【.*')
with io.open(aiml_usersentence_file_path,'w',encoding='utf8') as fout:
    for index,(ch2r,user,in_AIML,in_20150614To0615) in enumerate(aiml_sentences.values):
        # fout.write(u'---------第%d句\n'%(index+1))
        # print ch2r
        ch2r = patten.sub('',ch2r)
        # print ch2r
        # quit()
        fout.write(u'%d\tCh2R\t%s\t%s\t%s\n'%(index+1,ch2r,'nan','nan'))
        fout.write(u'%d\tUser\t%s\t%s\t%s\n'%(index+1,user,in_AIML,in_20150614To0615))


logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 取出所有进入协处理(AIML)的句子 ---------------
# ------------------------------------------------------------------------------


end_time = timeit.default_timer()
logging.debug('end! Running time:%ds!'%(end_time-start_time))
print 'end! Running time:%ds!'%(end_time-start_time)
logging.debug('='*20)
