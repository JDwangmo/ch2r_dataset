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
print '统计20150614-0615测试集的数据'
print('='*30)
print 'start running!'
logging.debug('='*30)
logging.debug('统计20150614-0615测试集的数据')
logging.debug('='*30)
logging.debug('start running!')
logging.debug('='*20)
import re

log_data_614_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/20150614-20150615测试集/20160125/data/20150614 log.csv'
log_data_615_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/20150614-20150615测试集/20160125/data/20150615 log.csv'

# 所有进入协处理的句子
corprocessor_sentence = pd.DataFrame(columns=['CUR_SENTENCE','PAST1_SENTENCE'])

# ------------------------------------------------------------------------------
# -------------- region start : log 2015年6月14日统计情况 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('log 2015年6月14日统计情况')


patten = re.compile('\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ')
pre_sentence = ''
count = 0
ood_614_sentences_aiml = []
ood_614_sentences_classify = []
for line in open(log_data_614_path,'r'):
    # 用户话语
    if patten.findall(line):
        # 去除两端空字符
        line = line.strip()
        # print line.split(',')
        line = ','.join(line.split(',')[:-10])
        line = line.strip('"')
    #     print line
        # quit()
        if line.__contains__('【AIML】'):

            corprocessor_sentence.loc[len(corprocessor_sentence)] = [line,pre_sentence]
            # count += 1
            # print pre_sentence
            # print line
            # quit()
            # pre_sentence = pre_sentence.split('|')[-1]
            ood_614_sentences_aiml.append(pre_sentence.split('|')[-1])
            if line.__contains__('【进入分类'):
                count += 1
                # print pre_sentence
                # print line
                # pre_sentence = pre_sentence.split('|')[-1]
                # print pre_sentence
                ood_614_sentences_classify.append(pre_sentence.split('|')[-1])

    pre_sentence = line

print count
logging.debug('2015年6月14日OOD话语有:%d句'%(len(ood_614_sentences_aiml)))
logging.debug('去重后:%d句'%(len(set(ood_614_sentences_aiml))))
print('2015年6月14日OOD话语有:%d句'%(len(ood_614_sentences_aiml)))
print('去重后:%d句'%(len(set(ood_614_sentences_aiml))))
logging.debug('2015年6月14日未被AIML匹配话语有:%d句'%(len(ood_614_sentences_classify)))
logging.debug('去重后:%d句'%(len(set(ood_614_sentences_classify))))
print('2015年6月14日未被AIML匹配话语有:%d句'%(len(ood_614_sentences_classify)))
print('去重后:%d句'%(len(set(ood_614_sentences_classify))))



logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : log 2015年6月14日统计情况 ---------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -------------- region start : log 2015年6月15日统计情况 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('log 2015年6月15日统计情况')


patten = re.compile('\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ')
pre_sentence = ''
count = 0
ood_615_sentences_aiml = []
ood_615_sentences_classify = []
for line in open(log_data_615_path,'r'):
    # 用户话语
    if patten.findall(line):
        # 去除两端空字符
        # count += 1
        line = line.strip()
        # print line.split(',')
        line = line.strip('"')
        # print line
        # quit()
        if line.__contains__('【AIML】'):
            corprocessor_sentence.loc[len(corprocessor_sentence)] = [pre_sentence,line]
            # count += 1
            # print pre_sentence
            # print line
            # quit()
            # pre_sentence = pre_sentence.split('|')[-1]
            ood_615_sentences_aiml.append(pre_sentence.split('|')[-1])

            if line.__contains__('【进入分类'):
                count += 1
                # print pre_sentence
                # print line
                # pre_sentence = pre_sentence.split('|')[-1]
                # print pre_sentence
                # ood_sentences.append(pre_sentence)
                ood_615_sentences_classify.append(pre_sentence.split('|')[-1])

    pre_sentence = line

print count
logging.debug('2015年6月15日OOD话语有:%d句'%(len(ood_615_sentences_aiml)))
logging.debug('去重后:%d句'%(len(set(ood_615_sentences_aiml))))
print('2015年6月15日OOD话语有:%d句'%(len(ood_615_sentences_aiml)))
print('去重后:%d句'%(len(set(ood_615_sentences_aiml))))

logging.debug('2015年6月15日未被AIML匹配话语有:%d句'%(len(ood_615_sentences_classify)))
logging.debug('去重后:%d句'%(len(set(ood_615_sentences_classify))))
print('2015年6月15日未被AIML匹配话语有:%d句'%(len(ood_615_sentences_classify)))
print('去重后:%d句'%(len(set(ood_615_sentences_classify))))

logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : log 2015年6月15日统计情况 ---------------
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# -------------- region start : 将所有协处理句子(20150614-0615)保存 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('将所有协处理句子(20150614-0615)保存')

ood_sentences_aiml = ood_614_sentences_aiml + ood_615_sentences_aiml
ood_sentences_classify = ood_614_sentences_classify + ood_615_sentences_classify
logging.debug('2015年6月14-15日OOD话语有:%d句'%(len(ood_sentences_aiml)))
logging.debug('去重后:%d句'%(len(set(ood_sentences_aiml))))
print('2015年6月14-15日OOD话语有:%d句'%(len(ood_sentences_aiml)))
print('去重后:%d句'%(len(set(ood_sentences_aiml))))

logging.debug('2015年6月14-15日未被AIML匹配话语有:%d句'%(len(ood_sentences_classify)))
logging.debug('去重后:%d句'%(len(set(ood_sentences_classify))))
print('2015年6月14-15日未被AIML匹配话语有:%d句'%(len(ood_sentences_classify)))
print('去重后:%d句'%(len(set(ood_sentences_classify))))



output_path = 'output/corprocessor_sentence.csv'
corprocessor_sentence.to_csv(output_path,sep='\t',encoding='utf8')

print corprocessor_sentence.shape

logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 将所有协处理句子(20150614-0615)保存 ---------------
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# -------------- region start : 去除OOD检测错误句子后再次统计 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('去除OOD检测错误句子后再次统计')
from jiebanlp.toolSet import seg
import io

# 使用现在的AIML训练库检验句子是否 AIML匹配
train_data_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160526/train_all.csv'

train_data = pd.read_csv(train_data_file_path,sep='\t',encoding='utf8')
train_data = train_data[train_data['LABEL']!=u'其他#其他']
# 分词
sentence_to_seg = lambda x: seg(x, sep=' ',
                               full_mode=False,
                               remove_stopword=False,
                               replace_number=False,
                               verbose=0)
train_data['WORDS'] = train_data['SENTENCE'].apply(sentence_to_seg)

ood_sentence = pd.read_csv('/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/20150614-20150615测试集/20160125/data/ood_sentence.csv',sep='\t',header=0,index_col=0)
patten = re.compile(u'【.*')
filter_sentence = lambda x: patten.sub('',''.join(x.decode('utf8').split('|')[1:])).replace(' ','')
ood_sentence['PAST1_SENTENCE'] = ood_sentence['PAST1_SENTENCE'].apply(filter_sentence)
ood_sentence['CUR_SENTENCE'] = ood_sentence['CUR_SENTENCE'].apply(filter_sentence)
ood_sentence['WORDS'] = ood_sentence['PAST1_SENTENCE'].apply(sentence_to_seg)

print ood_sentence.head()
print ood_sentence.shape
# print ood_sentence['PAST1_SENTENCE']
print '去重后'
print len(ood_sentence['WORDS'].unique())

# 将去重后的句子进行保存
with io.open('output/ood_sentence_no_repetition.csv','w',encoding='utf8') as fout:
    for item in sorted(ood_sentence['WORDS'].unique()):
        fout.write(u'%s\n'%(item))


def in_aiml(sentence):
    # print sentence

    if len(train_data[train_data['WORDS'] == sentence]) == 0:
        return 'None'
    else:
        return train_data[train_data['WORDS'] == sentence].as_matrix()[0][0]

ood_sentence['IN_AIML'] = ood_sentence['WORDS'].apply(lambda x : in_aiml(x))

ood_sentence_aiml = ood_sentence[ood_sentence['IN_AIML'] != 'None']
ood_sentence_classify = ood_sentence[ood_sentence['IN_AIML'] == 'None']

logging.debug('AIML匹配到的有:%d'%(len(ood_sentence_aiml)))
print 'AIML匹配到的有:%d'%(len(ood_sentence_aiml))

logging.debug('AIML未匹配到的有:%d'%(len(ood_sentence_classify)))
print 'AIML未匹配到的有:%d'%(len(ood_sentence_classify))

logging.debug(u'AIML匹配到的去重后有:%d'%(len(ood_sentence_aiml['WORDS'].unique())))
print u'AIML匹配到的去重后有:%d'%(len(ood_sentence_aiml['WORDS'].unique()))

logging.debug(u'AIML未匹配到的去重后有:%d'%(len(ood_sentence_classify['WORDS'].unique())))
print u'AIML未匹配到的去重后有:%d'%(len(ood_sentence_classify['WORDS'].unique()))
# ood_sentence_classify['PAST1_SENTENCE'].to_csv('output/temp.csv')
# temp = pd.read_csv('/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/训练和测试集综合/20150614-0615测试集/20160125/data/ood.csv',sep='\t')
# print temp['SENTENCE'].head()
# print len(sorted(temp['SENTENCE']))
ood_sentence_classify = ood_sentence_classify.sort_values(by=['WORDS'])
with io.open('output/ood_sentence_classify.csv','w',encoding='utf8') as fout:
    for ch2r,user,words in ood_sentence_classify[['CUR_SENTENCE','PAST1_SENTENCE','WORDS']].values:
        if words in ood_sentence_classify['WORDS'].unique():
            fout.write(u'%s\t%s\n'%(user,words))

ood_sentence_aiml = ood_sentence_aiml.sort_values(by=['WORDS'])
with io.open('output/ood_sentence_aiml.csv','w',encoding='utf8') as fout:
    for ch2r,user,words in ood_sentence_aiml[['CUR_SENTENCE','PAST1_SENTENCE','WORDS']].values:
        if words in ood_sentence_aiml['WORDS'].unique():
            fout.write(u'%s\t%s\n'%(user,words))

quit()


logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 去除OOD检测错误句子后再次统计 ---------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -------------- region start : 统计测试句子的长度情况 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('统计测试句子的长度情况')

# 指的是 2015年6月14-15日进入协处理的OOD话语,有431句
print len(ood_sentences_aiml)
count_sentence_length = lambda x: len(x.decode('utf8'))
ood_sentence['SENTENCE_LENGTH'] = map(count_sentence_length,ood_sentence['PAST1_SENTENCE'])
print ood_sentence.head()
print '句子平均长度:%f'%np.average(ood_sentence['SENTENCE_LENGTH'])
print '|'.join(ood_sentence['SENTENCE_LENGTH'].value_counts().sort_index().index.astype(dtype=str))
print '|'.join(ood_sentence['SENTENCE_LENGTH'].value_counts().sort_index().as_matrix().astype(dtype=str))

temp = pd.read_csv('/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/20150614-20150615测试集/20160125/data/ood_labeled.csv',sep='\t',header=0,index_col=None)
temp[['SENTENCE','LABEL']].to_csv('output/temp.csv',sep='\t')


logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 统计测试句子的长度情况 ---------------
# ------------------------------------------------------------------------------


end_time = timeit.default_timer()
print 'end! Running time:%ds!'%(end_time-start_time)
logging.debug('='*20)
logging.debug('end! Running time:%ds!'%(end_time-start_time))