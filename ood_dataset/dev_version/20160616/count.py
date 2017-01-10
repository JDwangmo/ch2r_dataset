#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-16'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
import yaml
verbose = 2
logging.basicConfig(filename='log/count.log',filemode = 'w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()
print('='*30)
print('='*30)
print 'start running!'
logging.debug('='*30)
import re
import io
# ------------------------------------------------------------------------------
# -------------- region start : 变量在这里定义 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('变量在这里定义')

aiml_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160616/data/aiml_merge.csv'
aiml_sortedByuser_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160616/output/aiml_merge_sortedByuser.csv'
aiml_dialogue_sortedByuser_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160616/output/aiml_dialogue_sortedByuser.csv'

label_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160616/data/OOD标注-label.csv'

labeled_aiml_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/dev_vesion/20160616/output/labeled_data.csv'

logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 变量在这里定义 ---------------
# ------------------------------------------------------------------------------

aiml_data = pd.read_csv(aiml_file_path,sep='\t',header=None,encoding='utf8')
aiml_data.columns=['CH2R','USER','USER_WORDS','IN_AIML','IN_20150614-0615']
# print aiml_data.head()

# ------------------------------------------------------------------------------
# -------------- region start : 按用户话语进行归类,目的在于使得同样的用户话语聚集在一起,方便对比 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('按用户话语进行归类,目的在于使得同样的用户话语聚集在一起,方便对比')

print aiml_data.head()
# 将3045句OOD话语按句子排序后，保存起来
aiml_data['USER'].sort_values().to_csv(aiml_sortedByuser_file_path,
                                             encoding='utf8',
                                             index=False)


# 按用户话语进行归类,目的在于使得同样的用户话语聚集在一起,方便对比.
# 以对话段的形式显示，方便标注
aiml_sentences = aiml_data.sort_values(by=['USER'])
patten = re.compile(u'【.*')
aiml_sentences['CH2R'] = aiml_sentences['CH2R'].apply(lambda x: patten.sub('',x))

with io.open(aiml_dialogue_sortedByuser_file_path,'w',encoding='utf8') as fout:
    for index,(ch2r,user,user_words,in_AIML,in_20150614To0615) in enumerate(aiml_sentences.values):
        # fout.write(u'---------第%d句\n'%(index+1))
        # print ch2r
        ch2r = patten.sub('',ch2r)
        # print ch2r
        # quit()
        # fout.write(u'%d\tCh2R\t%s\n'%(index+1,ch2r))
        fout.write(u'%d\tCh2R\t%s\t%s\t%s\n'%(index+1,ch2r,'nan','nan'))
        fout.write(u'%d\tUser\t%s\t%s\t%s\n'%(index+1,user,in_AIML,in_20150614To0615))


logging.debug('='*30)
logging.debug('start running!')
logging.debug('='*20)



logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 按用户话语进行归类,目的在于使得同样的用户话语聚集在一起,方便对比 ---------------
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# -------------- region start : 对OOD话语进行统计 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('对OOD话语进行统计')


logging.debug(u'OOD的句子有:%d,去重之后有:%d'%(len(aiml_sentences),len(aiml_sentences['USER_WORDS'].unique())))
print(u'OOD的句子有:%d,去重之后有:%d'%(len(aiml_sentences),len(aiml_sentences['USER_WORDS'].unique())))

logging.debug(u'出现AIML训练库中的句子有:%d,去重之后有:%d'%(len(aiml_sentences[aiml_sentences['IN_AIML']!='None']),len(aiml_sentences[aiml_sentences['IN_AIML']!='None']['USER_WORDS'].unique())))

print (u'出现AIML训练库中的句子有:%d,去重之后有:%d'%(len(aiml_sentences[aiml_sentences['IN_AIML']!='None']),len(aiml_sentences[aiml_sentences['IN_AIML']!='None']['USER_WORDS'].unique())))


logging.debug(u'出现在20150614-20150615测试集中的句子有:%d,去重之后有:%d'%(len(aiml_sentences[aiml_sentences['IN_20150614-0615']!='None']),len(aiml_sentences[aiml_sentences['IN_20150614-0615']!='None']['USER_WORDS'].unique())))
print(u'出现在20150614-20150615测试集中的句子有:%d,去重之后有:%d'%(len(aiml_sentences[aiml_sentences['IN_20150614-0615']!='None']),len(aiml_sentences[aiml_sentences['IN_20150614-0615']!='None']['USER_WORDS'].unique())))
#

aiml_sentences[(aiml_sentences['IN_AIML']=='None').as_matrix() * (aiml_sentences['IN_20150614-0615']!='None').as_matrix()]['USER_WORDS'].to_csv('temp.tmp',sep='\t',encoding='utf8',index=False)

logging.debug(u'出现在AIML训练库且出现20150614-20150615测试集中的句子有:%d,去重之后有:%d'%(len(aiml_sentences[(aiml_sentences['IN_AIML']!='None').as_matrix() * (aiml_sentences['IN_20150614-0615']!='None').as_matrix()]),len(aiml_sentences[(aiml_sentences['IN_AIML']!='None').as_matrix() * (aiml_sentences['IN_20150614-0615']!='None').as_matrix()]['USER_WORDS'].unique())))

print(u'出现在AIML训练库且出现20150614-20150615测试集中的句子有:%d,去重之后有:%d'%(len(aiml_sentences[(aiml_sentences['IN_AIML']!='None').as_matrix() * (aiml_sentences['IN_20150614-0615']!='None').as_matrix()]),len(aiml_sentences[(aiml_sentences['IN_AIML']!='None').as_matrix() * (aiml_sentences['IN_20150614-0615']!='None').as_matrix()]['USER_WORDS'].unique())))

logging.debug('交集中的句子分布情况如下:')
# print '交集中的句子分布情况如下:'

# print aiml_sentences[(aiml_sentences['IN_AIML']!='None').as_matrix() * (aiml_sentences['IN_20150614-0615']!='None').as_matrix()]['USER_WORDS'].value_counts()

logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 对OOD话语进行统计 ---------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -------------- region start : 将标注好的标签和数据合并,并统计 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('将标注好的标签和数据合并,并统计')

def merge_sentence_label():
    # -------------- region start : 加载人工标注的标签 -------------
    logging.debug('-' * 20)
    logging.debug('加载人工标注的标签')
    print '加载人工标注的标签'

    labels = []
    with open(label_file_path,'r') as fin:
        for line in fin:
            labels.append(line.strip())
    logging.debug('-' * 20)
    # -------------- region end : 加载人工标注的标签 ---------------
    # -------------- region start : 添加到ood数据中 -------------
    logging.debug('-' * 20)
    print('-' * 20)
    logging.debug('添加到ood数据中')
    print '添加到ood数据中'

    aiml_sentences['LABEL'] = labels

    logging.debug('-' * 20)
    # -------------- region end : 添加到ood数据中 ---------------
merge_sentence_label()
logging.debug('ID的句子有:%d条'%(sum(aiml_sentences['LABEL']=='ID')))
print 'ID的句子有:%d条'%(sum(aiml_sentences['LABEL']=='ID'))

aiml_sentences = aiml_sentences[aiml_sentences['LABEL']!='ID']

logging.debug('其他#其他 的句子有%d条'%(sum(aiml_sentences['LABEL']=='其他#其他')))
print '其他#其他 的句子有%d条'%(sum(aiml_sentences['LABEL']=='其他#其他'))
aiml_sentences =  aiml_sentences[aiml_sentences['LABEL']!='其他#其他']
print len(aiml_sentences)

logging.debug('去除ID和其他#其他的句子之后...')
print '去除ID和其他#其他的句子之后...'
print aiml_sentences.tail()

logging.debug('出现在20150614-20150615测试集中的句子有:%d,去重之后有:%d'%(
    sum(aiml_sentences['IN_20150614-0615']!='None'),
    len(aiml_sentences[aiml_sentences['IN_20150614-0615']!='None']['USER_WORDS'].unique())
)
              )
print '出现在20150614-20150615测试集中的句子有:%d,去重之后有:%d'%(sum(aiml_sentences['IN_20150614-0615']!='None'),len(aiml_sentences[aiml_sentences['IN_20150614-0615']!='None']['USER_WORDS'].unique()))

logging.debug('出现在AIML训练库中的句子有:%d'%(sum(aiml_sentences['IN_AIML']!='None')))
print '出现在AIML训练库中的句子有:%d,去重之后有:%d'%(sum(aiml_sentences['IN_AIML']!='None'),len(aiml_sentences[aiml_sentences['IN_AIML']!='None']['USER_WORDS'].unique()))

mask = (aiml_sentences['IN_AIML']=='None').as_matrix() * (aiml_sentences['IN_20150614-0615']=='None').as_matrix()
# print mask
logging.debug('不出现AIML训练库中也不出现在20150614-20150615测试集中的句子有%d句,去重之后有:%d'%(len(aiml_sentences[mask]),len(aiml_sentences[mask]['USER_WORDS'].unique())))
print '不出现AIML训练库中也不出现在20150614-20150615测试集中的句子有%d句,去重之后有:%d'%(len(aiml_sentences[mask]),len(aiml_sentences[mask]['USER_WORDS'].unique()))

aiml_sentences = aiml_sentences[mask]
aiml_sentences = aiml_sentences.sort_values(by=['LABEL'])
aiml_sentences[['LABEL','USER',]].to_csv(labeled_aiml_file_path,
                      sep='\t',
                      index=False,
                      header=True,
                      encoding='utf8')

logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 将标注好的标签和数据合并,并统计 ---------------
# ------------------------------------------------------------------------------

print aiml_sentences['LABEL'].value_counts()

end_time = timeit.default_timer()
print 'end! Running time:%ds!'%(end_time-start_time)
logging.debug('='*20)
logging.debug('end! Running time:%ds!'%(end_time-start_time))