#encoding=utf8
__author__ = 'jdwang'
__date__ = 'create date: 2016-06-15'
__email__ = '383287471@qq.com'
import numpy as np
import pandas as pd
import logging
import timeit
import yaml
logging.basicConfig(filename='log/data_clean.log',filemode = 'w',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
start_time = timeit.default_timer()
print('='*30)
print 'start running!'
logging.debug('='*30)
logging.debug('='*30)
logging.debug('start running!')
logging.debug('='*20)

import re
import io

log_data_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/start-20150613测试集/data/startTo20150613.log.utf8'
data_clean_file_path = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/start-20150613测试集/output/data_clean.csv'
# 对话段中,用户话语数量大于等于(ge,>=1)1句的对话段
dialogue_sentence_gt_1 = '/home/jdwang/PycharmProjects/corprocessor/coprocessor/Corpus/ood_dataset/start-20150613测试集/output/dialogue_usersentence_ge_1.csv'

# ------------------------------------------------------------------------------
# -------------- region start : 清理数据,使之变成pandas.DataFrame格式的数据 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('清理数据,使之变成pandas.DataFrame格式的数据')

# 清理数据
pattern = re.compile('^[a-z0-9]{24,32}\t')
with open(data_clean_file_path,'w') as fout:
    fin = open(log_data_file_path,'r')
    fout.write(fin.readline())
    temp = []
    for line in fin:
        line = line.strip()
        temp.append(line)
        if pattern.findall(line):
            # print line:
            # print ','.join(temp)
            # 以下输出进行处理,是由于部分记录会被截断成几句,所以进行下面处理
            if len(temp)>1:
                output = '\t'.join(temp[:-1]).split('\t')
                output = '\t'.join(output[0:3] + [','.join(output[3:])])
                fout.write('%s\n'%(output))

            temp=[temp[-1]]
            # pass
        else:
            # print line
            pass
    # 最后一条记录的输出
    output = '\t'.join(temp).split('\t')
    output = '\t'.join(output[0:3] + [','.join(output[3:])])
    fout.write('%s\n' % (output))

# quit()

logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 清理数据,使之变成pandas.DataFrame格式的数据 ---------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -------------- region start : 取出对话段中,用户话语 数量大于等于(ge,>=1)1句的对话段 -------------
# ------------------------------------------------------------------------------
logging.debug('=' * 20)
logging.debug('取出对话段中,用户话语 数量大于等于(ge,>=1)1句的对话段')


log_data = pd.read_csv(data_clean_file_path,sep='\t',header=0,encoding='utf8')

print log_data.head()
# print log_data.describe()
user_name = log_data['Name'].unique().tolist()
user_name.remove(u'Ch2R')
print set(user_name)
logging.debug('用户个数:%d'%(len(user_name)))
print '用户个数:%d'%(len(user_name))

logging.debug('对话中是否空句子的个数:%d'%(sum(log_data['Record'].isnull())))
print '对话中是否空句子的个数:%d'%(sum(log_data['Record'].isnull()))


logging.debug('对话段数有:%d'%(len(log_data['SessionID'].unique())))
print '对话段数有:%d'%(len(log_data['SessionID'].unique()))
# print log_data[log_data['Record'].isnull()]
dialogue_counts = 0
user_name_set = []
with io.open(dialogue_sentence_gt_1,'w',encoding='utf8') as fout:
    # 输出表头 column
    # print '\t'.join(log_data.columns)
    fout.write('%s\n' % '\t'.join(log_data.columns))

    for session_id, group in log_data.groupby(by=['SessionID']):
        # print session_id
        # print group
        # 用户数
        if session_id=='hbtusvrojcnapxnlncfw2cmq':
            pass
        user_name = group['Name'].unique().tolist()
        user_name.remove(u'Ch2R')
        # 回合数
        rounds_count = len(group)/2.0
        try:
            # print user_name[0]
            if rounds_count >= 1:
                user_name_set.extend(user_name)
                # fout.write(u'测试者:%s\n'%(user_name[0]))
                # fout.write(u'回合数:%.1f\n'%(rounds_count))
                # fout.write(u'-'*20+'\n')
                dialogue_counts += 1
        except:
            logging.debug('exception!')
            print 'exception!'

            # print group
            # quit()

        for item in group.as_matrix():
            # print item

            try:
                fout.write('%s\n' % ('\t'.join([unicode(i) for i in item])))
            except:
                print 'exception!'
        # quit()
        # fout.write(u'-' * 20 + '\n')
user_name_set = set(user_name_set)
logging.debug('经过处理,总共对话段数为:%d'%(dialogue_counts))
print '经过处理,总共对话段数为:%d'%(dialogue_counts)

logging.debug(u'再次确认,参与对话的用户数为:%d'%(len(user_name_set)))
print u'再次确认,参与对话的用户数为:%d'%(len(user_name_set))
logging.debug(u'用户有:%s'%(','.join(user_name_set)))
print u'用户有:%s'%(','.join(user_name_set))
# print user_name_set



logging.debug('=' * 20)
# ------------------------------------------------------------------------------
# -------------- region end : 取出对话段中,用户话语 数量大于等于(ge,>=1)1句的对话段 ---------------
# ------------------------------------------------------------------------------


end_time = timeit.default_timer()
print 'end! Running time:%ds!'%(end_time-start_time)
logging.debug('='*20)
logging.debug('end! Running time:%ds!'%(end_time-start_time))