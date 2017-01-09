## ID Data Set - Dev Version
### Describe
- ID 数据集的开发版本

### Project Structure
- [ch2r_log_files](https://github.com/JDwangmo/ch2r_dataset/id_dataset/dev_vesion/ch2r_log_files):
    - Ch2R 日志文件
    - 文件/子文件夹
        - data_util.py： 日志文件数据工具类  
        - origin：原始导出的日志文件
            - `1 ch2r_log_from20160601_to20170109.csv`: Ch2R 从 20160601 - 20170109 的对话记录日志文件;
        - after_clean：清理后的日志文件
            - `1 ch2r_log_from20160601_to20170109_id_sentences.csv`：
                - `origin/1 ch2r_log_from20160601_to20170109.csv`的提取出来ID句子和句型模式标签;
            - `2 ID句型模式标注（ch2r_log_from20160601_to20170109_id_sentences）.xls`：
                - `1 ch2r_log_from20160601_to20170109_id_sentences.csv`的excel版本;
### Dependence lib
- pandas

### User Manual
- 1 
- 2 