1. 执行1.数据预处理-单词分词.py中clear函数进行过滤，生成cleand_text_for_DP列

2. 对cleand_text_for_DP列进行人工核对，梳理表述，并替换一些
常见的特殊字符，如上所示。

3-1.添加hcq等药物到stopword（可以试着生成lda主题，看有高频的，
但对主题归纳没帮助的词，加入到stopword里面），
继续执行1.数据预处理-单词分词.py中proceed函数以便生成单词集合并保存
到“foranylsis"列。
3-2. 添加stopword，执行1.数据预处理-短语分词.py生成单词和短语集合。

4.在emeditor中针对两种方法的集合替换同义词，例如'realdonaldtrump',
'donald trump','donaldtrump','donald'替换成'trump'。

5.在emeditor软件中分别清除两种方法生成的集合中的重复项。

6.执行hlda_tomotopy.py生成hlda层次主题。从3开始往上调整层级数。

7.执行hop

8.执行读取top2vec主题.py训练主题模型，然后使用try_hierarchical_info.py
获得模型的层次主题词，使用getTopicnum_top2vec.py获得每个主题下文档的数量。

