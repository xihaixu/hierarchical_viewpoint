#!/usr/bin/env python3
# 过滤掉转发的推文，过滤掉非英文的推文
# packages to store and manipulate data
import pandas as pd
import numpy as np

# package to clean text
import json
import requests
import pandas as pd
import numpy as np
import emoji
import regex
import re
import string
from collections import Counter

# Natural Language Processing (NLP)
import spacy
import gensim
from spacy.tokenizer import Tokenizer
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from wordcloud import STOPWORDS
from nltk.stem import WordNetLemmatizer
from string import punctuation
from gingerit.gingerit import GingerIt

stopwords = set(STOPWORDS)
parser = GingerIt()
##表情和停用词可以自己定义
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']

urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*|(Doi:)[^ ]*)"
# userPattern = '@[^\s]+' #原来的
userPattern = '@[^\s]*'  # 我修改的
filter_userPattern = '@[^\s]*[\s]{1,1}[A-z]{1,}'
# great_userPattern=r'@\w+(?=\b[^"]*(("[^"]*){2})*$)'
great_userPattern = '^(?:\s*@\S*\s+){1,}'
alphaPattern = "[^a-zA-Z0-9]"  # 要注意，这样会把标点符号去掉。可以把要保留的标点符号直接加载列表里面，例如;表示保留分号
sequencePattern = r"(.)\1\1+"
seqReplacePattern = r"\1\1"
htmlTagPattern = r'<[^>]*>'


def give_emoji_free_text(text):
    """
    Removes emoji's from tweets
    Accepts:
        Text (tweets)
    Returns:
        Text (emoji free tweets)
    """
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text


def url_free_text(text):
    '''
    Cleans text from urls
    '''
    # text = re.sub(r'http\S+', '', text)  #这是我原来写的，jupter里面的
    text = re.sub(urlPattern, '', text)
    return text


def remove_xml(text):
    return re.sub(r'<[^<]+?>', '', text)


def remove_html(text):
    return re.sub(r'<[^>]*>', '', text)


def remove_selfdefine(text):  # &nbsp;|首空格->首标点符号|尾空格|@#|一个字母加标点:([^a^i]{1}[/])|尾部多余的标点符号
    return re.sub(r'(&.*?;)|(^(\s*([\s,-,.?:;\'"!`]+|(-{2})|(/.{3})|(/(/))|(/[/])|({}))\s*))|(\s+$)|(@|#)', '', text)


def replace_selfdefine(text):
    return re.sub(r'(w/)', ',', text)


def remove_newlines(text):
    return text.replace('\n', ' ')


def remove_manyspaces(text):
    return re.sub(r'\s+', ' ', text)


def clean_text(text):
    text = give_emoji_free_text(text)
    text = url_free_text(text)
    text = remove_xml(text)
    text = remove_newlines(text)
    text = remove_manyspaces(text)
    return text


def preprocess(textdata):
    processedText = []
    wordLemm = WordNetLemmatizer()

    for tweet in textdata:
        tweet = str(tweet).lower()
        print(tweet)
        # # Replace all URls with 'URL'
        # tweet = re.sub(urlPattern, 'URL', tweet)
        # # Replace all emojis.
        # for emoji in emojis.keys():
        #     tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        #     # Replace @USERNAME to 'USER'.
        # tweet = re.sub(userPattern, 'USER', tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # 删除末尾的标点符号
            word = re.sub("[^a-zA-Z0-9]+$|^[^a-zA-Z0-9]", "", word)
            # Checking if the word is a stopword.
            if word not in ALL_STOP_WORDS:  # 全单词匹配
                if len(word) > 3:  #原来的值为1，现在修改为3
                    # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweetwords += (word + ' ')
        # 避免covid-19这种，中间的-替换成空格变成covid 19.再执行拆分变成covid和19，再替换，那么就留下19这个数字
        # 类似情况还有i'm等。所以把去除非字符的代码移动下面
        # Replace all non alphabets.
        new_tweet = re.sub(alphaPattern, " ", tweetwords)
        new_tweetwords = ''
        for new_word in new_tweet.split():
            # Checking if the word is a stopword.
            if new_word not in ALL_STOP_WORDS:  # 全单词匹配
                if len(new_word) > 1:
                    new_tweetwords += (new_word + ' ')
        # 替换数字
        new_tweetwords = re.sub("^\d+|\s\d+", "", new_tweetwords)
        # 替换首行空格
        new_tweetwords = re.sub("^\s+", "", new_tweetwords)
        processedText.append(new_tweetwords)

    return processedText


def raw_clean(textdata):
    processedText = []
    for tweet in textdata:
        # delete all URls.
        tweet = url_free_text(str(tweet))
        # delete all emojis.
        tweet = give_emoji_free_text(tweet)
        # # delete all @users.
        # tweet = re.sub(userPattern, '', tweet)
        # delete @users in the front,not in text
        # searchObj = re.search(filter_userPattern, tweet)
        # if searchObj:
        #     secondObj = re.search(userPattern, searchObj.group())
        #     if secondObj:
        #         tweet=tweet[(searchObj.span()[0] + secondObj.span()[1] + 1):]
        tweet = re.sub(great_userPattern, '', tweet)
        # delete xml标记、空行、多空格
        tweet = remove_xml(tweet)
        tweet = remove_newlines(tweet)
        tweet = remove_manyspaces(tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        # delete其它自定义特殊字符
        tweet = remove_selfdefine(tweet)
        # replace其它自定义特殊字符
        tweet = replace_selfdefine(tweet)
        # 还原缩写词，忽略大小写进行替换
        for contraction_map in CONTRACTION_MAP.keys():
            print(r'(?i)' + contraction_map)
            tweet = re.sub(r'(?i)' + contraction_map, CONTRACTION_MAP[contraction_map], tweet)
        # 替换俚语
        try:
            tweet = parser.parse(tweet)['result']
            # 大小写有区别，所以尽量转换为小写再执行自动更正
            # 慎用，可能会修改句子原本的意思
            # 这里以大写的方式使用，可以保留HCQ这种大写的专有名词不被替换
        except:
            print("sorry!")
        finally:
            processedText.append(tweet)
            print(tweet)
    return processedText


if __name__ == "__main__":
    # data = pd.read_excel('filter_for_hcq.xlsx')
    data = pd.read_csv('1.7-1.21\\filter_for_hcq.csv')
    nlp = spacy.load('en_core_web_lg')
    # Custom stopwords
    custom_stopwords = ['rt', 'hi', '\n', '\n\n', '&amp;', ' ', '.', '-', 'got', "it's", 'it’s', "i'm", 'i’m', 'im',
                        'want', 'like', '$', '@', 'covid 19', 'covid19', 'covid-19', 'URL', 'EMOJI', 'USER',
                        '2019-ncov', '2019ncov', 'ncov2019', '2019cov', 'covid_19', '2019_ncov', 'ncov19', '2019n_cov',
                        'ncov-2019', 'covid2019', 'covid', 'corona', 'nCov', 'coronavirus', 'covid19',
                        'pandemic', 'ncov', 'cov', 'hydroxychloroquine', 'chloroquine', 'SARS', 'hcq', 'hydroxy','anti'
                        ]

    # Customize stop words by adding to the default list
    STOP_WORDS = nlp.Defaults.stop_words.union(custom_stopwords)

    # ALL_STOP_WORDS = spacy + gensim + wordcloud
    ALL_STOP_WORDS = STOP_WORDS.union(SW).union(stopwords).union(stopwordlist)

    # read stopword.txt
    morestopwords = []
    with open('stopword.txt', 'r') as f:
        for line in f:
            morestopwords.append(line.strip('\n'))
    ALL_STOP_WORDS = ALL_STOP_WORDS.union(morestopwords)

    # # # 下面是最新的清理方法
    # # text = list(data['slang_text_for_TOPIC_STANCE'])
    # # # 做一般的清理，为依存关系分析使用
    # # processedtext = raw_clean(text)
    # # name = ['cleand_text_for_DP']
    # # raw_text = pd.DataFrame(columns=name, data=processedtext)
    # # raw_text.to_csv("cat_for_hcq.csv", encoding="utf-8", index=False)

    # 做主题分析、立场识别前的分词、去停用词和词形还原处理
    text = list(data['text_with_DP'])
    processedtext = preprocess(text)
    name = ['foranylsis']
    raw_text = pd.DataFrame(columns=name, data=processedtext)
    raw_text.to_csv("topic_for_hcq.csv", encoding="utf-8", index=False)

    # #下面是做测试用的
    # text="Chloroquine scarcity hit Nigeria As The Deadly Coronavirus debut in the Western Africa nation. Female now sniff used sanitary pad to get high in Gombe, Insane right?"
    # for word in text.lower().split():
    #     # Checking if the word is a stopword.
    #     if word not in ALL_STOP_WORDS:
    #         print(word)
    # # print(parser.parse(text)['result'])
