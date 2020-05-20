# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:51:48 2020

@author: USER
"""

# Naive Bayes Classifier의 이해 - 한글
# 문장의 유사도 측정
'''
'메리가 좋아'
'고양이도 좋아'
'난 수업이 지루해'
'메리는 이쁜 고양이야'
'난 마치고 메리랑 놀거야'
'''

from konlpy.tag import Okt
from nltk.tokenize import word_tokenize
import nltk

pos_tagger = Okt()

train = [('메리가 좋아', 'pos'),
         ('고양이도 좋아', 'pos'),
         ('난 수업이 지루해', 'neg'),
         ('메리는 이쁜 고양이야', 'pos'),
         ('난 마치고 메리랑 놀거야', 'pos')]


all_words = set(word.lower()
                for sentence in train # sentence : '메리가 좋아'
                for word in word_tokenize(sentence[0])) # tokenize : 쪼개다! 분리하다! / word : '메리가', '좋아'

all_words
print(all_words)

t = [({word: (word in word_tokenize(x[0]))
        for word in all_words}, x[1]) for x in train]    # 원래는 붙여써야함!

t
print(t)

classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()

