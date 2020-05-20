# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:27:30 2020

@author: USER
"""
import sqlite3
import pandas as pd
import numpy as np

# DB 연결 (DB 파일 경로 넣기)
con = sqlite3.connect("C:/python_jar/20200421_python_pandas_bs_jar/digital.db")

# 테이블명 주의
read_df = pd.read_sql('select * from digital', con)
con.close()
print(read_df['가격']) #칼럼명 주의

# 문제1. 평균 / 최대 / 최소 / 4분위 값 각각 구하시오
mean_df = np.mean(read_df['가격']) # read_df.가격.mean()
max_df = np.max(read_df['가격'])
min_df = np.min(read_df['가격'])
first_df = np.percentile((read_df['가격']), 25)
second_df = np.percentile((read_df['가격']), 50)
third_df = np.percentile((read_df['가격']), 75)
fourth_df = np.percentile((read_df['가격']), 100)
print('평균값 : ', mean_df, '\n 최댓값 : ', max_df,
      '\n 최솟값 : ', min_df, '\n 1분위값 : ', first_df,
      '\n 2분위값 : ', second_df, '\n 3분위값 : ', third_df,
      '\n 4분위값 : ', fourth_df)

import matplotlib.pyplot as plt
plt.figure()
# 객체명.컬럼명 = 객체명['컬럼명']
plottingline = plt.plot(read_df.제품명, read_df.가격)
plt.xticks(rotation=90, fontsize=3)

plt.show()

