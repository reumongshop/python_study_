# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:18:51 2020

@author: USER
"""

# web 데이터 읽기
import pandas_datareader.data as web

import pandas as pd

import datetime

# 야후 증권
import yfinance

import sqlite3

# 추출한 시작 날짜 종료 날짜 설정
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2019, 1, 1)

# 야후증권으로부터 삼성전자 주식 추출
samsung = web.get_data_yahoo("005930.KS", start, end)

# 상위 5개만 출력
sf = samsung.tail(5)
print(sf)
print(type(samsung))

# 삼성전자 1년 데이터 데이터베이스에 저장
con = sqlite3.connect("c:/python_jar/kospi3.db")

# DataFrame은 DB에 저장할 수 있는 함수 제공 : to_sql()
# Auto Commit이 기본
samsung.to_sql('samsung', con, chunksize=1000)

# DataFrame은 DB를 통해 조회할 수 있는 쿼리도 제공
readed_df = pd.read_sql("SELECT * FROM samsung", con, index_col='Date')
print(readed_df)

'''
기본 설정값
DataFrame.to_sql(name,
                 con,
                 flavor='sqlite',
                 schema=None,
                 if_exists='fail',
                 index=True,
                 index_label=None,
                 chunksize=None,
                 dtype=None)
'''

'''
name : SQL 테이블 이름으로 파이썬 문자열 형태로 나타낸다
con : Cursor 객체
flavor : 사용한 DBMS를 지정할 수 있는데 'sqlite' 또는 'mysql'을 사용할 수 있다. 기본값는 sqlite

schema : Schema를 지정할 수 있는데 기본값은 None이다

if_exists : 데이터베이스에 테이블이 존재할 때 수행 동작을 지정한다. 'fail', 'replace', 'append' 중 하나를 사용할 수 있는데
            기본값은 'fail'이다.
            
            'fail'은 데이터베이스에 테이블이 있다면 아무 동작도 수행하지 않는다.
            
            'replace'는 테이블이 존재하면 기본 테이블을 삭제하고

index : DataFrame의 index를 데이터베이스에 칼럼으로 추가할지에 대한 여부를 지정한다. 기본값은 True이다.

index_label : 인덱스 칼럼에 대한 라벨을 지정할 수 있다. 기본값은 None이다.

chunksize : 한 번에 저장되는 로우의 크기를 정숫값으로 지정할 수 있다. 기본값은 None으로 DataFrame 내의 모든 로우가 한 번에 저장

dtype : 칼럼에 대한 SQL 타입을 파이썬 딕셔너리로 넘겨줄 수 있다.

'''

