# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:28:33 2020

@author: USER
"""

import sqlite3

print(sqlite3.version)
print(sqlite3.sqlite_version)

con = sqlite3.connect("c:/python_jar/kospi.db")
# kospi.db 파일은 작업 파일에 생성해야함
# 데이터베이스 파일명 확장자는 무조건 .db !!
# => 다른 데이터베이스에 이식 가능

print(type(con))

cursor = con.cursor() # 정적 커서 생성 / 접속으로부터 새 커서 개체를 반환
# cursor = con.cursor(True) # 부울 인수를 사용하여 동적 커서 생
# 자바에서 PreparedStatement 와 비슷!

# cursor.execute("CREATE TABLE kakao(Date text, Open int, High int, Low int, Closing int, Volumn int)")
                                # 한줄로 기입, 줄바꿈하면 오류발생!!
                                # 변수명, 데이터타입 / Date 라는 변수는 텍스트 타입으로 변수 선언 / close 라는 내장 함수가 있어, closing  이라고 사용
cursor.execute("INSERT INTO kakao VALUES('16.06.03', 97000, 98600, 96900, 98000, 321405)")

cursor.execute("SELECT * FROM kakao")
# print(cursor.fetchone()) # 한번에 한줄만 가져오는 함수
# print(cursor.fetchall()) # 한번에 여러줄 가져오는 함수

kakao = cursor.fetchall()
print(type(kakao))
print(kakao[0][0])
print(kakao[0][1])
print(kakao[0][2])
print(kakao[0][3])

con.commit()
con.close()
