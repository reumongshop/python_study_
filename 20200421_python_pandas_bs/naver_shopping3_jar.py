# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:37:03 2020

@author: USER
"""


# 네이버 쇼 썸네일 가져오기

# 제목과 썸네일이 같이 존재하는 영역

import errno # 예외처리 시 필요
from bs4 import BeautifulSoup 
import requests, re, os # os 운영체제 다룰 때
from urllib.request import urlretrieve # urlretrieve : 파일 다운로드 받을 때

# 저장폴더 생성
'''
여기서는 os모듈을 참조
os.path.isdir : 이미 디렉토리가 있는지 검사
os.path.join : 현재 경로를 계산하여 입력으로 들어온 텍스트를
               합하여 새로운 경로 만듦
os.makedirs : 입력으로 들어온 경로로 폴더 생성

모듈 참조와 아래 urlretrieve부분도 변
'''
try : 
    if not (os.path.isdir('image')):
        os.makedirs(os.path.join('image'))
        print("image 폴더 생성 성공!")
except OSError as e:
    if e.errono != errno.EEXIST:
        print("폴더 생성 실패")
        exit()
        
# 웹 페이지를 열고 소스 코드 읽어오는 작업
html = requests.get("https://comic.naver.com/webtoon/weekday.nhn")
soup = BeautifulSoup(html.text, "html.parser")
html.close()

# 요일별 웹툰영역 추출하기
data1_list = soup.findAll('div', {'class' : 'col_inner'})
print(data1_list)
print(type(data1_list))

# 전체 웹툰 리스트
'''
요일별 웹툰 영역 중 제목과 썸네일 영역을 하나의 리스트로
'''

li_list = []

for data1 in data1_list:
    # 제목 + 썸네일 영역 추출
    # 해당 부분을 찾아 li_list 와 병합
    li_list.extend(data1.findAll('li'))

print(li_list)

# 각각의 요소 중 <img> 태그의 제목과 썸네일(~.jpg)만 추출하기
# 다운로드하기
'''
이미지 또는 동영상 링크가 있다면 다운로드하는 방법은 쉽다.
from urllib.request import urlretrieve 를 추가한 뒤,
urlretrieve 호출 시에 링크와 저장할 파일명을 넣으면 된다.
'''

# 특수문자 처리
'''
도중에 에러가 난 부분을 보면 파일명에 특수문자가 있는 경우,
따라서 추출한 제목에서 특수문자는 다른 문자로 변경해주거나 삭제

변경은 replace를 하면 되는데,
여기서는 정규식 표현을 이용한 re모듈을 사용하여 삭제
따라서 re모듈을 import

'''

for li in li_list:
    img = li.find('img')
    title = img['title']
    img_src = img['src']
    # print(title, img_src)
    # 해당 영역의 글자가 아닌 것은 ''로 치환시킨다
    
    title = re.sub('[^0-9a-zA-Zㄱ-힗]', '', title) # 제목에 해당 정규표현식에 포함되는 내용이 ^ 아닌 것!
    
    # 주소, 파일경로+파일명+확장자
    urlretrieve(img_src, './image/' + title + '.jpg')
    



