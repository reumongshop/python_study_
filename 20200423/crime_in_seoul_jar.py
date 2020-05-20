# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:33:29 2020

@author: USER
"""

##############################################################################
# 강남 3구는 안전한가?
##############################################################################

'''
강남 3구의 주민들이 자신들이 거주하는 구의 체감 안전도를 높게 생각한다는 기사를 확인해보기
기사 원문 http://news1.kr/articles/?1911504

Matplotlib의 heatmap 등을 그릴 때
cmap의 디폴트 설정이 변경되어 heatmap 등에서 cmap을 적용할 때
옵션을 잡아주어야 동일한 효과가 나타난다

Folium이 0.4.0 으로 버전업 되면서
choropleth 명령에서 geo_str 옵션명이 geo_data 옵션명으로 변경됨
circle marker 적용할 때, fill=True 옵션을 반드시 사용해야함
'''

##############################################################################
# 데이터 정리하기
# 필요한 모듈을 import

import numpy as np
import pandas as pd

'''
다운받은 데이터(csv) 파일을 읽는다
콤마(,)로 천단위가 구분되어 있고, 한글 인코딩은 euc-kr
'''

crime_anal_police = pd.read_csv('c:/python_data/02. crime_in_Seoul.csv', thousands=',', encoding='euc-kr')

r = crime_anal_police
print(r)

'''
관서별로 되어 있는 데이터를 소속 구별로 변경
1. 경찰서 이름으로 구 정보 얻기
'''

# 구글 맵스 사용해서 경찰서 위치(위도, 경도) 정보를 받아온다
import googlemaps

# 자신의 key 사용
gmaps_key = "AIzaSyBiV7Q9ZswJ9QmsBOE6oZ2ls1iGIavXXSo"
gmaps = googlemaps.Client(key=gmaps_key)
r = gmaps.geocode('서울중부경찰서', language='ko')
print(r)

# 중부서, 수서서 => 서울** 경찰서로 변경
station_name=[]
for name in crime_anal_police['관서명']:
    station_name.append('서울' + str(name[:-1]) + '경찰서')
    
print(station_name)

# 경찰서 이름을 이용하여 주소 얻기
station_address = []
station_lat = []
station_lng = []

for name in station_name:
    tmp = gmaps.geocode(name, language='ko')
    station_address.append(tmp[0].get("formatted_address"))
    
    tmp_loc = tmp[0].get("geometry")
    
    station_lat.append(tmp_loc['location']['lat'])
    station_lng.append(tmp_loc['location']['lng'])
    
    print(name + '-->' + tmp[0].get("formatted_address"))
    
print(station_address)


'''
저장한 주소를 띄어쓰기, 공백으로 나누고,
구별이라는 컬럼으로 저장
'''

gu_name = []
for name in station_address:
    tmp = name.split()
    
    tmp_gu = [gu for gu in tmp if gu[-1] == '구'][0]
    
    gu_name.append(tmp_gu)
    
crime_anal_police['구별'] = gu_name
print(crime_anal_police.head())



'''
금천경찰서인 경우 관악구로 되어 있어서
금천구로 변경
'''

crime_anal_police[crime_anal_police['관서명'] == '금천서']
crime_anal_police.loc[crime_anal_police['관서명'] == '금천서',
                      ['구별']] = '금천구'
                      
crime_anal_police[crime_anal_police['관서명'] == '금천서']

# 현재까지 작업 내용 저장
crime_anal_police.to_csv('c:/python_data/02_crime_in_Seoul_include_gu_name.csv', sep=',', encoding='utf-8')

