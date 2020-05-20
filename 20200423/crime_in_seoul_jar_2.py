# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:38:59 2020

@author: USER
"""

################################
# 범죄 데이터 구별로 정리하기

import pandas as pd
import numpy as np

from crime_in_seoul_jar import station_lng, station_lat


crime_anal_raw = pd.read_csv('c:/python_data/02_crime_in_Seoul_include_gu_name.csv', encoding='utf-8')
crime_anal_raw.head()

### pandas의 pivot_table

'''
pd.pivot_table(df,             # 피벗 테이블 만들기 위한 기본 데이터
               index = [],     # pivot_table의 index를 설정(multi index도 가능)
               columns = [],   # 원하는 columns을 설정
               values = [],    # column에 해당하는 값
               aggfunc = [],   # 분석을 위한 파라미터 (예: np.sum, np.mean 사용)
               fill_value = 0, # Nan 값을 채우기
               margins = True) # 모든 데이터의 결과를 아래에 붙일 것인지 결정
'''

# pivot_table 이용
# 저장한 데이터를 관서별에서 구별로
crime_anal = pd.pivot_table(crime_anal_raw, index='구별', aggfunc=np.sum)
crime_anal.head()

print(crime_anal)

'''
각 범죄별 검거율을 계산하고,
검거 건수는 검거율로 대체한 후, 검거 건수는 삭제
'''

crime_anal['강간검거율'] = crime_anal['강간 검거'] / crime_anal['강간 발생'] * 100
crime_anal['강도검거율'] = crime_anal['강도 검거'] / crime_anal['강도 발생'] * 100
crime_anal['살인검거율'] = crime_anal['살인 검거'] / crime_anal['살인 발생'] * 100
crime_anal['절도검거율'] = crime_anal['절도 검거'] / crime_anal['절도 발생'] * 100
crime_anal['폭력검거율'] = crime_anal['폭력 검거'] / crime_anal['폭력 발생'] * 100

del crime_anal['강간 검거']
del crime_anal['강도 검거']
del crime_anal['살인 검거']
del crime_anal['절도 검거']
del crime_anal['폭력 검거']

print(crime_anal.head())

'''
100이 넘는 숫자들은 100으로 처리
'''

con_list = ['강간검거율', '강도검거율', '살인검거율', '절도검거율', '폭력검거율']

for column in con_list:
    crime_anal.loc[crime_anal[column] > 100, column] = 100
    
print(crime_anal.head())

# 컬럼 뒤에 발생이라는 단어 삭제 : rename()를 사용
crime_anal.rename(columns = {'강간 발생' : '강간',
                             '강도 발생' : '강도',
                             '살인 발생' : '살인',
                             '절도 발생' : '절도',
                             '폭력 발생' : '폭력'}, inplace=True)
    
print(crime_anal.head())

###############################
# 데이터 표현을 위해 전처리
'''
강도와 살인은 두 자리수, 절도와 폭력은 네 자리수로 구성 되어 있어
각각 비슷한 범위에 놓고 비교하는 것이 편리하기 때문에
각 컬럼별로 정규화(NORMALIZE) 작업

각 항목의 최대값을 1로 두면,
추후 범죄발생 건수를 종합적으로 비교할 때 편리

강간, 강도, 살인, 절도, 폭력에 대해
각 컬럼별로 정규화(NORMALIZE)
'''

'''
파이썬의 머신러닝에 관한 모듈 중 scikit learn에 있는 전처리(preprocessing) 도구에는
최소, 최대값을 이용하여 정규화 시키는 함수가 존재 : MinMaxScaler()
'''

from sklearn import preprocessing

col = ['강간', '강도', '살인', '절도', '폭력']

x = crime_anal[col].values
min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x.astype(float))

crime_anal_norm = pd.DataFrame(x_scaled, columns = col, index = crime_anal.index)

# 정규화된 데이터프레임에 검거율 추가
col2 = ['강간검거율', '강도검거율', '살인검거율', '절도검거율', '폭력검거율']

crime_anal_norm[col2] = crime_anal[col2]

print(crime_anal_norm.head())

# CCTV_result.csv 에서 구별 인구수와 CCTV 개수만 추가
result_CCTV = pd.read_csv('c:/python_data/01. CCTV_result.csv', encoding='utf-8', index_col='구별')

crime_anal_norm[['인구수', 'CCTV']] = result_CCTV[['인구수', '소계']]
print("인구수와 CCTV 개수 => ", crime_anal_norm.head())

# 발생 건수의 합을 '범죄'라는 컬럼으로 합하여 추가
col = ['강간', '강도', '살인', '절도', '폭력']

crime_anal_norm['범죄'] = np.sum(crime_anal_norm[col], axis=1)
print("범죄라는 컬럼으로 합 => ", crime_anal_norm.head())

# 검거율도 통합하여 추가
col = ['강간검거율', '강도검거율', '살인검거율', '절도검거율', '폭력검거율']
crime_anal_norm['검거'] = np.sum(crime_anal_norm[col], axis=1)
print("검거율도 통합 => ", crime_anal_norm.head())


###############################################################
import matplotlib.pyplot as plt
import seaborn as sns # 다수의 상관관계 표현하기 위한 모듈
import platform # 한글 폰트 설정하기 위한 모듈

# 한글 폰트 설정
path = "c:/Windows/Fonts/malgun.ttf"

# 폰트설정 (특히 한글 부분)
from matplotlib import font_manager, rc

plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... SORRY ~_~')
    
# pairplot() 상관관계 : "강도", "살인", "폭력"
sns.pairplot(crime_anal_norm, vars=["강도", "살인", "폭력"], kind='reg', size=3)
plt.show()
'''
강도와 폭력, 살인과 폭력, 강도와 살인 모두 양의 상관관계 보임
'''

# pairplot() 상관관계 : "인구수", "CCTV", "살인", "강도"
sns.pairplot(crime_anal_norm, x_vars=["인구수", "CCTV"],
             y_vars=["살인", "강도"], kind='reg', size=3)
plt.show()
'''
전체적인 상관계수는 CCTV와 살인의 관계가 낮을지 몰라도
CCTV가 없을 때 살인사건 많은 구간 있음
즉, CCTV수를 기준으로 좌측면에 살인과 강도의 높은 수를 갖는 데이터가 보임
'''

# pairplot() 상관관계 : "인구수", "CCTV", "살인검거율", "폭력검거율"
sns.pairplot(crime_anal_norm, x_vars=["인구수", "CCTV"],
             y_vars=["살인검거율", "폭력검거율"], kind='reg', size=3)
plt.show()

'''
살인 및 폭력 검거율과 CCCTV의 관계가 음의 상관계수도 보여줌
인구수와 살인 및 폭력 검거율도 음의 상관관계 보임
'''

# pairplot() 상관관게 : "인구수", "CCTV", "절도검거율", "강도검거율"
sns.pairplot(crime_anal_norm, x_vars=["인구수", "CCTV"],
             y_vars=["절도검거율", "강도검거율"], kind='reg', size=3)
plt.show()

# 검거율의 합계인 검거 항목 최고값을 100으로 한정한 후, 그 값으로 정렬
tmp_max = crime_anal_norm['검거'].max()

crime_anal_norm['검거'] = crime_anal_norm['검거'] / tmp_max *100

crime_anal_norm_sort = crime_anal_norm.sort_values(by='검거', ascending=False)

print(crime_anal_norm_sort.head())

# heatmap 으로 시각화
target_col = ['강간검거율', '강도검거율', '살인검거율', '절도검거율', '폭력검거율']
crime_anal_norm_sort = crime_anal_norm.sort_values(by='검거', ascending=False)

plt.figure(figsize = (10,10))

sns.heatmap(crime_anal_norm_sort[target_col],
            annot=True, fmt='f',
            linewidths=.5, # linewidth는 칸 간격 의미
            cmap='RdPu')

plt.title('범죄 검거 비율(정규화된 검거의 합으로 정렬)')
plt.show()

# 발생 건수 정렬하여 heatmap으로 시각화
target_col = ['강간', '강도', '살인', '절도', '폭력', '범죄']

crime_anal_norm['범죄'] = crime_anal_norm['범죄'] / 5

crime_anal_norm_sort = crime_anal_norm.sort_values(by='범죄', ascending=False)

plt.figure(figsize = (10, 10))

sns.heatmap(crime_anal_norm_sort[target_col],
            annot=True, fmt='f',
            linewidths=.5,
            cmap='RdPu')

plt.title('범죄비율(정규화된 발생 건수로 정렬)')
plt.show()

# 여기까지 작업 저장
crime_anal_norm.to_csv('c:/python_data/02_crime_in_Seoul_final.csv', sep=',', encoding='utf-8')





import json
geo_path = 'c:/python_data/02. skorea_municipalities_geo_simple.json'
geo_str = json.load(open(geo_path, encoding='utf-8'))


crime_anal_raw['lat'] = station_lat
crime_anal_raw['lng'] = station_lng

col = ['살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거']
tmp = crime_anal_raw[col] / crime_anal_raw[col].max()

crime_anal_raw['검거'] = np.sum(tmp, axis=1)

crime_anal_raw.head()


import folium
import webbrowser

########################
map = folium.Map(location=[37.5502, 126.982], zoom_start=11)

for n in crime_anal_raw.index:
    folium.Marker([crime_anal_raw['lat'][n],
                   crime_anal_raw['lng'][n]]).add_to(map)

map
map.save('folium_kr.html')
webbrowser.open_new("folium_kr.html")


########################
map = folium.Map(location=[37.5502, 126.982], zoom_start=11)

for n in crime_anal_raw.index:
    folium.CircleMarker([crime_anal_raw['lat'][n], crime_anal_raw['lng'][n]],
                        radius=crime_anal_raw['검거'][n] * 10,
                        color='#3186cc',
                        fill_color='#3186cc',
                        fill=True).add_to(map)
    
map
map.save('folium_kr2.html')
webbrowser.open_new("folium_kr2.html")


########################
map = folium.Map(location=[37.5502, 126.982], zoom_start=11)

map.choropleth(geo_data = geo_str,
               data = crime_anal_norm['범죄'],
               columns = [crime_anal_norm.index, crime_anal_norm['범죄']],
               fill_color='PuRd', # PuRd, YlGnBu
               key_on = 'feature.id')

for n in crime_anal_raw.index:
    folium.CircleMarker([crime_anal_raw['lat'][n], crime_anal_raw['lng'][n]],
                        radius = crime_anal_raw['검거'][n] * 10,
                        color = '#3186cc',
                        fill_color='#3186cc',
                        fill=True).add_to(map)
    
map.save('folium_kr3.html')
webbrowser.open_new("folium_kr3.html")


