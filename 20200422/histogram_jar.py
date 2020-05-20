# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:40:36 2020

@author: USER
"""

# matplotlib 히스토그램 그리기
# hist() 를 이용해서 히스토그램(histogram)을 그리기

# 1 - 값 입력하기
import matplotlib.pyplot as plt

''' weight 리스트는 몸무게 값을 나타냅니다. '''
weight = [68, 81, 64, 56, 78, 74, 61, 77, 66, 68, 59,
          71, 80, 59, 67, 81, 69, 73, 69, 74, 70, 65]

''' hist() 함수에 리스트의 형태로 값들을 직접 입력해주면 된다. '''
plt.hist(weight)
plt.show()

# 2 - 여러 개의 히스토그램 그리기
import matplotlib.pyplot as plt
import numpy as np

'''
numpy의 np.random.randn()와
np.random.standard_normal(), np.random.rand() 함수를 이용해서
임의의 값들 생성
'''

# array a는 표준편차 2.0, 평균 1.0을 갖는 정규분포
a = 2.0 * np.random.randn(10000) + 1.0

# array b는 표준정규분포를 따른다
b = np.random.standard_normal(10000)

# array c는 -10.0 에서 10.0 사이의 균일한 분포를 갖는 5000개의 임의의 값
c = 20.0 * np.random.rand(5000) - 10.0

'''
세 개의 분포를 동시에 그래프에 나타내기
plt.hist()

bins는 몇개의 영역으로 쪼갤지 설정

density=True로 설정해주면,
밀도함수가 되어서 막대의 아래 면적이 1이 된다

alpha는 투명도를 의미. 0.0에서 1.0 사이의 값 갖는다

histtype을 'step'으로 설정하면 막대 내부가 비어있고
'stepfilled'로 설정하면 막대 내부가 채워진다
'''

plt.hist(a, bins=100, density=True, alpha=0.7, histtype='step')
plt.hist(b, bins=50, density=True, alpha=0.5, histtype='stepfilled')
plt.hist(c, bins=100, density=True, alpha=0.9, histtype='step')

plt.show()


a = np.random.rand(1000)
b = np.random.rand(10000)
c = np.random.rand(100000)

plt.hist(a, bins=100, density=True, alpha=0.5, histtype='step', label='n=1000')
plt.hist(b, bins=100, density=True, alpha=0.75, histtype='step', label='n=10000')
plt.hist(c, bins=100, density=True, alpha=1.0, histtype='step', label='n=100000')

plt.legend()


import numpy as np
import matplotlib.pyplot as plt

''' a는 [0, 10) 범위의 임의의 정수 1000개 '''
a = np.random.randint(0, 10, 1000)

''' b는 [0, 20) 범위의 임의의 정수 1000개 '''
b = np.random.randint(0, 20, 1000)

''' c는 [0, 20) 범위의 임의의 정수 1000개 '''
c = np.random.randint(0, 30, 1000)

plt.hist(a, bins=100, density=False, alpha=0.5, histtype='step', label='0<=randinit<10')
plt.hist(b, bins=100, density=False, alpha=0.75, histtype='step', label='10<=randinit<20')
plt.hist(c, bins=100, density=False, alpha=1.0, histtype='step', label='0<=randinit<20')

plt.legend()
plt.show()




