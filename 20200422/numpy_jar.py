# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:07:28 2020

@author: USER
"""


# numpy 난수 생성(random 모듈)
# 난수 생성에 활용할 수 있는 numpy의 random모듈(numpy.random)

# 1 - random.rand() : 주어진 형태의 난수 생성
import numpy as np

# 예제
'''
만들어진 난수 array는 주어진 값에 의해 결정되며,
[0, 1] 범위에서 균일한 분포 갖는다
'''

a = np.random.rand(5)
print(a)

b = np.random.rand(2, 3)
print(b)

'''
random 모듈 주요 함수

random.rand() : 주어진 형태의 난수 array 생성
random.randitn() : (최저값, 최대값) 범위에서 임의의 정수
random.randn() : 표준정규분포(standard normal idstribution)를 갖는 난수 반환
random.standard_normal() : randn()과 standard_normal()은 기능이 비슷하지만,
                           standard_normal()은 튜플을 인자로 받는다는 점에서 차이
random.random_sample() : (0.0, 1.0) 범위의 임의의 실수 반환
random.choice() : 주어진 1차원 어레이에서 임의의 샘플을 생성
random.seed() : 난수 생성에 필요한 시드 정함, 코드를 실행할 때마다 똑같은 난수 생성
                난수는 임의의 값으로 매번 값을 정해줘 값이 변하지만, seed는 고정!
'''

# Matplotlib 산점도 그리기
# scatter() 를 이용해서 산점도 (scatter plot)를 그릴 수 있다

import matplotlib.pyplot as plt
import numpy as np

'''
np.random.seed()를 통해서 난수 생성의 시드를 설정하면,
같은 난수를 재사용할 수 있다.

seed() 에 들어갈 파라미터는 0에서 4294967295 사이의 정수여야 한다.
'''

np.random.seed(19680801)

'''
x, y의 위치, 마커의 색(color)과 면적(area)를 무작위로 지정
예를 들어, x는
[0.7003673, 0.74275081, ... , 0.23722644, 0.82394557]로
0에서 1사이의 무작위한 50개의 값을 갖는다
'''

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N)) **2


'''
scatter()에 x, y 위치 입력
s는 마커의 면적을,
c는 마커의 색을 지정
alpha는 마커색의 투명도 결정
'''

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()



