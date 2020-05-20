# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:38:07 2020

@author: USER
"""

# Matplotlib 기본 사용
'''
Matplotlib 라이브러리를 이용해서 그래프를 그리는 일반적인 방법
'''

# Pyplot 소개
'''
matplotlib.pyplot은
Matplotlib을 MATLAB과 비슷하게 동작하도록 하는 명령어 스타일의 함수의 모음

각각의 pyplot 함수를 사용해서 그림(figure)에 변화를 줄 수 있다

예를 들어,
그림을 만들어서 플롯 영역을 만들고,
몇 개의 라인을 플롯하고,
라벨(label)들로 꾸미는 등의 일을 할 수 있다.

'''

# 기본 그래프
'''
pyplot 으로 어떤 값들을 시각화하는 것은 매우 간단
pyplot.plot()에 하나의 리스트를 입력함으로써 그래프가 그려진다

matplotlib은
리스트의 값들이 y값들이라고 가정하고,
x 값들 ([0,1,2,3])을 자동으로 만들어낸다
'''

import matplotlib.pyplot as plt

plt.plot([1,2,3,4])
plt.ylabel('y-label')
plt.show()

'''
plot()은 다재다능한(versatile)한 명령어여서,
임의의 개수의 인자를 받을 수 있다.

예를 들어, 아래와 같이 입력하면, x와 y값을 그래프로 나타낼 수 있다.
'''

plt.plot([1,2,3,4], [1,4,9,16])

# 스타일 지정하기
'''
x, y값 인자에 대해
색상ㅇ과 선의 형태를 지정하는 포맷 문자열을 세번째 인자에 입력할 수 있다.
디폴트 포매 문자열은 'b-'인데 파란색(blue)의 선(line, '-')을 의미
아래의 'ro'는 빨간색(red)의 원형(circle, 'o') 마커를 의미
'''

import matplotlib.pyplot as plt

plt.plot([1,2,3,4], [1,4,9,16], 'ro')

# axis()를 이용해서 축의 [xmin, xmax, ymin, ymax] 범위 지정
plt.axis([0, 6, 0, 20])
plt.show()

# 여러 개의 그래프 그리기
'''
matplotlib에서 리스트만 가지고 작업하는 것은 제한적이기 때문에,
일반적으로 numpy 어레이 이용

사실, 모든 시퀀스는 내부적으로 numpy 어레이로 변환된다
'''

# 다양한 포맷 스타일의 여러 개의 라인을 하나의 그래프로 그리기
import matplotlib.pyplot as plt
import numpy as np

# 200ms 간격으로 균일한 샘플된 시간
t = np.arange(0., 5., 0.2)

# 빨간 대쉬, 파란 사각형, 녹색 삼각형
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

# matplotlib 라벨 설정
'''
xlabel(), ylabe() 함수를 사용해서
그래프의 x, y 축에 대한 라벨을 설정할 수 있다.
'''

import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16])

'''
xlabel()과 ylabel()에 텍스트를 입력해주면,
각각의 축에 라벨이 나타난다
'''

plt.xlabel('X-Label')
plt.ylabel('Y-Label')
plt.axis([0, 5, 0, 20])
plt.show()

'''
axis()에 [x_min, x_max, y_min, y_max] 의 형태로
x, y축 범위를 지정

입력 리스트는 꼭 네개의 값(x_min, x_max, y_min, y_max)이 있어야 한다

입력값이 없으면
데이터에 맞게 자동(Auto)으로 범위를 지정
'''

# Matplotlib 색깔 지정
# 자주 사용하는 색깔 외에도 다양한 색상 지정할 수 있다

import matplotlib.pyplot as plt
'''
plot()에 color='springgreen'과 같이 입력해주면,
springgreen에 해당하는 색깔이 지정된다
'''

plt.plot([1,2,3,4], [1,4,9,16], color='springgreen')
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
plt.axis([0, 5, 0, 20])
plt.show()

# Matplotlib 색깔 지정하기2
'''
16진수 코드(hex code)로도 색깔 지정할 수 있다
색깔, 마커와 선의 종류까지 모두 지정
'''

import matplotlib.pyplot as plt
'''
색깔은 '#e35f62'와 같이 16진수로, 마커는 circle,
선 종류는 대쉬(dashed)로 지정
'''

plt.plot([1,2,3,4], [1,4,9,16], color='#e35f62',
         marker='o', linestyle='--')
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
plt.axis([0, 5, 0, 20])
plt.show()

# matplotlib 여러 곡선 그리기
# 세개의 곡선을 하나의 그래프에 그리기
import matplotlib.pyplot as plt
import numpy as np

'''
numpy를 사용해서 array 생성
numpy.arange()
주어진 간격에 따라 균일한 array 생성
'''

a = np.arange(5) # 0~4 # 마지막 수는 X
b = np.arange(1, 5) # 1~4
c = np.arange(2, 10, 2) #2~8 / 간격:2

print(a)
print(b)
print(c)

a = np.arange(0, 2, 0.2) # 0~2미만 까지 0.2 간격으로 출력

'''
plot()에 x값, y값, 스타일을 순서대로 세 번씩 입력하면,
세 개의 곡선(y=x, y=x^2, y=x^3)이 동시에 그려진다
'''

plt.plot(a, a, 'r--',
         a, a**2, 'bo',
         a, a**3, 'g-.')

'''
'r-'은 빨간색(red)의 대쉬(dashed) 스타일의 선,
'bo'는 파란색(blue)의 circle 마커,
'g-.'은 녹색(green)의 대쉬-닷(dash-dot) 스타일 선 의미
'''
plt.show()

# 세 개의 곡선의 세세한 스타일을 설정할 수 있다
import matplotlib.pyplot as plt
import numpy as np

a = np.arange(0, 2, 0.2)

# 첫번째 곡선의 스타일은 'bo'로
plt.plot(a, a, 'bo')

# 두번째 곡선은 color='#e35f62', marker='*', linewidth=2 로
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)

# 세번째 곡선은 color='springgreen', marker='^', markersize=9 로
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)

plt.show()

# matplotlib 그리드와 틱 설정
'''
gird()와 tick_params()를 이용해서 그래프의 그리드와 틱 스타일을 설정할 수 있다.
'''

import matplotlib.pyplot as plt
import numpy as np

a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)

'''
그리드 함수 사용 방법!
그리드가 표시되도록 하려면,
gride()의 첫번째 파라미터를 True로 설정

axis='y'로 설정하면 y축 그리드만 표시
alpha는 투명도 설정
0으로 설정하면 투명하게, 1은 불투명하게 표시

linestyle을 대쉬(dashed)로 설정
'''

plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')


'''
tick_params()를 이용해서 그래프의 틱(tick)에 관련된 설정을 할 수 있다
axis='both'로 설정하면 x, y축의 틱에 모두 적용
direction='in'으로 틱의 방향을 그래프 안쪽으로 설정

틱의 길이(length)를 3만큼으로 하고,
틱과 라벨의 거리(pad)를 6만큼
틱 라벨의 크기(labelsize)를 14로 설정
'''

plt.tick_params(axis='both', direction='in', length=3, pad=65, labelsize=14)
plt.show()

# matplotlib 타이틀 설정하기
# title()을 이용해서 그래프의 제목(타이틀) 설정
'''
plt.title()을 이용해서 그래프의 타이틀을 'Sample graph'로 설정
'''
plt.title('Sample graph')
plt.show()

# 2 - 위치와 오프셋
plt.title('Sample graph', loc='right', pad=20)

'''
loc='right'로 설정하면,
타이틀이 그래프의 오른쪽 위에 나타나게 된다

'left', 'center', 'right'로 설정할 수 있으며 default값은 'center'

pad=20은
타이틀과 그래프와의 간격(오프셋)을 포인트 단위로 설정
'''

# 3 - 폰트 설정
'''
fontdict에 딕셔너리 형태로 폰트에 대한 설정을 입력할 수 있다.
'fontsize'를 16으로, 'fontweight'를 'bold'로 설정

'fontsize'는 포인트 단위의 숫자를 입력하거나,
'smaller', 'x-large' 등의 상대적인 설정을 할 수 있다.

'fontweight'에는 'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'의 설정을 할 수 있다.
'''

title_font = {
        'fontsize' : 16,
        'fontweight' : 'bold'
        }

plt.title('Sample graph', fontdict = title_font, loc='left', pad=20)

#Matplotlib 막대 그래프 그리기
'''
bar() 함수를 이용해서 막대 그래프(bar graph)를 그릴 수 있다
연도별 값을 갖는 데이터를 막대 그래프로 플롯
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(3)

# years는 x축에 표시될 연도이고, values는 y 값
years = ['2017', '2018', '2019']
values = [100, 400, 900]

# 먼저 bar() 함수에 x(=[0, 1, 2])와 값(=[100, 400, 900])을 입력
plt.bar(x, values)

'''
xticks()에 x 와 years 를 입력해주면
x축에 '2017', '2018', '2019' 가 순서대로 표시
'''

plt.xticks(x, years)
plt.show()

'''
막대 그래프에도 막대와 테두리의 색, 두께 등 다양한 스타일 적용할 수 있다.
우선 bar() 함수에 x, y(=values) 값 입력
'''
plt.bar(x, values,
        width=0.6,
        align='edge',
        color="springgreen",
        edgecolor="gray",
        linewidth=3,
        tick_label=years,
        log=True)

plt.show()

'''
barh()를 이용하면 수평 막대 그래프를 그릴 수 있다.
'''
y = np.arange(3)
plt.barh(y,values,
         height=-0.6,
         color='springgreen',
         edgecolor='gray',
         linewidth=3,
         tick_label=years,
         log=False)
plt.show()
