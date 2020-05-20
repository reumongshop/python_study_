# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:23:08 2020

@author: USER
"""

'''
Numpy 라이브러리

- Numpy 란 Numerical Python의 약자로
대규모 다차원 배열과 행렬 연산에 필요한 다양한 함수를 제공한다
데이터 분석할 때 사용되는 다른 라이브러리 pandas와 matplotlib의 기반이 된다
기본적으로 array 라는 단위로 데이터 관리하는데, 행렬 개념으로 생각하면 된다

- Numpy 특징 :
일반 list 에 비해 빠르고 메모리에 효율적
선형대수와 관련된 다양한 기능 제공하고, for문 while문 같은 반복문 없이 데이터 배열에 대한 처리 지원
  
- Numpy 빠른 이유 :
넘파이는 메모리에 차례대로 생성/ 할당 해준다
반면 기존의 list는 이 값(value) 가 어디에 있는지 주소만 저장을 해놓고 그 주소를 알려준다
그래서 list를 for 문 돌리면 그 주소마다 하나씩 하나씩 다 찾아가면서 연산을 해줘야 하는데
numpy는 같은 곳에 몰려 있기 때문에 연산이 더 빠르게 이뤄진다

- Numpy 호출 : "import numpy as np"로 넘파이 호출하는데
모두 np라는 별칭(alias)로 호출하지만 특별한 이유는 없다.

- Numpy로 array 생성하는 방법:
ex) test_array = np.array([1,3,5,7], float)
    type(test_array[3])을 하면 4바이트씩 numpy.float64 라는 값이 반환된다
    float32 같은 데이터 타입은 하나씩 모여서 메모리 블럭을 구성한다
    32bit(비트) = 4byte(바이트)이다. (8bit가 1byte)
    
    
'''

'''
import numpy as np

test_array = np.array([1,3,5,7], float)
print(type(test_array))


'''

'''
1차원 벡터 형식

vector는 일차원의 행렬을 말하고 하나의 행에 열만 있는 것
각 숫자는 value(요소) 라고 부른다


matrix =[[1,2,3,4], [5,6,7,8], [9,10,11,12]]
np.array(matrix, int).shape
# (3, 4)

매트릭스 : 행과 열이 같이 있는 것을 의미



tensor = [ [[1,2,3,4], [5,6,7,8], [9,10,11,12]],
           [[1,2,3,4], [5,6,7,8], [9,10,11,12]],
           [[1,2,3,4], [5,6,7,8], [9,10,11,12]]]   

np.array(tensor, int).shape
#(3, 3, 4)

np.array(tensor, int).ndim # number of dimension
# 3
np.array(tensor, int).size # data의 개수
# 36

tensor 는 매트릭스가 여러개 있는 것으로 3차원, 4차원, 5차원...이 다 표현된다


넘파이 데이터타입 :
    Ndarray의 single element가 가지는 data type
    각 element가 차지하는 memory 크기 결정
    
np.array( [[1, 2.6, 3.2], [4, 5.1, 6]], dtype='int)
# array([[1,2,3], [4,5,6]])

np.array( [[1, 2.6, 3.2], [4, "5", 6]], dtype=np.float32)
# array([[1., 2.6, 3.2], [4., 5., 6. ]].dtype=float32)

각 요소마다 데이터 타입을 지정해주면 그 데이터 타입으로 변환이 되는 걸 볼 수 있다

- nbyte : ndarray object의 메모리 크기 리턴
np.array([[1, 2.6, 3.2], [4, "5", 6]], dtype=np.float32).nbytes
# 24 -> 32bits = 4bytes -> 6 * 4 bytes
np.array([[1, 2.6, 3.2], [4, "5", 6]], dtype=np.float64).nbytes
# 48 -> 64bits = 8bytes -> 6 * 8 bytes
np.array([[1, 2.6, 3.2], [4, "5", 6]], dtype=np.int8).nbytes
# 6 -> 8bits = 1 bytes -> 6 * 1bytes

하나의 value가 4바이트를 가지는데
요소가 6개 있으니까, 이게 메모리에서 차지하는 건 총 24바이트가 된다.
그 다음 타입은 하나가 8바이트이니까 48바이트를 차지한다.


- Array의 shape 크기를 변경(element의 개수는 동일)

t_matrix =[[1,2,3,4], [5,6,7,8]]
np.array(t_matrix).shape
# (2,4)

np.array(t_matrix).reshqpe(8, )
# array([1,2,3,4,5,6,7,8])

np.array(t_matrix).reshape(8, ).shape
# (8, )


- Array의 size만 같다면 다차원으로 자유롭게 변형 가능
np.array(t_matrix).reshape(2, 4).shape
# (2,4)

np.array(t_matirx).reshape(-1, 2).shape
@ (4, 2)  => -1 은 size를 기반으로 row 개수 산정

np.array(t_matrix).reshape(2,2,2)
# array( [[[1,2],
            [3,4],
            
        [[5, 6]],
        [7, 8]]])
    
    
flatten (많이 쓰임!!!★)
- 다차원 array를 1차원 array로 변환
t_matrix=[ [[1,2], [3,4]], [[1,2], [3,4]], [[1,2], [3,4]] ]
np.array(t_matrix).flatten()
# array([1,2,3,4,1,2,3,4,1,2,3,4])

'''
'''
# indexing
import numpy as np

a = np.array([1, 2.2, 3], [4, 5, 6.3], int)
print(a)

print(a[0,0])

a[0, 0] = 7
print(a)

a[0][0] = 8
print(a)

# slicing
list와 달리 행과 열 부분을 나눠서 slicing이 가능
matrix 부분 집합 추출할 때 유용

a = np.array([[1,2,3,4,5], [6,7,8,9,10], int)
a[:, 1:] #전체 row의 1열 이상
a[1, 2:4] # 1 row의 2열~3열
a[1:3] # 1 row~ 로우 전체


- array 의 범위를 지정하여, 값의 list를 생성하는 명령어 : arange
np.arange(20)  # list의 range와 같은 역할, intege로 0부터 19까지 배열 추출
np.arange(0, 1, 0.2) #float 기능
np.arange(20).reshape(4,5)

arange(20)은 총 20개의 value를 vector 형식으로 쭉 뽑게 된다

np.zeros(shape=(5,2), dtype=np.int8) # 5 by 2 zero matrix 생성, int8
np.ones(shape=(5,2), dtype=np.int8) # 5 by 2 one matrix 생성, int8

np.empty(shape=(3,2), dtype=np.int8) # empty : 실행할 때마다 달라질 확률이 높다

empty 는 주어진 shape 대로 비어있는 것을 생성
이런 식으로 array 를 만드는데 메모리에 어느 정도 할당 시켜준다
메모리 기존에 있던 값을 보여준다

zeros나 ones는 0과 1로 메모리 할당 값을 초기화 시켜주는데
empty는 초기화시키지 않고 기존 메모리에 있는 찌꺼기 그대로 보여준다


- 기존 ndarray의 shape 크기만큼 1 or 0 or empty array 반환
t_matrix = np.arange(15).reshape(3,5)
np.ones_lie(t_matrix)

t_matrix1 = np.arange(15).reshape(3, 5)
np.zeros_like(t_matrix1) # 채워져있는 구조를 이용해서 0으로 바꾼 것, 원본 반영되는게 아님


t_matrix2 = np.arange(15).reshape(3,5)
np.empty_like(t_matrix2)



np.identity(n=3, dtype=np.int8)
#array([[1, 0, 0],
#       [0, 1, 0],
#       [0, 0, 1]], dtype=int8)

np.identity(n=5) #정사각형 행렬
#array([[1., 0., 0., 0., 0.],
#       [0., 1., 0., 0., 0.],
#       [0., 0., 1., 0., 0.],
#       [0., 0., 0., 1., 0.],
#       [0., 0., 0., 0., 1.]])

np.eye(N=3, M=4, dtype=np.int) # N값과 M 값을 변경시켜서 직사각형 형태로 만들 수 있다.
#array([[1, 0, 0, 0],
#       [0, 1, 0, 0],
#       [0, 0, 1, 0]])

np.eye(4) # identity행렬과 같게 출력
#array([[1., 0., 0., 0.],
#       [0., 1., 0., 0.],
#       [0., 0., 1., 0.],
#       [0., 0., 0., 1.]])

np.eye(3, 6, k=3) # k --> start index
# 기준 열에서 1을 시작점으로 찍는 옵션(3칸 건너 뛰고 시작한다.)
#array([[0., 0., 0., 1., 0., 0.],
#       [0., 0., 0., 0., 1., 0.],
#       [0., 0., 0., 0., 0., 1.]])


행렬 중 대각선 값만 뽑아내는 함수
t_matrix = np.arange(16).reshape(4,4)
np.diag(t_matrix)
# array([ 0,  5, 10, 15])

np.diag(t_matrix, k=1) # k옵션은 출력하는 열의 시작 위치를 나타낸다.


Random 

axis
: 모든 operation function 실행시 기준이 되는 dimension 축

t_array=np.arange(1,13).reshape(3,4)
t_array
# array([[1,2,3,4],
       [5,6,7,8],
       [9,10,11,12]])
    
t_array.sum(axis=0), t_array.sum(axis=1)
# (array([15,18,21,24]), array([10,26,42]))
   

tensor = np.array([t_array, t_array, t_array])
tensor

# cnn 에서 많이 쓰임(신경망 모델)

표준편차 보여주는 함수
t_array=np.arange(1,13).reshape(3,4)
t_array

t_array.mean(), t_array.mean(axis=0)
t_array.std(), t_array.std(axis=0)



딥러닝에서 자주 볼 수 있는 함수
np.exp(t_array) #지수함수

np.sqrt(t_array) #루트

np.sin(t_array) #sin함수


- numpy array를 합치는 함수
a = np.array([1,2,3])
b = np.array([4,5,6])
np.vstack((a,b))
#array([[1, 2, 3],
       [4, 5, 6]])

a = np.array([ [1], [2], [3]])
b = np.array([ [4], [5], [6]])
np.hstack((a,b))
# array([[1, 4],
#        [2, 5],
#        [3, 6]])


concatenate
- numpy array 를 합치는 함수
a=np.array([1,2,3])
b=np.array([4,5,6])
np.concatenate((a,b), axis=0)

a=np.array([[1,2],[3,4]])
b=np.array([[5,6]])
np.concatenate((a,b.T), axis=1)
# b.T는 b의 역행렬

vstack이랑 hstack 똑같은 함수인데 axis로 결정된다


Operations between arrays
numpy는 array간 기본적인 사칙연산 지원
a=np.array([[1,2,3], [4,5,6]], float)

a-a #matrix-matrix 연산

a*a #matrix 내 요소들 간 같은 위치에 있는 값들끼리 연산


같은 index에 있는 것끼리 더하고 빼고 곱해줘서
그 자리에 
= element-wise operation 이라고 한다

dot_a = np.arrange(1, 7).reshape(2,3)
dot_b = np.arrange(1, 7).reshape(3,2)
dot_a.dot(dot_b)

t_matrix = np.array([[1,2], [3,4]], float)
scalar = 2
t_matrix + scalar #matrix, scalar 덧셈


scalar-matrix 외에도,
vector-matrix 간의 연산도 지원

t_matrix = np.arange(1, 13).reshape(3,4)
t_vector = np.arange(100, 400, 100)
t_matrix + t_vector


All, Any
All : Array의 데이터가 전부 조건에 만족하면 True
Any : Array의 데이터 중 하나라도 조건에 만족하면 True
a = np.arange(5)
a
# array([0, 1, 2, 3, 4])

np.all(a>3)
# False

np.all(a<5)
# True

np.any(a>3)
# True

np.any(a>5)
# True

all은 말 그대로 모든 조건 만족하면 True가 나오고
any는 하나라도 만족하면 True 추출해내는 함수


a=np.array([1,5,3], float)
b=np.array([4,7,2], float)
a>b
# array([False, False, True])

a == b
# array([False, False, True])

(a>b).any()
# True

(a>b).all()
# False

a=np.array([2, 3, 1], float)
np.logical_and(a>0, a<3) #and조건의 비교
# array([True, False, True])

b=np.array([False, True, True], bool)
np.logical_not(b) # not 조건 비교
# array([True, False, False])

c=np.array([False, False, False], bool)
np.logical_or(b,c) # or 조건의 비교

logical_and란 함수의 2가지 조건을 넣을 수 있다.


np.where
where(조건, True, False)

a = np.array([2,3,1], float)
np.where(a>1,0,3)
# array([0, 0, 3])

a=np.arange(3, 10)
np.where(a>6) # True값의 index값 반환
# (array([4,5,6], dtype=int64),)

a=np.array([2, np.NaN, np.Inf], float)
np.isnan(a) # null인 경우 True
# array([False, True, False])

np.isfinite(a) # 한정된 수의 경우 True
# array([True, False, False])

np.where은 우리가 생각하는 if문의 역할을 한다
조건이 있으면 결과값으로 어떤 걸 리턴하고 아니면 else 써서
다른 ~~...(설명 잘림)

isnan은 null값인 경우에만 True가 나온다
np.Nan은 numpy의 null 값을 입력하는 함수이고,
null값이니까 True

np.Inf 는 무한대

np.isinfite()는 한정된 수의 경우 True가 나오고
한정되지 않은 NaN이나 Inf의 경우에는 False가 나온다

argmax, argmin
array내 최대값 또는 최소값의 index 반환
a=np.array([2,3,1,5,6,22,11])
np.argmax(a), np.argmin(a)
# (5,2)

axis 기반의 반환
a=np.array([[1,4,2,22], [45,32,4,7], [34,54,9,8]])
np.argmax(a, axis=0), np.argmin(a, axis=1)
# (array([1,2,2,0], dtype=int64), array([0,2,3], dtype=int64))

boolean index
# -numpy의 배열은 특정 조건에 따른 값을 배열 형태로 추출 가능
# -comparison operation 함수들도 모두 사용 가능
t_a = np.array( [3,5,8,0,7,4], float)
t_a > 4
#array([False,  True,  True, False,  True, False])

t_a[t_a>4] # 조건이 True인 index 의 요소값만 추출
#array([5., 8., 7.])

t_c = t_a <4
t_c
# array([ True, False, False,  True, False, False])

'''







