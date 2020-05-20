# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:31:54 2020

@author: USER
"""

# 1. 붓꽃의 품종 분류
# (1) 데이터 적재
# scikit-learn의 dataset 모듈에 포함되어 있다

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("iris_dataset의 키 : \n{}".format(iris_dataset.keys()))

print(iris_dataset.DESCR)

print(iris_dataset['DESCR'][:200] + "\n...") # 데이터셋 설명 앞부분만


# 예측하려는 붓꽃 품종의 이름을 가지고 있는 key : target_names
format("타깃의 이름 : {}".format(iris_dataset['target_names']))

#특성을 설명하는 문자열 리스트 : feature_names
format("특성의 이름 : {}".format(iris_dataset['feature_names']))

# 실제 데이터(target, data) 중 data는
# 꽃잎의 길이와 폭, 꽃받침의 길이와 폭을 수치값으로 가지고 있는 Numpy 배열
print("data의 타입 : {}".format(type(iris_dataset['data'])))

print("data의 크기 : {}".format(iris_dataset['data'].shape))
# 배열의 행은 개개의 꽃, 열은 각 꽃의 측정치
# 이 배열은 150개의 붓꽃 데이터를 가지고 있으며, 각 붓꽃마다 4개의 측정치를 가지고 있음
# 머신러닝에서 각 아이템은 샘플이라 하고 속성은 특성이라 부름
# 그러므로 data 배열의 크기는 150 x 4 가 됨
# 이는 scikit-learn의 스타일이며 항상 데이터가 이런 구조일 거라 가정하고 있음 

print("data의 처음 다섯 행 : \n{}".format(iris_dataset['data'][:5]))
# 1열 : 꽃받침의 길이
# 2열 : 꽃받침의 폭
# 3열 : 꽃잎의 길이
# 4열 : 꽃잎의 폭

# target 배열 : 샘플 붓꽃의 품종을 담은 Numpy 배열
print("data의 타입 : {}".format(type(iris_dataset['target'])))

print("data의 타입 : {}".format(iris_dataset['target'].shape))

print("타깃 : \n{}".format(iris_dataset['target']))


# (2) 훈련데이터와 텍스트 데이터
# 머신러닝 모델 만들 때 사용하는 훈련데이터와 몯레이 얼마나 잘 작동하는지 측정하는 테스트데이터로 나눈다
# scikit-learn은 데이터셋을 섞어서 나눠주는 train_test_split 함수 제공
# (훈련세트 : 75%, 테스트세트 : 25%)

# scikit-learn에서 데이터는 대문자 X로 표시하고 레이블은 소문자 y로 표기한다
# 이는 수학에서 함수의 입력을 x, 출력을 y로 나타내는 표준공식 f(x)=y에서 유래된 것이다

# 수학의 표기 방식을 따르되 데이터는 2차원 배열(행렬)이므로 대문자 X를, 타깃은 1차원 배열(벡터)이므로 소문자 y를 사용

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                    iris_dataset['target'],
                                                    random_state=0)

# train 데이터와 test 데이터로 나누기 전에 무작위로 섞어주지 않으면
# 순서대로 나누어지기 때문에 y_test(테스트레이블) 값이 모두 2가 나오게 된다
# 세 클래스(품종) 중 하나만 포함한 테스트 세트를 사용하면 모델이 얼마나 잘 일반화 되었는지 알 수 없다
# 테스트 세트는 모든 클래스의 데이터를 포함하도록 잘 섞어야 한다
# random_state=0 은 이 함수를 여러번 실행해도 같은 랜덤값이 리턴된다


# 분리된 값 확인
print("X_train 크기 : {}".format(X_train.shape))
print("y_train 크기 : {}".format(y_train.shape))

print("X_test 크기 : {}".format(X_test.shape))
print("y_test 크기 : {}".format(y_test.shape))

# (3) 데이터 살펴보기
# 머신러닝 모델을 만들기 전에 머신러닝 없이도 풀 수 있는 문제가 아닌지,
# 혹은 필요한 정보가 누락되어 있는지 데이터를 조사해 보는 것이 좋다
# 실제 데이터에는 일관성이 없거나 이상한 값이 들어가 있는 경우가 종종 있다

# 산점도 행렬을 통해 데이터 특성 찾기!!
# 산점도 : 여러 변수로 이루어진 자료에서 두 변수끼리 짝을 지어 작성된 산점도 행렬 형태로 배열

# X_train 데이터를 사용해서 데이터프레임 만든다
iris_dataframe = pd.DataFrame(X_train,
                              columns = iris_dataset.feature_names)

iris_dataframe.head()

pd.plotting.scatter_matrix(iris_dataframe, c=y_train,
                           figsize=(15,15),
                           marker='o', hist_kwds={'bins':20},
                           s=60, alpha=.8)

# 세 클래스가 꽃잎과 꽃받침의 측정값에 따라 비교적 잘 구분되어 있는 것을 볼 수 있다
# 클래스 구분을 위한 머신러닝 기법을 사용하면 잘 구분될 것이다


# (4) K-최근접 이웃(K-nearest neighbors, k-nn) 알고리즘 이용한 머신러닝
# 훈련 데이터를 통해 모델이 만들어지고
# 새로운 데이터가 들어오면 가까운 훈련 데이터 포인트를 찾아 분류한다

# scikit-learn의 모든 머신러닝 모델은
# Estimator 라는 파이썬 클래스로 각각 구현되어 있다
# k-최근접 이웃 분류 알고리즘은 neighbors 모듈 아래 KNeighborsClassifier 클래스에 구현되어 있다

# 모델을 사용하기 위해 클래스로부터 객체를 만들고 paramiter를 설정한다
# 가장 중요한 이웃의 개수를 1로 지정하고 모델을 만들어보자

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

# 훈련 데이터셋으로부터 모델을 만들기 위해 fit 메소드 사용
print(knn.fit(X_train, y_train))


# fit 메소드는 knn 객체 자체를 변환시키면서 반환시킨다

# (5) 예측하기
# 위에서 만든 모델을 사용해서 새 데이터에 대한 예측을 만들 수 있다
# 야생에서 꽃받침 길이는 3cm, 폭은 4.2cm
# 꽃잎의 길이는 0.8cm, 폭은 0.4cm인 붓꽃을 찾았다 가정하고 이 붓꽃의 품종을 찾아보자

# 측정값은 numpy배열로 만드는데
# 하나의 붓꽃 샘플(1)에 4가지 특성(4)이 있으므로 1 by 4 배열을 만들어야 함
# 붓꽃 하나의 측정값은 2차원 numpy 배열에 행으로 들어가므로,
# scikit-learn은 항상 데이터가 2차원 배열일 것으로 예상
X_new = np.array([[3, 4.2, 0.8, 0.4]])
X_new

print("X_new.shape : {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("에측 : {}".format(prediction))

print("예측한 붓꽃의 이름 : {}".format(iris_dataset['target_names'][prediction]))

'''
# 하나의 입력, 특성을 가진 값이 아니기 때문에 아래와 같이 벡터형태로 나타내면 에러 발생한다
X_new2 = np.array([3, 4.2, 0.8, 0.4])
X_new2prediction = knn.predict(X_new2)

# ValueError: Expected 2D array, got 1D array instead:
# array=[3.  4.2 0.8 0.4].
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

# X_new = np.array([[3, 4.2, 0.8, 0.4]]) 이와 같은 형태의 배열이어야 함
'''


# (6) 모델 평가
# 앞에서 만든 테스트 셋을 가지고 현재 만든 학습모델이 잘 만들어졌는지 확인해보기

y_pred = knn.predict(X_test)
# 만들어진 학습모델은 가지고 테스트 데이터의 붓꽃품종을 예측한다

print(y_pred)
# 테스트 데이터의 예측 값

print(y_pred == y_test)
# 예측 품종과 실제 품종이 같으면 True


# 테스트 세트의 정확도
# y_pred = knn.predict(X_test)
print("테스트 세트의 정확도 : {:.4f}%".format(np.mean(y_pred == y_test) * 100))

# knn 객체의 score 메소드 사용
print("테스트 세트의 정확도 : {:.4f}%".format(knn.score(X_test, y_test) * 100))

# sklearn.metrics 의 accuracy_score 사용
from sklearn import metrics

# y_pred = knn.predict(X_test)
print("테스트 세트의 정확도 : {:.4f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))



# (7) k값 변경
accuracy_set = []
k_set = [1,3,5,7,9,11]

for k in k_set:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy_set.append(accuracy)
    
from pprint import pprint
pprint(accuracy_set)

max(accuracy_set)



'''
데이터 스케일링 (Data Scaling)
데이터 스케일링이란 데이터 전처리 과정의 하나입니다.
데이터 스케일링을 해주는 이유는 데이터의 값이 너무 크거나 혹은 작은 경우에 모델 알고리즘 학습과정에서
0 으로 수렴하거나 무한으로 발산해버릴 수 있기 때문입니다.
따라서, scaling 은 데이터 전처리 과정에서 굉장히 중요한 과정입니다.
가볍게 살펴보도록 하겠습니다.


(1) StandardScaler
각 feature 의 평균을 0, 분산을 1 로 변경합니다. 모든 특성들이 같은 스케일을 갖게 됩니다.
(2) RobustScaler
모든 특성들이 같은 크기를 갖는다는 점에서 StandardScaler 와 비슷하지만,
평균과 분산 대신 median 과 quartile 을 사용합니다.
RobustScaler 는 이상치에 영향을 받지 않습니다.
(3) MinMaxScaler
모든 feature 가 0 과 1 사이에 위치하게 만듭니다.
데이터가 2 차원 셋일 경우,
모든 데이터는 x 축의 0 과 1 사이에, y 축의 0 과 1 사이에 위치하게 됩니다.
(4) Normalizer
StandardScaler, RobustScaler, MinMaxScaler 가 각 columns 의 통계치를 이용한다면
Normalizer 는 row 마다 각각 정규화됩니다.
Normalizer 는 유클리드 거리가 1 이 되도록 데이터를 조정합니다.
(유클리드 거리는 두 점 사이의 거리를 계산할 때 쓰는 방법, L2 Distance)

Normalize 를 하게 되면
Spherical contour(구형 윤곽)을 갖게 되는데,
이렇게 하면 좀 더 빠르게 학습할 수 있고 과대적합 확률을 낮출 수 있습니다

2. Code
scikit-learn 에 있는 유방암 데이터셋으로 데이터 스케일링을 해보겠습니다.
데이터를 학습용과 테스트용으로 분할했습니다.
scaler 를 사용하기 이전에 주의 해야될 점을 먼저 살펴보겠습니다.
scaler 는 fit 과 transform 메서드를 지니고 있습니다.
fit 메서드로 데이터 변환을 학습하고,
transform 메서드로 실제 데이터의 스케일을 조정합니다.
이때, fit 메서드는 학습용 데이터에만 적용해야 합니다.
그 후, transform 메서드를 학습용 데이터와 테스트 데이터에 적용합니다.
scaler 는 fit_transform()이란 단축 메서드를 제공합니다.
학습용 데이터에는 fit_transform()메서드를 적용하고,
테스트 데이터에는 transform()메서드를 적용합니다

'''

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    stratify = cancer.target,
                                                    random_state=2019)



# (1) StandardScaler code
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)

print('스케일 조정 전 features MIN value : \n {}'.format(X_train.min(axis=0)))
print('스케일 조정 전 features MAX value : \n {}'.format(X_train.max(axis=0)))
print('스케일 조정 전 features MIN value : \n {}'.format(X_train_scale.min(axis=0)))
print('스케일 조정 전 features MAX value : \n {}'.format(X_train_scale.max(axis=0)))



# (2) RobustScaler code
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scale = scaler.fit_transform(X_train)

print('스케일 조정 전 features MIN value : \n {}'.format(X_train.min(axis=0)))
print('스케일 조정 전 features MAX value : \n {}'.format(X_train.max(axis=0)))
print('스케일 조정 전 features MIN value : \n {}'.format(X_train_scale.min(axis=0)))
print('스케일 조정 전 features MAX value : \n {}'.format(X_train_scale.max(axis=0)))



# (3) MinMaxScaler code
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)

print('스케일 조정 전 features MIN value : \n {}'.format(X_train.min(axis=0)))
print('스케일 조정 전 features MAX value : \n {}'.format(X_train.max(axis=0)))
print('스케일 조정 전 features MIN value : \n {}'.format(X_train_scale.min(axis=0)))
print('스케일 조정 전 features MAX value : \n {}'.format(X_train_scale.max(axis=0)))



# (4) Normalizer code
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
X_train_scale = scaler.fit_transform(X_train)

print('스케일 조정 전 features MIN value : \n {}'.format(X_train.min(axis=0)))
print('스케일 조정 전 features MAX value : \n {}'.format(X_train.max(axis=0)))
print('스케일 조정 전 features MIN value : \n {}'.format(X_train_scale.min(axis=0)))
print('스케일 조정 전 features MAX value : \n {}'.format(X_train_scale.max(axis=0)))


'''
3. 적용해보기
SVC 로 cancer 데이터셋을 학습해보겠습니다.
먼저, 데이터 스케일링을 적용하지 않은 채 진행하겠습니다
'''

from sklearn.svm import SVC
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)

svc = SVC()

svc.fit(X_train, Y_train)

print('test accuracy : %.3f' % svc.score(X_test, Y_test))



# 데이터를 MinMaxScaler 로 스케일을 조정하고 SVC 모델로 학습시키기
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)
svc.fit(X_train_scale, Y_train)

print('Scaled test accuracy : %.3f' % svc.score(X_test_scale, Y_test))
# 성능이 더 좋아짐!





#LinearRegression 학습
from sklearn.linear_model import LinearRegression
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)

lr=LinearRegression()
lr.fit(X_train, Y_train)#학습
print(lr.score(X_test,Y_test)) # 0.7291학습후 모델성능 확인

#LinearSVC 학습
from sklearn.svm import LinearSVC
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)

lr=LinearSVC()
lr.fit(X_train, Y_train)#학습
print(lr.score(X_test,Y_test)) # 0.874 학습후 모델성능 확인

#KNeighborsClassifier 학습
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)

lr=KNeighborsClassifier()
lr.fit(X_train, Y_train)#학습
print(lr.score(X_test,Y_test)) # 0.937학습후 모델성능 확인

#DecisionTreeClassifier 학습
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)

lr=DecisionTreeClassifier()
lr.fit(X_train, Y_train)#학습
print(lr.score(X_test,Y_test))# 0.874 학습후 모델성능 확인

#RandomForestClassifier 학습
from sklearn.ensemble  import RandomForestClassifier
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)

lr=RandomForestClassifier()
lr.fit(X_train, Y_train)#학습
print(lr.score(X_test,Y_test)) # 0.972 학습후 모델성능 확인
