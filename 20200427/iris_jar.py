# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:46:02 2020

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

print(iris)
print(iris.keys())
print(iris.target_names) # 품종
# setosa(부채붓꽃), versicolor(흰여로), virginica(질경이)
print(iris.feature_names) # 특성
# sepal length(꽃받침 길이)
# sepal width(꽃받침 너비)
# petal length(꽃잎 길이)
# petal width(꽃잎 너비)

print(iris.data[:10])

# 품종의 값 확인 (label)
print(iris.target)

'''
머신러닝 모델을 만들 때 데이터를 인풋으로 넣고 학습시키는 것도 중요하지만,
그렇게 학습시킨 모델의 성능을 평가하는 것도 중요

모델의 성능을 평가하지 않으면,
모델이 과적합(overfittin)되었는지 일반화 되지 않았는지 판단할 수 없다

가령, 데이터를 전부 학습하는데 사용한다면 모델의 정확도는 높을 수 있으나
새로운 데이터가 주어졌을 때 예측도가 현저히 떠어지는, 과적합 발생할 수 있다
(즉, 일반화에 실패할 수 있다)

머신러닝의 목적은 성능을 높이고 일반화하는데 있다
이를 위해 주어진 데이터를 학습데이터(training data)와
테스트 데이터(test data, hold-out set)으로 나누어야 한다

scikit-learn에서는
데이터셋을 훈련셋과 테스트셋으로 분할하는데 용이한 train_test_split 이란 함수를 제공한다
'''

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.3,
                                                    random_state=2019)


# X_train 에는 전체 데이터 셋의 70%를, X_test에는 나머지 30% 데이터를 가진다
# 원래 머신러닝 모델에 데이터를 학습시킬 때 feature engineering 을 거쳐야 하는데
# iris 셋에선 큰 필요가 없을 것 같아 그냥 있는 그대로의 데이터를 넣고 학습 시킨다

# 각각의 값 확인
print("-----------------------------------------------")
print(X_train)
print("-----------------------------------------------")
print(X_test)
print("-----------------------------------------------")
print(Y_train)
print("-----------------------------------------------")
print(Y_test)
print("-----------------------------------------------")


# KNN 모델의 fit 메소드를 통해 입력(X_train)과 정답(Y_train)을 넣고 학습시킨다
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

# 학습을 시켰으므로 모델의 성능을 평가해본다
print('accurarcy : {:.2f}'.format(knn.score(X_test, Y_test)))



'''

전체 데이터셋에서 30% 데이터를 분할하여 test 데이터로 만들었다
그 test 데이터에는 {입력:정답}을 모두 맞춘 데이터셋이므로 학습시킨 모델을 평가하기에 적절하다

knn.score(X_test, Y_test)는 X_test 데이터를 아까 학습시킨 knn 모델에 넣고
Y_test와 비교하여 정확도를 출력한다.
100% 정확도를 얻었으며, 이는 테스트 세트에 있는 샘플 데이터들의 100%를 정확히 맞췄다는 뜻이다

'''






