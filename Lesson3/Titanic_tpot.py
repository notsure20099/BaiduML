# 使用TPOT自动机器学习工具对MNIST进行分类
from tpot import TPOTClassifier
import tpot as tp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

# 加载数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 使用平均年龄来填充年龄中的nan值
train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的nan值
train['Fare'].fillna(train['Fare'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)

# 使用登录最多的港口来填充登录港口的nan值
train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S',inplace=True)
# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train[features]
train_labels = train['Survived']
test_features = test[features]

train_labels.map(lambda x: 1 if x=='Yes' else 0)
print(train_labels)
# 特征向量化，不产生稀疏矩阵

for attr in [ 'Sex','Embarked']:
    oh=LabelEncoder()
    print(type(train_features[attr]))
    print(len(train_features[attr]))
    # train_features[feature]=oh.fit_transform(np.array(train_features[feature]).reshape(-1,1))
    # test_features[feature]=oh.transform(np.array(test_features[feature]).reshape(-1,1))
    train_features[attr] = oh.fit_transform(train_features[attr].values.reshape(-1,1))
    test_features[attr] = oh.transform(test_features[attr].values.reshape(-1,1))
# dvec=DictVectorizer(sparse=True)
# train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
# print(dvec.feature_names_)
print('特征值')
print(train_features.head())
print(test_features.head())
X_train, X_test, y_train, y_test = train_test_split(train_features,train_labels, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')