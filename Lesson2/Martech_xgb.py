import pandas as pd
import numpy as np

train_data = pd.read_csv('train_label.csv',index_col=0)
print(train_data.columns)
train = train_data.drop(['customer_province','member_status','is_member_actived','order_pay_time','goods_list_time','goods_delist_time','goods_price','goods_id','order_detail_id','order_id','member_id','customer_gender'],axis=1)

test_data = pd.read_csv('train.csv')
print(test_data.columns)
test = test_data.drop(['customer_province','member_status','is_member_actived','order_pay_time','goods_list_time','goods_delist_time','goods_price','goods_id','order_detail_id','order_id','member_id','customer_gender'],axis=1)

test.drop_duplicates(['customer_id'],inplace=True)
test['customer_city'].fillna('nowhere',inplace=True)
train['customer_city'].fillna('nowhere',inplace=True)


print(len(train))
print((len(test)))
print('load success')

from sklearn.preprocessing import LabelEncoder

#print(train.isna().sum())
print(train.isna().sum())
print(test.isna().sum())
# 对于分类特征进行特征值编码
attr=['is_customer_rate','goods_status','goods_has_discount','customer_city','order_status','order_count','order_detail_status']
lbe_list=[]
for feature in attr:
    print(feature)
    lbe=LabelEncoder()
    train[feature] = lbe.fit_transform(train[feature])
    test[feature] = lbe.transform(test[feature])
    lbe_list.append(lbe)
#train.to_csv('temp.csv')
print('labelSuccess')

import xgboost as xgb
from sklearn.model_selection import train_test_split

param = {'boosting_type':'gbdt',
                         'objective' : 'binary:logistic', #
                         'eval_metric' : 'auc',
                         'eta' : 0.01,
                         'max_depth' : 15,
                         'colsample_bytree':0.8,
                         'subsample': 0.9,
                         'subsample_freq': 8,
                         'alpha': 0,
                         'lambda': 1,
        }
X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['label','customer_id'],axis=1), train['label'], test_size=0.05, random_state=42)

train_data = xgb.DMatrix(X_train, label=y_train)
valid_data = xgb.DMatrix(X_valid, label=y_valid)
test_data = xgb.DMatrix(test.drop('customer_id',axis=1))

model = xgb.train(param, train_data, evals=[(train_data, 'train'), (valid_data, 'valid')], num_boost_round = 200, early_stopping_rounds=10, verbose_eval=10)

predict=model.predict(test_data)
print(predict)

test['result']=predict
test[['customer_id','result']].to_csv('Martech_xgb.csv')
print('success')
# 转化为二分类输出
#200组最大，earlystop10情况下成绩最好，说明在比赛结果验证中很可能有过拟合情况出现   w=1.11 s=44.89
