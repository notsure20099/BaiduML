# 随机生成8万个样本，用于二分类训练
import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

n_estimator = 10
# 生成样本集
X, y = make_classification(n_samples=80000, n_features=20)
#print(X)
# 将样本集分成测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)


# 基于GBDT监督变换
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd.fit(X_train, y_train)
# 得到OneHot编码
grd_enc = OneHotEncoder(categories='auto')

temp = grd.apply(X_train)
np.set_printoptions(threshold=np.inf)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
#print(grd_enc.get_feature_names()) # 查看每一列对应的特征
# 使用OneHot编码作为特征，训练LR
grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

# 使用LR进行预测
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]

# 直接使用GBDT进行预测  计算baseline
y_pred_grd = grd.predict_proba(X_test)[:, 1]

print(len(y_pred_grd_lm))
print(len(y_pred_grd))
NE = (-1) / len(y_pred_grd) * sum(((1+y_pred_grd_lm)/2 * np.log(y_pred_grd) +  (1-y_pred_grd_lm)/2 * np.log(1 - y_pred_grd)))
print("Normalized Cross Entropy " + str(NE*100)+"%")


