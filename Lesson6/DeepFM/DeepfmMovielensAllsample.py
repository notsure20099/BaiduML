import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat,get_feature_names

def DataProcess():
    movies = pd.read_csv("movies.dat", sep="::", header=None)
    ratings = pd.read_csv("ratings.dat", sep="::", header=None)
    users = pd.read_csv("users.dat", sep="::", header=None)
    ratings.columns = ['userID', 'movieID', 'rating', 'timestamp']
    movies.columns = ['movieID', 'name', 'tag']
    users.columns = ['userID', 'sex', 'age', 'occupation', 'zipcode']

    df1 = pd.merge(ratings, movies, on='movieID')
    Allsamples = pd.merge(df1, users, on='userID')

    Allsamples.to_csv('MovielensAllsamples.csv', index=False)

#数据加载
data = pd.read_csv("MovielensAllsamples.csv")

sparse_features = ['userID', 'movieID', 'name', 'tag', 'sex', 'age', 'occupation', 'zipcode']
target = ['rating']

# 对特征标签进行编码
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature])
# 计算每个特征中的 不同特征值的个数
fixlen_feature_columns = [SparseFeat(feature, data[feature].nunique()) for feature in sparse_features]
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}

# 使用DeepFM进行训练
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'], )
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=True, validation_split=0.2, )
# 使用DeepFM进行预测
pred_ans = model.predict(test_model_input, batch_size=256)
# 输出RMSE或MSE
mse = round(mean_squared_error(test[target].values, pred_ans), 4)
rmse = mse ** 0.5
print("test RMSE", rmse)