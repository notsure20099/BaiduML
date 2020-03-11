from surprise import KNNBaseline
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import KFold
from surprise import model_selection
from surprise import accuracy
def dataprocess():
    data = pd.read_csv('MovielensAllsamples.csv')
    dataPro = data[['userID','movieID','rating','timestamp']]
    dataPro.to_csv('MovielenLessFearture.csv',index=False)

reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)
data = Dataset.load_from_file('MovielenLessFearture.csv',reader=reader)
kf = KFold(n_splits=3)
algo = KNNBaseline(k=50, sim_options={'user_based': False, 'verbose': 'True'})

preTemp = []
for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)
    preTemp = preTemp +predictions

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)

result = map(lambda x: x/3,preTemp)