import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

from surprise import BaselineOnly, Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate, GridSearchCV
from surprise.prediction_algorithms.matrix_factorization import SVD

my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

file_path = os.path.expanduser("/Users/jessicahalterman/Documents/MachineLearning/Assignment3/ratings_small.csv")
reader = Reader(line_format="user item rating timestamp", sep=",", rating_scale=(1, 5), skip_lines=1)
dataset = Dataset.load_from_file(file_path, reader=reader)

#Item-based CF
icf = KNNBasic(user_based=False)

icf_results = cross_validate(icf, dataset, measures=["RMSE", "MAE"], cv=5)
print(icf_results["test_rmse"].mean())
print(icf_results["test_mae"].mean())

#User-based CF
ucf = KNNBasic(user_based=True)

ucf_results = cross_validate(icf, dataset, measures=["RMSE", "MAE"], cv=5)
print(ucf_results["test_rmse"].mean())
print(ucf_results["test_mae"].mean())

#PMF
pmf = SVD(biased=False) #when biased=False, it is equivalent to PMF

pmf_results = cross_validate(pmf, dataset, measures=["RMSE", "MAE"], cv=5)
print(pmf_results["test_rmse"].mean())
print(pmf_results["test_mae"].mean())

#Different similarities
#Item-based CF
sim_options = {
    "name": "cosine",
    "user_based": False,  # compute  similarities between items
}
icf = KNNBasic(sim_options=sim_options)
icf_results_cosine = cross_validate(icf, dataset, measures=["RMSE", "MAE"], cv=5)

sim_options = {
    "name": "msd",
    "user_based": False,  # compute  similarities between items
}
icf = KNNBasic(sim_options=sim_options)
icf_results_msd = cross_validate(icf, dataset, measures=["RMSE", "MAE"], cv=5)

sim_options = {
    "name": "pearson",
    "user_based": False,  # compute  similarities between items
}
icf = KNNBasic(sim_options=sim_options)
icf_results_pearson = cross_validate(icf, dataset, measures=["RMSE", "MAE"], cv=5)

#User-based CF
sim_options = {
    "name": "cosine",
    "user_based": True
}
ucf = KNNBasic(sim_options=sim_options)
ucf_results_cosine = cross_validate(icf, dataset, measures=["RMSE", "MAE"], cv=5)

sim_options = {
    "name": "msd",
    "user_based": True
}
ucf = KNNBasic(sim_options=sim_options)
ucf_results_msd = cross_validate(icf, dataset, measures=["RMSE", "MAE"], cv=5)

sim_options = {
    "name": "pearson",
    "user_based": True
}
ucf = KNNBasic(sim_options=sim_options)
ucf_results_pearson = cross_validate(icf, dataset, measures=["RMSE", "MAE"], cv=5)

data = {'Model':  ['User-Based CF', 'User-Based CF', 'User-Based CF', 'Item-Based CF', 'Item-Based CF', 'Item-Based CF'],
        'Similarity': ['cosine', 'msd', 'pearson', 'cosine', 'msd', 'pearson'],
        'Average RMSE': [ucf_results_cosine["test_rmse"].mean(), ucf_results_msd["test_rmse"].mean(), ucf_results_pearson["test_rmse"].mean(), icf_results_cosine["test_rmse"].mean(), icf_results_msd["test_rmse"].mean(), icf_results_pearson["test_rmse"].mean()],
        'Average MAE': [ucf_results_cosine["test_mae"].mean(), ucf_results_msd["test_mae"].mean(), ucf_results_pearson["test_mae"].mean(), icf_results_cosine["test_mae"].mean(), icf_results_msd["test_mae"].mean(), icf_results_pearson["test_mae"].mean()]}

print('RMSE')
print(icf_results_cosine['test_rmse'].mean())
print(icf_results_msd['test_rmse'].mean())
print(icf_results_pearson['test_rmse'].mean())

print(ucf_results_cosine['test_rmse'].mean())
print(ucf_results_msd['test_rmse'].mean())
print(ucf_results_pearson['test_rmse'].mean())

print('MAE')
print(icf_results_cosine['test_mae'].mean())
print(icf_results_msd['test_mae'].mean())
print(icf_results_pearson['test_mae'].mean())

print(ucf_results_cosine['test_mae'].mean())
print(ucf_results_msd['test_mae'].mean())
print(ucf_results_pearson['test_mae'].mean())


df = pd.DataFrame(data)
sb.barplot(data=df, x='Similarity', y='Average MAE', hue='Model')
sb.barplot(data=df, x='Similarity', y='Average RMSE', hue='Model')
plt.show()

ib_results = [0] * 21
ub_results = [0] * 21
for i in range(1, 21):
    icf = KNNBasic(user_based=False, k=i)
    results = cross_validate(icf, dataset, measures=["rmse"], cv=5)
    ib_results[i] = results["test_rmse"].mean()
    ucf = KNNBasic(user_based=True, k=i)
    results = cross_validate(ucf, dataset, measures=["rmse"], cv=5)
    ub_results[i] = results["test_rmse"].mean()

df = pd.DataFrame(columns=["Model", "K", "Average RMSE"])

for i in range(1, 21):
    entry = pd.DataFrame.from_dict({
        "Model": ["User-Based CF"],
        "K":  [i],
        "Average RMSE": ub_results[i]
    })
    df = pd.concat([df, entry], ignore_index=True)
for i in range(1, 21):
    entry = pd.DataFrame.from_dict({
        "Model": ["Item-Based CF"],
        "K":  [i],
        "Average RMSE": ib_results[i]
    })
    df = pd.concat([df, entry], ignore_index=True)

sb.barplot(data=df, x='K', y='Average RMSE', hue='Model')
plt.show()

param_grid = {"k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "user-based": [False]}
grid_search = GridSearchCV(KNNBasic, param_grid, measures=["rmse"], cv=5)
grid_search.fit(dataset)
print(grid_search.best_params["rmse"])

param_grid_ub = {"k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "user-based": [True]}
grid_search_ub = GridSearchCV(KNNBasic, param_grid_ub, measures=["rmse"], cv=5)
grid_search_ub.fit(dataset)
print(grid_search_ub.best_params["rmse"])