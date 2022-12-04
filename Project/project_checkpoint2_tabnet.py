import pandas as pd
import numpy as np
import dask as dsk
import torch
import joblib
import time
import dask.dataframe as dd
import dask_ml as dml
from dask.distributed import Client
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFRegressor
from xgboost.dask import DaskXGBClassifier, DaskXGBRFClassifier
from sklearn.model_selection import KFold
from skfeature.function.similarity_based import lap_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.neighbors import KNeighborsClassifier
#from dask_ml.model_selection import KFold, HyperbandSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline, make_pipeline

pd.options.display.max_columns = None
pd.options.display.max_rows = None

dsk.config.set({"optimization.fuse.active": True})

combined_ddf = dd.read_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/allData.csv', low_memory=False)

#Mostly empty, so drop feature
combined_ddf = combined_ddf.drop(['FirstDepTime'], axis=1)

combined_ddf = combined_ddf.drop(['ArrivalDelayGroups', 'ArrDelayMinutes', 'ArrDel15', 'DepDelayMinutes', 'DepDel15', 'DepartureDelayGroups'], axis=1)
#Remove diverted flights and flights cancelled due to weather
combined_ddf = combined_ddf[(combined_ddf.Diverted == 0) & (combined_ddf.CancelledDueToWeather == 0) & (combined_ddf.Year > 2019)]

combined_df = combined_ddf.compute()

#map plane tail numbers to integers
unique_planes = combined_df['Tail_Number'].unique()
combined_df['Tail_Number'] = combined_df['Tail_Number'].fillna('Unknown')
unique_planes = np.append(unique_planes, 'Unknown')
plane_mapping = pd.Series(data=np.arange(0, 6416, 1), index=unique_planes)
combined_df['Tail_Number'] = combined_df['Tail_Number'].map(plane_mapping).astype(int)

#map time blocks to integers
unqiue_time_blocks = combined_df['ArrTimeBlk'].unique()
time_block_mapping = pd.Series(data=np.arange(0, 19, 1), index=unqiue_time_blocks)
combined_df['ArrTimeBlk'] = combined_df['ArrTimeBlk'].map(time_block_mapping).astype(int)
combined_df['DepTimeBlk'] = combined_df['DepTimeBlk'].map(time_block_mapping).astype(int)

combined_df = combined_df.drop(['AirTime'], axis=1)

#for on-time flights, estimate actual timings based on the scheduled timings
combined_df['ArrTime'] = np.where(pd.isna(combined_df['ArrTime']) & (combined_df['OnTime'] == 1), combined_df['CRSArrTime'], combined_df['ArrTime'])
combined_df['ArrDelay'] = np.where(pd.isna(combined_df['ArrDelay']) & (combined_df['OnTime'] == 1), 0, combined_df['ArrDelay'])
combined_df['ActualElapsedTime'] = np.where((pd.isna(combined_df['ActualElapsedTime'])) & (combined_df['OnTime'] == 1), combined_df['CRSElapsedTime'], combined_df['ActualElapsedTime'])

combined_df['ArrDelay'] = combined_df['ArrDelay'].fillna(99999)
combined_df['ArrTime'] = combined_df['ArrTime'].fillna(0)
combined_df['ActualElapsedTime'] = combined_df['ActualElapsedTime'].fillna(0)
combined_df['DepTime'] = combined_df['DepTime'].fillna(0)
combined_df['DepDelay'] = combined_df['DepDelay'].fillna(99999)
combined_df['CRSElapsedTime'] = combined_df['CRSElapsedTime'].fillna(0)

boolean_mapping = {False: 0, True: 1}
combined_df['Holiday'] = combined_df['Holiday'].map(boolean_mapping)

#Remove flights delayed due only to weather
#combined_df['DelayedDueToWeather'] = 0

#combined_df.loc[(combined_df['OnTime'] == 0) & (combined_df.WeatherDelay > 0) & (combined_df.NASDelay == 0) & (combined_df.SecurityDelay == 0) & (combined_df.CarrierDelay == 0) & (combined_df.LateAircraftDelay == 0), 'DelayedDueToWeather'] = 1

#combined_df = combined_df[combined_df.DelayedDueToWeather == 0]

#print(combined_df[['DepTimeBlk', 'OnTime']].groupby(['DepTimeBlk'], as_index=False).mean().sort_values(by='OnTime', ascending=False))

#combined_df.to_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/processed_data.csv', index=False)
#processed_ddf = dd.read_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/processed_data.csv', low_memory=False)

#X_test = combined_df[(combined_df.Year == '2022') & (combined_df.Month == 8)]

y = combined_df.OnTime
X = combined_df.drop(['Year', 'DepTime', 'DepDelay', 'ActualElapsedTime', 'ArrTime', 'ArrDelay', 'Cancelled', 'ActualElapsedTime', 'CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay', 'CancelledDueToWeather', 'CancelledDueToSecurity', 'CancelledDueToNAS', 'CancelledDueToCarrier', 'Diverted', 'CRSElapsedTime'], axis=1).copy()
#X = combined_df[['Quarter', 'DayOfWeek', 'Month', 'DepartureHour', 'ArrivalHour', 'OriginAirportID', 'DOT_ID_Reporting_Airline', 'DistanceGroup']]
X = combined_df[['DepartureHour', 'ArrivalHour', 'Tail_Number', 'OriginAirportID', 'Month', 'DayofMonth', 'DayOfWeek', 'DOT_ID_Reporting_Airline', 'Flight_Number_Reporting_Airline','DistanceGroup']]

tabnetClf = TabNetClassifier(optimizer_fn=torch.optim.Adam)

X = X.to_numpy()
y = y.to_numpy()

class roc_score(Metric):
    def __init__(self):
        self._name = "roc_score"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return roc_auc_score(y_true, y_score[:, 1])

class precision(Metric):
    def __init__(self):
        self._name = "precision"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return precision_score(y_true, y_score[:, 1].round())

class recall(Metric):
    def __init__(self):
        self._name = "recall"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return recall_score(y_true, y_score[:, 1].round())

#undersampling
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
kf = KFold(n_splits=2, random_state=42, shuffle=True)
cv_results_us = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)
    classifier = TabNetClassifier(verbose=1,seed=42, optimizer_fn=torch.optim.Adam)
    classifier.fit(X_train=X_train_resampled, y_train=y_train_resampled,
              eval_set=[(X_test, y_test)],
              patience=5, max_epochs=100,
              eval_metric=['accuracy', 'balanced_accuracy', recall, precision, roc_score],
              batch_size=4096, virtual_batch_size=256)
    print(classifier.best_cost)
    cv_results_us.append(classifier.best_cost)
print(cv_results_us)

#oversampling and undersampling
oversample = RandomOverSampler(sampling_strategy=0.35, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
sampling_pipeline = make_pipeline(oversample, undersample)
kf = KFold(n_splits=2, random_state=42, shuffle=True)
cv_results_uos = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train_resampled, y_train_resampled = sampling_pipeline.fit_resample(X_train, y_train)
    classifier = TabNetClassifier(verbose=1,seed=42, optimizer_fn=torch.optim.Adam)
    classifier.fit(X_train=X_train_resampled, y_train=y_train_resampled,
              eval_set=[(X_test, y_test)],
              patience=5, max_epochs=100,
              eval_metric=['accuracy', 'balanced_accuracy', recall, precision, roc_score],
              batch_size=4096, virtual_batch_size=256)
    print(classifier.best_cost)
    cv_results_uos.append(classifier.best_cost)
print(cv_results_uos)

#original data
kf = KFold(n_splits=2, random_state=42, shuffle=True)
cv_results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier = TabNetClassifier(verbose=1,seed=42, optimizer_fn=torch.optim.Adam)
    classifier.fit(X_train=X_train, y_train=y_train,
              eval_set=[(X_test, y_test)],
              patience= 5, max_epochs=100,
              eval_metric=['accuracy', 'balanced_accuracy', recall, precision, roc_score],
              batch_size=4096, virtual_batch_size=256)
    print(classifier.best_cost)
    cv_results.append(classifier.best_cost)
print(cv_results)