import pandas as pd
import numpy as np
import dask as dsk
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
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.neighbors import KNeighborsClassifier
from dask_ml.model_selection import KFold, HyperbandSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from sklearn.naive_bayes import CategoricalNB
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

client = Client(processes=False)

X_ddf = dd.from_pandas(X, chunksize=4000)
y_ddf = dd.from_pandas(y, chunksize=4000)

#X_Selected_Features = SelectPercentile(chi2, percentile=20).fit_transform(X, y)
#print(X_Selected_Features.shape)

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

""" logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(logreg.score(X_test, y_test))

coeff_df = pd.DataFrame(X.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending=False)) """

""" varThreshold = VarianceThreshold(threshold=0.01)
varThreshold.fit(combined_df)
print(varThreshold.get_feature_names_out()) """

random_forest = RandomForestClassifier(random_state=42, max_depth=6) #0.6789
#0.593645013184229
#0.8194169681908734
#0.5328106815110077
#13.46 min
#fit = random_forest.fit(X_train, y_train)
#model = SelectFromModel(random_forest, prefit=True)
#features_selected = model.get_support()
#print(X_train.columns[features_selected])
#random_forest.fit(X_train, y_train)
#print(random_forest.score(X_test, y_test))
#cvr_random_forest = cross_validate(estimator=random_forest, X=X, y=y, cv=5, scoring='accuracy', return_train_score=True)
#print(cvr_random_forest['test_score'].mean())

counter = Counter(y)
# estimate scale_pos_weight value                  
estimate = counter[0]/counter[1]

xgbClassifier = XGBClassifier(random_state=42) #, scale_pos_weight=estimate)  #0.545   #booster='gbtree', tree_method='hist', grow_policy='lossguide')

xgb_rf = XGBRFClassifier(random_state=42, scale_pos_weight=estimate)

sgdc = SGDClassifier(class_weight='balanced', shuffle=True, random_state=42, penalty='elasticnet') #0.6755

knn = KNeighborsClassifier()

dask_xgbclassifier = DaskXGBClassifier(n_estimators=100, tree_method="hist") #a0.8183376754909081
dask_rf = DaskXGBRFClassifier()

#X_train, X_test, y_train, y_test = dml.model_selection.train_test_split(X_ddf, y_ddf, random_state=42, shuffle=True, test_size=0.3)

"""dask_xgbclassifier.client = client
#dask_xgbclassifier.fit(X_train, y_train)
#print(dask_xgbclassifier.score(X_test, y_test))

kfold = KFold(shuffle=False, random_state=42, n_splits=5)

kfold.split(X=X_ddf.to_dask_array(lengths=True), y=y_ddf.to_dask_array(lengths=True))

X_da = X_ddf.to_dask_array(lengths=True)
y_da = y_ddf.to_dask_array(lengths=True)

for train_index, test_index in kfold.split(X=X_da, y=y_da):
    X_train_ddf = dd.io.from_dask_array(X_da[train_index])
    y_train_ddf = dd.io.from_dask_array(y_da[train_index])
    X_test_ddf = dd.io.from_dask_array(X_da[test_index])
    y_test_ddf = dd.io.from_dask_array(y_da[test_index])
    train_split_ddf = dd.io.from_dask_array(train_index)
    dask_xgbclassifier.fit(X_train_ddf, y_train_ddf)
    print(dask_xgbclassifier.score(X_test_ddf, y_test_ddf))

scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)} """

parameters = {
    'loss': ['hinge'], # 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    #'rho': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1],
    'fit_intercept': [True, False],
    'max_iter': [1000, 1500],
    'shuffle': [True, False],
    #'eta0': [0, 1, 5, 10, 15],
    #'power_t': [-1, 0.5, 1, 2, 5, 10],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'class_weight': ['balanced', None],
    'random_state': [42]
}

#clf = GridSearchCV(sgdc, parameters)
#clf.fit(X_scaled, y)
#print(clf.best_params_)

undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
sgdc_pipeline = make_pipeline(undersample, SGDClassifier(shuffle=True, random_state=42, penalty='elasticnet'))
rf_pipeline = make_pipeline(undersample, random_forest)

#undersampling for random forest
start_time = time.time()  
cv_rf_undersample = cross_validate(estimator=rf_pipeline, X=X, y=y, cv=2, scoring=('roc_auc', 'accuracy','balanced_accuracy', 'recall', 'precision'))
print(cv_rf_undersample['test_roc_auc'].mean())
print(cv_rf_undersample['test_accuracy'].mean())
print(cv_rf_undersample['test_balanced_accuracy'].mean())
print(cv_rf_undersample['test_recall'].mean())
print(cv_rf_undersample['test_precision'].mean())
print(time.time() - start_time)

#undersampling for sgdclassifier
start_time = time.time()  
cv_sgd_undersample = cross_validate(estimator=sgdc_pipeline, X=X_scaled, y=y, cv=2, scoring=('roc_auc', 'accuracy','balanced_accuracy', 'recall', 'precision'))
print(cv_sgd_undersample['test_roc_auc'].mean())
print(cv_sgd_undersample['test_accuracy'].mean())
print(cv_sgd_undersample['test_balanced_accuracy'].mean())
print(cv_sgd_undersample['test_recall'].mean())
print(cv_sgd_undersample['test_precision'].mean())
print(time.time() - start_time)

#over and under sampling
oversample = RandomOverSampler(sampling_strategy=0.35, random_state=42)
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
sgdc_pipeline_combo = make_pipeline(oversample, undersample, sgdc)
rf_pipeline_combo = make_pipeline(oversample, undersample, random_forest)
start_time = time.time()  
cv_sgd_over_undersample = cross_validate(estimator=sgdc_pipeline_combo, X=X_scaled, y=y, cv=2, scoring=('roc_auc', 'accuracy','balanced_accuracy', 'recall', 'precision'))
print(cv_sgd_over_undersample['test_roc_auc'].mean())
print(cv_sgd_over_undersample['test_accuracy'].mean())
print(cv_sgd_over_undersample['test_balanced_accuracy'].mean())
print(cv_sgd_over_undersample['test_recall'].mean())
print(cv_sgd_over_undersample['test_precision'].mean())
print(time.time() - start_time)

rf_pipeline_combo = make_pipeline(oversample, undersample, random_forest)
start_time = time.time()  
cv_rf_over_undersample = cross_validate(estimator=rf_pipeline_combo, X=X, y=y, cv=2, scoring=('roc_auc', 'accuracy','balanced_accuracy', 'recall', 'precision'))
print(cv_rf_over_undersample['test_roc_auc'].mean())
print(cv_rf_over_undersample['test_accuracy'].mean())
print(cv_rf_over_undersample['test_balanced_accuracy'].mean())
print(cv_rf_over_undersample['test_recall'].mean())
print(cv_rf_over_undersample['test_precision'].mean())
print(time.time() - start_time)

#original data
start_time = time.time()  
cross_validate_rf = cross_validate(estimator=random_forest, X=X, y=y, cv=2, scoring=('roc_auc', 'accuracy','balanced_accuracy', 'recall', 'precision'))
print(cross_validate_rf['test_roc_auc'].mean())
print(cross_validate_rf['test_accuracy'].mean())
print(cross_validate_rf['test_balanced_accuracy'].mean())
print(cross_validate_rf['test_recall'].mean())
print(cross_validate_rf['test_precision'].mean())
print(time.time() - start_time)

start_time = time.time()  
cross_validate_sgdc = cross_validate(estimator=sgdc, X=X_scaled, y=y, cv=2, scoring=('roc_auc', 'accuracy','balanced_accuracy', 'recall', 'precision'))
print(cross_validate_sgdc['test_roc_auc'].mean())
print(cross_validate_sgdc['test_accuracy'].mean())
print(cross_validate_sgdc['test_balanced_accuracy'].mean())
print(cross_validate_sgdc['test_recall'].mean())
print(cross_validate_sgdc['test_precision'].mean())
print(time.time() - start_time)

