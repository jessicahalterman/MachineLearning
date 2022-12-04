from cmath import inf
from random import random
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import dask.dataframe as dd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import get_scorer_names
from collections import Counter

pd.options.display.max_columns = None
pd.options.display.max_rows = None

combined_ddf = dd.read_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/allData.csv', low_memory=False)

#Mostly empty, so drop feature
combined_ddf = combined_ddf.drop(['FirstDepTime'], axis=1)

combined_ddf = combined_ddf.drop(['ArrivalDelayGroups', 'ArrDelayMinutes', 'ArrDel15', 'DepDelayMinutes', 'DepDel15', 'DepartureDelayGroups'], axis=1)
#Remove diverted flights and flights cancelled due to weather

combined_ddf = combined_ddf[(combined_ddf.Diverted == 0) & (combined_ddf.CancelledDueToWeather == 0) & (combined_ddf.Year > 2019)]

combined_df = combined_ddf.compute()

#combined_df['DelayedDueToWeather'] = 0

#combined_df.loc[(combined_df['OnTime'] == 0) & (combined_df.WeatherDelay > 0) & (combined_df.NASDelay == 0) & (combined_df.SecurityDelay == 0) & (combined_df.CarrierDelay == 0) & (combined_df.LateAircraftDelay == 0), 'DelayedDueToWeather'] = 1

#print(combined_df[(combined_df.DelayedDueToWeather == 1) | combined_df.CancelledDueToWeather == 1].count())
#print(combined_df[(combined_df.DelayedDueToWeather == 0) & (combined_df.OnTime == 0)].count())

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

combined_df = combined_df.drop(['DepTime', 'DepDelay', 'ActualElapsedTime', 'ArrTime', 'ArrDelay', 'Cancelled', 'ActualElapsedTime', 'CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay', 'CancelledDueToWeather', 'CancelledDueToSecurity', 'CancelledDueToNAS', 'CancelledDueToCarrier', 'Diverted', 'CRSElapsedTime', 'AirTime'], axis=1)

boolean_mapping = {False: 0, True: 1}
combined_df['Holiday'] = combined_df['Holiday'].map(boolean_mapping)

#combined_df = combined_df[combined_df.DelayedDueToWeather == 0]

#print(combined_df[['CRSElapsedTime', 'OnTime']].groupby(['CRSElapsedTime'], as_index=False).mean().sort_values(by='OnTime', ascending=False))

#Rename some features so that they are shorter
heatmap_df = combined_df
heatmap_df['FlightNumber'] = heatmap_df['Flight_Number_Reporting_Airline']
heatmap_df['Airline'] = heatmap_df['DOT_ID_Reporting_Airline']
heatmap_df = heatmap_df.drop(['Flight_Number_Reporting_Airline', 'DOT_ID_Reporting_Airline'], axis=1)

"""#correlation head map
corrmat = heatmap_df.corr()
features = corrmat.index
plt.figure()
heatmap = sb.heatmap(heatmap_df[features].corr(), annot=True, cmap="RdYlGn", annot_kws={"size":8})
plt.tight_layout()
plt.show()"""

"""scalar = MinMaxScaler()
scaled_df = scalar.fit_transform(combined_df)

#pca to reduce feature count
pca = PCA(n_components=10)
transformed_df = pca.fit_transform(scaled_df)

pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
plt.legend('')
plt.xlabel('Principal Components')
plt.ylabel('Explained Varience')
plt.show()"""

airline_mapping = {'9E': 'Endeavor', 'AA': 'American', 'AS': 'Alaska', 'B6': 'JetBlue', 'DL': 'Delta', 'F9': 'Frontier', 'G4': 'Allegiant', 'HA': 'Hawaiian', 'NK': 'Spirit', 'OO': 'SkyWest', 'QX': 'Horizon Air', 'UA': 'United', 'WN': 'Southwest', 'YV': 'Mesa', 'YX': 'Republic'}

onTimeByDepartureHour = combined_df.groupby('DepartureHour')["OnTime"].agg("mean").reset_index()
#sb.lineplot(data=onTimeByDepartureHour, x='DepartureHour', y='OnTime', sort=True)

combined_df['Airline'] = combined_df['Reporting_Airline'].map(airline_mapping)

onTimeByAirline = combined_df.groupby('Airline')["OnTime"].agg("mean").reset_index()
print(onTimeByAirline.head())
onTimeByAirline = onTimeByAirline.sort_values(by=['OnTime'], ascending=False)
sb.barplot(data=onTimeByAirline, x='Airline', y='OnTime', palette='crest')
#plt.show()

OnTimeByDistance = combined_df.groupby('DistanceGroup')["OnTime"].agg("mean").reset_index()
#sb.scatterplot(data=OnTimeByDistance, x='DistanceGroup', y='OnTime')

print(combined_df.groupby('DepartureHour')["OnTime"].agg("mean"))
print(combined_df.groupby('Airline')["OnTime"].agg("mean"))

#plt.show()

#print(get_scorer_names())

y = combined_df.OnTime

counter = Counter(y)
# estimate scale_pos_weight value                  
total_percent_minority = counter[0]/(counter[1]+counter[0])
estimate = counter[0]/(counter[1])
print(estimate)
print(total_percent_minority)
print(counter[0])
print(counter[1])
print(counter[0]+counter[1])