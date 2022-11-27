from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import torch
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.cluster import DBSCAN

train_df = pd.read_csv('/Users/jessicahalterman/Documents/MachineLearning/Competition2/train.csv')
test_df = pd.read_csv('/Users/jessicahalterman/Documents/MachineLearning/Competition2/test.csv')
combine = [train_df, test_df]

boolean_mapping = {False: 0, True: 1}
home_planet_mapping = {'Europa': 0, 'Earth': 1, 'Mars': 2}
destination_mapping = {'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2}
side_of_ship_mapping = {'P': 0, 'S': 1}
deck_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}

for dataset in combine:
    dataset = dataset.astype({'VIP': bool, 'CryoSleep': bool})

for dataset in combine:
    dataset[['GroupNumber', 'PassengerWithinGroup']] = dataset.PassengerId.str.split('_', n=1, expand=True)
    dataset[['Deck', 'CabinNumber', 'SideOfShip']] = dataset.Cabin.str.split('/', expand=True)
    #dataset['RoomService'] = dataset['RoomService'].fillna(0)
    #dataset['FoodCourt'] = dataset['FoodCourt'].fillna(0)
    #dataset['ShoppingMall'] = dataset['ShoppingMall'].fillna(0)
    #dataset['Spa'] = dataset['Spa'].fillna(0)
    #dataset['VRDeck'] = dataset['VRDeck'].fillna(0)
    dataset['VIP'] = dataset['VIP'].map(boolean_mapping)
    dataset['CryoSleep'] = dataset['CryoSleep'].map(boolean_mapping)
    dataset['HomePlanet'] = dataset['HomePlanet'].map(home_planet_mapping, na_action='ignore')
    dataset['Destination'] = dataset['Destination'].map(destination_mapping, na_action='ignore')

train_df = train_df.drop(['Cabin', 'Name'], axis='columns')
test_df = test_df.drop(['Cabin', 'Name'], axis='columns')
combine = [train_df, test_df]

train_df['Transported'] = train_df['Transported'].map(boolean_mapping)

for dataset in combine:
    dataset['Deck'] = dataset['Deck'].map(deck_mapping, na_action='ignore')
    dataset['SideOfShip'] = dataset['SideOfShip'].map(side_of_ship_mapping, na_action='ignore')
    dataset['AgeBand'] = 0
    dataset['Under9'] = 0
    dataset[['AgeBetween18And26']] = 0
    dataset['SpentUnder100'] = 0
    dataset['GroupSize'] = 1
    dataset['VRGreaterThan1000'] = 0
    dataset['AteMoreInRoom'] = 0
    dataset['SpendMoneyAtSpaOrMall'] = 0
    dataset['TotalSpent'] = 0
    dataset['SpentNothing'] = 0

imputer = KNNImputer(n_neighbors=1)
train_new = imputer.fit_transform(train_df)
test_new = imputer.fit_transform(test_df)
train_new_df = pd.DataFrame(train_new, index=train_df.index, columns=train_df.columns)
test_new_df = pd.DataFrame(test_new, index=test_df.index, columns=test_df.columns)
combine = [train_new_df, test_new_df]

for dataset in combine:
    dataset.loc[(dataset['Age'] < 9), 'AgeBand'] = 1
    dataset.loc[(dataset['Age'] >= 9) & (dataset['Age'] < 18), 'AgeBand'] = 2
    dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] < 27), 'AgeBand'] = 3
    dataset.loc[(dataset['Age'] >= 27), 'AgeBand'] = 4

for dataset in combine:
    dataset.loc[(dataset['Age'] < 9), 'Under9'] = 1
    dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] < 27), 'AgeBetween18And26'] = 1
    dataset.loc[(dataset['RoomService'] < 100) & (dataset['FoodCourt'] < 100) & (dataset['ShoppingMall'] < 100) & (dataset['Spa'] < 100) & (dataset['VRDeck'] < 100), 'SpentUnder100'] = 1
    dataset.loc[(dataset['VRDeck'] > 1000), 'VRGreaterThan1000'] = 1
    dataset.loc[(dataset['RoomService'] > dataset['FoodCourt']), 'AteMoreInRoom'] = 1
    dataset.loc[(dataset['Spa'] > 500) | (dataset['ShoppingMall'] > 500), 'SpendMoneyAtSpaOrMall'] = 1

for dataset in combine:
    for index, row in dataset.iterrows():
        current_group = row['GroupNumber']
        matching_subset = dataset.loc[(dataset['GroupNumber'] == current_group)]
        if (len(matching_subset) > 1):
            dataset.at[index, 'GroupSize'] = len(matching_subset)
        dataset.at[index, 'TotalSpent'] = row['RoomService'] + row['FoodCourt'] + row['ShoppingMall'] + row['Spa'] + row['VRDeck']

for dataset in combine:
    dataset.loc[dataset['TotalSpent'] == 0, 'SpentNothing'] = 1

""" plt.scatter(train_df['Age'],train_df['VIP'])
plt.xlabel('Age')
plt.ylabel('VIP')
plt.title('Scatter plot on training dataset')
plt.show() """

#sb.displot(train_df, x="Spa", hue="Transported", stat="density")

#sb.set_style("whitegrid")
#sb.pairplot(train_new_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']], hue="Transported")
#plt.show()

#Plot age distribution for each Transported value
#sb.FacetGrid(train_new_df,hue='Transported').map(sb.distplot,'TotalSpent').add_legend()
#plt.show()

#grid = sb.FacetGrid(train_new_df)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend()
#plt.show()
""" print(train_new_df.groupby('SpendMoneyAtSpaOrMall', sort=False)['Transported'].mean())
print(train_new_df.groupby('AteMoreInRoom', sort=False)['Transported'].mean())
print(train_new_df.groupby('VRGreaterThan1000', sort=False)['Transported'].mean())
print(train_new_df.groupby('Under9', sort=False)['Transported'].mean())
print(train_new_df.groupby('AgeBand', sort=False)['Transported'].mean())
print(train_new_df.groupby('SpentUnder100', sort=False)['Transported'].mean())
print(train_new_df.groupby('GroupSize', sort=False)['Transported'].mean())
print(train_new_df.groupby('SpentUnder100', sort=False)['Transported'].mean()) """

y = train_new_df.Transported
X = train_new_df.drop(['Transported', 'PassengerId'], axis=1).copy()
X_test = test_new_df.drop(['PassengerId'], axis=1).copy()
#X_test = X_test[['CryoSleep', 'SpentNothing', 'VRGreaterThan1000', 'AteMoreInRoom', 'Under9', 'VIP', 'Destination', 'Deck', 'HomePlanet']]
#X = X[['CryoSleep', 'SpentNothing', 'VRGreaterThan1000', 'AteMoreInRoom', 'Under9', 'VIP', 'Destination', 'Deck', 'HomePlanet']]

X = X.to_numpy()
y = y.to_numpy()
X_test = X_test.to_numpy()

X_train, X_train_test, y_train, y_train_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.fit_transform(X_test)

classifier = TabNetClassifier(verbose=1,seed=42, optimizer_fn=torch.optim.Adam)
classifier.fit(X_train=X_train, y_train=y_train,
    eval_set=[(X_train_test, y_train_test)],
    patience= 10, max_epochs=100,
    eval_metric=['accuracy'])
    #batch_size=4096, virtual_batch_size=256)

y_test_pred = classifier.predict(X_test)

output_mapping = {0: 'False', 1: 'True'}

y_output = np.vectorize(output_mapping.get)(y_test_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": y_output
    })
submission.to_csv('/Users/jessicahalterman/Documents/MachineLearning/Competition2/submission12.csv', index=False)