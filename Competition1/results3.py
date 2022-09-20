from asyncio import SendfileNotAvailableError
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

train_df = pd.read_csv('/Users/jessicahalterman/Documents/MachineLearning/Assignment1/train.csv')
test_df = pd.read_csv('/Users/jessicahalterman/Documents/MachineLearning/Assignment1/test.csv')
combine = [train_df, test_df]

train_df = train_df.drop(['Cabin', 'Fare', 'Ticket'], axis='columns')
test_df = test_df.drop(['Cabin', 'Fare', 'Ticket'], axis='columns')
combine = [train_df, test_df]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_df = train_df.drop(['Name'], axis='columns')
test_df = test_df.drop(['Name'], axis='columns')
combine = [train_df, test_df]

for dataset in combine:
    for index, row in dataset.iterrows():
        if pd.isnull(row['Age']):
            current_title = row['Title']
            current_class = row['Pclass']
            current_sex = row['Sex']
            matching_subset = dataset.loc[(dataset['Title'] == current_title) & (dataset['Pclass'] == current_class) & (dataset['Sex'] == current_sex)]
            dataset.at[index, 'Age'] = matching_subset['Age'].median().astype(int)
            if pd.isnull(row['Age']):
                matching_subset = dataset.loc[(dataset['Title'] == current_title)]
                dataset.at[index, 'Age'] = matching_subset['Age'].median().astype(int)
    dataset['Age'] = dataset['Age'].astype(int)


#print(train_df.describe())

#print(train_df[['SibSp','Survived', 'Parch']].groupby(['SibSp','Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#children = train_df.loc[train_df['Age'] < 18]

#print(train_df.loc[(train_df['Age'] >= 18) & (train_df['SibSp'] > 1)])

for dataset in combine:
    dataset['Infant'] = 0
    dataset['Child'] = 0
    dataset['Adult'] = 0
    dataset['ElderAdult'] = 0

for dataset in combine:
    dataset.loc[(dataset['Age'] < 2), 'Infant'] = 1
    dataset.loc[(dataset['Age'] >= 2) & (dataset['Age'] < 18), 'Child'] = 1
    dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] < 60), 'Adult'] = 1
    dataset.loc[(dataset['Age'] >= 60), 'ElderAdult'] = 1

#for dataset in combine:
#    dataset['ChildParentCount'] = 0
#    dataset['ChildSiblingCount'] = 0

#for dataset in combine:
#    dataset.loc[(dataset['Age'] < 18), 'ChildParentCount'] = dataset['Parch']
#    dataset.loc[(dataset['Age'] < 18), 'ChildSiblingCount'] = dataset['SibSp']

# Create new feature FamilySize from Parch and SibSp
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Jonkheer'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dona'], 'AristocratFemale')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major'], 'Military')
    dataset['Title'] = dataset['Title'].replace(['Sir', 'Don'], 'AristocratMale')

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5, 'AristocratFemale': 6, 'Military': 7, 'AristocratMale': 8}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# replace Embarked with numerical value
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(int)

train_df = train_df.drop(['Age'], axis='columns')
test_df = test_df.drop(['Age'], axis='columns')
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

#print(notMarried[['SibSp', 'Survived', 'Pclass']].groupby(['SibSp', 'Pclass']).mean().sort_values(by='Survived', ascending=False))

#print(notMarried[['Parch', 'Survived', 'Pclass']].groupby(['Parch', 'Pclass']).mean().sort_values(by='Survived', ascending=False))

y = train_df.Survived
X = train_df.drop(['Survived', 'PassengerId'], axis=1).copy()
X_final_test = test_df.drop(['PassengerId'], axis=1,).copy()

print(X.describe())
 
logRegression = LogisticRegression(max_iter=300)
xgbcClassifier = XGBClassifier(booster='gbtree', tree_method='hist', grow_policy='lossguide')
randomForest = RandomForestClassifier(max_depth=3)
 
final_model = VotingClassifier(
    estimators=[('lr', logRegression), ('xgb', xgbcClassifier), ('rf', randomForest)], voting='hard')
 
final_model.fit(X, y)
y_test_pred = final_model.predict(X_final_test)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_test_pred
    })
submission.to_csv('/Users/jessicahalterman/Documents/MachineLearning/Competition1/submission_4.csv', index=False)