import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

train_df = pd.read_csv('/Users/jessicahalterman/Documents/MachineLearning/Assignment1/train.csv')
test_df = pd.read_csv('/Users/jessicahalterman/Documents/MachineLearning/Assignment1/test.csv')
combine = [train_df, test_df]

train_df = train_df.drop(['Cabin', 'Fare', 'Ticket', 'Embarked'], axis='columns')
test_df = test_df.drop(['Cabin', 'Fare', 'Ticket', 'Embarked'], axis='columns')
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
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

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

train_df = train_df.drop(['SibSp', 'Parch', 'Age'], axis='columns')
test_df = test_df.drop(['SibSp', 'Parch', 'Age'], axis='columns')
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

#print(notMarried[['SibSp', 'Survived', 'Pclass']].groupby(['SibSp', 'Pclass']).mean().sort_values(by='Survived', ascending=False))

#print(notMarried[['Parch', 'Survived', 'Pclass']].groupby(['Parch', 'Pclass']).mean().sort_values(by='Survived', ascending=False))

y = train_df.Survived
X = train_df.drop(['Survived'], axis=1)

print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
 
logRegression = LogisticRegression(max_iter=300)
xgbcClassifier = XGBClassifier()
randomForest = RandomForestClassifier()
 
final_model = VotingClassifier(
    estimators=[('lr', logRegression), ('xgb', xgbcClassifier), ('rf', randomForest)], voting='hard')
 
final_model.fit(X_train, y_train)
print(final_model.score(X_test, y_test)) 
 
y_test_pred = final_model.predict(test_df)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_test_pred
    })
submission.to_csv('/Users/jessicahalterman/Documents/MachineLearning/Competition1/submission.csv', index=False)