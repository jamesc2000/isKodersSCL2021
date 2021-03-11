import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def test_score(X_train, X_valid, y_train, y_valid, est):
    model = RandomForestRegressor(n_estimators=est, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Pipeline Stuff
numerical_preprocessor = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='mean'))
    ]
)

categorical_preprocessor = Pipeline(
    [
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ],
)

transformer = ColumnTransformer(
    [
        ('categoricals', categorical_preprocessor, [2, 7]),
        ('numericals', numerical_preprocessor, [0, 1, 3, 4, 5, 6])
    ]
)

output = 'solution.csv'

train_full = pd.read_csv('train.csv')
test_full = pd.read_csv('test.csv')

features = [
            'MSSubClass',
            'LotArea',
            'Street',
            'BsmtFullBath',
            '1stFlrSF',
            '2ndFlrSF',
            'YearRemodAdd',
            'CentralAir'
            ]

X = train_full[features]
X_test = test_full[features]
y = train_full.SalePrice

print('Training Data')
print(X.head())
X.to_csv('scratch/training_data.csv', index=False)
#  print(pd.DataFrame(transformer.fit_transform(X), columns=[i for i in range(0, 10)]))

print('Training Target')
print(y.head())

print('Shape of X')
print(X.shape)

print('Testing Data')
print(X_test.head())
X_test.to_csv('scratch/testing_data.csv', index=False)

print('Shape of X_test')
print(X_test.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

model = Pipeline(
    [
        ('transformer', transformer),
        ('model', RandomForestRegressor(n_estimators=350))
    ]
)

model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({'Id': test_full.Id, 'SalePrice': predictions})

#  output.to_csv('submission.csv', index=False)

