import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def test_score(X_train, X_valid, y_train, y_valid, est):
    model = RandomForestRegressor(n_estimators=est, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


output = 'solution.csv'

X_t = pd.read_csv('train.csv')
X_v = pd.read_csv('test.csv')

features = ['MSSubClass',
            'LotArea',
            'Street',
            'BsmtFullBath',
            '1stFlrSF',
            '2ndFlrSF',
            'YearRemodAdd',
            'CentralAir']

y_t = X_t.SalePrice

print(y_t.head())

X_t = X_t[features]

print(X_t.head())

X_test = X_v[features]

#  X_t = X_t.dropna(0)
#  X_test = X_test.dropna(0)


s = (X_t.dtypes == 'object')
categorical = list(s[s].index)

label_enc = LabelEncoder()
OH_enc = OneHotEncoder()

for col in categorical:
    X_t[col] = label_enc.fit_transform(X_t[col])
    X_test[col] = label_enc.transform(X_test[col])

X_train, X_valid, y_train, y_valid = train_test_split(X_t, y_t,
                                                      train_size=0.8,
                                                      test_size=0.2)

print('1000 Estimators')
print(test_score(X_train, X_valid, y_train, y_valid, 1000))

model = RandomForestRegressor(n_estimators=500)

model.fit(X_t, y_t)

predictions = model.predict(X_test)

output_data = pd.DataFrame({'Id': X_v.Id,
                            'SalePrice': predictions})

output_data.to_csv(output, index=False)

#  X_train.to_csv('label_enc.csv')
