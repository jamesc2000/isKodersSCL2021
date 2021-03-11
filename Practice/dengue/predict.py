import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def test_score(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


X = pd.read_csv('denguecases.csv')
y = X.Dengue_Cases

print('Description of denguecases.csv')
print(X.describe())
print(X.head())
print(y.head())

X.drop(['Dengue_Cases'], axis=1, inplace=True)

# For test/validation purposes only, in actual predictions use entire dataset
# Split X into training and testing
train_X, valid_X, train_y, valid_y = train_test_split(X, y,
                                                      train_size=0.8,
                                                      test_size=0.2)


# Label categorical data
categoricals = ['Month', 'Region']
label_encoder = LabelEncoder()
for col in categoricals:
    train_X[col] = label_encoder.fit_transform(train_X[col])
    valid_X[col] = label_encoder.transform(valid_X[col])

print('With Label Encoding')
print(test_score(train_X, valid_X, train_y, valid_y))


print('Dropping Month and Region')
train_X.drop(categoricals, axis=1, inplace=True)
valid_X.drop(categoricals, axis=1, inplace=True)
print(test_score(train_X, valid_X, train_y, valid_y))
