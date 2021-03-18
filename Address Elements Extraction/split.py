import pandas as pd

actual_train = pd.read_csv('actual_train.csv')

actual_train = actual_train.sample(frac=1).reset_index(drop=True)

actual_test = actual_train.iloc[:80000, :]
actual_train = actual_train.iloc[80001:, :]

actual_train.to_csv('actual_train.csv')
actual_test.to_csv('actual_test.csv')
