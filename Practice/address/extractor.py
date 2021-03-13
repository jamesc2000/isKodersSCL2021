import pandas as pd

raw = pd.read_csv('fastfood.csv')

print('Reading csv ', raw.shape, 'Head(5):')
print(raw.head(5))

addresses = raw[['address', 'city', 'country']]

print('Saved new csv as address.csv')
addresses.to_csv('addresses.csv', index=True)
