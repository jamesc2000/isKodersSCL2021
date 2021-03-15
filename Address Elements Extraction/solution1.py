import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

# Separate POI and street by replacing / to , in csvs, uncomment in final
try:
    train_raw = open('train.csv', encoding='utf-8')
    test_raw = open('test.csv', encoding='utf-8')
    train_processed = open('train_proc.csv', 'w', encoding='utf-8')
    test_processed = open('test_proc.csv', 'w', encoding='utf-8')
    data_train = train_raw.read()
    data_train = data_train.replace('/', ',')
    train_processed.write(data_train)
    data_test = test_raw.read()
    data_test = data_test.replace('/', ',')
    test_processed.write(data_test)
finally:
    train_raw.close()
    train_processed.close()

raw_addresses = pd.read_csv('train_proc.csv')
poi = raw_addresses['POI']
train_poi = pd.DataFrame(data={
    'selected_text': poi.dropna(),
    'category': 'poi'
})
street = raw_addresses['street']
train_street = pd.DataFrame(data={
    'selected_text': street.dropna(),
    'category': 'street'
})
train = pd.concat([train_poi, train_street], axis=0)
train = train.reset_index(drop=True)
print(train.head(10))
print(train.tail(10))

#  Independent
#  X = addresses['raw_address']
#  Target
#  y = addresses['POI/street']

#  Split data into training and validation sets for metrics
#  X_train, y_train, X_valid, y_valid = train_test_split(
    #  X, y
#  )

#  Create a text categorizer
#  can categorize into either a POI or a street
#  textcat = nlp.create_pipe(
    #  "textcat",
    #  config={
        #  "exclusive_classes": True,
        #  "architecture": "bow"})

#  nlp.add_pipe(textcat)

#  textcat.add_label('POI')
#  textcat.add_label('street')

#  print(X_train.head())
