import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

addresses = pd.read_csv('train.csv')
nlp = spacy.blank('en')

# Independent
X = addresses['raw_address']
# Target
y = addresses['POI/street']

# Split data into training and validation sets for metrics
X_train, y_train, X_valid, y_valid = train_test_split(
    X, y
)

# Create a text categorizer
# can categorize into either a POI or a street
textcat = nlp.create_pipe(
    "textcat",
    config={
        "exclusive_classes": True,
        "architecture": "bow"})

nlp.add_pipe(textcat)

textcat.add_label('POI')
textcat.add_label('street')

print(X_train.head())
