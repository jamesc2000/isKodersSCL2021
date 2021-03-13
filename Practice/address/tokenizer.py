import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

addresses = pd.read_csv('addresses.csv')

buffer = {'Token': [], 'Pos': []}


for row in range(0, 9999):
    test = addresses['address'][row] + ' ' + addresses['city'][row] + ' ' + addresses['country'][row]
    doc = nlp(test)
    for token in doc:
        buffer['Token'].append(token.text)
        buffer['Pos'].append(token.pos_)


tokens = pd.DataFrame(buffer)
tokens.to_csv('tokens.csv')
