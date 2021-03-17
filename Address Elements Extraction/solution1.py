import pandas as pd
import spacy
from spacy.util import minibatch, compounding
#  from thinc.api import compounding
#  from thinc.api.model.ops import minibatch
from numpy.random import shuffle
#  from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data_spacy(file_path):
    train_data = pd.read_csv(file_path)
    train_data.dropna(axis=0, how='any', inplace=True)
    train_data['Num_words_text'] = train_data['selected_text'].apply(lambda x: len(str(x).split()))
    mask = train_data['Num_words_text'] > 2
    train_data = train_data[mask]
    print(train_data['category'].value_counts())

    train_texts = train_data['selected_text'].tolist()
    train_cats = train_data['category'].tolist()

    # One hot encode poi/street
    final_train_categories = []
    for cat in train_cats:
        cat_list = {}
        if cat == 'poi':
            cat_list['poi'] = 1
            cat_list['street'] = 0
            cat_list['misc'] = 0
        if cat == 'street':
            cat_list['poi'] = 0
            cat_list['street'] = 1
            cat_list['misc'] = 0
        if cat == 'misc':
            cat_list['poi'] = 0
            cat_list['street'] = 0
            cat_list['misc'] = 1
        final_train_categories.append(cat_list)

    training_data = list(zip(train_texts, [{'categories': cats} for cats in final_train_categories]))
    return training_data, train_texts, train_cats


def sort_scores(li):
    return(sorted(li, key=lambda x: x[1], reverse=True))


def evaluate(tokenizer, textcat, test_texts, test_cats):
    docs = (tokenizer(text) for text in test_texts)
    preds = []
    for i, doc in enumerate(textcat.pipe(docs)):
        scores = sort_scores(doc.cats.items())
        cat_list = []
        for score in scores:
            cat_list.append(score[0])
        preds.append(cat_list[0])

    labels = ['poi', 'street', 'misc']

    print(classification_report(test_cats, preds, labels=labels))


def train_spacy(
        train_data,
        iterations,
        test_texts,
        test_cats,
        dropout=0.3,
        model=None,
        init_tok2vec=None):
    nlp = spacy.load('xx_ent_wiki_sm')

    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            'textcat', config={'threshold': 0.5}
        )
        textcat = nlp.add_pipe('textcat')
    else:
        textcat = nlp.get_pipe('textcat')

    textcat.add_label('poi')
    textcat.add_label('street')
    textcat.add_label('misc')

    pipe_exceptions = ['textcat', 'trf_wordpiecer', 'trf_tok2vec']
    exclude = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*exclude):
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open('rb') as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print('Training...')
        # print(something)
        batch_sizes = compounding(16.0, 64.0, 1.5)
        for i in range(iterations):
            print('i: ' + str(i))
            losses = {}
            shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                evaluate(nlp.tokenizer, textcat, test_texts, test_cats)
            #Print elapsed time
        with nlp.use_params(optimizer.averages):
            model_name = model_arch + 'AddressElementExtraction'
            path = './' + model_name
            nlp.to_disk(path)
    return nlp


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
train.to_csv('actual_train.csv', index=False)

training_data, train_texts, train_cats = load_data_spacy('actual_train.csv')
print(training_data[:10])
print(len(training_data))
test_data, test_texts, test_cats = load_data_spacy('actual_valid.csv')
print(len(test_data))

nlp = train_spacy(training_data, 10, test_texts, test_cats)

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
