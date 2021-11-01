import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

data = pd.read_csv('training.1600000.processed.noemoticon.csv', usecols=[0, 5], header=None, encoding='latin-1') 
values = data[5]
keys = data[0]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(values)
clf = MultinomialNB().fit(X_train_counts, keys)

testing_data = pd.read_csv('testdata.manual.2009.06.14.csv', usecols=[0, 5], header=None, encoding='latin-1') 

testing_values = testing_data[5]
testing_keys = testing_data[0]

test_vect = count_vect.transform(testing_values)
predicted = clf.predict(test_vect)

print(f"f1_score, average = None: {f1_score(testing_keys, predicted, average=None)}")
print(f"f1_score, average = micro: {f1_score(testing_keys, predicted, average='micro')}")
print(f"f1_score, average = macro: {f1_score(testing_keys, predicted, average='macro')}")
print(f"f1_score, average = weighted: {f1_score(testing_keys, predicted, average='weighted')}")