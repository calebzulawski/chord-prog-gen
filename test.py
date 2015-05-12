#!/usr/bin/env python
import chord_learning as cl
import sklearn.svm
import sklearn.preprocessing
import pickle

cl.randomSelection('training.csv','_training.csv',1000)
cl.randomSelection('training.csv','_test.csv',1000)

#keepLabels = ['min','maj','min7','maj7','dim','dim7','aug']
keepLabels = ['min','maj']

print('Augmenting training set')
cl.augmentation('_training.csv','_training-aug.csv')

cl.reduceClasses('_training-aug.csv','_train-final.csv', keepLabels)
cl.reduceClasses('_test.csv','_test-final.csv', keepLabels)

print('Loading training data...')
features, labels = cl.csv2ldata('_train-final.csv')
print('Loading test data...')
features_test, labels_test = cl.csv2ldata('_test-final.csv')

le = sklearn.preprocessing.LabelEncoder()
le.fit(labels + labels_test)
labels = le.transform(labels)
labels_test = le.transform(labels_test)

classifier = sklearn.svm.SVC()
classifier.probability = True
classifier.kernel = 'linear'
print('Training SVM...')
classifier.fit(features, labels)

print('Predicting...')
results = classifier.predict(features_test)

numCorrect = 0
for i in range(len(results)):
	if labels_test[i] == results[i]:
		numCorrect += 1

print(str(numCorrect) + ' of ' + str(len(results)) + ' classified correctly: ' + str(numCorrect*100 / len(labels_test)) + '%')

saveObj = {'classifier': classifier, 'labelencoder': le}

with open('model.pkl', 'wb') as fp:
	pickle.dump(saveObj, fp)