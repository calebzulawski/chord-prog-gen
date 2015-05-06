#!/usr/bin/env python
import chord_learning as cl
import sklearn.svm

cl.randomSelection('training.csv','training-10k.csv',10000)
cl.randomSelection('training.csv','test.csv',10000)

print('Augmenting training set')
cl.augmentation('training-10k.csv','training-10k-aug.csv')

print('Loading training data...')
features, labels = cl.csv2ldata('training-10k-aug.csv')
print('Loading test data...')
features_test, labels_test = cl.csv2ldata('test.csv')

classifier = sklearn.svm.SVC()
print('Training SVM...')
classifier.fit(features, labels)

print('Predicting...')
results = classifier.predict(features_test)

numCorrect = 0
for i in range(len(results)):
	if labels_test[i] == results[i]:
		numCorrect += 1

print(str(numCorrect) + ' of ' + str(len(results)) + ' classified correctly: ' + str(numCorrect*100 / len(labels_test)) + '%')

print(results)

cl.saveObject(classifier,'svm.pkl')