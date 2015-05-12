#!/usr/bin/env python
import sys
import chord_learning as cl
import sklearn.svm
import sklearn.preprocessing
import pickle
import numpy as np

print('Loading Model...')
with open(sys.argv[1], 'rb') as fp:
	loadObj = pickle.load(fp)

classifier = loadObj['classifier']
le = loadObj['labelencoder']

print('Extracting features...')
features = cl.featuresFromWav(sys.argv[2])

print('Predicting...')
results = classifier.predict_proba(features)

chords = []
stepsize = int(sys.argv[3])
for i in range(int(np.floor(results.shape[0]/stepsize))):
	tempsum = np.sum(results[i*stepsize:i*stepsize+stepsize-1,],axis=0)
	chords.append(np.argmax(tempsum))

print(chords)
print(le.inverse_transform(chords))