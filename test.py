#!/usr/bin/env python
import chord_learning as cl
import sklearn.linear_model

# Commented since already generated
#cl.beatles('training.csv')

# Commented since already cleaned
#cl.cleancsv('training.csv','training-cleaned.csv')

#cl.augmentation('training-cleaned.csv','training-aug.csv')
#print(cl.uniquelabels('training-aug.csv'))

cl.randomSelection('training-aug.csv','training-20k',20000)

#classifier = sklearn.linear_model.SGDClassifier()

#classifier = cl.batchTrain(classifier,'training-aug.csv',10,20000,holdout=100000)

#cl.saveObject(classifier,'linearsvm.pkl')