#!/usr/bin/env python

import chord_learning as cl

print('Building training set...')
cl.beatles('training-raw.csv')

print('Cleaning bad labels...')
cl.cleancsv('training-raw.csv','training-cleaned.csv')

print('Normalizing samples...')
cl.normalize('training-cleaned.csv', 'training.csv')