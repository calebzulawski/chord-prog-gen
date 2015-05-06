#!/usr/bin/env python

import chord_learning as cl

#cl.beatles('training-raw.csv')

#cl.cleancsv('training-raw.csv','training-cleaned.csv')

cl.normalize('training-cleaned.csv', 'training.csv')