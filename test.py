#!/usr/bin/env python
import chord_learning as cl

# Commented since already generated
#cl.beatles('training.csv')

cl.cleancsv('training.csv','training-cleaned.csv')

print(cl.uniquelabels('training-cleaned.csv'))