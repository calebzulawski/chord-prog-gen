#!/usr/bin/env python
import numpy as np
from scipy.io import wavfile
import cq_tools as cq
import chord_learning as cl

# fs, data = wavfile.read('classical.wav')

# c = cq_tools.chromagram(data, fs)

# c = cq_tools.normalize(c)

# cq_tools.chromagramviz(c)

cl.beatles('training.csv')