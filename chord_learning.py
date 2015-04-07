import numpy as np
from scipy.io import wavfile
import cq_tools
import glob
import csv

def beatles(filename, minFreq=27.5,octaves=9,bins=12,thresh=0):
	wavdir = './data/beatles/data/'
	labeldir = './data/beatles/labels/chordlab/The Beatles/'

	fs = 44100
	L = 882
	kernel = cq_tools.kernel(minFreq, octaves, fs, bins=bins, thresh=thresh)

	for album in glob.glob(wavdir + '*'):
		albumname = album.split('/')[-1]
		print(albumname)
		songs = glob.glob(wavdir + albumname +'/*.wav')
		songs.sort()

		songlabels = glob.glob(labeldir + albumname + '/*.lab')
		songlabels.sort()

		if len(songs) != len(songlabels):
			raise Exception('Labels do not match wav files!')

		for songind in range(len(songs)):
			print('\t',songs[songind].split('/')[-1],':',songlabels[songind].split('/')[-1])
			songfs, data = wavfile.read(songs[songind])
			if songfs != fs:
				raise Exception('File sampling rate does not match kernel rate!')
			
			c = cq_tools.chromagram(data, fs, length=L, k=kernel)
			c = cq_tools.normalize(c)

			times = (np.arange(1,c.shape[0] + 1,1) * L/2)/fs
			labels = getlabels(times,songlabels[songind])

			with open('training.csv', 'a') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')
				for ind in range(c.shape[0]):
					row = c[ind,].tolist()
					row.append(labels(ind))
					writer.writerow(row)


def getlabels(times,labelfile):
	labeldata = []
	outputlabels = []
	with open(labelfile) as csvfile:
		reader = csv.reader(csvfile,delimiter=' ')
		for row in reader:
			labeldata.append(row)
	for time in times:
		foundlabel = False
		for labelset in labeldata:
			if time >= float(labelset[0]) and time < float(labelset[1]):
				outputlabels.append(labelset[2])
				foundlabel = True
				break
		if not foundlabel:
			raise Exception('Time was not found in label file!')
	return outputlabels