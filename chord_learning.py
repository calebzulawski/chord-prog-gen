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
			print('Labels do not match wav files!')
			continue

		for songind in range(len(songs)):
			print('\t',songs[songind].split('/')[-1],':',songlabels[songind].split('/')[-1])
			songfs, data = wavfile.read(songs[songind])
			if songfs != fs:
				print('\t\tFile sampling rate does not match kernel rate!')
				continue
			
			c = cq_tools.chromagram(data, fs, length=L, k=kernel)
			c = cq_tools.normalize(c)

			times = (np.arange(1,c.shape[0] + 1,1) * L/2)/fs
			labels = getlabels(times,songlabels[songind])

			with open('training.csv', 'a') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')
				for ind in range(c.shape[0]):
					row = c[ind,].tolist()
					row.append(labels[ind])
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
			print('\t\tTimes were not found in label file...')
			outputlabels.append('N')
	return outputlabels

def cleancsv(inputcsv,outputcsv):
	keytonum = {
		'A'  : 1,
		'A#' : 2,
		'Bb' : 2,
		'B'  : 3,
		'Cb' : 3,
		'B#' : 4,
		'C'  : 4,
		'C#' : 5,
		'Db' : 5,
		'D'  : 6,
		'D#' : 7,
		'Eb' : 7,
		'E'  : 8,
		'Fb' : 8,
		'E#' : 9,
		'F'  : 9,
		'F#' : 10,
		'Gb' : 10,
		'G'  : 11,
		'G#' : 12,
		'Ab' : 12
	}
	with open(inputcsv,'r') as incsvfile:
		with open(outputcsv,'a') as outcsvfile:
			reader = csv.reader(incsvfile,delimiter=',',quotechar='"')
			writer = csv.writer(outcsvfile,delimiter=',')
			for row in reader:
				if row[-1] != 'N':
					if len(row[-1]) < 3:
						row[-1] = str(keytonum[row[-1]])
					else:
						if row[-1][1] == '#' or row[-1][1] == 'b':
							row[-1] = str(keytonum[row[-1][0:2]]) + row[-1][2:]
						else:
							row[-1] = str(keytonum[row[-1][0]]) + row[-1][1:]
					row[-1] = row[-1].split('/')[0]
					writer.writerow(row)				
			
def uniquelabels(inputcsv):
	labels = {}
	with open(inputcsv,'r') as csvfile:
		reader = csv.reader(csvfile,delimiter=',',quotechar='"')
		for row in reader:
			labels[row[-1]] = 0
	labels = list(labels.keys())
	return labels

def transpose1(row):
	chroma = row[0:-1]
	label = row[-1]