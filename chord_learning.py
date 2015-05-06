import numpy as np
from scipy.io import wavfile
import cq_tools
import glob
import csv
import pickle
import gc

def beatles(filename, minFreq=27.5,octaves=9,bins=12,thresh=0,padding=0.1):
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
			# c = cq_tools.normalize(c)

			times = (np.arange(1,c.shape[0] + 1,1) * L/2)/fs
			labels = getlabels(times,songlabels[songind],padding)

			with open(filename, 'a') as csvfile:
				writer = csv.writer(csvfile, delimiter=',')
				for ind in range(c.shape[0]):
					row = c[ind,].tolist()
					row.append(labels[ind])
					writer.writerow(row)


def getlabels(times,labelfile,padding):
	labeldata = []
	outputlabels = []
	with open(labelfile) as csvfile:
		reader = csv.reader(csvfile,delimiter=' ')
		for row in reader:
			labeldata.append(row)
	for time in times:
		foundlabel = False
		for labelset in labeldata:
			if time >= (float(labelset[0]) + padding) and time < (float(labelset[1]) - padding):
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
			writer = csv.writer(outcsvfile,delimiter=',',quotechar='"')
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

def augmentation(inputcsv,outputcsv):
	with open(inputcsv,'r') as incsvfile:
		with open(outputcsv,'w') as outcsvfile:
			reader = csv.reader(incsvfile,delimiter=',',quotechar='"')
			writer = csv.writer(outcsvfile,delimiter=',',quotechar='"')
			for row in reader:
				writer.writerow(row)
				chroma = row[:-1]
				label = row[-1]
				for i in range(11):
					chroma = [chroma[-1]] + chroma[:-1]
					splitlabel = label.split(':')
					if len(splitlabel) == 1:
						label = str(int(splitlabel[0]) % 12 + 1)
					else:
						label = str(int(splitlabel[0]) % 12 + 1) + ':' + splitlabel[1]
					writer.writerow(chroma + [label])

def csv2ldata(csvfile):
	with open(csvfile,'r') as fp:
		features = []
		labels = []
		reader = csv.reader(fp,delimiter=',',quotechar='"')
		for row in reader:
			features.append(tuple(row[:-1]))
			labels.append(row[-1])
	return features, labels

def trainClassifier(classifier,inputcsv):
	features = []
	labels = []
	with open(inputcsv,'r') as csvfile:
		reader = csv.reader(csvfile,delimiter=',',quotechar='"')
		for row in reader:
			features.append(tuple(row[:-1]))
			labels.append(row[-1])
	classifier.fit(features,labels)
	return classifier

def batchTrain(classifier,inputcsv,batchsize,epochs,holdout=0):
	features = []
	labels = []
	print('Reading CSV...')
	#gc.disable()
	with open(inputcsv,'r') as csvfile:
		reader = csv.reader(csvfile,delimiter=',',quotechar='"')
		for row in reader:
			features.append(row[:-1])
			labels.append(row[-1])
	#gc.enable()
	print('Finding unique labels...')
	unique_labels = np.unique(labels)
	print('Finished.')
	if holdout > 0:
		features_h = []
		labels_h = []
		ind = np.arange(len(labels))
		np.random.shuffle(ind)
		ind = ind[:holdout]
		ind.sort()
		ind = ind[::-1]
		for i in range(holdout):
			features_h.append(features[ind[i]])
			labels_h.append(labels[ind[i]])
			del features[ind[i]]
			del labels[ind[i]]
	ind = np.arange(len(labels))
	print('Training with size ' + str(batchsize) + ' batches, ' + str(epochs) + ' epochs:')
	for epoch in range(epochs):
		np.random.shuffle(ind)
		num_batches = str(np.floor(len(labels) / batchsize))
		for ind_batch in range(int(np.floor(len(labels) / batchsize))):
			print('\tBatch ' + str(ind_batch) + ' of ' + num_batches )
			classifier.partial_fit( [features[i] for i in ind[ind_batch*batchsize:ind_batch*batchsize+batchsize]], [labels[i] for i in ind[ind_batch*batchsize:ind_batch*batchsize+batchsize]], classes=unique_labels )
		print('Epoch ' + str(epoch) + ' score: ' + str(classifier.score(features_h,labels_h)))
	return classifier

def randomSelection(inputcsv,outputcsv,n):
	with open(inputcsv,'r') as inputfile:
		nsamp = sum(1 for line in inputfile) - 1
		print(str(nsamp) + ' samples.')
		inds = np.arange(nsamp)
		np.random.shuffle(inds)
		inds = inds[:n]
		inds.sort()
		print('Subset selected')
		i = 0;
		count = 0;
		inputfile.seek(0)
		with open(outputcsv,'w') as outputfile:
			for line in inputfile:
				if inds[i] == count:
					i += 1
					outputfile.write(line)
					if i == len(inds):
						return
				count += 1

def saveObject(obj,picklefile):
	with open(picklefile,'w') as f:
		pickle.dump(obj,f)

def loadObject(picklefile):
	with open(picklefile,'r') as f:
		obj = pickle.load(f)
	return obj

def normalize(inputcsv, outputcsv):
	with open(inputcsv,'r') as inputfile:
		with open(outputcsv, 'w') as outputfile:
			reader = csv.reader(inputfile,delimiter=',',quotechar='"')
			writer = csv.writer(outputfile,delimiter=',',quotechar='"')
			for row in reader:
				floatrow = [float(i) for i in row[:-1]]
				newrow = np.divide(floatrow,np.sqrt(np.sum(np.power(floatrow,2)))).tolist()
				newrow.append(row[-1])
				writer.writerow(newrow)