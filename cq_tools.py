import numpy as np
import scipy
#import matplotlib.pyplot as plt
import sklearn.preprocessing

def nextpow2(n):
    npow = 2
    while npow <= n: 
        npow = npow * 2
    return npow

def kernel(minFreq, octaves, fs, bins=12, thresh=0):
	maxFreq = minFreq * (2**octaves)
	print('Creating kernel for',minFreq,'Hz to',maxFreq,'Hz.')
	Q = 1/(2**(1/bins)-1)
	K = np.ceil(bins * np.log2(maxFreq/minFreq))
	fftLen = nextpow2(np.ceil(Q*fs/minFreq))
	tempKernel = np.zeros((fftLen,1)) + 0j
	kernel = []
	for kcq in range(int(K),0,-1):
		length = np.ceil(Q * fs / (minFreq * 2**((kcq-1)/bins)))
		tempKernel[0:length,0] = np.hamming(length)/length * np.exp(2*np.pi*1j*Q*np.arange(0,length)/length)
		specKernel = np.fft.fft(tempKernel)
		for ii in range(len(specKernel)):
			if abs(specKernel[ii]) <= thresh:
				specKernel[ii] = 0
		if isinstance(kernel,np.ndarray):
			kernel = np.hstack((specKernel,kernel))
		else:
			kernel = specKernel
	kernel = np.conj(kernel)/fftLen
	kernel = scipy.sparse.coo_matrix(kernel)
	kernel = kernel.transpose() #transpose for numpy's sparse dot() implementation
	print('Finished creating kernel.')
	return kernel

def cqt(x,kernel):
	fft = np.fft.fft(x,kernel.shape[1])
	cq = kernel.dot(fft)
	return cq

def chroma(x,kernel,bins=12):
	cq = np.absolute(cqt(x,kernel))
	c = np.zeros(bins)
	for n in range(int(kernel.shape[0]/bins)):
		c += cq[(n*bins):(n*bins+bins)]
	return c

def chromagram(x,fs,length=[],minFreq=27.5,octaves=9,bins=12,thresh=0,window=[],step=[],k=[],verbose=False):
	# Setup variables
	if not length:
		length = np.ceil(fs/50)
	if not isinstance(window,np.ndarray):
		window = np.hamming(length)
	if window.size != length:
		raise Exception('Window lengths do not match!')
	if not step:
		step = np.floor(length/2)
	if len(x.shape) == 1:
		x = np.expand_dims(x,1)
	nsteps = int(np.floor((x.shape[0]-length)/step) + 1)
	c = np.zeros((nsteps,bins))

	# Create kernel
	if k == []:
		k = kernel(minFreq,octaves,fs,bins=bins,thresh=thresh)

	for ind in range(nsteps):
		if verbose:
			print(ind,'/',nsteps)
		for channel in range(x.shape[1]):
			selection = x[ind*step:ind*step+length,channel]
			#selection = np.multiply(selection,window)
			c[ind,] += chroma(selection,k,bins=bins)
	return c

# def normalize(c):
# 	c = sklearn.preprocessing.normalize(c, axis=1)
# 	return c

# def chromagramviz(c):
# 	plt.pcolor(np.fliplr(c.transpose()))
# 	plt.show()