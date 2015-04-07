import numpy as np

def nextpow2(n):
    npow = 2
    while npow <= n: 
        npow = npow * 2
    return npow

def kernel(minFreq, octaves, fs, bins=12, thresh=0):
	maxFreq = minFreq * (2**octaves)
	Q = 1/(2**(1/bins)-1)
	K = np.ceil(bins * np.log2(maxFreq/minFreq))
	fftLen = nextpow2(np.ceil(Q*fs/minFreq))
	tempKernel = np.zeros((fftLen,1))
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
	return kernel

def cqt(x,kernel):
	cq = np.dot(np.fft(x,kernel.shape[0]),kernel)
	return cq

def chroma():
	
