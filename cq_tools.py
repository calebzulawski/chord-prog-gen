import numpy as np

def nextpow2(n):
    npow = 2
    while npow <= n: 
        npow = npow * 2
    return npow

def cq_kernel(minFreq, maxFreq, bins, fs, thresh=0.0054):
	Q = 1/(2^(1/bins)-1)
	K = np.ceil(bins * np.log2(maxFreq/minFreq))
	fftLen = 2^nextpow2(np.ceil(Q*fs/minFreq))
	tempKernel = np.zeros((fftLen,1))
	kernel = []
	for kcq in range(K,0,-1):
		length = np.ceil(Q * fs / (minFreq * 2^((k-1)/bins)))
		tempKernel[0:length] = (np.hamming(length)/length) * np.exp(2*np.pi*j*Q*np.linspace(0:(len-1))/length)
		specKernel = np.fft.fft(tempKernel)
		for ii in range(len(specKernel)):
			if abs(specKernel[ii]) <= thresh:
				specKernel[ii] = 0
		if isinstance(kernel,np.ndarray):
			kernel = np.vstack(specKernel,kernel)
		else:
			kernel = specKernel
	kernel = np.transpose(kernel)
	kernel = np.conj(kernel)/fftLen