import numpy as np
import scipy.stats as stats
from scipy.stats import linregress
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')
import spatialsignificance as sp


def gen_randomfield(alpha = 3.0, size = 128, norm = True):
		
	phase = np.random.normal(size = (size, size)) + 1j * np.random.normal( size = (size, size))
	#phase = np.random.uniform(low=-1, high=1, size = (size, size)) + 1j * np.random.uniform(low=-1, high=1, size = (size, size))
	phase /= np.abs(phase)

	freq_xy = [np.fft.fftfreq(size) for _ in range(2)]    
	freq_grid = np.meshgrid(*freq_xy, indexing="ij") #makes a frequency grid
	power_spec = np.sqrt(np.sum(np.power(freq_grid, 2), axis=0))	
	amplitude = np.power(power_spec, -alpha/2) #generates warning, ignore
	amplitude[0,0] = 0
		
	field = np.fft.ifft2(amplitude * phase).real

	if norm:
		field -= np.mean(field)
		field /= np.std(field)

	return field

def gen_randomfield_moran(grid, m, dm, alpha = 3.0, size = 128, norm = True):		
	x1 = gen_randomfield(alpha=alpha, size=size, norm=norm)
	while abs( sp.moran(grid, x1) - m ) > dm:
		x1 = gen_randomfield(alpha=alpha, size=size, norm=norm)
	return x1
	
#Works well enough for my purposes...
#based on https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
def power_exponent(data, fit_from_id=0):
	
	size = data.shape[0]
	
	ft_data = np.fft.fftn(data)
	ft_amp = (np.abs(ft_data)**2).flatten()

	#too fancy for its own good
	freq_xy = [np.fft.fftfreq(size)*size for _ in range(2)]    
	freq_grid = np.meshgrid(*freq_xy, indexing="ij") 
	freq_norm = np.sqrt(np.sum(np.power(freq_grid, 2), axis=0)).flatten()	

	#constructing the bins, and their centres, VERY explicitly
	kbins = np.arange(0.5, size//2+1, 1.)
	kvals = 0.5 * (kbins[1:] + kbins[:-1])
	
	#don't use this often - given x,y data figure out which bin the x value is in, and compute the mean of the ys in that bin
	amplitudes, bin_edges, bin_number = stats.binned_statistic(freq_norm, ft_amp, statistic = "mean", bins = kbins)
	
	lr = linregress( np.log(kvals[fit_from_id:]), np.log(amplitudes[fit_from_id:]))
	return lr.slope, lr.stderr, (np.log(kvals), np.log(amplitudes))

	


if __name__ == '__main__':
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	np.random.seed(123456789)
	

	size=40
	alpha=4
	example = gen_randomfield(alpha=alpha, size=size)
	
	alpha_estimate, alpha_err, pspec = power_exponent(example)
	print("alpha ~= {} +/- {}".format( alpha_estimate, alpha_err) )
	plt.imshow(example)
	plt.show()
	plt.close()

	
