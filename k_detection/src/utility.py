import numpy as np

def histogram(theta, nbins=None, verb=True):
    if nbins is None:
        nbins = len(theta)
    # Return evenly spaced numbers over a specified interval.
    # start = 0
    # stop = 2*PI
    # nbins = Number of samples to generate
    binsLIM = np.linspace(0,2*np.pi,nbins)
    hist, bins = np.histogram(theta, binsLIM)
    return hist, bins 