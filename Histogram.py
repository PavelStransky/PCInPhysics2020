import numpy as np

def Histogram(data, minValue=None, maxValue=None, numBins=100, normalize=False):
    """ Calculates a histogram of the input array.

        Arguments:
        data -- input data
        minValue, maxValue -- minimum and maximum value of the histogram 
                              (if not specified, taken as the minimum and maximum value of the input dats)
        numBins -- final number of bins in the histogram
        normalize -- True if the final values of the histogram shall be normalized to get probability density

        Returns:
        Position of the centres of bins, histogram
    """
    histogram = np.zeros(numBins)

    if minValue is None:
        minValue = min(data)
    if maxValue is None:
        maxValue = max(data)

    for d in data:
        if minValue <= d < maxValue:
            index = int((d - minValue) / (maxValue - minValue) * numBins)
            histogram[index] += 1

    binXValues = np.linspace(minValue, maxValue, numBins + 1)   # Border positions of bins
    binWidth = binXValues[1] - binXValues[0]

    binXValues = binXValues[0:-1] + 0.5 * binWidth              # Middle positions of bins

    if normalize:                                               # Integral of the histogram normalize to 1
        histogram /= (maxValue - minValue) * len(data) / numBins  

    return binXValues, histogram
