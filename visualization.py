import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import colors

def compareDistributions(arr1, arr2, path=None, show_fig=True, plot_title="linear initialization", 
	arr1Title="Initial Centroids", arr2Title="Post K-means Centroids"):
	""" Graph for comparing distributions of two Numpy arrays. Used in this codebase to compare initial 
	centroid values to those after kmeans is ran. Adapted from https://matplotlib.org/gallery/statistics/hist.html
	
	Args:
		arr1 (ndarray): 1d numpy array of floats, set to initial kmeans centroids in our code.
		arr2 (ndarray): 1d numpy array of floats, set to post kmeans centroids in our code.
		path (String): path to save graph
		show_fig (bool): Flag to show figure.
		plot_title (String): title of graph. Set to init method in our code.
		arr1Title (String): title for first subplot.
		arr2Title (String): title for second subplot.
	"""
	figure, axs = plt.subplots(1, 2, tight_layout=True)#, sharey=True)
	figure.suptitle(plot_title, fontsize=12, y=0.03)

	N, bins, patches = axs[0].hist(arr1, bins='auto', density=True)
	axs[0].set_title(arr1Title)
	fracs = N / N.max()
	norm = colors.Normalize(fracs.min(), fracs.max())
	for tf, tp in zip(fracs, patches):
		color = plt.cm.winter(norm(tf))
		tp.set_facecolor(color)
	axs[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))

	figure.subplots_adjust(hspace=0.5)

	N_2, bins_2, patches_2 = axs[1].hist(arr2, bins='auto', density=True)
	axs[1].set_title(arr2Title)
	fracs_2 = N_2 / N_2.max()
	norm_2 = colors.Normalize(fracs_2.min(), fracs_2.max())
	for tf_2, tp_2 in zip(fracs_2, patches_2):
		color_2 = plt.cm.winter(norm(tf_2))
		tp_2.set_facecolor(color_2)
	axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

	if path: 
		figure.savefig(path)
	if show_fig:
		plt.show()
def graphCDF(weights, path=None, title=None, show_fig=True, plot_title="Weights CDF Estimate", bins_factor=10):
	"""
	Graph for graphing CDF of the values in a numpy array. Used in this codebase to graph
	CDF of weights of a layer. Adapted from https://matplotlib.org/examples/statistics/histogram_demo_cumulative.html

	Args:
		weights (ndarray): 1d numpy array of floats, set to weights of a layer in our code.
		path (String): path to save graph
		title (String): title of graph. Set to layer name in our code.
		show_fig (bool): Flag to show figure.
		plot_title (String): title of graph. Set to "Weights CDF Estimate" in our code.
		bins_factor (int): factor for the number of bins in the CDF histogram. More bins = less smooth curve.
	"""
	figure, ax = plt.subplots(figsize=(8, 4))
	figure.suptitle(plot_title, fontsize=12)
	num_bins = int(weights.size/bins_factor)
	ax.hist(weights, bins=num_bins, density=1, histtype='step', cumulative=True, label='weights')
	if path:
		figure.savefig(path)
	if show_fig:
		plt.show()

if __name__=="__main__":
	x = np.random.randn(500)
	y = np.random.randn(7000)
	compareDistributions(x, y)
	graphCDF(x)