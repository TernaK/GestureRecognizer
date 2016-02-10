"""
This script is used to read the training and test images, convert to greyscale
then vectorize the data.
"""

import pickle as pickle
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#convert rgb to greyscale
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#get all gestore data
def readGestures():
	alldata = np.zeros([1000, 32*32])
	allLabels = np.zeros([1000, 2])
	index = 0
	for i in range(1,501):
		fileNames = ['one/'+str(i)+'.png', 'two/'+str(i)+'.png']
		img = mpimg.imread(fileNames[0])
		img = rgb2gray(img)
		alldata[index] = np.array(img).reshape([1,32*32])
		allLabels[index] = np.array([1., 0.])
		index = index + 1

		img = mpimg.imread(fileNames[1])
		img = rgb2gray(img)
		alldata[index] = np.array(img).reshape([1,32*32])
		allLabels[index] = np.array([0., 1.])
		index = index + 1
	return (alldata, allLabels)

#prepare a dataset which will be shuffled
def getDataset(allData, allLabels):
	dataset = {'train':np.zeros([800,32*32]), 'trainLabels':np.zeros([800,2]), 'test':np.zeros([800,32*32]), 'testLabels':np.zeros([200,2])}
	randIndexes = np.arange(1000)
	np.random.shuffle(randIndexes)

	for i in range(0, 800):
		dataset['train'][i] = np.array(allData[randIndexes[i]])
		dataset['trainLabels'][i] = np.array(allLabels[randIndexes[i]])
		pass

	for i in range(0,200):
		dataset['test'][i] = np.array(allData[randIndexes[i+800]])
		dataset['testLabels'][i] = np.array(allLabels[randIndexes[i+800]])
		pass

	return dataset

allData, allLabels = readGestures()
dataset = getDataset(allData, allLabels)

for key in dataset.keys():
	print(key + ": ", dataset[key].shape)

#dump in a pikle file
pickle.dump(dataset, open('dataset.pickle', 'wb'))