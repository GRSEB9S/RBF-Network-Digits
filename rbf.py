# Implementation of Radial Basis Function Network for Digit Classification
# Jai Khanna


import random as rnd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import math

def main():
	t=Tools()
	data  = pd.read_csv('mnist.txt',sep=r"\s+", header=None)
	nlabels=10                #total number of labels
	samplesPerLabel=200       #number of samples for each label
	nclusters=50              #number of clusters for k-means
	ksplits=10                #ksplits-fold cross validation

	#Labelling the data
	trueLabels= np.array([])
	for x in range(nlabels):
		for y in range(samplesPerLabel):
			trueLabels=np.append(trueLabels,[x])
	data['label']=trueLabels

	#Splitting data into train and test
	(df_train,df_test)=t.equalSplit(data,nlabels)

	#CrossValidate
	print('Performing k-fold cross validation using RBF network...')
	validationData=df_train.sample(frac=1).reset_index(drop=True)
	i=0;
	betaValues=range(80,300,20) #range decided upon by looking at graph of a wider range
	accuracy=np.zeros(len(betaValues))
	for beta in betaValues:
		accuracy[i]=t.kFoldValidation(validationData,ksplits,nlabels,nclusters,beta)
		i=i+1
	print('Cross validation complete.')

	#Visualize optimal beta
	plt.plot(betaValues,accuracy,'ro')
	plt.ylabel('Accuracy %')
	plt.xlabel('Beta')
	plt.show()
	optimalBeta=betaValues[(np.argmax(accuracy))]
	maxAccuracy=np.amax(accuracy)
	print('Optimal beta and validation accuracy for optimal beta',optimalBeta,maxAccuracy)

	#Train
	print('training...')
	(predictedLabels_train,centers_train,centroidLabel_train)=t.trainRBF(df_train,nclusters,optimalBeta,nlabels)

	#Test
	print('testing...')
	predictedTestLabels=t.RBF(df_train.iloc[:,0:240],optimalBeta,centers_train,centroidLabel_train,nlabels)

	#Test accuracy
	df_train['predicted']=predictedTestLabels
	testLabels=df_train['label'].values
	print('Comparing predicted labels to actual labels: ')
	print(df_train[['label','predicted']])
	accuracy=0
	for y in range(len(predictedTestLabels)):
		if predictedTestLabels[y] == testLabels[y]:
			accuracy+=1
	accuracy=(accuracy/len(predictedTestLabels))*100
	print('Percent accuracy on test data:',accuracy)


class Tools():
	def __init__(self):
		pass

	def calcKmean(self,data,n):
		kmeanz = KMeans(n_clusters=n).fit(data)
		centers=np.array(kmeanz.cluster_centers_)
		closest,_=pairwise_distances_argmin_min(kmeanz.cluster_centers_,data)
		closest=np.array(closest)
		return (centers,closest)

	def RBF(self,data,beta,centers,centroidLabels,nlabels):

		x = data.as_matrix()
		(nrows,_)=x.shape
		rbf_matrix=np.zeros(shape=(nrows,len(centers)))
		for i in range(nrows):
			for j in range(len(centers)):
				rbf_matrix[i][j]=(np.exp(-(np.linalg.norm(np.subtract(x[i],centers[j])))*(np.linalg.norm(np.subtract(x[i],centers[j])))/beta))

		hyp=np.zeros(shape=(nrows,nlabels))
		predictedLabels=np.zeros(nrows)
		for i in range(nrows):
			for j in range(len(centroidLabels)):
				hyp[i][int(centroidLabels[j])]+=rbf_matrix[i][j]
			predictedLabels[i]=np.argmax(hyp[i])
		return predictedLabels

	def trainRBF(self,data,k,beta,nlabels):
		#k-means: Getting centroids and the row indices (in df_train) of the data points closest to the centroids
		t=Tools()
		(centers,indices)=t.calcKmean(data.iloc[:,0:240],k)
		#The label of each centroid according training data
		centroidLabel=np.zeros(len(centers))
		for x in range(len(centers)):
			centroidLabel[x]=data['label'][indices[x]]
		predictedLabels=t.RBF(data.iloc[:,0:240],beta,centers,centroidLabel,nlabels)
		return (predictedLabels,centers,centroidLabel)

	def kFoldValidation(self,data,k,nlabels,nclusters,beta):
		t=Tools()
		(size,_)=data.shape
		binSize=math.floor(size/k)
		data=data.iloc[0:binSize*k,:]
		accuracy=np.zeros(k)
		for x in range(k):
			dataTest=data.iloc[x:x+binSize,:]
			dataTrain=pd.concat([data.iloc[0:x,:],data.iloc[x+binSize:binSize*k,:]],ignore_index=True)
			(predictedLabels_train,centers_train,centroidLabel_train)=t.trainRBF(dataTrain,nclusters,beta,nlabels)
			predictedLabels_test=t.RBF(dataTest.iloc[:,0:240],beta,centers_train,centroidLabel_train,nlabels)
			testLabels=dataTest['label'].values
			for y in range(len(predictedLabels_test)):
				if predictedLabels_test[y] == testLabels[y]:
					accuracy[x]+=1
			accuracy[x]=(accuracy[x]/len(predictedLabels_test))*100
		meanAccuracy=np.mean(accuracy)
		return meanAccuracy

	def equalSplit(self,data,nlabels):
		df_train= pd.DataFrame()
		df_test=pd.DataFrame()
		for x in range(nlabels):
			a=(range((2*x*100),((2*x +1)*100)))
			slice_df=data.loc[a]
			df_train=pd.concat([df_train,slice_df],ignore_index=True)
		for x in range(nlabels):
			a=(range((2*x*100+100),((2*x +1)*100+100)))
			slice_df=data.loc[a]
			df_test=pd.concat([df_test,slice_df],ignore_index=True)
		return (df_train,df_test)

if __name__ == "__main__":
	main()
