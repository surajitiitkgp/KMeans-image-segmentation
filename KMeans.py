'''
    @Author: Surajit Kundu
    @Email: surajit.113125@gmail.com
    @Defination: This code is written for segmenting the 2-d gray images using k-means algorithm. 
    We mainly focus on segmenting the DICOM images. Although it can be applicable on other images

'''

## import the required libraries
from numpy import dot
from numpy.linalg import norm
import math
from pydicom import dcmread
import matplotlib.pyplot as plt
import numpy as np

## define the class
class KMeans:  
    def __init__(self, metric='euclidean', K=2):
        """
            @Defination:
                Creating a parameterized constructer.
            @Input:
                distance > Which distance matrix is used for clustering
                K > number of clusteres
        """          
        self.metric = metric
        self.K = K
        self.distance_matrices = {'euclidean': self.euclidean, 'cosine': self.cosine}
    '''
        Defining all distance functions below
    '''
    def euclidean(self, point, centroids):
        """
            @Defination:
                Calculating the euclidean distances from a pixel to all centroids
            @Input:
                point > pixel value in 1-d
                centroids > K number of centroids
            @Return:
                np.array(distances) > euclidean distances from the given point to all the centroids
        """          
        distances = []
        for i in centroids:
            dist = np.sqrt(np.sum((point-i)**2))
            distances.append(dist)
        return np.array(distances)

    def cosine(self, point, centroids):
        """
            @Defination:
                Calculating the cosine distances from a pixel to all centroids
            @Input:
                point > pixel value in 1-d
                centroids > K number of centroids
            @Return:
                cos_sim > cosine distances from the given point to all the centroids                
        """         
        cos_sim = dot(point, centroids)/(norm(point)*norm(centroids))
        return cos_sim

    def cluster(self, train_data, points):
        """
            @Defination:
                Clustering the pixels of a image
            @Input:
                train_data > all pixels in 1-d
                points > K number of centroids
            @Return:
                clusters, indeces > return the clustered pixels and corresponding index values               
        """          
        ## Initilizing K-d array
        clusters = [[] for _ in range(self.K)]
        indeces = [[] for _ in range(self.K)]
        #print("points ", points)
        counter = 0
        ## Finding distance from each pixel data
        for single_pixel in train_data:
            ## finding distance from a single point to k points using the selected distance metric. Ex. single_pixel=23, points = [2, 4, 9] 
            distances = self.distance_matrices[self.metric](single_pixel, points)
            ## taking the index value having minimum distance
            min_distance_index = np.argmin(distances)
            ## storing the pixel value into the belonging cluster
            clusters[min_distance_index].append(single_pixel)
            ## Storing the index value of the corresponding pixel
            indeces[min_distance_index].append(counter)
            counter = counter+1
        clusters = np.array(clusters)
        ## taking the mean of each cluster
        means = [np.mean(i) for i in clusters]
        ## mean equals to the clustering k porints (mean after the next iteration remains same)
        if(np.array(points).all()==np.array(means).all()):
            print("Final Centroids", means)
            return clusters, indeces
        else:
            ## If mean varies after the clustering
            self.cluster(train_data, means)   

    def clusteredImages(self, image_1d, shape=(1,1), centroids='random'):
        """
            @Defination:
                form the clustered images after the clustering
            @Input:
                image_1d > all pixels in 1-d
                shape > shape of actual 2-d image
                centroids > Initilization of centroids
            @Return:
                clustered_images > return all the clustered images               
        """      
        if len(image_1d.shape) >1: 
            image_1d = self.convertToGray1D(image_1d)
        ## Randomly k number of points for k-means clustering if centroids are not set
        points = np.random.choice(image_1d[image_1d>0], size=(self.K,1)) if centroids == 'random' else centroids
        #points = np.array([np.sum(image1d)/len(image1d), np.sum(image1d)/len(image1d)*2, np.sum(image1d)/len(image1d)*4])
        #points = np.array([math.log(np.sum(image1d),np.sqrt(np.mean(image1d))), np.mean(image1d)*2, np.max(image1d)*0.88])
        print("Initial Controids", points)
        ## getting the clustered value and corresponding indeces
        clusters, indeces = self.cluster(image_1d, points)
        clustered_images = []
        # iterating all clusters 1 to k
        for i in range(self.K):
            img = []
            # creating a 1-d images with all zero pixels
            for p in range(len(image_1d)):
                img.insert(p, 0)
            # replaceing the value with actual pixels as achived after clustering
            for c in range(len(indeces[i])):
                img[indeces[i][c]] = clusters[i][c]
            ## Reshaping the images with actual size
            img = (np.reshape(np.array(img), shape))
            clustered_images.append(img)
        ## return the clustered images
        return clustered_images   

    def convertToGray1D(self, img_array):
        """
            @Defination:
                convert 2-d gray image to 1-d
            @Input:
                img_array > image pixels in 2-d
            @Return:
                np.array(img_arr_1d) > return all pixels in 1-d               
        """         
        ## convert the image to gray if 3 channel images
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        dim = img_array.shape[0]*img_array.shape[1]
        ## converting from 2-d to 1-d
        img_arr_1d = np.reshape(img_array,(dim,))
        return np.array(img_arr_1d)

import cv2    
img = dcmread("Sample/2013032202202100000.1.2.15.dcm")
img_arr = img.pixel_array     
#img_arr = cv2.imread("Sample/1.png", 0)
kmeans = KMeans(K=3)
plt.imshow(np.array(img_arr), cmap='gray')
images = kmeans.clusteredImages(img_arr, shape=img_arr.shape)
plt.figure(figsize=(15, 15)) 
for i in range(len(images)):
    plt.subplot(1, len(images), i+1)
    image = np.array(images[i])
    plt.imshow(image, cmap='gray')
plt.show()
