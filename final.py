# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:22:23 2021

@author: riddl
"""

import logging
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2

from PIL import Image, ImageChops

import math

# Homomorphic filter class
class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.
    
    High-frequency filters implemented:
        butterworth+
        
        gaussian
    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H
        
        .
    """


    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, filter='butterworth', H = None):
        """
        Method to apply homormophic filter on an image
        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency 
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)

img = cv2.imread('C:/Users/riddl/.spyder-py3/IMG_0059.jpg')

img = cv2.resize(img,(640,480))
img1 = cv2.resize(img,(640,480))
img = img[:, :, 0]

#preprocessing 

homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
img_filtered = homo_filter.filter(I=img, filter_params=[30,2])
#cv2.imshow("Homo Morphic",img_filtered)

cv2.imshow("Image",img)

    
#plt.imshow(img)
img_filtered = cv2.GaussianBlur(img_filtered, (5, 5), 0)


#segmentation

ret, thresh = cv2.threshold(img_filtered,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

thresh_inv = cv2.bitwise_not(thresh)




contours, hierarchy = cv2.findContours(thresh_inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(contours)

area = 0
cnt = 0
for c in range(len(contours)):
    
    area1 = cv2.contourArea(contours[c])
    if(area1>area):
        area=area1
        cnt = c 

x,y,w,h = cv2.boundingRect(contours[cnt])
print(x,y,w,h)

crop_img = img1[y:y+h, x:x+w]
cv2.imshow("Thresh",thresh_inv)

cv2.imshow("Cropped",crop_img)
#cv2.imshow("Mask",mask)
#som implementation for color extraction

width_c, height_c, channel_c = crop_img.shape
#print(width_c, height_c)
#print(width_c*height_c)
mulWidthHeight = width_c*height_c
sqrtMul = math.sqrt(mulWidthHeight)*5
doubleSqrtMul = round(math.sqrt(sqrtMul))
print(doubleSqrtMul)



crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
crop_reshaped = crop_rgb.reshape(crop_rgb.shape[0] * crop_rgb.shape[1], 3)
print(crop_reshaped.shape)

print(crop_reshaped)

cv2.waitKey(0)
cv2.destroyAllWindows()


#Image Clustering and Color filtering 
#New SOM

from matplotlib import pyplot as plt
import sys
import numpy as np

import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from skimage.color import rgba2rgb
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#old_v = tf.logging.get_verbosity()
#tf.logging.set_verbosity(tf.logging.ERROR)
plt.style.use('seaborn-white')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.random.seed(6278)


class SOM_Layer(): 

    def __init__(self,m,n,dim,learning_rate_som = 0.5,radius_factor = 1.3):
        
        self.m = m
        self.n = n
        self.dim = dim
        self.map = tf.Variable(tf.random.uniform(shape=[m*n,dim],minval=0,maxval=1,seed=2))
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
        self.alpha = learning_rate_som
        self.sigma = max(m,n)*1.1

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getmap(self): return self.map
    def getlocation(self): return self.bmu_locs
    def getIndices(self): return self.bmu_indices
    def getWeights(self): return self.new_weights
    

    def feedforward(self,input):
    
        self.input = input
        self.squared_distance = tf.reduce_sum(tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0),tf.expand_dims(self.input, axis=1)), 2), 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])

    def backprop(self,iter,num_epoch):

        # Update the weigths 
        radius = tf.subtract(self.sigma,
                                tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                    tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        alpha = tf.subtract(self.alpha,
                            tf.multiply(iter,
                                            tf.divide(tf.cast(tf.subtract(self.alpha, 1),tf.float32),
                                                      tf.cast(tf.subtract(num_epoch, 1),tf.float32))))

        self.bmu_distance_squares = tf.reduce_sum(
                tf.pow(tf.subtract(
                    tf.expand_dims(self.location_vects, axis=0),
                    tf.expand_dims(self.bmu_locs, axis=1)), 2), 
            2)

        self.neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
                self.bmu_distance_squares, "float32")), tf.multiply(
                tf.square(tf.multiply(radius, 0.08)), 2)))

        self.learning_rate_op = tf.multiply(self.neighbourhood_func, alpha)
        
        self.numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.learning_rate_op, axis=-1),
            tf.expand_dims(self.input, axis=1)), axis=0)

        self.denominator = tf.expand_dims(
            tf.reduce_sum(self.learning_rate_op,axis=0) + float(1e-20), axis=-1)

        self.new_weights = tf.div(self.numerator, self.denominator)
        self.update = tf.assign(self.map, self.new_weights)

        return self.update
    
dim  = 36
SOM_layer = SOM_Layer(dim,dim,3)
num_epoch = 20
batch_size = len(crop_reshaped)

print(batch_size)

# create the graph
x = tf.placeholder(shape=[batch_size,3],dtype=tf.float32)
current_iter = tf.placeholder(shape=[],dtype=tf.float32)


# graph
SOM_layer.feedforward(x)
map_update=SOM_layer.backprop(current_iter,num_epoch)

# session
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    # start the training
    for iter in range(num_epoch):
        for current_train_index in range(0,len(crop_reshaped),batch_size):
            currren_train = crop_reshaped[current_train_index:current_train_index+batch_size]
            sess.run(map_update,feed_dict={x:currren_train,current_iter:iter})

    # get the trained map and normalize
    trained_map = sess.run(SOM_layer.getmap()).reshape(dim,dim,3)
    #trained_map[:,:,0] = (trained_map[:,:,0]-trained_map[:,:,0].min())/(trained_map[:,:,0].max()-trained_map[:,:,0].min())
    trained_map[:,:,1] = (trained_map[:,:,1]-trained_map[:,:,1].min())/(trained_map[:,:,1].max()-trained_map[:,:,1].min())
    trained_map[:,:,2] = (trained_map[:,:,2]-trained_map[:,:,2].min())/(trained_map[:,:,2].max()-trained_map[:,:,2].min())

    # after training is done get the cloest vector
    locations = sess.run(SOM_layer.getlocation(),feed_dict={x:crop_reshaped})
    indices = sess.run(SOM_layer.getIndices(),feed_dict={x:crop_reshaped})
    weights = sess.run(SOM_layer.getIndices(),feed_dict={x:crop_reshaped})

    plt.imshow(trained_map.astype(float))
#    for i, m in enumerate(locations):
 #       plt.text(m[1], m[0], color_names[i], ha='center', va='center',bbox=dict(facecolor='white', alpha=0.5,lw=0)) 
    plt.axis('off')
    plt.title('Color SOM')
    plt.show()
    plt.close('all')


#class SOM(object):
#    """
#    2-D Self-Organizing Map with Gaussian Neighbourhood function
#    and linearly decreasing learning rate.
#    """
# 
#    #To check if the SOM has been trained
#    _trained = False
# 
#    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
#        """
#        Initializes all necessary components of the TensorFlow
#        Graph.
# 
#        m X n are the dimensions of the SOM. 'n_iterations' should
#        should be an integer denoting the number of iterations undergone
#        while training.
#        'dim' is the dimensionality of the training inputs.
#        'alpha' is a number denoting the initial time(iteration no)-based
#        learning rate. Default value is 0.3
#        'sigma' is the the initial neighbourhood value, denoting
#        the radius of influence of the BMU while training. By default, its
#        taken to be half of max(m, n).
#        """
# 
#        #Assign required variables first
#        self._m = m
#        self._n = n
#        if alpha is None:
#            alpha = 0.3
#        else:
#            alpha = float(alpha)
#        if sigma is None:
#            sigma = max(m, n) / 2.0
#        else:
#            sigma = float(sigma)
#        self._n_iterations = abs(int(n_iterations))
# 
#        ##INITIALIZE GRAPH
#        self._graph = tf.Graph()
# 
#        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
#        with self._graph.as_default():
# 
#            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
# 
#            #Randomly initialized weightage vectors for all neurons,
#            #stored together as a matrix Variable of size [m*n, dim]
#            self._weightage_vects = tf.Variable(tf.random_normal(
#                [m*n, dim]))
# 
#            #Matrix of size [m*n, 2] for SOM grid locations
#            #of neurons
#            self._location_vects = tf.constant(np.array(
#                list(self._neuron_locations(m, n))))
# 
#            ##PLACEHOLDERS FOR TRAINING INPUTS
#            #We need to assign them as attributes to self, since they
#            #will be fed in during training
# 
#            #The training vector
#            self._vect_input = tf.placeholder("float32", [dim])
#            #Iteration number
#            self._iter_input = tf.placeholder("float32")
# 
#            ##CONSTRUCT TRAINING OP PIECE BY PIECE
#            #Only the final, 'root' training op needs to be assigned as
#            #an attribute to self, since all the rest will be executed
#            #automatically during training
# 
#            #To compute the Best Matching Unit given a vector
#            #Basically calculates the Euclidean distance between every
#            #neuron's weightage vector and the input, and returns the
#            #index of the neuron which gives the least value
#            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
#                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
#                    [self._vect_input for i in range(m*n)])), 2), 1)),
#                                  0)
# 
#            #This will extract the location of the BMU based on the BMU's
#            #index
#            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
#                                 np.array([[0, 1]]))
#            slice_input = tf.cast(slice_input, tf.int32)
#            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
#                                          tf.constant(np.array([1, 2]))),
#                                 [2])
# 
#            #To compute the alpha and sigma values based on iteration
#            #number
#            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,
#                                                  self._n_iterations))
#            _alpha_op = tf.multiply(alpha, learning_rate_op)
#            _sigma_op = tf.multiply(sigma, learning_rate_op)
# 
#            #Construct the op that will generate a vector with learning
#            #rates for all neurons, based on iteration number and location
#            #wrt BMU.
#            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
#                self._location_vects, tf.stack(
#                    [bmu_loc for i in range(m*n)])), 2), 1)
#            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
#                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
#            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
# 
#            #Finally, the op that will use learning_rate_op to update
#            #the weightage vectors of all neurons based on a particular
#            #input
#            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
#                learning_rate_op, np.array([i]), np.array([1])), [dim])
#                                               for i in range(m*n)])
#            weightage_delta = tf.multiply(
#                learning_rate_multiplier,
#                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
#                       self._weightage_vects))                                         
#            new_weightages_op = tf.add(self._weightage_vects,
#                                       weightage_delta)
#            self._training_op = tf.assign(self._weightage_vects,
#                                          new_weightages_op)                                       
# 
#            ##INITIALIZE SESSION
#            self._sess = tf.Session()
# 
#            ##INITIALIZE VARIABLES
#            init_op = tf.initialize_all_variables()
#            self._sess.run(init_op)
# 
#    def _neuron_locations(self, m, n):
#        """
#        Yields one by one the 2-D locations of the individual neurons
#        in the SOM.
#        """
#        #Nested iterations over both dimensions
#        #to generate all 2-D locations in the map
#        for i in range(m):
#            for j in range(n):
#                yield np.array([i, j])
# 
#    def train(self, input_vects):
#        """
#        Trains the SOM.
#        'input_vects' should be an iterable of 1-D NumPy arrays with
#        dimensionality as provided during initialization of this SOM.
#        Current weightage vectors for all neurons(initially random) are
#        taken as starting conditions for training.
#        """
# 
#        #Training iterations
#        for iter_no in range(self._n_iterations):
#            #Train with each vector one by one
#            for input_vect in input_vects:
#                self._sess.run(self._training_op,
#                               feed_dict={self._vect_input: input_vect,
#                                          self._iter_input: iter_no})
# 
#        #Store a centroid grid for easy retrieval later on
#        centroid_grid = [[] for i in range(self._m)]
#        self._weightages = list(self._sess.run(self._weightage_vects))
#        self._locations = list(self._sess.run(self._location_vects))
#        for i, loc in enumerate(self._locations):
#            centroid_grid[loc[0]].append(self._weightages[i])
#        self._centroid_grid = centroid_grid
# 
#        self._trained = True
# 
#    def get_centroids(self):
#        """
#        Returns a list of 'm' lists, with each inner list containing
#        the 'n' corresponding centroid locations as 1-D NumPy arrays.
#        """
#        if not self._trained:
#            raise ValueError("SOM not trained yet")
#        return self._centroid_grid
# 
#    def map_vects(self, input_vects):
#        """
#        Maps each input vector to the relevant neuron in the SOM
#        grid.
#        'input_vects' should be an iterable of 1-D NumPy arrays with
#        dimensionality as provided during initialization of this SOM.
#        Returns a list of 1-D NumPy arrays containing (row, column)
#        info for each input vector(in the same order), corresponding
#        to mapped neuron.
#        """
# 
#        if not self._trained:
#            raise ValueError("SOM not trained yet")
# 
#        to_return = []
#        for vect in input_vects:
#            min_index = min([i for i in range(len(self._weightages))],
#                            key=lambda x: np.linalg.norm(vect-
#                                                         self._weightages[x]))
#            to_return.append(self._locations[min_index])
# 
#        return to_return
#    
#som = SOM(20,30,3,20)
#som.train(crop_reshaped)
#
#image_grid = som.get_centroids()
#print(image_grid)
#print(len(image_grid[1]))
#
#trained_map = som.map_vects(crop_reshaped)
#print(trained_map)
#print(len(trained_map[0]))
#print(len(trained_map))
#print(len(image_grid))
#print(locations)
#print(trained_map.astype(float))
#print(trained_map[0].shape)
#print(crop_reshaped.shape)
rgb_crop_float = []
for i in trained_map:
    rgb_crop_float.append(i*255)

#print(rgb_crop_float)
#print(len(rgb_crop_float))
#print(len(rgb_crop_float[0]))

np_mapped = np.asarray(rgb_crop_float, dtype='float32')


print(weights)
print(indices)

print(len(indices))
for i in indices:
    print(i)
#print(np_mapped)

#print(len(np_mapped[0]))
#new_clusterImage = []
#
#for i in np_mapped:    
#    little_img = [i[x:x+8] for x in range(0,len(i),8)]
#    print(little_img)
#    print(len(little_img))
#    print(len(little_img[0]))
#    
#    for j in range(0,len(i),36):
#        temp = i[j:j+36]
#        intermidiate.append(temp)
#    little_img = [intermidiate[x:x+6] for x in range(0,len(intermidiate),6)]
#    print(little_img)
#    new_clusterImage.append(little_img[0])
#  
#print(len(new_clusterImage)) 
#print(len(new_clusterImage[0]))
#print(len(np_mapped))      
    
#let the war begin

print(len(locations))


import os
from collections import Counter
from scipy import spatial
from numpy import dot
from numpy.linalg import norm

dirname = "C:/Users/riddl/Pictures/Project"
subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
print(subfolders)

type_list = ["Leukocytes","Nitrite","Urobilinogen","Protein","pH","Blood","Specific Gravity","Ketone","Bilirubin","Glucose"]

def load_images_from_folder(folder, map_vectors):
    selected_sims = {}
    
    for filename in os.listdir(folder):
        img_samp = cv2.imread(os.path.join(folder,filename))
        if img_samp is not None:
            image_name = filename.split(".")[0]
            print(image_name)
            similarity_vector = []
            img_samp = cv2.resize(img_samp, (6,6), interpolation = cv2.INTER_AREA)
            img_samp = cv2.GaussianBlur(img_samp,(5,5),0)
            #img_samp = cv2.cvtColor(img_samp, cv2.COLOR_BGR2RGB)
            img_samp = img_samp.astype(np.float32)
            img_samp = img_samp.reshape(img_samp.shape[0]*img_samp.shape[1],3)
            print(img_samp.shape)
            
            
            for i in map_vectors:
                err = np.sum((i - img_samp) **2 )
                err /= i.shape[0] * img_samp.shape[1]
                similarity_vector.append(err)
            selected_sims.update({image_name: min(similarity_vector)})

        
    selected_type = min(selected_sims, key=selected_sims.get)
    print(selected_sims)
    
    return selected_type



#def load_images_from_folder_hist(folder, map_vectors):
#    selected_sims = {}
#    
#    for filename in os.listdir(folder):
#        img_samp = cv2.imread(os.path.join(folder,filename))
#        if img_samp is not None:
#            image_name = filename.split(".")[0]
#            print(image_name)
#            similarity_vector = []
#            img_samp = cv2.resize(img_samp, (8,8), interpolation = cv2.INTER_AREA)
#            #img_samp = cv2.GaussianBlur(img_samp,(2,2),0)
#            #img_samp = cv2.cvtColor(img_samp, cv2.COLOR_BGR2RGB)
#            img_samp = img_samp.astype(np.float32)
#            hist = cv2.calcHist([img_samp], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
#            hist = cv2.normalize(hist, hist).flatten()
#            #img_samp = img_samp.reshape(img_samp.shape[0]*img_samp.shape[1],3)
#            print(img_samp.shape)
#            
#            for i in map_vectors:
#                little_img = [i[x:x+8] for x in range(0,len(i),8)]
#                little_img = np.asarray(little_img, dtype='float32')
#                hist_i = cv2.calcHist([little_img], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
#                hist_i = cv2.normalize(hist_i, hist_i).flatten()
#                
#                
#                d = cv2.compareHist(hist_i, hist, cv2.HISTCMP_CHISQR)
#                print(d)
#                #err = np.sum((i - img_samp) **2 )
#                #err /= i.shape[0] * img_samp.shape[1]
#                similarity_vector.append(d)
#            selected_sims.update({image_name: min(similarity_vector)})
#
#        
#    selected_type = min(selected_sims, key=selected_sims.get)
#    print(selected_sims)
#    
#    return selected_type


#hist_res = load_images_from_folder_hist(subfolders[0], np_mapped)

#print(hist_res)
print(len(indices))
   
result_lists = []

for d in range(len(type_list)):
    if(d==0):
        print(type_list[d])
        result_l = load_images_from_folder(subfolders[d], np_mapped)
        print(result_l)
        result_lists.append(result_l)
    if(d==1):
        print(type_list[d])
        result_n = load_images_from_folder(subfolders[d], np_mapped)
        print(result_n)
        result_lists.append(result_n)
    if(d==2):
        print(type_list[d])
        result_U = load_images_from_folder(subfolders[d], np_mapped)
        print(result_U)
        result_lists.append(result_U)
    if(d==3):
        print(type_list[d])
        result_P = load_images_from_folder(subfolders[d], np_mapped)
        print(result_P)
        result_lists.append(result_P)
    if(d==4):
        print(type_list[d])
        result_pH = load_images_from_folder(subfolders[d], np_mapped)
        print(result_pH)
        result_lists.append(result_pH)
    if(d==5):
        print(type_list[d])
        result_B = load_images_from_folder(subfolders[d], np_mapped)
        print(result_B)
        result_lists.append(result_B)
    if(d==6):
        print(type_list[d])
        result_S = load_images_from_folder(subfolders[d], np_mapped)
        print(result_S)
        result_lists.append(result_S)
    if(d==7):
        print(type_list[d])
        result_K = load_images_from_folder(subfolders[d], np_mapped)
        print(result_K)
        result_lists.append(result_K)
    if(d==8):
        print(type_list[d])
        result_Bi = load_images_from_folder(subfolders[d], np_mapped)
        print(result_Bi)
        result_lists.append(result_Bi)
    if(d==9):
        print(type_list[d])
        result_G = load_images_from_folder(subfolders[d], np_mapped)
        print(result_G)
        result_lists.append(result_G)
    
        
        
print("Voila devi jones locker")
print(result_lists)



cv2.waitKey(0)
cv2.destroyAllWindows()