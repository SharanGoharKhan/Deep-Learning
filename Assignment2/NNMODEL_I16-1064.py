# Importing required modules
import numpy as np
import scipy.io as sio
from sklearn.neural_network import MLPClassifier
import sys
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

# Give feedback to user that it's training
print('Please wait for about 30 second for the modal to train')

# Supress warning to have required output
warnings.filterwarnings("ignore")

# Scales values between 0 and 1
def featureScaling(X):
    for image in range(0,m):
        min_value = np.min(X[image])
        max_value = np.max(X[image])
        for pixel in range(0,n):
            X[image,pixel] = (X[image,pixel] - min_value)/(max_value-min_value)
    return X

# Importing dataset
mat_data = sio.loadmat('ex3data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
m, n = X.shape

# Call feature scaling function for X
X = featureScaling(X)

# Divide to training and set good for randomizing it as well
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=102)

# Flip half of values to introduce variation
temp_data = X_train[0:m//2,:]
X_train[0:m//2,:] = np.flip(temp_data,1)

# Train the modal using Multilayer Layer Perceptron Classifier hopefully a type of neural network
clf = MLPClassifier(solver='sgd',tol=1e-4, max_iter = 400, alpha = 1e-4, hidden_layer_sizes = (30,),random_state=1)
clf.fit(X_train,y_train)

# Get image file name from given user using sys module
# Am expecting the image to be 20*20 with values between 0 and 1 in .png format
image_file_name = sys.argv[1]

# Read the image using matplotlib
actual_image = mpimg.imread(image_file_name)

# Reshape the image dimension to work in my modal use order='F' to correctly reshape it
pred_image = actual_image.reshape((1,400),order='F')

# Image is predicted in one line only the digit
print('Number predicted is: ',clf.predict(pred_image)[0])