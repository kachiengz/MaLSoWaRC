"""
*************************************************************************************************************************************************************************************************
Kevin Achieng
kachieng@uwyo.edu
Department of Civil and Artichectural Engineering
University of Wyoming

October 29th, 2018
December 17th, 2018
December 30th, 2018

#1. Single-layer artificial neural network (ANN) Prediction of Soil Moisture (given suction) & thus Soil Water Retention Curve  for soil (e.g. Loamy Sand) subjected to 
	subjected to monotonic wetting
#2. Deep Neural Network (DNN) (2-10-Layer artificial neural network ) Prediction of Soil Moisture (given suction) & thus Soil Water Retention Curve  for soil (e.g. Loamy Sand) 
	subjected to subjected to monotonic wetting
#   Deep Neural Network (DNN) is an artificial neural network (ANN) with multiple (>=2) layers between the input and output layers.
*************************************************************************************************************************************************************************************************

"""


# 1.1 Import the primary libraries 
import tensorflow as tf
import numpy
import numpy as np
import matplotlib.pyplot as plt
#1.2. Import the secondary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
from tensorflow import set_random_seed 

#Fix a random seed  to initialize random no. generator
# initialized random number generator to ensure that the results are REPRODUCIBLE
np.random.seed(1)
set_random_seed(2)
 
# 2. Define learning parameters i.e. learning rate (for gradient descent) & the number of steps (epochs)
learning_rate   = 0.01
learning_rate_ANN   = 0.01
momentum        = 0.1 #Optional --- used if using tensorflow "MomentumOptimizer" instead of "GradientDescentOptimizer"
training_epochs =5000
training_epochs_Art =5000
 
# 3. Import the soil Data (measured/observed suction and soil moisture)
soilData = pd.read_csv('E:/Fall2018/SupportVectorRegression/SoilMositureData/Data/Submit/Code/Test_data/Wetting2.csv') #change this to your own directory where the SWRC data is stored
X = soilData.iloc[:, 2:3].values # Soil suction (cm): This creates x as matrix instead of vector
X.shape
print(X)
y = soilData.iloc[:, 0:1].values #  Observed Soil moisture ($cm^3$/$cm^3$) : create y as matrix instead of vector
y_mean = soilData['Observed Water Content'].mean()
#y=y.ravel()
y.shape
print(y) 

# 4. Feature Scaling (Standardizing the data i.e. xnew=(x-mean)/std)
# Feature Scaling (Standardizing) is a process of transforming variables to have values in same range so that no variable is dominated by the other.
# tensorflow libraries do not apply feature scaling by default. Therefore, apply feature scaling to the X,Y dataset using sklearn libraries

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # StandardScalar() is the class used for feature scaling from sklearn library.
sc_y = StandardScaler()
X = sc_X.fit_transform(X) # sc_x.fit_transform(X) transforms and fits X into scaled data
y = sc_y.fit_transform(y)

# 5. define training & test data:
# Typically, the data is subdivided into train and test data in the ration of 0.9 to 0.1, respectively.
# However, since the measured dataset is small and monotonic in nature, the same data is used to train and test the model
train_X = X
print("\ntrain_X=", train_X)
train_Y = y
 
test_X = X
print("\ntest_X=", test_X)
test_Y = y

# 7. define placeholders (input nodes of the graph)
#	 placeholders are where training samples (x,f(x)) are placed
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 8. setup the models: 1 ANN and 8 DNN models
 
# Artificial NN model generation helper function 
def Artificial_NN(x, weights_Art, biases_Art):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
    # searched for a long time for an error:
    # reshaped_x helped to solve the error of wrong input dimension!
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
	
    # output layer with linear activation : out_layer=W_out*layer1+b_out
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
      
    # return the DNN model
    return out_layer
 
# Deep NN model generation helper function
def Deep_NN(x, weights, biases):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# output layer #6 with linear: layer_6=linear(W_out*layer_5+b_out)	

    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
      
    # return the DNN model
    return out_layer


def Deep_NN_2HiddenLayers(x, weights2, biases2):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# output layer #3 with linear: layer_out=linear(W_out*layer_2+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights2['h1']), biases2['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights2['h2']), biases2['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_2, weights2['out']) + biases2['out']
      
    # return the DNN model
    return out_layer
		
def Deep_NN_3HiddenLayers(x, weights3, biases3):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# output layer #4 with linear: layer_out=linear(W_out*layer_3+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights3['h1']), biases3['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights3['h2']), biases3['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights3['h3']), biases3['b3'])
    layer_3 = tf.nn.relu(layer_3)

	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_3, weights3['out']) + biases3['out']
      
    # return the DNN model
    return out_layer	

def Deep_NN_4HiddenLayers(x, weights4, biases4):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# Output layer #5 with linear: layer_out=linear(W_out*layer_4+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights4['h1']), biases4['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights4['h2']), biases4['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights4['h3']), biases4['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights4['h4']), biases4['b4'])
    layer_4 = tf.nn.relu(layer_4)

	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_4, weights4['out']) + biases4['out']
      
    # return the DNN model
    return out_layer	

def Deep_NN_5HiddenLayers(x, weights5, biases5):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# output layer #6 with linear: layer_out=linear(W_out*layer_5+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights5['h1']), biases5['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights5['h2']), biases5['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights5['h3']), biases5['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights5['h4']), biases5['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights5['h5']), biases5['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
      
    # return the DNN model
    return out_layer

def Deep_NN_6HiddenLayers(x, weights6, biases6):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# hidden layer #6 with RELU: layer_6=relu(W6*layer_5+b6)
	# output layer #7 with linear: layer_out=linear(W_out*layer_6+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights6['h1']), biases6['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights6['h2']), biases6['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights6['h3']), biases6['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights6['h4']), biases6['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights6['h5']), biases6['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_6 = tf.add(tf.matmul(layer_5, weights6['h6']), biases6['b6'])
    layer_6 = tf.nn.relu(layer_6)
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_6, weights6['out']) + biases6['out']
      
    # return the DNN model
    return out_layer

def Deep_NN_7HiddenLayers(x, weights7, biases7):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# hidden layer #6 with RELU: layer_6=relu(W6*layer_5+b6)
	# hidden layer #7 with RELU: layer_7=relu(W7*layer_6+b7)
	# output layer #8 with linear: layer_out=linear(W_out*layer_7+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights7['h1']), biases7['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights7['h2']), biases7['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights7['h3']), biases7['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights7['h4']), biases7['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights7['h5']), biases7['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_6 = tf.add(tf.matmul(layer_5, weights7['h6']), biases7['b6'])
    layer_6 = tf.nn.relu(layer_6)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_7 = tf.add(tf.matmul(layer_6, weights7['h7']), biases7['b7'])
    layer_7 = tf.nn.relu(layer_7)
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_7, weights7['out']) + biases7['out']
      
    # return the DNN model
    return out_layer	

def Deep_NN_8HiddenLayers(x, weights8, biases8):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# hidden layer #6 with RELU: layer_6=relu(W6*layer_5+b6)
	# hidden layer #7 with RELU: layer_7=relu(W7*layer_6+b7)
	# hidden layer #8 with RELU: layer_8=relu(W8*layer_7+b8)	
	# output layer #9 with linear: layer_out=linear(W_out*layer_8+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights8['h1']), biases8['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights8['h2']), biases8['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights8['h3']), biases8['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights8['h4']), biases8['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights8['h5']), biases8['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_6 = tf.add(tf.matmul(layer_5, weights8['h6']), biases8['b6'])
    layer_6 = tf.nn.relu(layer_6)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_7 = tf.add(tf.matmul(layer_6, weights8['h7']), biases8['b7'])
    layer_7 = tf.nn.relu(layer_7)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_8 = tf.add(tf.matmul(layer_7, weights8['h8']), biases8['b8'])
    layer_8 = tf.nn.relu(layer_8)	
    
	# output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_8, weights8['out']) + biases8['out']
      
    # return the DNN model
    return out_layer	
	

def Deep_NN_9HiddenLayers(x, weights9, biases9):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# hidden layer #6 with RELU: layer_6=relu(W6*layer_5+b6)
	# hidden layer #7 with RELU: layer_7=relu(W7*layer_6+b7)
	# hidden layer #8 with RELU: layer_8=relu(W8*layer_7+b8)
	# hidden layer #9 with RELU: layer_9=relu(W9*layer_8+b9)	
	# output layer #10 with linear: layer_out=linear(W_out*layer_9+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights9['h1']), biases9['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights9['h2']), biases9['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights9['h3']), biases9['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights9['h4']), biases9['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights9['h5']), biases9['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_6 = tf.add(tf.matmul(layer_5, weights9['h6']), biases9['b6'])
    layer_6 = tf.nn.relu(layer_6)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_7 = tf.add(tf.matmul(layer_6, weights9['h7']), biases9['b7'])
    layer_7 = tf.nn.relu(layer_7)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_8 = tf.add(tf.matmul(layer_7, weights9['h8']), biases9['b8'])
    layer_8 = tf.nn.relu(layer_8)	
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_9 = tf.add(tf.matmul(layer_8, weights9['h9']), biases9['b9'])
    layer_9 = tf.nn.relu(layer_9)	
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_9, weights9['out']) + biases9['out']
      
    # return the DNN model
    return out_layer	

def Deep_NN_10HiddenLayers(x, weights10, biases10):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# hidden layer #6 with RELU: layer_6=relu(W6*layer_5+b6)
	# hidden layer #7 with RELU: layer_7=relu(W7*layer_6+b7)
	# hidden layer #8 with RELU: layer_8=relu(W8*layer_7+b8)
	# hidden layer #9 with RELU: layer_9=relu(W9*layer_8+b9)
	# hidden layer #10 with RELU: layer_10=relu(W10*layer_9+b10)	
	# output layer #11 with linear: layer_out=linear(W_out*layer_9+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights10['h1']), biases10['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights10['h2']), biases10['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights10['h3']), biases10['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights10['h4']), biases10['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights10['h5']), biases10['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
	# hidden layer #5 with RELU: layer_2=relu(W2*layer1+b2)
    layer_6 = tf.add(tf.matmul(layer_5, weights10['h6']), biases10['b6'])
    layer_6 = tf.nn.relu(layer_6)
	
	# hidden layer #6 with RELU: layer_2=relu(W2*layer1+b2)
    layer_7 = tf.add(tf.matmul(layer_6, weights10['h7']), biases10['b7'])
    layer_7 = tf.nn.relu(layer_7)
	
	# hidden layer #7 with RELU: layer_2=relu(W2*layer1+b2)
    layer_8 = tf.add(tf.matmul(layer_7, weights10['h8']), biases10['b8'])
    layer_8 = tf.nn.relu(layer_8)	
	
	# hidden layer #8 with RELU: layer_2=relu(W2*layer1+b2)
    layer_9 = tf.add(tf.matmul(layer_8, weights10['h9']), biases10['b9'])
    layer_9 = tf.nn.relu(layer_9)	
	
	# hidden layer #9 with RELU: layer_2=relu(W2*layer1+b2)
    layer_10 = tf.add(tf.matmul(layer_9, weights10['h10']), biases10['b10'])
    layer_10 = tf.nn.relu(layer_10)	
	
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_10, weights10['out']) + biases10['out']
      
    # return the DNN model
    return out_layer	
	
# 9. Specify the number of neurons per layer i.e. neurons in input layer, hidden layer(s), and output layer
dim_in = 1
dim1 = 7
dim2 = 7
dim3 = 7
dim4 = 7
dim5 = 7
dim6 = 7
dim7 = 7
dim8 = 7
dim9 = 7
dim10 = 7
dim_out = 1
      
# 10. Create dictionaries to hold the weights & biases of all layers
weights = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
    'out': tf.Variable(tf.random_normal([dim2, dim_out]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights_Art = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
	'out': tf.Variable(tf.random_normal([dim2, dim_out]))
}
biases_Art = {
    'b1': tf.Variable(tf.random_normal([dim1])),
	'out': tf.Variable(tf.random_normal([dim_out]))
}   


weights2 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
    'out': tf.Variable(tf.random_normal([dim2, dim_out]))
}
biases2 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights3 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
    'out': tf.Variable(tf.random_normal([dim3, dim_out]))
}
biases3 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}
weights4 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
    'out': tf.Variable(tf.random_normal([dim4, dim_out]))
}
biases4 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights5 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
    'out': tf.Variable(tf.random_normal([dim2, dim_out]))
}
biases5 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights6 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
	'h6': tf.Variable(tf.random_normal([dim5, dim6])),
    'out': tf.Variable(tf.random_normal([dim6, dim_out]))
}
biases6 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
	'b6': tf.Variable(tf.random_normal([dim6])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights7 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
	'h6': tf.Variable(tf.random_normal([dim5, dim6])),
	'h7': tf.Variable(tf.random_normal([dim6, dim7])),
    'out': tf.Variable(tf.random_normal([dim7, dim_out]))
}
biases7 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
	'b6': tf.Variable(tf.random_normal([dim6])),
	'b7': tf.Variable(tf.random_normal([dim7])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights8 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
	'h6': tf.Variable(tf.random_normal([dim5, dim6])),
	'h7': tf.Variable(tf.random_normal([dim6, dim7])),
	'h8': tf.Variable(tf.random_normal([dim7, dim8])),
    'out': tf.Variable(tf.random_normal([dim8, dim_out]))
}
biases8 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
	'b6': tf.Variable(tf.random_normal([dim6])),
	'b7': tf.Variable(tf.random_normal([dim7])),
	'b8': tf.Variable(tf.random_normal([dim8])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights9 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
	'h6': tf.Variable(tf.random_normal([dim5, dim6])),
	'h7': tf.Variable(tf.random_normal([dim6, dim7])),
	'h8': tf.Variable(tf.random_normal([dim7, dim8])),
	'h9': tf.Variable(tf.random_normal([dim8, dim9])),
    'out': tf.Variable(tf.random_normal([dim9, dim_out]))
}
biases9 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
	'b6': tf.Variable(tf.random_normal([dim6])),
	'b7': tf.Variable(tf.random_normal([dim7])),
	'b8': tf.Variable(tf.random_normal([dim8])),
	'b9': tf.Variable(tf.random_normal([dim9])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights10 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
	'h6': tf.Variable(tf.random_normal([dim5, dim6])),
	'h7': tf.Variable(tf.random_normal([dim6, dim7])),
	'h8': tf.Variable(tf.random_normal([dim7, dim8])),
	'h9': tf.Variable(tf.random_normal([dim8, dim9])),
	'h10': tf.Variable(tf.random_normal([dim9, dim10])),
    'out': tf.Variable(tf.random_normal([dim10, dim_out]))
}
biases10 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
	'b6': tf.Variable(tf.random_normal([dim6])),
	'b7': tf.Variable(tf.random_normal([dim7])),
	'b8': tf.Variable(tf.random_normal([dim8])),
	'b9': tf.Variable(tf.random_normal([dim9])),
	'b10': tf.Variable(tf.random_normal([dim10])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}
# 11. Now use helper function that was previously defined to generate a DNN/ANN models
	#ANN and DNN
pred = Deep_NN(X, weights, biases) 
pred_Artificial = Artificial_NN(X, weights_Art, biases_Art)

	#2 to 10 Hidden Layers
pred_2Layers_DNN = Deep_NN_2HiddenLayers(X, weights2, biases2) 
pred_3Layers_DNN = Deep_NN_3HiddenLayers(X, weights3, biases3) 
pred_4Layers_DNN = Deep_NN_4HiddenLayers(X, weights4, biases4)
pred_5Layers_DNN = Deep_NN_5HiddenLayers(X, weights5, biases5)
pred_6Layers_DNN = Deep_NN_6HiddenLayers(X, weights6, biases6) 
pred_7Layers_DNN = Deep_NN_7HiddenLayers(X, weights7, biases7)
pred_8Layers_DNN = Deep_NN_8HiddenLayers(X, weights8, biases8)
pred_9Layers_DNN = Deep_NN_9HiddenLayers(X, weights9, biases9)
pred_10Layers_DNN = Deep_NN_10HiddenLayers(X, weights10, biases10)

# 12. Define the DNN/ANN cost function that needs to be optimzed in order to get optimal weights and biases for the DNN/ANN layers:
# This is acieved by minimizing the sum of squared errors (SSE)
n_samples=len(test_X)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
cost_Art = tf.reduce_sum(tf.pow(pred_Artificial-Y, 2))/(2*n_samples)

cost2 = tf.reduce_sum(tf.pow(pred_2Layers_DNN -Y, 2))/(2*n_samples) 
cost3 = tf.reduce_sum(tf.pow(pred_3Layers_DNN -Y, 2))/(2*n_samples)  
cost4 = tf.reduce_sum(tf.pow(pred_4Layers_DNN -Y, 2))/(2*n_samples) 
cost5 = tf.reduce_sum(tf.pow(pred_5Layers_DNN -Y, 2))/(2*n_samples) 
cost6 = tf.reduce_sum(tf.pow(pred_6Layers_DNN -Y, 2))/(2*n_samples) 
cost7 = tf.reduce_sum(tf.pow(pred_7Layers_DNN -Y, 2))/(2*n_samples) 
cost8 = tf.reduce_sum(tf.pow(pred_8Layers_DNN -Y, 2))/(2*n_samples) 
cost9 = tf.reduce_sum(tf.pow(pred_9Layers_DNN -Y, 2))/(2*n_samples) 
cost10 = tf.reduce_sum(tf.pow(pred_10Layers_DNN -Y, 2))/(2*n_samples) 

# 13. Generate optimizer node for gradient descent algorithm using tensorflow GradientDescentOptimizer() function
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)  #Method 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)		 #Method 2
optimizer_Art = tf.train.GradientDescentOptimizer(learning_rate_ANN).minimize(cost_Art)

optimizer2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost2)
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost3) 
optimizer4 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost4)
optimizer5 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost5)
optimizer6 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost6)
optimizer7 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost7)
optimizer8 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost8)
optimizer9 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost9)
optimizer10 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost10)
 
# 14. initialize the variables (to be optimized) & launch optimization 
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

init_Art = tf.initialize_all_variables()
sess_Art = tf.InteractiveSession()
sess_Art.run(init_Art) 

init2 = tf.initialize_all_variables()
sess2 = tf.InteractiveSession()
sess2.run(init2) 

init3 = tf.initialize_all_variables()
sess3 = tf.InteractiveSession()
sess3.run(init3)

init4 = tf.initialize_all_variables()
sess4 = tf.InteractiveSession()
sess4.run(init4)

init5 = tf.initialize_all_variables()
sess5 = tf.InteractiveSession()
sess5.run(init5) 

init6 = tf.initialize_all_variables()
sess6 = tf.InteractiveSession()
sess6.run(init6)

init7 = tf.initialize_all_variables()
sess7 = tf.InteractiveSession()
sess7.run(init7)

init8 = tf.initialize_all_variables()
sess8 = tf.InteractiveSession()
sess8.run(init8)

init9 = tf.initialize_all_variables()
sess9 = tf.InteractiveSession()
sess9.run(init9)

init10 = tf.initialize_all_variables()
sess10 = tf.InteractiveSession()
sess10.run(init10)

# 15. Kick off the training of the ANN/DNN models ...
print("Starting with training in Deep_NN...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess.run(optimizer, feed_dict={X: x, Y: y})        
    # display epoch nr
    if epoch % 20 == 0:
        c = sess.run(cost, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN Optimization Finished!")

print("Starting with training in Artificial_NN...")
for epoch in range(training_epochs_Art):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess_Art.run(optimizer_Art, feed_dict={X: x, Y: y})        
    # display epoch nr
    if epoch % 20 == 0:
        c = sess_Art.run(cost_Art, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Artificial_NN Optimization Finished!")

print("Starting with training in Deep_NN_2HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess2.run(optimizer2, feed_dict={X: x, Y: y})       
    # display epoch nr
    if epoch % 20 == 0:
        c2 = sess2.run(cost2, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_2HiddenLayers Optimization Finished!")

print("Starting with training in Deep_NN_3HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess3.run(optimizer3, feed_dict={X: x, Y: y})        
    # display epoch nr
    if epoch % 20 == 0:
        c3 = sess2.run(cost3, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_3HiddenLayers Optimization Finished!")

print("Starting with training in Deep_NN_4HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess4.run(optimizer4, feed_dict={X: x, Y: y})       
    # display epoch nr
    if epoch % 20 == 0:
        c4 = sess4.run(cost4, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_4HiddenLayers Optimization Finished!")

print("Starting with training in Deep_NN_5HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess5.run(optimizer5, feed_dict={X: x, Y: y})       
    # display epoch nr
    if epoch % 20 == 0:
        c5 = sess5.run(cost5, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_5HiddenLayers Optimization Finished!")

print("Starting with training in Deep_NN_6HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess6.run(optimizer6, feed_dict={X: x, Y: y})         
    # display epoch nr
    if epoch % 20 == 0:
        c6 = sess6.run(cost6, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_6HiddenLayers Optimization Finished!")

print("Starting with training in Deep_NN_7HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess7.run(optimizer7, feed_dict={X: x, Y: y})        
    # display epoch nr
    if epoch % 20 == 0:
        c7 = sess7.run(cost7, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_7HiddenLayers Optimization Finished!")

print("Starting with training in Deep_NN_8HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess8.run(optimizer8, feed_dict={X: x, Y: y})  
    # display epoch nr
    if epoch % 20 == 0:
        c8 = sess8.run(cost8, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_8HiddenLayers Optimization Finished!")

print("Starting with training in Deep_NN_9HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess9.run(optimizer9, feed_dict={X: x, Y: y})  
    # display epoch nr
    if epoch % 20 == 0:
        c9 = sess9.run(cost9, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_9HiddenLayers Optimization Finished!")

print("Starting with training in Deep_NN_10HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess10.run(optimizer10, feed_dict={X: x, Y: y})  
    # display epoch nr
    if epoch % 20 == 0:
        c10 = sess10.run(cost10, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN_10HiddenLayers Optimization Finished!")

 
# 16. Display final costs
test_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN costs=", test_cost, "\n")
test_cost_Art = sess_Art.run(cost_Art, feed_dict={X: test_X, Y: test_Y})
print("Test dataset ANN costs=", test_cost_Art, "\n")

test_cost2 = sess2.run(cost2, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN2 costs=", test_cost2, "\n")
test_cost3 = sess3.run(cost3, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN3 costs=", test_cost3, "\n")
test_cost4 = sess4.run(cost4, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN4 costs=", test_cost4, "\n")
test_cost5 = sess5.run(cost5, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN5 costs=", test_cost5, "\n")
test_cost6 = sess6.run(cost6, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN6 costs=", test_cost6, "\n")
test_cost7 = sess7.run(cost7, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN7 costs=", test_cost7, "\n")
test_cost8 = sess8.run(cost8, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN8 costs=", test_cost8, "\n")
test_cost9 = sess9.run(cost9, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN9 costs=", test_cost9, "\n")
test_cost10 = sess10.run(cost10, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN10 costs=", test_cost10, "\n")
 
# 17. Calculate the prediction by ANN/DNN models 
predicted_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value = sess.run(pred, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_Y[i] = y_value
predicted_Y=sc_y.inverse_transform(predicted_Y)
#print(predicted_Y)

predicted_ANN_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value = sess_Art.run(pred_Artificial, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_ANN_Y[i] = y_value
predicted_ANN_Y=sc_y.inverse_transform(predicted_ANN_Y)
print(pd.DataFrame(predicted_Y)) 
print(pd.DataFrame(predicted_Y).dtypes)

predicted_DNN2_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value2 = sess2.run(pred_2Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN2_Y[i] = y_value2
predicted_DNN2_Y=sc_y.inverse_transform(predicted_DNN2_Y)
print(pd.DataFrame(predicted_DNN2_Y)) 
print(pd.DataFrame(predicted_DNN2_Y).dtypes)

predicted_DNN3_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value3 = sess3.run(pred_3Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN3_Y[i] = y_value3
predicted_DNN3_Y=sc_y.inverse_transform(predicted_DNN3_Y)
print(pd.DataFrame(predicted_DNN3_Y)) 
print(pd.DataFrame(predicted_DNN3_Y).dtypes)

predicted_DNN4_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value4 = sess4.run(pred_4Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN4_Y[i] = y_value4
predicted_DNN4_Y=sc_y.inverse_transform(predicted_DNN4_Y)
print(pd.DataFrame(predicted_DNN4_Y)) 
print(pd.DataFrame(predicted_DNN4_Y).dtypes)

predicted_DNN5_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value5 = sess5.run(pred_5Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN5_Y[i] = y_value5
predicted_DNN5_Y=sc_y.inverse_transform(predicted_DNN5_Y)
print(pd.DataFrame(predicted_DNN5_Y)) 
print(pd.DataFrame(predicted_DNN5_Y).dtypes)

predicted_DNN6_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value6 = sess6.run(pred_6Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN6_Y[i] = y_value6
predicted_DNN6_Y=sc_y.inverse_transform(predicted_DNN6_Y)
print(pd.DataFrame(predicted_DNN6_Y)) 
print(pd.DataFrame(predicted_DNN6_Y).dtypes)

predicted_DNN7_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value7 = sess7.run(pred_7Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN7_Y[i] = y_value7
predicted_DNN7_Y=sc_y.inverse_transform(predicted_DNN7_Y)
print(pd.DataFrame(predicted_DNN7_Y)) 
print(pd.DataFrame(predicted_DNN7_Y).dtypes)

predicted_DNN8_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value8 = sess8.run(pred_8Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN8_Y[i] = y_value8
predicted_DNN8_Y=sc_y.inverse_transform(predicted_DNN8_Y)
print(pd.DataFrame(predicted_DNN8_Y)) 
print(pd.DataFrame(predicted_DNN8_Y).dtypes)

predicted_DNN9_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value9 = sess9.run(pred_9Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN9_Y[i] = y_value9
predicted_DNN9_Y=sc_y.inverse_transform(predicted_DNN9_Y)
print(pd.DataFrame(predicted_DNN9_Y)) 
print(pd.DataFrame(predicted_DNN9_Y).dtypes)

predicted_DNN10_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value10 = sess10.run(pred_10Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN10_Y[i] = y_value10
predicted_DNN10_Y=sc_y.inverse_transform(predicted_DNN10_Y)
print(pd.DataFrame(predicted_DNN10_Y)) 
print(pd.DataFrame(predicted_DNN10_Y).dtypes)



# 18. Save predicted SWC

predicted_SWC=soilData.copy()
predicted_SWC['SWC_DNN']=pd.DataFrame(predicted_Y).iloc[:, 0:1].values
predicted_SWC['SWC_ANN']=pd.DataFrame(predicted_ANN_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_2HiddenLayers']=pd.DataFrame(predicted_DNN2_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_3HiddenLayers']=pd.DataFrame(predicted_DNN3_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_4HiddenLayers']=pd.DataFrame(predicted_DNN4_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_5HiddenLayers']=pd.DataFrame(predicted_DNN5_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_6HiddenLayers']=pd.DataFrame(predicted_DNN6_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_7HiddenLayers']=pd.DataFrame(predicted_DNN7_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_8HiddenLayers']=pd.DataFrame(predicted_DNN8_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_9HiddenLayers']=pd.DataFrame(predicted_DNN9_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_10HiddenLayers']=pd.DataFrame(predicted_DNN10_Y).iloc[:, 0:1].values

print(predicted_SWC)
predicted_SWC.to_csv('./Output_files/ANN_DNN_Layer1-10_predicted_Wetting2_SWC_Epoch=5000.csv', encoding='utf-8', index=False)

# Willmot's index of agreement d1
d1_ANN = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_ANN']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_ANN']-y_mean)).sum())
d1_DNN = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN']-y_mean)).sum())
d1_DNN2 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_2HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_2HiddenLayers']-y_mean)).sum())
d1_DNN3 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_3HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_3HiddenLayers']-y_mean)).sum())
d1_DNN4 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_4HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_4HiddenLayers']-y_mean)).sum())
d1_DNN5 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_5HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_5HiddenLayers']-y_mean)).sum())
d1_DNN6 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_6HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_6HiddenLayers']-y_mean)).sum())
d1_DNN7 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_7HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_7HiddenLayers']-y_mean)).sum())
d1_DNN8 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_8HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_8HiddenLayers']-y_mean)).sum())
d1_DNN9 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_9HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_9HiddenLayers']-y_mean)).sum())


print('d1_ANN='+str(d1_ANN)+ ': and d1_DNN='+str(d1_DNN))

# 19. plot measured/observed SWC/SWRC and ANN/DNN predicted SWC/SWRC

fig = plt.figure()
ax = plt.subplot(111)

plt.scatter(soilData.iloc[:, 0:1].values, soilData.iloc[:, 2:3].values, label='Observed Data, '+'N='+str(n_samples), facecolors='none', edgecolors='r')

plt.semilogy(predicted_ANN_Y, soilData.iloc[:, 2:3].values, 'kx' ,label='ANN: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_ANN_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_ANN_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_ANN,2)))
plt.semilogy(predicted_DNN2_Y, soilData.iloc[:, 2:3].values, 'bs' ,label='DNN2: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN2_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN2_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN2,2)))
plt.semilogy(predicted_DNN3_Y, soilData.iloc[:, 2:3].values, 'g+' ,label='DNN3: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN3_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN3_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN3,2)))
plt.semilogy(predicted_DNN4_Y, soilData.iloc[:, 2:3].values, 'kv' ,label='DNN4: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN4_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN4_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN4,2)))
plt.semilogy(predicted_DNN5_Y, soilData.iloc[:, 2:3].values, 'k^' ,label='DNN5: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN5_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN5_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN5,2)))
plt.semilogy(predicted_DNN6_Y, soilData.iloc[:, 2:3].values, 'ko' ,label='DNN6: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN6_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN6_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN6,2)))
plt.semilogy(predicted_DNN7_Y, soilData.iloc[:, 2:3].values, 'c>' ,label='DNN7: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN7_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN7_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN7,2)))
plt.semilogy(predicted_DNN8_Y, soilData.iloc[:, 2:3].values, 'm<' ,label='DNN8: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN8_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN8_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN8,2)))
plt.semilogy(predicted_DNN9_Y, soilData.iloc[:, 2:3].values, 'bp' ,label='DNN9: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN9_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN9_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN9,2)))

plt.xlabel('Soil Water Content ($cm^3$/$cm^3$)')
plt.ylabel('Suction (cm)')
#plt.xlim([0, 0.5])
plt.ylim([0, 100])
from matplotlib.ticker import LogLocator

#plt.legend(loc='lower left')
#plt.legend(loc='upper right')

ax.legend(ncol=2,handleheight=2.1, labelspacing=0.05, bbox_to_anchor=(0.5, -0.1), loc='upper center')


plt.tight_layout()
# 20. Saave the plots
plt.savefig('./Output_files/NeuralNetwork_Wetting2_SWC_1-10Hidden-Layers_Epoch=5000.png', bbox_inches="tight")
plt.savefig('./Output_files/NeuralNetwork_Wetting2_SWC_1-10Hidden-Layers_Epoch=5000.pdf', bbox_inches="tight")
plt.show()
