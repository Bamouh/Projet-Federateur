
# coding: utf-8

# In[23]:


#importations
from matplotlib import pyplot as plt
import numpy as np
#import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import cv2
import glob
import pydot
from IPython.display import SVG


# In[65]:


NUMBER_OF_CLASSES = 14
CLASSES = ("-", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "x")
TRAINING_IMAGES_PER_CLASS = 500
EPOCHS = 5


# In[66]:


pathTrain = "sample/Train/*"
pathTest = "sample/Test/*"
def readThenErode(file):
    image = cv2.imread(file, 0)
    image = cv2.copyMakeBorder(image,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    return image

X_Train = np.array([readThenErode(file) for file in glob.glob(pathTrain)])
for i in range(100):
    plt.imshow(X_Train[i*50], cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()

X_Train = np.expand_dims(X_Train, axis=3)
X_Train = X_Train/255

X_Test = np.array([readThenErode(file) for file in glob.glob(pathTest)])
X_Test = np.expand_dims(X_Test, axis=3)
X_Test = X_Test/255

print(X_Train.shape)

j = 0
Y_Train = np.zeros((X_Train.shape[0], NUMBER_OF_CLASSES))
Y_Train[0][0] = 1
for i in range(1, X_Train.shape[0]):
    if i%TRAINING_IMAGES_PER_CLASS == 0:
        j = j + 1
    Y_Train[i][j] = 1

print(Y_Train.shape)
print(Y_Train)


# In[67]:


def ProjectModel(input_shape):
    
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((2, 2))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(6, (5, 5), strides = (1, 1), name = 'conv0')(X)
    X = Activation('relu')(X)

    # AVERAGEPOOL
    X = AveragePooling2D(pool_size = (2, 2), name='average_pool0')(X)
    
    # Zero-Padding
    X = ZeroPadding2D((2, 2))(X)
    
    # CONV -> RELU Block applied to X
    X = Conv2D(16, (5, 5), strides = (1, 1), name = 'conv1')(X)
    X = Activation('relu')(X)

    # AVERAGEPOOL
    X = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), name='average_pool1')(X)

    # FLATTEN X + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(120, activation='relu', name='fc0')(X)
    
    # FULLYCONNECTED
    X = Dense(84, activation='relu', name='fc1')(X)
    
    # FULLYCONNECTED (SOFTMAX)
    X = Dense(NUMBER_OF_CLASSES, activation='softmax', name='output')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='ProjectModel')

    return model


# In[68]:


model = ProjectModel(X_Train.shape[1:])


# In[69]:


model.compile("Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[76]:


model.fit(x = X_Train, y = Y_Train, epochs = EPOCHS)


# In[77]:


model.summary()


# In[78]:


preds = model.predict(X_Test)


# In[79]:


preds = (preds > 0.5).astype(int)
for i in preds:
    print(i)


# In[80]:


preds = preds.tolist()
j = 0
for i in range(len(preds)):
    if i%10==0 and i!=0:
        j = j + 1
    print("Predicted : " + CLASSES[preds[i].index(1)] +" / Real : "+ CLASSES[j])


# In[81]:


from keras.models import load_model

model.save('digit_model.h5')  # creates a HDF5 file 'my_model.h5'

