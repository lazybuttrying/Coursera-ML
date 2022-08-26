# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 125
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_7_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    input_shape = (64,64,3)
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tf.keras.layers.ZeroPadding2D(padding=(3, 3), input_shape=(64,64,3)),
            ## Conv2D with 32 7x7 filters and stride of 1
            tf.keras.layers.Conv2D(32, 7, strides=(1,1)),
            ## BatchNormalization for axis 3
            tf.keras.layers.BatchNormalization(axis=3),
            ## ReLU
            tf.keras.layers.ReLU(),
            ## Max Pooling 2D with default parameters
            tf.keras.layers.MaxPool2D(),
            ## Flatten layer
            tf.keras.layers.Flatten(),
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    
    return model
```


```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d (ZeroPadding2 (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu (ReLU)                 (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 32768)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 103ms/step - loss: 1.4076 - accuracy: 0.6700
    Epoch 2/10
    38/38 [==============================] - 4s 100ms/step - loss: 0.2355 - accuracy: 0.9067
    Epoch 3/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.3037 - accuracy: 0.8950
    Epoch 4/10
    38/38 [==============================] - 4s 98ms/step - loss: 0.2220 - accuracy: 0.9217
    Epoch 5/10
    38/38 [==============================] - 4s 100ms/step - loss: 0.1218 - accuracy: 0.9550
    Epoch 6/10
    38/38 [==============================] - 4s 98ms/step - loss: 0.1214 - accuracy: 0.9533
    Epoch 7/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0911 - accuracy: 0.9717
    Epoch 8/10
    38/38 [==============================] - 4s 98ms/step - loss: 0.0734 - accuracy: 0.9750
    Epoch 9/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.0739 - accuracy: 0.9783
    Epoch 10/10
    38/38 [==============================] - 4s 100ms/step - loss: 0.1020 - accuracy: 0.9683





    <tensorflow.python.keras.callbacks.History at 0x7f92fe23c090>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 31ms/step - loss: 0.2309 - accuracy: 0.8867





    [0.23089326918125153, 0.8866666555404663]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 4



![png](output_28_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(8, 4, strides=(1,1), padding="SAME")(input_img)
    ## RELU
    A1 = tfl.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tfl.MaxPool2D(pool_size=(8,8), strides=(8,8), padding="SAME")(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tfl.Conv2D(16, 2, strides=(1,1), padding="SAME")(P1)
    ## RELU
    A2 = tfl.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tfl.MaxPool2D(pool_size=(4,4), strides=(4,4),padding="SAME")(A2)
    ## FLATTEN
    F = tfl.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tfl.Dense(6, activation="softmax")(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 64, 64, 8)         392       
    _________________________________________________________________
    re_lu_3 (ReLU)               (None, 64, 64, 8)         0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 8, 8, 16)          528       
    _________________________________________________________________
    re_lu_4 (ReLU)               (None, 8, 8, 16)          0         
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 2, 2, 16)          0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)

history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7992 - accuracy: 0.1343 - val_loss: 1.7881 - val_accuracy: 0.1583
    Epoch 2/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7880 - accuracy: 0.2148 - val_loss: 1.7813 - val_accuracy: 0.2583
    Epoch 3/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7824 - accuracy: 0.2417 - val_loss: 1.7783 - val_accuracy: 0.2083
    Epoch 4/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7764 - accuracy: 0.2833 - val_loss: 1.7742 - val_accuracy: 0.3000
    Epoch 5/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7695 - accuracy: 0.3074 - val_loss: 1.7688 - val_accuracy: 0.2583
    Epoch 6/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.7619 - accuracy: 0.3509 - val_loss: 1.7621 - val_accuracy: 0.3417
    Epoch 7/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7530 - accuracy: 0.3491 - val_loss: 1.7532 - val_accuracy: 0.3667
    Epoch 8/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7425 - accuracy: 0.3741 - val_loss: 1.7431 - val_accuracy: 0.4000
    Epoch 9/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7292 - accuracy: 0.4083 - val_loss: 1.7299 - val_accuracy: 0.4333
    Epoch 10/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7132 - accuracy: 0.4019 - val_loss: 1.7137 - val_accuracy: 0.4333
    Epoch 11/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.6922 - accuracy: 0.4130 - val_loss: 1.6947 - val_accuracy: 0.4417
    Epoch 12/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.6661 - accuracy: 0.4500 - val_loss: 1.6701 - val_accuracy: 0.4667
    Epoch 13/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.6333 - accuracy: 0.4870 - val_loss: 1.6439 - val_accuracy: 0.4833
    Epoch 14/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.5951 - accuracy: 0.5222 - val_loss: 1.6137 - val_accuracy: 0.4667
    Epoch 15/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.5494 - accuracy: 0.5565 - val_loss: 1.5741 - val_accuracy: 0.5417
    Epoch 16/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.4984 - accuracy: 0.5935 - val_loss: 1.5280 - val_accuracy: 0.5250
    Epoch 17/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.4468 - accuracy: 0.6009 - val_loss: 1.4816 - val_accuracy: 0.5250
    Epoch 18/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.3905 - accuracy: 0.6167 - val_loss: 1.4369 - val_accuracy: 0.5500
    Epoch 19/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.3350 - accuracy: 0.6259 - val_loss: 1.3886 - val_accuracy: 0.5667
    Epoch 20/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.2827 - accuracy: 0.6407 - val_loss: 1.3463 - val_accuracy: 0.5750
    Epoch 21/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.2319 - accuracy: 0.6546 - val_loss: 1.3010 - val_accuracy: 0.5917
    Epoch 22/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.1876 - accuracy: 0.6630 - val_loss: 1.2633 - val_accuracy: 0.5917
    Epoch 23/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.1427 - accuracy: 0.6759 - val_loss: 1.2236 - val_accuracy: 0.6083
    Epoch 24/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.1038 - accuracy: 0.6769 - val_loss: 1.1899 - val_accuracy: 0.6083
    Epoch 25/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.0669 - accuracy: 0.6898 - val_loss: 1.1548 - val_accuracy: 0.6250
    Epoch 26/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.0343 - accuracy: 0.6954 - val_loss: 1.1258 - val_accuracy: 0.6083
    Epoch 27/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9998 - accuracy: 0.7009 - val_loss: 1.0940 - val_accuracy: 0.6583
    Epoch 28/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9734 - accuracy: 0.7120 - val_loss: 1.0726 - val_accuracy: 0.6333
    Epoch 29/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9427 - accuracy: 0.7176 - val_loss: 1.0409 - val_accuracy: 0.6667
    Epoch 30/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9211 - accuracy: 0.7185 - val_loss: 1.0225 - val_accuracy: 0.6583
    Epoch 31/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.8935 - accuracy: 0.7176 - val_loss: 0.9932 - val_accuracy: 0.6833
    Epoch 32/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8743 - accuracy: 0.7241 - val_loss: 0.9778 - val_accuracy: 0.6750
    Epoch 33/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8500 - accuracy: 0.7315 - val_loss: 0.9526 - val_accuracy: 0.6917
    Epoch 34/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.8340 - accuracy: 0.7324 - val_loss: 0.9382 - val_accuracy: 0.6750
    Epoch 35/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8132 - accuracy: 0.7389 - val_loss: 0.9160 - val_accuracy: 0.7000
    Epoch 36/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.7980 - accuracy: 0.7417 - val_loss: 0.9030 - val_accuracy: 0.6833
    Epoch 37/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.7800 - accuracy: 0.7491 - val_loss: 0.8851 - val_accuracy: 0.7083
    Epoch 38/100
    17/17 [==============================] - 2s 117ms/step - loss: 0.7663 - accuracy: 0.7546 - val_loss: 0.8713 - val_accuracy: 0.6917
    Epoch 39/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7500 - accuracy: 0.7620 - val_loss: 0.8554 - val_accuracy: 0.7083
    Epoch 40/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7377 - accuracy: 0.7574 - val_loss: 0.8453 - val_accuracy: 0.7000
    Epoch 41/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.7233 - accuracy: 0.7750 - val_loss: 0.8339 - val_accuracy: 0.7083
    Epoch 42/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.7115 - accuracy: 0.7750 - val_loss: 0.8231 - val_accuracy: 0.7083
    Epoch 43/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.7001 - accuracy: 0.7796 - val_loss: 0.8124 - val_accuracy: 0.7167
    Epoch 44/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.6908 - accuracy: 0.7796 - val_loss: 0.8042 - val_accuracy: 0.7083
    Epoch 45/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6789 - accuracy: 0.7898 - val_loss: 0.7943 - val_accuracy: 0.7083
    Epoch 46/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.6688 - accuracy: 0.7870 - val_loss: 0.7855 - val_accuracy: 0.7250
    Epoch 47/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6600 - accuracy: 0.7926 - val_loss: 0.7782 - val_accuracy: 0.7250
    Epoch 48/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.6498 - accuracy: 0.7991 - val_loss: 0.7704 - val_accuracy: 0.7250
    Epoch 49/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.6396 - accuracy: 0.8009 - val_loss: 0.7631 - val_accuracy: 0.7167
    Epoch 50/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.6316 - accuracy: 0.7991 - val_loss: 0.7558 - val_accuracy: 0.7250
    Epoch 51/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6238 - accuracy: 0.8019 - val_loss: 0.7486 - val_accuracy: 0.7250
    Epoch 52/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.6151 - accuracy: 0.8056 - val_loss: 0.7419 - val_accuracy: 0.7333
    Epoch 53/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.6075 - accuracy: 0.8074 - val_loss: 0.7352 - val_accuracy: 0.7333
    Epoch 54/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6009 - accuracy: 0.8074 - val_loss: 0.7287 - val_accuracy: 0.7333
    Epoch 55/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5933 - accuracy: 0.8093 - val_loss: 0.7242 - val_accuracy: 0.7333
    Epoch 56/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5862 - accuracy: 0.8120 - val_loss: 0.7170 - val_accuracy: 0.7333
    Epoch 57/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5803 - accuracy: 0.8111 - val_loss: 0.7116 - val_accuracy: 0.7333
    Epoch 58/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5732 - accuracy: 0.8176 - val_loss: 0.7069 - val_accuracy: 0.7333
    Epoch 59/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5665 - accuracy: 0.8176 - val_loss: 0.6999 - val_accuracy: 0.7333
    Epoch 60/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5610 - accuracy: 0.8185 - val_loss: 0.6968 - val_accuracy: 0.7417
    Epoch 61/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5535 - accuracy: 0.8185 - val_loss: 0.6906 - val_accuracy: 0.7417
    Epoch 62/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5478 - accuracy: 0.8222 - val_loss: 0.6863 - val_accuracy: 0.7333
    Epoch 63/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5420 - accuracy: 0.8250 - val_loss: 0.6809 - val_accuracy: 0.7250
    Epoch 64/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5362 - accuracy: 0.8259 - val_loss: 0.6767 - val_accuracy: 0.7250
    Epoch 65/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5309 - accuracy: 0.8269 - val_loss: 0.6724 - val_accuracy: 0.7250
    Epoch 66/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5251 - accuracy: 0.8324 - val_loss: 0.6677 - val_accuracy: 0.7250
    Epoch 67/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5203 - accuracy: 0.8324 - val_loss: 0.6637 - val_accuracy: 0.7250
    Epoch 68/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5146 - accuracy: 0.8343 - val_loss: 0.6593 - val_accuracy: 0.7333
    Epoch 69/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5102 - accuracy: 0.8352 - val_loss: 0.6557 - val_accuracy: 0.7250
    Epoch 70/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5047 - accuracy: 0.8380 - val_loss: 0.6511 - val_accuracy: 0.7333
    Epoch 71/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5003 - accuracy: 0.8370 - val_loss: 0.6470 - val_accuracy: 0.7333
    Epoch 72/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4950 - accuracy: 0.8398 - val_loss: 0.6440 - val_accuracy: 0.7333
    Epoch 73/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4903 - accuracy: 0.8407 - val_loss: 0.6391 - val_accuracy: 0.7333
    Epoch 74/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4854 - accuracy: 0.8417 - val_loss: 0.6360 - val_accuracy: 0.7333
    Epoch 75/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4815 - accuracy: 0.8444 - val_loss: 0.6328 - val_accuracy: 0.7333
    Epoch 76/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4766 - accuracy: 0.8454 - val_loss: 0.6290 - val_accuracy: 0.7333
    Epoch 77/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4726 - accuracy: 0.8454 - val_loss: 0.6256 - val_accuracy: 0.7333
    Epoch 78/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.4680 - accuracy: 0.8481 - val_loss: 0.6219 - val_accuracy: 0.7250
    Epoch 79/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4640 - accuracy: 0.8491 - val_loss: 0.6192 - val_accuracy: 0.7250
    Epoch 80/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4593 - accuracy: 0.8509 - val_loss: 0.6163 - val_accuracy: 0.7250
    Epoch 81/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4549 - accuracy: 0.8500 - val_loss: 0.6118 - val_accuracy: 0.7250
    Epoch 82/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4507 - accuracy: 0.8565 - val_loss: 0.6089 - val_accuracy: 0.7250
    Epoch 83/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.4474 - accuracy: 0.8556 - val_loss: 0.6066 - val_accuracy: 0.7250
    Epoch 84/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4428 - accuracy: 0.8583 - val_loss: 0.6027 - val_accuracy: 0.7167
    Epoch 85/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4389 - accuracy: 0.8556 - val_loss: 0.5981 - val_accuracy: 0.7167
    Epoch 86/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4349 - accuracy: 0.8630 - val_loss: 0.5954 - val_accuracy: 0.7167
    Epoch 87/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4314 - accuracy: 0.8620 - val_loss: 0.5931 - val_accuracy: 0.7167
    Epoch 88/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4273 - accuracy: 0.8639 - val_loss: 0.5897 - val_accuracy: 0.7250
    Epoch 89/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4233 - accuracy: 0.8648 - val_loss: 0.5861 - val_accuracy: 0.7250
    Epoch 90/100
    17/17 [==============================] - 2s 113ms/step - loss: 0.4197 - accuracy: 0.8685 - val_loss: 0.5842 - val_accuracy: 0.7333
    Epoch 91/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.4159 - accuracy: 0.8704 - val_loss: 0.5794 - val_accuracy: 0.7417
    Epoch 92/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4121 - accuracy: 0.8704 - val_loss: 0.5769 - val_accuracy: 0.7417
    Epoch 93/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4093 - accuracy: 0.8731 - val_loss: 0.5764 - val_accuracy: 0.7417
    Epoch 94/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4052 - accuracy: 0.8722 - val_loss: 0.5723 - val_accuracy: 0.7500
    Epoch 95/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.4018 - accuracy: 0.8750 - val_loss: 0.5705 - val_accuracy: 0.7500
    Epoch 96/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.3985 - accuracy: 0.8769 - val_loss: 0.5678 - val_accuracy: 0.7500
    Epoch 97/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.3952 - accuracy: 0.8778 - val_loss: 0.5656 - val_accuracy: 0.7500
    Epoch 98/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.3919 - accuracy: 0.8787 - val_loss: 0.5618 - val_accuracy: 0.7583
    Epoch 99/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3884 - accuracy: 0.8787 - val_loss: 0.5597 - val_accuracy: 0.7583
    Epoch 100/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.3855 - accuracy: 0.8806 - val_loss: 0.5574 - val_accuracy: 0.7583


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [1.7992241382598877,
      1.7880146503448486,
      1.7824026346206665,
      1.7763657569885254,
      1.7694742679595947,
      1.7619385719299316,
      1.7529631853103638,
      1.7424933910369873,
      1.7291513681411743,
      1.7131800651550293,
      1.6922276020050049,
      1.6660795211791992,
      1.6333431005477905,
      1.5951355695724487,
      1.5494462251663208,
      1.4983614683151245,
      1.4467620849609375,
      1.3905093669891357,
      1.3350377082824707,
      1.2827115058898926,
      1.2319250106811523,
      1.187567114830017,
      1.1426621675491333,
      1.1037678718566895,
      1.066894769668579,
      1.0343338251113892,
      0.9997803568840027,
      0.9734225869178772,
      0.9426987171173096,
      0.9211150407791138,
      0.893471360206604,
      0.8742808103561401,
      0.8500359654426575,
      0.8340196013450623,
      0.8132370710372925,
      0.797957181930542,
      0.77997225522995,
      0.7663032412528992,
      0.7499666213989258,
      0.7376540303230286,
      0.7232588529586792,
      0.7115364074707031,
      0.7000799179077148,
      0.69078528881073,
      0.6789236664772034,
      0.6687679886817932,
      0.6600324511528015,
      0.6498334407806396,
      0.6396331191062927,
      0.6316466927528381,
      0.623787522315979,
      0.6150725483894348,
      0.6074920892715454,
      0.600877583026886,
      0.5932682156562805,
      0.5862027406692505,
      0.5802762508392334,
      0.5732230544090271,
      0.5665279030799866,
      0.5609622597694397,
      0.5534861087799072,
      0.5478466749191284,
      0.5420185327529907,
      0.5362350940704346,
      0.5308548212051392,
      0.5250979661941528,
      0.5203108191490173,
      0.5146010518074036,
      0.5102217197418213,
      0.5046712160110474,
      0.5003193020820618,
      0.49504441022872925,
      0.4903186857700348,
      0.48539161682128906,
      0.4815157353878021,
      0.4766291379928589,
      0.47255274653434753,
      0.4679624140262604,
      0.4640160799026489,
      0.45934081077575684,
      0.4548870921134949,
      0.4506534934043884,
      0.4473620057106018,
      0.442782998085022,
      0.4389062225818634,
      0.43488165736198425,
      0.4313916265964508,
      0.42727726697921753,
      0.4232839047908783,
      0.41971513628959656,
      0.4158731997013092,
      0.412137508392334,
      0.40930891036987305,
      0.4051778018474579,
      0.4018465578556061,
      0.3985363245010376,
      0.39517709612846375,
      0.391910195350647,
      0.3883870244026184,
      0.3855150640010834],
     'accuracy': [0.13425925374031067,
      0.21481481194496155,
      0.24166665971279144,
      0.28333333134651184,
      0.307407408952713,
      0.35092592239379883,
      0.3490740656852722,
      0.3740740716457367,
      0.40833333134651184,
      0.4018518626689911,
      0.41296297311782837,
      0.44999998807907104,
      0.4870370328426361,
      0.5222222208976746,
      0.5564814805984497,
      0.5935184955596924,
      0.6009259223937988,
      0.6166666746139526,
      0.6259258985519409,
      0.6407407522201538,
      0.654629647731781,
      0.6629629731178284,
      0.6759259104728699,
      0.6768518686294556,
      0.6898148059844971,
      0.6953703761100769,
      0.7009259462356567,
      0.7120370268821716,
      0.7175925970077515,
      0.7185184955596924,
      0.7175925970077515,
      0.7240740656852722,
      0.7314814925193787,
      0.7324073910713196,
      0.7388888597488403,
      0.7416666746139526,
      0.7490741014480591,
      0.7546296119689941,
      0.7620370388031006,
      0.7574074268341064,
      0.7749999761581421,
      0.7749999761581421,
      0.779629647731781,
      0.779629647731781,
      0.789814829826355,
      0.7870370149612427,
      0.7925925850868225,
      0.7990740537643433,
      0.8009259104728699,
      0.7990740537643433,
      0.8018518686294556,
      0.8055555820465088,
      0.8074073791503906,
      0.8074073791503906,
      0.8092592358589172,
      0.8120370507240295,
      0.8111110925674438,
      0.8175926208496094,
      0.8175926208496094,
      0.8185185194015503,
      0.8185185194015503,
      0.8222222328186035,
      0.824999988079071,
      0.8259259462356567,
      0.8268518447875977,
      0.8324074149131775,
      0.8324074149131775,
      0.8342592716217041,
      0.835185170173645,
      0.8379629850387573,
      0.8370370268821716,
      0.8398148417472839,
      0.8407407402992249,
      0.8416666388511658,
      0.8444444537162781,
      0.845370352268219,
      0.845370352268219,
      0.8481481671333313,
      0.8490740656852722,
      0.8509259223937988,
      0.8500000238418579,
      0.8564814925193787,
      0.855555534362793,
      0.8583333492279053,
      0.855555534362793,
      0.8629629611968994,
      0.8620370626449585,
      0.8638888597488403,
      0.864814817905426,
      0.8685185313224792,
      0.8703703880310059,
      0.8703703880310059,
      0.8731481432914734,
      0.8722222447395325,
      0.875,
      0.8768518567085266,
      0.8777777552604675,
      0.8787037134170532,
      0.8787037134170532,
      0.8805555701255798],
     'val_loss': [1.788120150566101,
      1.7813236713409424,
      1.7782987356185913,
      1.7741551399230957,
      1.7688246965408325,
      1.762108325958252,
      1.7532109022140503,
      1.743101954460144,
      1.7298680543899536,
      1.7136504650115967,
      1.6947156190872192,
      1.6701380014419556,
      1.6438933610916138,
      1.6137351989746094,
      1.5740580558776855,
      1.5279947519302368,
      1.4815845489501953,
      1.4369488954544067,
      1.3885716199874878,
      1.3462510108947754,
      1.3009836673736572,
      1.2633134126663208,
      1.2235982418060303,
      1.1898661851882935,
      1.1548469066619873,
      1.125799536705017,
      1.0940206050872803,
      1.0726377964019775,
      1.0409142971038818,
      1.0225120782852173,
      0.9932495355606079,
      0.9777698516845703,
      0.9526475071907043,
      0.9381572604179382,
      0.9160484075546265,
      0.9029920697212219,
      0.8851423859596252,
      0.8712813258171082,
      0.8553799986839294,
      0.8453130125999451,
      0.8339168429374695,
      0.8231012225151062,
      0.8123864531517029,
      0.8042476773262024,
      0.7943283915519714,
      0.7855059504508972,
      0.7782176733016968,
      0.7703738808631897,
      0.7630566954612732,
      0.7557861804962158,
      0.7485539317131042,
      0.741930365562439,
      0.7352241277694702,
      0.7286636233329773,
      0.7242158055305481,
      0.7170084118843079,
      0.7115857601165771,
      0.7068690061569214,
      0.6998856067657471,
      0.6967511773109436,
      0.6906266808509827,
      0.6862800121307373,
      0.6809337735176086,
      0.676744818687439,
      0.6724124550819397,
      0.667730987071991,
      0.6636728644371033,
      0.6592533588409424,
      0.6556581258773804,
      0.6510794162750244,
      0.6470034122467041,
      0.6440392136573792,
      0.6391295790672302,
      0.6359968781471252,
      0.6328279376029968,
      0.6290140748023987,
      0.62559574842453,
      0.6218599081039429,
      0.619182825088501,
      0.6162673234939575,
      0.611807107925415,
      0.6089489459991455,
      0.6066218614578247,
      0.602698028087616,
      0.5980885624885559,
      0.5953559875488281,
      0.5930554866790771,
      0.5897226929664612,
      0.5860980153083801,
      0.5841519832611084,
      0.5794268250465393,
      0.5769485235214233,
      0.5763704180717468,
      0.5723090767860413,
      0.5705164074897766,
      0.5677958726882935,
      0.5656316876411438,
      0.5618104338645935,
      0.5596863031387329,
      0.5573686361312866],
     'val_accuracy': [0.15833333134651184,
      0.25833332538604736,
      0.2083333283662796,
      0.30000001192092896,
      0.25833332538604736,
      0.34166666865348816,
      0.36666667461395264,
      0.4000000059604645,
      0.4333333373069763,
      0.4333333373069763,
      0.4416666626930237,
      0.46666666865348816,
      0.4833333194255829,
      0.46666666865348816,
      0.5416666865348816,
      0.5249999761581421,
      0.5249999761581421,
      0.550000011920929,
      0.5666666626930237,
      0.574999988079071,
      0.5916666388511658,
      0.5916666388511658,
      0.6083333492279053,
      0.6083333492279053,
      0.625,
      0.6083333492279053,
      0.6583333611488342,
      0.6333333253860474,
      0.6666666865348816,
      0.6583333611488342,
      0.6833333373069763,
      0.675000011920929,
      0.6916666626930237,
      0.675000011920929,
      0.699999988079071,
      0.6833333373069763,
      0.7083333134651184,
      0.6916666626930237,
      0.7083333134651184,
      0.699999988079071,
      0.7083333134651184,
      0.7083333134651184,
      0.7166666388511658,
      0.7083333134651184,
      0.7083333134651184,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7166666388511658,
      0.7250000238418579,
      0.7250000238418579,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7416666746139526,
      0.7416666746139526,
      0.7333333492279053,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7333333492279053,
      0.7250000238418579,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7250000238418579,
      0.7250000238418579,
      0.7333333492279053,
      0.7416666746139526,
      0.7416666746139526,
      0.7416666746139526,
      0.75,
      0.75,
      0.75,
      0.75,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![png](output_41_1.png)



![png](output_41_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
