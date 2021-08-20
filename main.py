#Dataset was converted to CSV with this script: https://pjreddie.com/projects/mnist-in-csv/

import os
for dirname, _, filenames in os.walk('fashion'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as kr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
#load CSV files as Pandas Dataframe
train_data = pd.read_csv("fashion-mnist_train.csv")
test_data = pd.read_csv("fashion-mnist_test.csv")
#make train data and test data
x_train = train_data.drop(["label"], axis =1)
y_train = train_data["label"]
x_test = test_data.drop(["label"], axis = 1)
y_test = test_data["label"]
#change train data and test data into float32 and divide by 255
#That normalizes data from 1 to 0.
x_train = x_train.astype('float32')/255
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')/255
#change data into numpy array
x_train= x_train.to_numpy()
y_train= y_train.to_numpy()
x_test = x_test.to_numpy()
#reshape train data and test data
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
#convert y_train into 10 categories
y_train = kr.utils.to_categorical(y_train, 10)

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title("No." + str(i))
    plt.imshow(x_train[i].reshape(28,28), cmap='Greys')

#Build 1st model.
# Here we are doing conv2d layer with input image size of 28,28,1.
# Then 32 hidden neurons
# Then Maxpool2d layer for finding max value from strides of size(2,2)
# Flattening it
# Then adding 128 hidden neurons
# Then adding 10 hidden neurons
# Then compiling using accuracy metrics with activation categorical_crossentropy for multiclass labels and finally using optimizer
# as Adam for minimising loss it is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments
# This method is "computationally efficient, has little memory requirement, invariant to diagonal rescaling of gradients, and is well suited for problems 
# that are large in terms of data/parameters"
model = Sequential()
model.add(Conv2D(32,3, activation='relu',padding='same', input_shape=(28, 28,1)))
model.add(Conv2D(32,3,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

result1=model.fit(x_train, y_train,batch_size=128, epochs=20)

#Build 2nd model.
# Here we are doing conv2d layer with input image size of 28,28,1.
# Then 32 hidden neurons
# Then Maxpool2d layer for finding max value from strides of size(2,2)
# Flattening it
# Then adding 128 hidden neurons
# Then adding 10 hidden neurons
# Then compiling using accuracy metrics with activation categorical_crossentropy for multiclass labels and finally using optimizer
# as RMSprop algorithm for minimising loss which is to Maintain a moving (discounted) average of the square of gradients and to divide the gradient by the root of this average
# It uses plain momentum, not Nesterov momentum.
model = Sequential()
model.add(Conv2D(32,3, activation='relu',padding='same', input_shape=(28, 28,1)))
model.add(Conv2D(32,3,padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#model fitting
#We call fit(), which will train the model by slicing the data into "batches" of size batch_size, 
# and repeatedly iterating over the entire dataset for a given number of epochs.
result2=model.fit(x_train, y_train,batch_size=128, epochs=20)

#Evaluate the two models by using two metrics, loss and accuracy.
metrics = ['loss', 'accuracy']
#show the evaluation result by using matoplot.
plt.figure(figsize=(10, 5))
#Use "For Loop".
for i in range(len(metrics)):
    metric = metrics[i]
    #set subplots to show the result
    plt.subplot(1, 2, i+1)
    #Titles of subplots are "loss" and "accuracy"
    #The returned history object holds a record of the loss values and metric values during training     
    plt.title(metric)
    plt_result1 = result1.history[metric]
    plt_result2 = result2.history[metric]

    #plot them all
    plt.plot(plt_result1, label='1st model')
    plt.plot(plt_result2, label='2nd model')
    plt.legend()
plt.show()

plt.imshow(x_test[[98]].reshape(28,28),cmap='Greys')

prediction=model.predict(x_test[[98]])
prediction

names=["T-shirt or Top","Pants","Pullover","Dress","Coat","Sandal","Shirt","Shoes","Bag","Boot"]
#Preparation for this predction.
list1=[]
[list1.append(i) for i in range(26)]
list2=[]
[list2.append(i) for i in names]
dic = dict(zip(list1, list2))
#Here we check the result.
print("The answer is",dic[np.argmax(prediction)],"!")

#Predicts the whole test data!
predictions = model.predict(x_test)
results = np.argmax(predictions,axis=1)
results = pd.Series(results, name="Label")
results.tail()
