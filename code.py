# Importing the required libraries
import cv2
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tqdm import tqdm
import numpy as np
from random import shuffle
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1


TRAIN_DIR = 'C:/Users/Sully/OneDrive/HonorsProgramming/train/train' #File path to training images
TEST_DIR = 'C:/Users/Sully/OneDrive/HonorsProgramming/test1/test1' #File path to test images
IMG_SIZE = 50 #Image size restriction
class EarlyStoppingAtValAcc(tf.keras.callbacks.Callback): #Class to stop early once validation accuracy is reached
    def __init__(self, accuracy_threshold=0.95): #Validation accuracy threshold is 95%
        super(EarlyStoppingAtValAcc, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None): #If reached, print percentage and stop training
        val_accuracy = logs.get("val_accuracy")
        if val_accuracy is not None and val_accuracy >= self.accuracy_threshold:
            print(f"\nValidation accuracy reached {self.accuracy_threshold * 100}%. Stopping training!")
            self.model.stop_training = True
#Code to process images into dataset inspired from GeeksForGeeks
# Function to extract class label from filename
def label_img(img):
    word_label = img.split('.')[-3] #Just get the three letters "cat" or "dog"
    # DIY One hot encoder
    if word_label == 'cat': return [1, 0] #Label cat as [1, 0]
    elif word_label == 'dog': return [0, 1] #Label dog as [0, 1]
 
'''Creating the training data'''
def create_train_data():
    # Creating an empty list where we should store the training data after a little preprocessing of the data
    training_data = []
    # tqdm is only used for interactive loading(gives the cool loading bar)
    # loading the training data
    for img in tqdm(os.listdir(TRAIN_DIR)): #For each image in the training
 
        # labeling the images
        label = label_img(img)
 
        path = os.path.join(TRAIN_DIR, img)
       
        # loading the image from the path and then converting them into grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
 
        # resizing the image for processing them in the covnet
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
 
        # final step-forming the training data list with numpy array of the images
        training_data.append([np.array(img), np.array(label)])
 
    # shuffling of the training data to preserve the random state of our data
    shuffle(training_data)
 
    # saving our trained data for further uses if required
    return training_data
 
'''Processing the given test data'''
# Same as processing the training data but we dont have to label it.
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    return testing_data
 
#Running the training and the testing in the dataset for our model'''
train_data = create_train_data() #Create the training dataset
test_data = process_test_data() #Create the validation dataset
#Create the CNN sequential model
model = Sequential([
    #Conv2d = 2d convolution layer. This one has 32 filers and a filter size of 3x3
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(), #Batch normalization to make the training faster and improve performance
    MaxPooling2D((2, 2)), #Reduces spatial dimensions of features by selecting the maximum value in a window - helps suppress noise
    Dropout(0.25),  #Dropout after pooling layer to prevent overfitting
    #Convolution layer with 64 filters of 3x3
    Conv2D(64, (3, 3), activation='relu'), #Kernel regularizer could be used to penalize large weights and reduce overfitting
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  


    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  


    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),


    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Increased dropout rate to prevent overfitting and reduce complexity
    Dense(2, activation='softmax')
])


model.compile(optimizer='adam',#Uses the adam optimizer to optimize weights
              loss='categorical_crossentropy',#Loss function used for multi-class classification
              metrics=['accuracy'])#Provides additional accuracy information


train = train_data[:-12500] #Set both training and testing set with an equal number of pictures
test = train_data[-12500:]
# Convert numpy arrays to TensorFlow Dataset objects
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array([i[1] for i in train])
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = np.array([i[1] for i in test])


train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
validation_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))#change to test_y


# Batch and prefetch the datasets for improved memory efficiency, generalization, and convergence
BATCH_SIZE = 32
train1_dataset = train_dataset.shuffle(len(X)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation1_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


#If it crosses validation threshold, stop early to prevent overfitting
early_stopping_at_val_acc = EarlyStoppingAtValAcc(accuracy_threshold=0.95)
# Train the model
history = model.fit(
    train1_dataset, #Dataset used to train
    epochs=50, #Number of run throughs
    validation_data=validation1_dataset, #Data used to validate
    callbacks=[early_stopping_at_val_acc], #Stop early
    verbose=1 #Information displayed
)


# Save the model
model.save('dog_cat_classifier.h5')


# Evaluate the model on the validation data
model.summary()
