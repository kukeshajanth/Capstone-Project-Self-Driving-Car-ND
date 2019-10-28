import numpy as np
import matplotlib.pyplot as plt
import os, time
import random
import glob

import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import  train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model


def build_model():
    
    num_classes = 3
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same',
                     input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    model.summary()
    
    return model
    
    
def train_model(data, model, hyper_params, model_names):   
    
    
    #Unpack the data
    x_train, y_train = data[0],data[1]
    
    # Unpack the hyper-parameters
    lr = hyper_params[0]
    epochs = hyper_params[1]
    batch_size = hyper_params[2]

    # Upack the model names
    model_best_name = model_names[0]
    model_final_name = model_names[1]

    
 
    # Select the optimizer and set the learning rate
    opt=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, \
                              epsilon=1e-08, decay=0.0)
    # Set the loss function
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
    # Set the check point
    checkpoint = ModelCheckpoint(filepath = model_best_name, verbose=0, save_best_only=True)
    
    # Create tensorboard  
    tensorboard =TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, \
                             write_images =True)
    start_time = time.time()
    # Fit the model and record the loss
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_split=0.1, shuffle=True).history
    end_time = time.time()
    # Save the model at the end of training
    model.save(model_final_name)
    print('')
    print('Training time (seconds): ', end_time - start_time)
    print('Final trained model saved at %s ' % model_final_name)
    return  history, model
  

def randombrightness(image):
    aug_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)  #randomising brighness value
    brightness = .25 + np.random.uniform()
    aug_img[::2] =  aug_img[::2] * brightness
    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_HSV2RGB)
    
    return aug_img

    


if __name__=='__main__':

    # The paths to the image dataset and trained models
    base_image_path = '/content/drive/My Drive/Colab Notebooks/training_images/'
    light_colors = ["red", "green", "yellow"]
    saved_model_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(saved_model_dir):
           os.makedirs(saved_model_dir)
    
    

    # Resize the images to the size of 32X32 and save the in the list named data
    data=[]
    color_counts = np.zeros(3)
    for color in light_colors:
        for img_file in glob.glob(os.path.join(base_image_path, color, "*")):
            img = cv2.imread(img_file)
            if not img is None:
                img = cv2.resize(img, (32, 32))
                label = light_colors.index(color)
                assert(img.shape == (32, 32, 3))
                data.append((img, label, img_file))
                color_counts[light_colors.index(color)]+=1
                             
    #Divide the data into training, validation, and test sets.                         
    random.shuffle(data)
    x, y, files = [], [],[]
    for sample in data:
        x.append(sample[0])
        x.append(randombrightness(sample[0]))
        y.append(sample[1])
        y.append(sample[1])
        files.append(sample[2])
        files.append(sample[2])
       
    
    # Data preprocessing: converting to numpy array, normalizing data, and creating
    # one-hot labels.
    X_train=np.array(x)
    
    X_train = X_train.astype('float32')
    
    encoder = LabelBinarizer()
    encoder.fit(y)
    X_train /= 255
    
    y_train_onehot = encoder.transform(y)
    
    data_train=[X_train, y_train_onehot]
    

    batch_size = 32 # batch size
    lr = -4  # learning rate
    epochs = 140  # number of training epochs
    hyper_params = [pow(10,lr), epochs, batch_size]
    
    
    model_best_name='model_b_lr-'+str(abs(lr))+'_ep'+str(epochs)+'_ba'+str(batch_size)\
                   +'.h5'
    model_best_name = os.path.join(saved_model_dir, model_best_name)             
    model_final_name='model_f_lr-'+str(abs(lr))+'_ep'+str(epochs)+'_ba'+str(batch_size)\
                   +'.h5'
    model_final_name = os.path.join(saved_model_dir, model_final_name)               
    model_names = [model_best_name, model_final_name]
    

    # Build model
    model = build_model()
    
    
    
    # Training
    history, model = train_model(data_train, model, hyper_params, model_names)

    # Plot the loss
    plt.plot(history['loss'],linewidth=3.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


    