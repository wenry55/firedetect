import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import shutil

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout

import cv2
import numpy as np
from PIL import Image
import os
import numpy as np


#TRAINING_DIR = "/content/FIRE-SMOKE-DATASET/Train"
TRAINING_DIR = "./FIRE-SMOKE-DATASET/Train"

training_datagen = ImageDataGenerator(rescale=1./255,
																			zoom_range=0.15,
																			horizontal_flip=True,
                                      fill_mode='nearest')

#VALIDATION_DIR = "/content/FIRE-SMOKE-DATASET/Test"
VALIDATION_DIR = "./FIRE-SMOKE-DATASET/Test"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(224,224),
	shuffle = True,
	class_mode='categorical',
  batch_size = 128
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(224,224),
	class_mode='categorical',
	shuffle = True,
  batch_size= 14
)

input_tensor = Input(shape=(224, 224, 3))

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)
#predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#for layer in base_model.layers:
#    layer.trainable = False

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# from keras.preprocessing import image
#Load the saved model
model = tf.keras.models.load_model('trained_fs')

rootd = 'BK'
flists = os.listdir('BK')
for imgp in flists:
    path = 'BK/' + imgp
    #print(path)
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) /255
    classes = model.predict(x)
    print(imgp, np.argmax(classes[0])==0, max(classes[0]), classes)

video = cv2.VideoCapture(0)
while True:
        _, frame = video.read()
#Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
#Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224,224))
        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)# [0]
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        print(prediction)
        #if prediction is 0, which means there is fire in the frame.
        #if prediction == 0:
        #        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #        print(probabilities[prediction])
#cv2.imshow("Capturing", frame)
        sleep(3)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break

video.release()
cv2.destroyAllWindows()
