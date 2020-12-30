import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


batch_size = 32
epochs = 2
IMAGE_SIZE = [100,100]

dataset = 'dataset'
categories = os.listdir(os.path.join(dataset, 'training'))
valid_path = os.path.join(dataset,'evaluation')
train_path = os.path.join(dataset,'training')
print('number of categories:',len(categories))
print(categories)


vgg = VGG16(include_top=False,
            weights='imagenet',
            input_shape=(100,100,3))
for layer in vgg.layers:
   layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(len(categories),
                   activation='softmax')(x)
model = Model(inputs=vgg.input,
              outputs=prediction)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


gen=ImageDataGenerator(rotation_range=20,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       shear_range=0.1,
                       zoom_range=0.2,
                       horizontal_flip=True,
                       vertical_flip=True,
                       preprocessing_function=preprocess_input)

train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)

r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs
)
