import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model
# from keras.utils import  plot_model
# model = VGG16()

vgg = VGG16(include_top=False,weights='imagenet',input_shape=(100,100,3))

for layer in vgg.layers:
   layer.trainable = False
#Now we will be training only the classifiers (FC layers)

dataset = 'Food-5K'
categories = os.path.join(dataset, 'training')
print('number of categories:',len(categories))

x = Flatten()(vgg.output)
prediction = Dense(len(dataset),
                   activation='softmax')(x)
model = Model(inputs=vgg.input,
              outputs=prediction)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
