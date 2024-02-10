### Add lines to import modules as needed
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
#from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
## 

def build_model1():
  model = keras.Sequential([
    layers.Conv2D(32, (3,3), strides=(2, 2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), strides=(2, 2), padding='same', activation='relu'),
    layers.BatchNormalization(),
  ])
    # Add four more pairs of Conv2D+Batchnorm with no striding option (so stride defaults to 1)
  for _ in range(4):
      model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
      model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
  model.add(layers.Flatten(input_shape=(32, 32, 3)))
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(10, activation='softmax'))
  
  return model

def build_model2():
  model = keras.Sequential([
    layers.Conv2D(32, (3,3), strides=(2, 2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, (3,3), strides=(2, 2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, (3,3), strides=(2, 2), padding='same', activation='relu'),
    layers.BatchNormalization(),
  ])
    # Add four more pairs of Conv2D+Batchnorm with no striding option (so stride defaults to 1)
  for _ in range(4):
      model.add(layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'))
      model.add(layers.BatchNormalization())

  model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
  model.add(layers.Flatten(input_shape=(32, 32, 3)))
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(10, activation='softmax'))

  return model

def build_model3():
  inputs = tf.keras.Input(shape=(32, 32, 3))

  # Define the layers
  x = layers.Conv2D(32, (3, 3), strides=2, padding='same')(inputs)
  residual = x
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.5)(x)

  x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.5)(x)

  x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.5)(x)

  residual = layers.Conv2D(128, (1, 1), strides=4, padding='same')(residual)  
  x = layers.add([x, residual])
  residual = x

  x = layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.5)(x)

  x = layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.5)(x)

  residual = layers.Conv2D(128, (1, 1), strides=4, padding='same')(residual)  
  x = layers.add([x, residual])
  residual = x


  x = layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.5)(x)

  x = layers.Conv2D(128, (3, 3), strides=1, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.5)(x)

  residual = layers.Conv2D(128, (1, 1), strides=4, padding='same')(residual)  
  x = layers.add([x, residual])
  residual = x

  x = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(128)(x)
  x = layers.BatchNormalization()(x)
  outputs = layers.Dense(10)(x)

  # Create the model
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model

def build_model50k():
  model = keras.Sequential([
    layers.SeparableConv2D(32, (3,3), strides=(2, 2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
  ])

  model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
  model.add(layers.Flatten(input_shape=(32, 32, 3)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(10, activation='softmax'))

  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':
  #print("fine")
  ########################################
  ## Add code here to Load the CIFAR10 data set
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)

  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model1.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2)
  


  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
                                             
  model2.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

  

  ### Repeat for model 3 and your best sub-50k params model
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  model3 = build_model3()

  model3.compile(optimizer='adam',
            loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

  model3.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), validation_split=0.2)


  model50k = build_model50k()

  model50k.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
  # Train model
  checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5',
                                             monitor='val_loss',
                                             save_best_only=True)
                                             
  model50k.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2, callbacks=[checkpoint])

  
  
