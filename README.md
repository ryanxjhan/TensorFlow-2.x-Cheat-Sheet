# TensorFlow Cheat Sheet

<a href="https://www.linkedin.com/in/ryanxjhan/" target="_blank">LinkedIn</a>

### Table of Contents

* [Layers](#layers)
* [Models](#models)
* [Activation Functions](#activations)
* [Optimizers](#optimizers)
* [Loss Functions](#loss)
* [Hyperparameters](#parameters)
* [Preprocessing](#preprocessing)
* [I/O](#io)
* [Plotting](#plotting)
* [Callbacks](#callbacks)
* [Common Architectures](#architectures)
* [Advanced Architectures](#aarchitectuers)



<a name="headers"/>

### Layers

| Layers       | Code                                                         | Usage                                                        |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Dense        | `tf.keras.layers.Dense(units, activation, input_shape)`      | Dense layer is the regular deeply connected neural network layer. It is most common and frequently used layer. |
| Flatten      | `tf.keras.layers.Flatten()`                                  | Flatten is used to flatten the input. For example, if flatten is applied to layer having input shape as (batch­_size, 2,2), then the output shape of the layer will be (batch­_size, 4). |
| Conv2D       | `tf.keras.layers.Conv2D(filters, kernel_size, activation, input_shape)` | Filter for two-di­men­sional image data.                     |
| MaxPooling2D | `tf.keras.layers.MaxPool2D(pool_size)`                          | Max pooling for two-di­men­sional image data.                |

<a name="models"/>

### Models

| Code                                                         | Usage                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `model = tf.ker­as.S­eq­uen­tia­l(l­ayers)`                  | Sequential groups a linear stack of layers into a tf.ker­as.M­odel. |
| `model.co­mpi­le(­opt­imizer, loss, metrics)`                | Configures the model for training.                           |
| `history = model.fit(x, y, epoch)`                                     | Trains the model for a fixed number of epochs (itera­tions on a dataset). |
| `history = model.fit_generator(train_generator, steps_per_epoch, epochs, validation_data, validation_steps)` | Fits the model on data yielded batch-­by-­batch by a Python generator. |
| `model.ev­alu­ate(x, y)`                                     | Returns the loss value & metrics values for the model in test mode. |
| `model.pr­edi­ct(x)`                                         | Generates output predic­tions for the input samples.         |
| `model.su­mma­ry()`                                          | Prints a string summary of the network.                      |
| `model.save(path)`                                           | Saves a model as a TensorFlow SavedModel or HDF5 file.       |
| `model.stop_training`                                        | Stops training when true.                                    |

<a name="activations"/>

### Activation Functions

| Name    | Usage                                     |
| ------- | ----------------------------------------- |
| relu    | the default activation for hidden layers. |
| sigmoid | binary classi­fic­ation.                  |
| tanh    | faster conver­gence than sigmoid.         |
| softmax | multiclass classi­fic­ation.              |

<a name="optimizers"/>

### Optimizers

| Name     | Usage                                                        |
| -------- | ------------------------------------------------------------ |
| Adam     | Adam combines the good properties of Adadelta and RMSprop and hence tend to do better for most of the problems. |
| SGD      | Stochastic gradient descent is very basic and works well for shallow networks. |
| AdaGrad  | Adagrad can be useful for sparse data such as tf-idf.        |
| AdaDelta | Extension of AdaGrad which tends to remove the decaying learning Rate problem of it. |
| RMSprop  | Very similar to AdaDelta.                                    |

<a name="loss"/>

### Loss Functions

| Name                          | Usage                                                        |
| ----------------------------- | ------------------------------------------------------------ |
| MeanSquaredError              | Default loss function for regression problems.               |
| MeanSquaredLogarithmicError   | For regression problems with large spread.                   |
| MeanAbsoluteError             | More robust to outliers.                                     |
| BinaryCrossEntropy            | Default loss function to use for binary classi­fic­ation problems. |
| Hinge                         | It is intended for use with binary classi­fic­ation where the target values are in the set {-1, 1}. |
| SquaredHinge                  | If using a hinge loss does result in better perfor­mance on a given binary classi­fic­ation problem, is likely that a squared hinge loss may be approp­riate. |
| CategoricalCrossEntropy       | Default loss function to use for multi-­class classi­fic­ation problems. |
| SparseCategoricalCrossEntropy | Sparse cross-­entropy addresses the one hot encoding frustr­ation by performing the same cross-­entropy calcul­ation of error, without requiring that the target variable be one hot encoded prior to training. |
| KLD                           | KL divergence loss function is more commonly used when using models that learn to approx­imate a more complex function than simply multi-­class classi­fic­ation, such as in the case of an autoen­coder used for learning a dense feature repres­ent­ation under a model that must recons­truct the original input. |

<a name="parameters"/>

### Hyperparameters

| Parameter       | Tips                                                         |
| --------------- | ------------------------------------------------------------ |
| Hidden Neurons  | The number of hidden neurons should be between the size of the input layer and the size of the output layer, and 2/3 the size of the input layer, plus the size of the output layer. |
| Learning Rate   | [0.1, 0.01, 0.001, 0.0001]                                   |
| Momentum        | [0.5, 0.9, 0.99]                                             |
| Batch Size      | Small values give a learning process that converges quickly at the cost of noise in the training process. Large values give a learning process that converges slowly with accurate estimates of the error gradient. The typical sizes are [32, 64, 128, 256, 512] |
| Conv2D Filters  | Earlier 2D convol­utional layers, closer to the input, learn less filters, while later convol­utional layers, closer to the output, learn more filters. The number of filters you select should depend on the complexity of your dataset and the depth of your neural network. A common setting to start with is [32, 64, 128] for three layers, and if there are more layers, increasing to [256, 512, 1024], etc. |
| Kernel Size     | (3, 3)                                                       |
| Pool Size       | (2, 2)                                                       |
| Steps per Epoch | train_­length // batch_size                                  |
| Epoch           | Use callbacks                                                |

<a name="preprocessing"/>

### Preprocessing

**ImageDataGenerator**

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/tmp/validation-horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
```

<a name="io"/>

### I/O

**zip files** 

```python
import os
import zipfile

local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
local_zip = '/tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()
```

<a name="plotting"/>

### Plotting

**Accuracy and Loss**

```python
import matplotlib.pyplot as plt
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs, acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs, loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss')
```

**Intermediate Representations**

```python
import numpy as np
import random
from   tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]

#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

img_path = random.choice(cat_img_files + dog_img_files)
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255.0

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# -----------------------------------------------------------------------
# Now let's display our representations
# -----------------------------------------------------------------------
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    
    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    
    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------

    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
```



<a name="callbacks"/>

### Callbacks

**on_epoch_end**

```python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```



<a name="architectures"/>

### Common Architectures

| Name    | Layers |
| ------- | ----- |
| ConvNet | Convolution + ReLU, MaxPooling, Flatten, Dense |

<a name="aarchitectures"/>

### Advanced Architectures

| Name | Usage |
| ---- | ----- |
|      |       |

