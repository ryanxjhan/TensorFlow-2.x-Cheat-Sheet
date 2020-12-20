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
* [Callbacks](#callbacks)
* [Architectures](#architectures)
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
| `model.fit(x, y, epoch)`                                     | Trains the model for a fixed number of epochs (itera­tions on a dataset). |
| `model.fit_generator(train_generator, steps_per_epoch, epochs, validation_data, validation_steps)` | Fits the model on data yielded batch-­by-­batch by a Python generator. |
| `model.ev­alu­ate(x, y)`                                     | Returns the loss value & metrics values for the model in test mode. |
| `model.pr­edi­ct(x)`                                         | Generates output predic­tions for the input samples.         |
| `model.su­mma­ry()`                                          | Prints a string summary of the network.                      |
| `model.save(path)`                                           | Saves a model as a TensorFlow SavedModel or HDF5 file.       |
| `model.stop_training`                                        | Stops training.                                              |

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

| Method             | Usage                                                        |
| ------------------ | ------------------------------------------------------------ |
| ImageDataGenerator | Generate batches of tensor image data with real-time data augmen­tation. |

<a name="callbacks"/>

### Callbacks

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

### Architectures

| Name    | Usage |
| ------- | ----- |
| ConvNet |       |

<a name="aarchitectures"/>

### Advanced Architectures

| Name | Usage |
| ---- | ----- |
|      |       |

