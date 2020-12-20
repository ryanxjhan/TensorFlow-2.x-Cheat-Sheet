# TensorFlow-Cheat-Sheet
### Table of Contents

* [Layers](#layers)
* [Models](#models)
* [Activation Functions](#activation)
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
| MaxPooling2D | `tf.keras.layers.Conv2D(pool_size)`                          | Max pooling for two-di­men­sional image data.                |

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



