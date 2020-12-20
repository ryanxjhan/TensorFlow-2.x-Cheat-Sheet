# TensorFlow-Cheat-Sheet
### Table of Contents

* [Layers](#layers)
* [Models](#models)
* [Activation Functions](#activation)
* [Optimizers](#optimizers)
* Loss Functions
* Hyperparameters
* Preprocessing
* Callbacks
* Architectures
* Advanced Architectures



<a name="headers"/>

### Layers

| Layers       | Code                                                         | Usage                                                        |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Dense        | `tf.keras.layers.Dense(units, activation, input_shape)`      | Dense layer is the regular deeply connected neural network layer. It is most common and frequently used layer. |
| Flatten      | `tf.keras.layers.Flatten()`                                  | Flatten is used to flatten the input. For example, if flatten is applied to layer having input shape as (batch­_size, 2,2), then the output shape of the layer will be (batch­_size, 4) |
| Conv2D       | `tf.keras.layers.Conv2D(filters, kernel_size, activation, input_shape)` | Filter for two-di­men­sional image data                      |
| MaxPooling2D | `tf.keras.layers.Conv2D(pool_size)`                          | Max pooling for two-di­men­sional image data                 |



