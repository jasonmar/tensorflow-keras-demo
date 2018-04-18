# Tensorflow and Keras Examples

## Description

This project demonstrates commonly used Tensorflow and Keras functionality in addition to some less commonly used TensorFlow Java library which demonstrates interoperability with Scala.

Model definition, training, monitoring with TF-Serving, save/export, freeze, optimize, load and predict examples are provided.


## Motivation

The goal of this project was to demonstrate the full lifecycle of a TensorFlow model and compare the usability of Keras and TensorFlow APIs.


## Features

* [Define neural network layers using Keras API](example/build_train_save_predict.py)
* [Randomly generated input data](example/sequence_demo.py)
* [Compile and Train using Keras API](nn/model/testmodel.py#151)
* [Export model to Keras JSON format](example/build_train_save_predict.py#92)
* [Export model to TensorFlow SavedModelBundle ProtoBuf format](nn/model/testmodel.py#176)
* [Optimize model for inference (strip training nodes)](nn/model/testmodel.py#262)
* [Set Callback on model training to output logs for TensorBoard visualization](example/write_summary.py)
* [Scala example usage of TensorFlow Java library to load SavedModelBundle](scoring/src/main/scala/tfs/Model.scala#7)
* [Scala example usage of TensorFlow Java library to generate predictions](scoring/src/main/scala/tfs/Model.scala#44)
* [Scala convenience methods for working with TensorFlow types](scoring/src/main/scala/tfs/package.scala)


