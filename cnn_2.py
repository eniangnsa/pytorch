import logging
import os
import sys

import tensorflow as tf
import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

logger = logging.getLogger(__name__)

class SimpleCNN(object):
    def __init__(self, learning_rate, num_epochs, beta, batch_size):
        self.learning_rate = learning_rate
        self.num_epoch = num_epochs
        self.beta = beta
        self.batch_size = batch_size
        self.logs_dir = "logs"
        self.save_dir = "saves"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, "simple_cnn.h5")
        self.logs_path = os.path.join(self.logs_dir, "simple_cnn")
        
    def build(self, input_shape, num_classes):
        """Builds a convolutional neural network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.beta), input_shape=input_shape),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.beta)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.beta)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', 
                                   kernel_regularizer=tf.keras.regularizers.l2(self.beta)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.beta)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def fit(self, X_train, X_test, y_train, y_test):
        input_shape = X_train.shape[1:]  # Get input shape (height, width, channels)
        num_classes = y_train.shape[1]   # Get the number of classes
        
        model = self.build(input_shape=input_shape, num_classes=num_classes)
        logger.info("Training CNN for {} epochs".format(self.num_epoch))
        
        # Training the model using `fit`
        model.fit(X_train, y_train, epochs=self.num_epoch, batch_size=self.batch_size, 
                  validation_data=(X_test, y_test))
        
        # Save the model after training
        model.save(self.save_path)
        logger.info("Model saved at {}".format(self.save_path))


# Practice
if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Loading Fashion MNIST data")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    logger.info("Shape of training data: ")
    logger.info("Train: {}".format(X_train.shape))
    logger.info("Test: {}".format(X_test.shape))
    
    # Preprocessing: normalize and expand dimensions for channels
    logger.info("Simple transformations by dividing pixels by 255")
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
    X_test = np.expand_dims(X_test, axis=-1)
    
    y_train = to_categorical(y_train, 10)  # One-hot encode labels
    y_test = to_categorical(y_test, 10)
    
    cnn_params = {
        "learning_rate": 3e-4,
        "num_epochs": 10,
        "beta": 1e-4,
        "batch_size": 32
    }
    
    logger.info("Initialization of the CNN")
    simple_cnn = SimpleCNN(**cnn_params)
    logger.info("Training CNN")
    simple_cnn.fit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
