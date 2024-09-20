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
        self.save_path = os.path.join(self.save_dir, "simple_cnn")
        self.logs_path = os.path.join(self.logs_dir, "simple_cnn")
        
    def build(self, input_tensor, num_classes):
        """Builds a convolutional neural network."""
        
        self.is_training = tf.Variable(True, name="is_training")
        
        # Convolutional layers
        conv_1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.beta),
            name="conv_1")(input_tensor)
        
        conv_2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.beta),
            name="conv_2")(conv_1)
        
        # Max Pooling Layer
        pool_3 = tf.keras.layers.MaxPooling2D(
            pool_size=2, 
            strides=2, 
            padding="same",
            name="pool_3")(conv_2)
        
        # Dropout Layer
        drop_4 = tf.keras.layers.Dropout(
            rate=0.5, 
            name="drop_4")(pool_3, training=self.is_training)
        
        # Convolutional Layer 5
        conv_5 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.beta),
            name="conv_5")(drop_4)
        
        # Convolutional Layer 6
        conv_6 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.beta),
            name="conv_6")(conv_5)
        
        # Max Pooling Layer
        pool_7 = tf.keras.layers.MaxPooling2D(
            pool_size=2, 
            strides=1, 
            padding="same",
            name="pool_7")(conv_6)
        
        # Dropout Layer
        drop_8 = tf.keras.layers.Dropout(
            rate=0.5, 
            name="drop_8")(pool_7, training=self.is_training)

        # Fully-connected layers
        flattened  = tf.layers.flatten(
            drop_8,
            name="flatten")
        fc_9 = tf.layers.dense(
            flattened,
            units = 1024,
            activation = tf.layers.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(self.beta),
            name = "fc_9")
        drop_10 = tf.layers.dropout(
            fc_9, 
            training = self.is_training,
            name = "drop_10")
        logits = tf.layers.dense(
            drop_10,
            units = num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(self.beta),
            name = 'logits'
        )
        
        return logits
    
    def _create_tf_dataset(self, x, y):
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(x),
                tf.data.Dataset.from_tensor_slices(y),
            )
        ).shuffle(50).repeat().batch(self.batch_size)
        
        return dataset
    
    def _log_loss_and_acc(self, epoch, loss, acc, suffix):
        summary = tf.summary(value=[
            tf.summary.value(tag="Loss_{}".format(suffix), simple_value=float(loss)),
            tf.summary.value(tag="acc_{}".format(suffix), simple_value=float(acc))
        ])
        self.summary_writer.add_summary(summary, epoch)
        
    
    # define the fit method
    def fit(self, X_train, X_test, y_train, y_test):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
        
        train_dataset = self._create_tf_dataset(X_train, y_train)
        test_dataset = self._create_tf_dataset(X_test, y_test)
        
        # Create a generic iterator
        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes
        )
        
        next_tensor_batch = iterator.get_next()
        
        # separate training and testing set init ops
        train_init_ops = iterator.make_initializer(train_dataset)
        test_init_ops = iterator.make_initializer(test_dataset)
        
        input_tensor, labels = next_tensor_batch
        
        num_classes = y_train.shape[1]

        
        # Building the network
        logits = self.build(input_tensor=input_tensor, num_classes=num_classes)
        logger.info("Build network")
        
        prediction = tf.nn.softmax(logits, name="predictions")
        loss_ops = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        ),
                                  name="loss")

        # define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_ops = optimizer.minimizer(loss_ops)
        
        correct = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(labels, 1),
            name="correct")
        accuracy_ops = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        
        
        initializer = tf.global_variables_initializer()
        logger.info("Initializing all variables")
        sess.run(initializer)
        logger.info("Initialized all variables")
        
        sess.run(train_init_ops)
        logger.info("Initialized dataset iterator")
        self.savers = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logs_path)
        
        logger.info("Training CNN for {} epochs".format(self.num_epoch))
        for epoch_idx in range(1, self.num_epoch+1):
            loss, _, accuracy = sess.run([
                loss_ops, train_ops, accuracy_ops
            ])
            self._log_loss_and_acc(epoch_idx, loss, accuracy, "train")
            
            if epoch_idx % 10 == 0:
                sess.run(test_init_ops)
                test_loss, test_accuracy = sess.run([
                    loss_ops, accuracy_ops],
                                                    feed_dict = {self.is_training:False})
                logger.info("==============> Epoch {}".format(epoch_idx))
                logger.info("\tTraining accuracy: {:.3f}".format(accuracy))
                logger.info("\tTraining loss : {:.6f}".format(loss))
                logger.info("\tTest Accuracy : {:.3f}".format(test_accuracy))
                logger.info("\tTest loss :  {:.6f}".format(test_loss))
                self._log_loss_and_acc(epoch_idx, test_loss, test_accuracy, "Test")
                
                
            # Create a checkpoint at every epoch
            self.saver.save(sess, self.save_path)
            
            
# time for practice
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
    
    
    logger.info("Simple transformations by dividing pixels by 255")
    X_train = X_train/255
    X_test = X_test/255
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    num_classes = len(np.unique(y_train))
    
    logger.info("Turning categories into one-hot encoding")
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    
    cnn_params = {
        "learning_rate": 3e-4,
        "num_epochs": 100,
        "beta": 1e-4,
        "batch_size": 32
    }
    
    
    logger.info("Initialization of the CNN")
    simple_cnn = SimpleCNN(**cnn_params)
    logger.info("Training CNN")
    simple_cnn.fit(
        X_train = X_train,
        X_test = X_test,
        y_train = y_train,
        y_test = y_test
    )