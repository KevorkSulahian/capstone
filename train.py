import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import signal

from dataloader import *
from tensorflow import keras
# from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from model.unet import *
# from tensorflow.keras.callbacks import *
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
# from keras.callbacks import TensorBoard

# Model
resume = None
weight_dir = pathlib.Path('weights').absolute()
weights_file = weight_dir.joinpath('u2net.h5')
default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)

# Training
batch_size = 10
epochs = 10000
learning_rate = 0.001
save_interval = 1000

# Dataset 
current_location = pathlib.Path(__file__).absolute().parents[0]
root_data_dir = pathlib.Path('data')
dataset_dir = root_data_dir.joinpath('DUTS-TR')
image_dir = dataset_dir.joinpath('DUTS-TR-Image')
mask_dir = dataset_dir.joinpath('DUTS-TR-Mask')

# Evaluation
output_dir = pathlib.Path('out')

# Arguments
parser = argparse.ArgumentParser(description="Salient object detection")
parser.add_argument('--batch_size', default = None, type = int)
parser.add_argument('--lr', default = None, type = float)
parser.add_argument('--resume', default=None, type = str)
args = parser.parse_args()

if args.batch_size:
    batch_size = args.batch_size

if args.lr:
    learning_rate = args.lr

if args.resume:
    resume = args.resume


# weight_data_dir = pathlib.Path('weights')
# resume = weight_data_dir.joinpath('u2net.h5')

# # Overwrite the default optimizer
adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_file, save_weights_only=True, verbose=1)

def save_df(loss):

    f=open('logs.txt','ab')
    a=np.array([[loss]],dtype=np.float32)
    np.savetxt(f,a)
    f.close()

def train():
    inputs = keras.Input(shape = default_in_shape)
    net = UNET()
    out = net(inputs)
    model = keras.Model(inputs = inputs, outputs = out, name = 'unet')
    model.compile(optimizer=adam, loss = bce_loss,metrics=None) 
    model.summary()

    # resume the training
    if resume:
        print('loading weights from %s' %resume)
        model.load_weights(resume)
    
    # save the weights
    def save_weights():
        print("Saving current state of the model to %s", weights_file)
        model.save_weights(str(weights_file))

    # early stop handler
    def autosave(sig,frame):
        print("stoping the training")
        save_weights()
        exit(0)
    
    for sig in [signal.SIGABRT, signal.SIGINT]:
        signal.signal(sig, autosave)

    # start the damn training
    print('Trainnig has been started')
    for e in range(epochs):
        try:
            feed,out =load_training_batch(batch_size=batch_size)
            loss = model.train_on_batch(feed,out)
            save_df(loss) # make sure to update this function to clean the data if its not resumed
        except KeyboardInterrupt:
            save_weights()
            return
        except ValueError:
            print("Value error")
            continue
        if e % 10 == 0:
            print('[%05d] Loss: %.4f' % (e,loss))

        if save_interval and e > 0 and e % save_interval == 0:
            save_weights()

if __name__=='__main__':
    train()