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

def str2bool(v):
    return v is not None and v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='UNET Testing')
parser.add_argument('--images', default=None, type=str)
parser.add_argument('--output', default=None, type=str)
parser.add_argument('--weights', default=None, type=str)
parser.add_argument('--merged', default=True, type=str2bool)
parser.add_argument('--apply_mask', default=True, type=str2bool)
args = parser.parse_args()

# python test.py --weights=weights/u2net.h5 --images=examples

if args.output:
    output_dir = pathlib.Path(args.output)

def apply_mask(img, mask):
    return np.multiply(img, mask)

def main():
    input_images = []

    # check and get the images in the folder
    if args.images:
        input_dir = pathlib.Path(args.images)
        if not input_dir.exists():
            input_dir.mkdir()
        imgs = glob.glob(str(input_dir.joinpath('*png'))) + glob.glob(str(input_dir.joinpath('*.jpg')))
        assert len(imgs) > 0, 'No images found in directory %s' % str(input_dir)
        input_images.extend(imgs)
    
    # Meh
    if not output_dir.exists():
        output_dir.mkdir()

    if len(input_images) == 0:
        return

    inputs = keras.Input(shape=default_in_shape)
    net = UNET()
    out = net(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name='unetmodel')
    model.compile(optimizer=adam, loss=bce_loss, metrics=None)


    # In case args don't exist if images are deformed check the size of the weights
    # anything less that 10 kb means it's not trained at all
    if args.weights:
        assert os.path.exists(args.weights), 'Model weights path must exist: %s' % args.weights
        model.load_weights(args.weights)

    # evaluate each image
    for img in input_images:
        image = Image.open(img).convert('RGB')
        input_image = image
        if image.size != default_in_shape:
            input_image = image.resize(default_in_shape[:2], Image.BICUBIC)
        
        input_tensor = format_input(input_image)
        fused_mask_tensor = model(input_tensor, Image.BICUBIC)[0][0]
        output_mask = np.asarray(fused_mask_tensor)
        
        if image.size != default_in_shape:
            output_mask = cv2.resize(output_mask, dsize=image.size)
        
        output_mask = np.tile(np.expand_dims(output_mask, axis=2), [1, 1, 3])
        output_image = np.expand_dims(np.array(image)/255., 0)[0]
        if args.apply_mask:
            output_image = apply_mask(output_image, output_mask)
        else:
            output_image = output_mask

        if args.merged:
            output_image = np.concatenate((output_mask, output_image), axis=1)

        output_image = cv2.cvtColor(output_image.astype('float32'), cv2.COLOR_BGR2RGB) * 255.
        output_location = output_dir.joinpath(pathlib.Path(img).name)
        cv2.imwrite(str(output_location), output_image)
        
if __name__=='__main__':
    main()