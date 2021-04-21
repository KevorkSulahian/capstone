import os
import pathlib
import random
import numpy as np
import wget
import zipfile
import glob

from PIL import Image 

# Model
resume = None
weight_dir = pathlib.Path('weights').absolute()
weights_file = weight_dir.joinpath('u2net.h5')
default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)

# Training
batch_size = 2
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

cache = None 

# some cleaning after aborting the training
def clean_dataloader():
    for tmp_file in glob.glob('*tmp'):
        os.remove(tmp_file)

# Standard resize with extra assertions
def format_input(input_image):
    # make sure data is in right shape
    assert(input_image.size == default_in_shape[:2] or input_image.shape == default_in_shape)
    inp = np.array(input_image)
    if inp.shape[-1] == 4:
        input_image = input_image.convert('RGB')
    return np.expand_dims(np.array(input_image)/255.,0)


# takes img and it's mask and maybe augments them
def get_image_mask_pair(img_name, in_resize = None, out_resize = None, augment = True):
    in_img = image_dir.joinpath(img_name)
    out_img = mask_dir.joinpath(img_name.replace('jpg', 'png'))

    if not in_img.exists() or not out_img.exists():
        return None

    img = Image.open(in_img)
    mask = Image.open(out_img)

    if in_resize:
        img = img.resize(in_resize[:2], Image.BICUBIC)
    if out_resize:
        mask = mask.resize(out_resize[:2], Image.BICUBIC)
    
    # image augmentation, similar to the paper
    if augment and bool(random.getrandbits(1)):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return (np.asarray(img, dtype=np.float32), np.expand_dims(np.asarray(mask, dtype=np.float32), -1))

# Simply picks K random pics
def load_training_batch(batch_size = 12, in_shape=  default_in_shape, out_shape = default_out_shape):
    global cache
    if cache is None:
        cache = os.listdir(image_dir)

    imgs = random.choices(cache, k=batch_size)
    image_list = [get_image_mask_pair(img, in_resize=default_in_shape, out_resize=default_out_shape) for img in imgs]
    
    tensor_in  = np.stack([i[0]/255. for i in image_list])
    tensor_out = np.stack([i[1]/255. for i in image_list])
    
    return (tensor_in, tensor_out)