#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7

import tensorflow as tf
from os import path, getcwd, chdir
import os
import zipfile

# local_zip = '/Users/ghost/Documents/ML/Learning-ML/images/horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('/Users/ghost/Documents/ML/Learning-ML/images/horse-or-human')
# local_zip = '/Users/ghost/Documents/ML/Learning-ML/images/validation-horse-or-human.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('//Users/ghost/Documents/ML/Learning-ML/images/validation-horse-or-human')
# zip_ref.close()


# Directory with our training horse pictures
train_horse_dir = os.path.join('/Users/ghost/Documents/ML/Learning-ML/images/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('/Users/ghost/Documents/ML/Learning-ML/images/horse-or-human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('/Users/ghost/Documents/ML/Learning-ML/images/validation-horse-or-human/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('/Users/ghost/Documents/ML/Learning-ML/images/validation-horse-or-human/humans')


train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
print('total validation horse images:', len(os.listdir(validation_horse_dir)))
print('total validation human images:', len(os.listdir(validation_human_dir)))


# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 16
next_horse_pix = [os.path.join(train_horse_dir, fname) 
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) 
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

# train_data_gen = ImageDataGenerator(rescale=1./255)
# train_dir = ./images
# train_generator = train_datagen.flow_flow_ditectory(
#     train_dir,
#     target_size=(300, 300)
#     batch_size=128,
#     class_mode='binary')


