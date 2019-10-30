from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import csv
import glob
import os
import shutil
import math
import time

# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder

# adjust width and height to your needs
width = 180
height = 180
# use one of these filter options to resize the image
count = 0
for photocode in sorted(glob.glob('photos/*.jpg')):
    imageFile = photocode
    im1 = Image.open(imageFile)
    im5 = im1.resize((width, height), Image.ANTIALIAS)# best down-sizing filter
    if not os.path.exists("processedphotos"):
        os.makedirs("processedphotos")
    
    if os.path.exists("processedphotos"):
        count = count + 1
        print(count)
        im5.save("processed" + photocode)
    

with open('output.csv', mode='rt') as f, open('temp.csv',  'w', newline='') as final:
    writer = csv.writer(final, delimiter=',')
    reader = csv.reader(f, delimiter=',')
    sorted1 = sorted(reader, key=lambda row2: (str(row2[2])))        
    for row2 in sorted1:
        row2.append(math.ceil(100*(int(row2[3])/int(row2[1]))))
        writer.writerow(row2)

with open('temp.csv', mode='rt') as f, open('sorted.csv',  'w', newline='') as final:
    writer = csv.writer(final, delimiter=',')
    reader = csv.reader(f, delimiter=',')
    sorted2 = sorted(reader, key=lambda row: (int(row[4])))
    for row in sorted2:
        temp = row[4]
        row[4] = row[3]
        row[3] = temp
        writer.writerow(row)

os.replace('sorted.csv','output.csv')

count = 0
for line in open("output.csv"):
    count = count+1
    print(count)
    csv_row = line.split(',') #returns a list ["1","50","60"],
    #print(csv_row)
    filename = csv_row[2]
    #print(filename)
    
    if not os.path.exists("processed_labeled/"):
        os.makedirs("processed_labeled/")
        
    if int(csv_row[3]) >= 30:
        dir = "processed_labeled/" + "30"
        #print(dir)
    else:
        dir = "processed_labeled/" + csv_row[3]
        #print(dir)
        
    if not os.path.exists(dir):
        os.makedirs(dir)

    if os.path.exists(dir):
        file_path = "processedphotos/" + filename + ".jpg"
        #print(dir)
        
        # move files into created directory
        if os.path.exists(file_path):
            shutil.copy(file_path, dir)  
        else:
            for x in range(1,10):
                file_path = "processedphotos/" + filename + "_" + str(x) + ".jpg"
                if os.path.exists(file_path):
                    shutil.copy(file_path, dir)
    


dir = "processed_labeled/"
outputdir = "final/"
count = 10

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    os.makedirs(outputdir + "train")
    os.makedirs(outputdir + "validation")
    
for folder in os.listdir(dir):
    for file in os.listdir(dir + folder):
        
        dir_name = folder + "/" + file
        print(f'dir_name: {dir_name}')
                
        dir_path = dir + dir_name
        print(f'dir_path: {dir_path}')
                        
        dir_train = "final/train/" + folder + "/"
        dir_validation = "final/validation/" + folder + "/"
        
        if count < 10:
            # check if directory exists or not yet
            if not os.path.exists(dir_train):
                os.makedirs(dir_train)
                    
            if os.path.exists(dir_train):
                count = count+1
                file_path = dir_train + "/" + file
                print(f'file_path: {file_path}')
                # move files into created directory
                shutil.copy(dir_path, dir_train)
        else:
            if not os.path.exists(dir_validation):
                os.makedirs(dir_validation)
                    
            if os.path.exists(dir_validation):
                file_path = dir_validation + "/" + file
                print(f'file_path: {file_path}')
                # move files into created directory
                shutil.copy(dir_path, dir_validation)
                count = 1


PATH = "final"

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

batch_size = 128
epochs = 15
IMG_HEIGHT = 180
IMG_WIDTH = 180

CLASS_NAMES = ['1', '2', '3', '4', '5',
               '11', '12', '13', '14', '15',
               '21', '22', '23', '24', '25',
               '16', '17', '18', '19', '20',
               '26', '27', '28', '29', '30',
               '6', '7', '8', '9', '10']

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           classes = list(CLASS_NAMES))

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              classes = list(CLASS_NAMES))

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(30, activation='softmax')
])
    
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

total_train = 27759
total_val = 3085

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

model.save_weights(
    "model_weights",
    overwrite=True,
    save_format=None
)