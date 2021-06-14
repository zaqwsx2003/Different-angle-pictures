import tensorflow as tf
from PIL import Image
import numpy as np
import os

#기본 크기 400x400 설정 시작

image1 = Image.open('1.jpg')
image1 = image1.resize((400, 400))
image1 = np.array(image1)
print(image1.shape) 

image2 = Image.open('2.jpg')
image2 = image2.resize((400, 400))
image2 = np.array(image2)
print(image2.shape) 

image3 = Image.open('3.jpg')
image3 = image3.resize((400, 400))
image3 = np.array(image3)
print(image3.shape) 

x_train = np.array([image1, image2, image3])

y_train = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

labels = ['1', '2', '3']

print("밝기 랜덤 생성")
#밝기 랜덤
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(brightness_range=[0.7, 1.3]) 

train_image_data_generator_iterator = train_image_data_generator.flow(
    x_train, 
    y_train,
    batch_size=1,
    shuffle=False, 
    save_prefix='밝기_', 
    save_to_dir='.') 

x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape)
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
print("밝기 끝 ")


print("좌우 랜덤 시작")
#좌우
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)

train_image_data_generator_iterator = train_image_data_generator.flow(
    x_train, 
    y_train,
    batch_size=1,
    shuffle=False, 
    save_prefix='좌우_',
    save_to_dir='.') 

x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape)
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape)
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
print("좌우 끝")

print("상하 랜덤 시작")
#상하 
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(vertical_flip=True) 

train_image_data_generator_iterator = train_image_data_generator.flow(
    x_train, 
    y_train,
    batch_size=1,
    shuffle=False, 
    save_prefix='상하_', 
    save_to_dir='.') 

x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape)
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
print("상하 끝")

print("좌우 랜덤 시작")
#좌우
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.3) 

train_image_data_generator_iterator = train_image_data_generator.flow(
    x_train, 
    y_train,
    batch_size=1,
    shuffle=False, 
    save_prefix='좌우_', 
    save_to_dir='.') 

x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
print("좌우 끝")

print("상하 랜덤 시작")
#상하
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=0.3) 

train_image_data_generator_iterator = train_image_data_generator.flow(
    x_train, 
    y_train,
    batch_size=1,
    shuffle=False, 
    save_prefix='상하_', 
    save_to_dir='.') 

x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape)
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
print("상하 끝")

print("회전 랜덤 시작")
 #회전 랜덤
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30) 

train_image_data_generator_iterator = train_image_data_generator.flow(
    x_train, 
    y_train,
    batch_size=1,
    shuffle=False, 
    save_prefix='회전_', 
    save_to_dir='.') 

x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape) 
print("회전 끝")

print("기울기 랜덤 시작")
#기울기 랜덤 
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(shear_range=30)

train_image_data_generator_iterator = train_image_data_generator.flow(
    x_train, 
    y_train,
    batch_size=1,
    shuffle=False, 
    save_prefix='기울기_', 
    save_to_dir='.') 

x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape)
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
print("기울기 끝")

print("확대 축소 랜덤 시작")
#확대 축소 랜덤
train_image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.3) 

train_image_data_generator_iterator = train_image_data_generator.flow(
    x_train, 
    y_train,
    batch_size=1,
    shuffle=False, 
    save_prefix='확대,축소_', 
    save_to_dir='.') 

x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape) 
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 
print(y_train_batch.shape)
x_train_batch, y_train_batch = next(train_image_data_generator_iterator)
print(x_train_batch.shape) 

print("확대 축소 끝\n")


print("----끝----")