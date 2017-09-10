
import os 
from matplotlib import pyplot as plt
import tensorflow as tf

FOLDER_TRAINING_SET='/Users/dboudeau/depot/leaf/'

filenames=os.listdir(FOLDER_TRAINING_SET+'THUYA/') 

# step 2
filename_queue = tf.train.string_input_producer(filenames)

# step 3: read, decode and resize images
reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)
image = tf.image.decode_jpeg(content, channels=3)
image = tf.cast(image, tf.float32)
resized_image = tf.image.resize_images(image, [224, 224])

# step 4: Batching
image_batch = tf.train.batch([resized_image], batch_size=8)


with tf.Session() as session:
    result = session.run(image_batch)