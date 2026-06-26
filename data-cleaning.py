import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")
import cv2
import imghdr
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


data_dir='data'
image_exts=['png','jpg','webp','jpeg']

#uymayan resimleri kaldırma yeri. bi kere calistirdik.
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path=os.path.join(data_dir, image_class, image)
        try:
            img=cv2.imread(image_path)
            if img is None:
                print("broken name or somethng:", image_path)
                os.remove(image_path)
            tip=imghdr.what(image_path)
            if tip not in image_exts:
                print("images ext not match".format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("issues with image".format(image_path))

for root, dirs, files in os.walk("data"):
    for file in files:
        path = os.path.join(root, file)

        try:
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3)
        except:
            print("Siliniyor:", path)
            os.remove(path)

