import tensorflow as tf
from tensorflow import keras
import os
import cv2
import pandas as pd
import numpy as np

def pred(f, dirpath):
    img = cv2.imread(os.path.join(dirpath, f))
    weight = df.loc[f]['weight']
    prob = model.predict([np.expand_dims(img, 0), np.expand_dims(weight, 0)])
    result = prob.argmax(-1)[0]
    return all_label[result]

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    model = keras.models.load_model('model_without_preprocess_finetuned.h5')
    all_label = ['can', 'paper_cup', 'paper_box', 'paper_milkbox', 'plastic']
    df = pd.read_csv('test_data/weights_test.csv')
    df = df.set_index('name')

    for dirpath, dirnames, filenames in os.walk('test_data'):
        for f in filenames:
            if f.endswith('.jpg'):
                print(f'{f}: {pred(f, dirpath)}')