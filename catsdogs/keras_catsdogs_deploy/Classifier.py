import os
import sys
import wget

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
K.clear_session()

class Classifier(object):

    def __init__(self):
        # K.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        set_session(self.sess)
        # load the trained model
        self.net = load_model('./model-inception_resnet_v2-final.h5')
        self.graph = tf.get_default_graph()
        print("model loaded")
        pass

    def evaluate(self, file):
        cls_list = ['cats', 'dogs']
        img = image.load_img(file, target_size=(299,299))
        if img is None:
            return 'unknown'
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        pred = self.net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        return cls_list[0]

    def predict(self, X, features_names, **kwargs):
        
#        Return a prediction.
#        Parameters
#        ----------
#        X : array-like
#        feature_names : array of feature names (optional)
        result = []
        print(X)
        for x in X:
            print(x)
            filename = wget.download(x)
            print(filename)
            cls_list = ['cats', 'dogs']
            img = image.load_img(filename, target_size=(299,299))
            if img is None:
                return 'unknown'
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            with self.graph.as_default():
                set_session(self.sess)
                pred = self.net.predict(x)[0]
            top_inds = pred.argsort()[::-1][:5]
            result.append(cls_list[0])
            # result.append(self.evaluate(filename))
        print("Predict called - will run identity function")
        return result

    # if __name__ == '__main__':
    #     args = parse_args()
    #     files = get_files(args.path)
    #     cls_list = ['cats', 'dogs']
    #     # loop through all files and make predictions
    #     for f in files:
    #         img = image.load_img(f, target_size=(299,299))
    #         if img is None:
    #             continue
    #         x = image.img_to_array(img)
    #         x = preprocess_input(x)
    #         x = np.expand_dims(x, axis=0)
    #         pred = net.predict(x)[0]
    #         top_inds = pred.argsort()[::-1][:5]
    #         print(f)
    #         for i in top_inds:
    #             print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
