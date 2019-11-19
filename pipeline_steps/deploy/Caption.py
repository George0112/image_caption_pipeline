import pickle
import json
import models
import numpy as np
import tensorflow as tf
from io import BytesIO
from tensorflow.python.lib.io import file_io
import wget

class Caption(object):
    """
    Model template. You can load your model parameters in __init__ from a location accessible at runtime
    """

    def __init__(self):
        """
        Add any initialization parameters. These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest.
        """
        
        print("Initializing")

        self.max_length = 50
        
        tokenizer_path =  '/mnt/ms-coco/tokenize/tokenizer.pickle'
        # tokenizer_path =  '/home/chao/ms-coco-all/tokenize/tokenizer.pickle'

        model_train_output_dir = '/mnt/ms-coco/train'
        # model_train_output_dir = '/home/chao/ms-coco-all/train'

        print("tokenizer_path: ", tokenizer_path)

        # Load tokenizer, model, test_captions, and test_imgs
        
        # Load tokenizer
        with file_io.FileIO(tokenizer_path, 'rb') as src:
            self.tokenizer = pickle.load(src)
        
        vocab_size = len(self.tokenizer.word_index) + 1

        print("vocab_size: ", vocab_size)

        # Shape of the vector extracted from InceptionV3 is (64, 2048)
        self.attention_features_shape = 64
        self.features_shape = 2048
        
        embedding_dim = 256
        units = 512
        
        self.encoder = models.CNN_Encoder(embedding_dim)
        self.decoder = models.RNN_Decoder(embedding_dim, units, vocab_size)
        
        # Load model from checkpoint (encoder, decoder)
        optimizer = tf.keras.optimizers.Adam()
        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,
                            decoder=self.decoder, optimizer=optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, model_train_output_dir + '/checkpoints', max_to_keep=5)
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output

        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
    # Preprocess the images using InceptionV3
    def load_image(self, img):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img #, image_path
    
    # Run predictions
    def evaluate(self, image):

        hidden = self.decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(self.load_image(image), 0)
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)

        return result

    def predict(self,X,features_names):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        for x in X:
            print(x)
            filename = wget.download(x)
            result = self.evaluate(filename)
        
        print("Predict called - will run identity function")
        return result