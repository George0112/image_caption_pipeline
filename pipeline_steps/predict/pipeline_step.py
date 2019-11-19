import pickle
import json
import models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
from io import BytesIO
from tensorflow.python.lib.io import file_io
from ast import literal_eval as make_tuple
import click
from collections import namedtuple

@click.command()
@click.option('--dataset-path', default="/mnt/ms-coco")
@click.option('--max-length', default=50)
@click.option('--units', default=512)
@click.option('--embedding-dim', default=256)

def predict(dataset_path: str, 
        embedding_dim: int, units: int, max_length: int):
    print(models)
    # if tokenizing_output != 'default':
    #     tokenizing_output = make_tuple(tokenizing_output)
    #     max_length = int(tokenizing_output[0])
    
    preprocess_output_dir = dataset_path + '/preprocess'
    
    valid_output_dir = dataset_path + '/valid'
    
    tokenizer_path = dataset_path + '/tokenize/tokenizer.pickle'

    model_train_output_dir = dataset_path + '/train'

    print("tokenizer_path: ", tokenizer_path)
    
    val_cap_path = valid_output_dir + '/captions.npy'
    val_img_path = valid_output_dir + '/images.npy'
    tensorboard_dir = valid_output_dir + '/logs/' #+ datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)

    # Load tokenizer, model, test_captions, and test_imgs
    
    # Load tokenizer
    with file_io.FileIO(tokenizer_path, 'rb') as src:
        tokenizer = pickle.load(src)
    
    vocab_size = len(tokenizer.word_index) + 1

    print("vocab_size: ", vocab_size)

    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    attention_features_shape = 64
    features_shape = 2048
    
    encoder = models.CNN_Encoder(embedding_dim)
    decoder = models.RNN_Decoder(embedding_dim, units, vocab_size)
    
    # Load model from checkpoint (encoder, decoder)
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_train_output_dir + '/checkpoints', max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    # Load test captions
    f = BytesIO(file_io.read_file_to_string(val_cap_path, 
                                            binary_mode=True))
    cap_val = np.load(f)
    
    # load test images
    f = BytesIO(file_io.read_file_to_string(val_img_path, 
                                            binary_mode=True))
    img_name_val = np.load(f)
    
    # To get original image locations, replace .npy extension with .jpg and 
    # replace preprocessed path with path original images
    PATH = dataset_path + '/train2014/train2014/'
    print(img_name_val[0])
    img_name_val = [img.replace('.npy', '.jpg') for img in img_name_val]
    img_name_val = [img.replace(preprocess_output_dir, PATH) for img in img_name_val]
    print(preprocess_output_dir)
    print(img_name_val[0])

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
    # Preprocess the images using InceptionV3
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path
    
    # Run predictions
    def evaluate(image):
        attention_plot = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot
    
    # Modified to plot images on tensorboard
    def plot_attention(image, result, attention_plot):
        img = tf.io.read_file(image)
        img = tf.image.decode_jpeg(img, channels=3)
        temp_image = np.array(img.numpy())
        
        len_result = len(result)
        for l in range(min(len_result, 10)): # Tensorboard only supports 10 imgs
            temp_att = np.resize(attention_plot[l], (8, 8))
            plt.title(result[l])
            img = plt.imshow(temp_image)
            plt.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
            
            # Save plt to image to access in tensorboard
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            final_im = tf.image.decode_png(buf.getvalue(), channels=4)
            final_im = tf.expand_dims(final_im, 0)
            with summary_writer.as_default():
                tf.summary.image("attention", final_im, step=l)
    
    # Select a random image to caption from validation set
    rid = np.random.randint(0, len(img_name_val))
    image = img_name_val[rid]
    print(image)
    real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    result, attention_plot = evaluate(image)
    print ('Image:', image)
    print ('Real Caption:', real_caption)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot)
    
    # Plot attention images on tensorboard
    metadata = {
        'outputs': [
            {
                'storage': 'inline',
                'source': ('# Predicted figure: ![](http://140.114.79.72:31381/ms-coco/train2014/train2014/' + image.split('/')[-1] + ')' + 
                    '\n  # Real Caption: ' + real_caption + '\n # Predicted Caption: ' + ' '.join(result)),
                'type': 'markdown',
            }
        ]
    }
    print(metadata)
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)
        
    divmod_output = namedtuple('output', ['mlpipeline_ui_metadata'])
    return divmod_output(json.dumps(metadata))

if __name__ == "__main__":
    predict()