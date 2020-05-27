import json
import time
import pickle
import models
import numpy as np
import tensorflow as tf
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.python.lib.io import file_io
from ast import literal_eval as make_tuple
import click
import os
import matplotlib.pyplot as plt
from collections import namedtuple

@click.command()
@click.option('--dataset-path', default="/mnt/ms-coco")
@click.option('--batch-size', default=8)
@click.option('--embedding-dim', default=256)
@click.option('--units', default=512)
@click.option('--epochs', default=20)

def train_model(dataset_path: str,  
        batch_size: int, embedding_dim: int, units: int, epochs: int):
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Convert output from string to tuple and unpack
    # preprocess_output = make_tuple(preprocess_output)
    # tokenizing_output = make_tuple(tokenizing_output)
    
    # Unpack tuples
    # preprocessed_imgs_path = preprocess_output[1]
    # tokenizer_path = tokenizing_output[1]
    # cap_vector_file_path = tokenizing_output[2]
    preprocessed_imgs_path = dataset_path + '/preprocess/preprocessed_imgs.npy'
    tokenizer_path = dataset_path + '/tokenize/tokenizer.pickle'
    cap_vector_file_path = dataset_path + '/tokenize/cap_vector.npy'
    valid_output_dir = dataset_path + '/valid/'
    train_output_dir = dataset_path + '/train/'

    if not (os.path.isdir(valid_output_dir)):
        os.mkdir(valid_output_dir)

    if not (os.path.isdir(train_output_dir)):
        os.mkdir(train_output_dir)
    
    # load img_name_vector
    # f = BytesIO(file_io.read_file_to_string(preprocessed_imgs_path, binary_mode=True))
    img_name_vector = np.load(preprocessed_imgs_path)
    print(img_name_vector[:5])
    
    # Load cap_vector
    # f = BytesIO(file_io.read_file_to_string(cap_vector_file_path, binary_mode=True))
    cap_vector = np.load(cap_vector_file_path)
    
    # Load tokenizer
    with file_io.FileIO(tokenizer_path, 'rb') as src:
        tokenizer = pickle.load(src)
    
    # Split data into training and testing
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(
                                                            img_name_vector,
                                                            cap_vector,
                                                            test_size=0.2,
                                                            random_state=0)
    
    # Create tf.data dataset for training
    BUFFER_SIZE = 1000 # common size used for shuffling dataset
    vocab_size = len(tokenizer.word_index) + 1
    num_steps = len(img_name_train) // batch_size

    print("num steps: ", num_steps)
    print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))
    
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    features_shape = 2048
    
    # Load the numpy files
    def map_func(img_name, cap):
        f = BytesIO(file_io.read_file_to_string(img_name.decode('utf-8'), binary_mode=True))
        img_tensor = np.load(f)
        return img_tensor, cap
    
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # get models from models.py
    encoder = models.CNN_Encoder(embedding_dim)
    decoder = models.RNN_Decoder(embedding_dim, units, vocab_size)
    
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    # Create loss function
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    
    # Create check point for training model
    ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, train_output_dir + 'checkpoints/', max_to_keep=5)
    start_epoch = 0
    print('latest checkpoint: ', ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])*5
        ckpt.restore(ckpt_manager.latest_checkpoint)
            
    # Create training step
    loss_plot = []
    @tf.function
    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss
    
    # Create summary writers and loss for plotting loss in tensorboard
    tensorboard_dir = train_output_dir + 'logs/' #+ datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    
    # Train model
    path_to_most_recent_ckpt = None
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    for epoch in range(start_epoch, epochs):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            if batch >= num_steps:
                break
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss
            train_loss(t_loss)
            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        
        # Storing the epoch end loss value to plot in tensorboard
        loss_plot.append(total_loss / num_steps)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss per epoch', train_loss.result(), step=epoch)
        
        train_loss.reset_states()
        
        if epoch % 5 == 0:
            path_to_most_recent_ckpt = ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.xticks(range(start_epoch, epochs))
    plt.show()

    # Save validation data to use for predictions
    val_cap_path = valid_output_dir + '/captions.npy'
    np.save(file_io.FileIO(val_cap_path, 'w'), cap_val)
    
    val_img_path = valid_output_dir + '/images.npy'
    np.save(file_io.FileIO(val_img_path, 'w'), img_name_val)

    # Save train data
    val_cap_path = valid_output_dir + '/train_captions.npy'
    np.save(file_io.FileIO(val_cap_path, 'w'), cap_train)
    
    val_img_path = valid_output_dir + '/train_images.npy'
    np.save(file_io.FileIO(val_img_path, 'w'), img_name_train)

    # Add plot of loss in tensorboard
    metadata = {
        'outputs': [
            {
                'storage': 'inline',
                'source': '# Loss: ' + str(float(total_loss/num_steps)),
                'type': 'markdown',
            }
        ]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    print(float(total_loss / num_steps))
    
    divmod_output = namedtuple('output', ['mlpipeline_ui_metadata'])
    return divmod_output(json.dumps(metadata))
    return path_to_most_recent_ckpt, val_cap_path, val_img_path

if __name__ == "__main__":
    train_model()