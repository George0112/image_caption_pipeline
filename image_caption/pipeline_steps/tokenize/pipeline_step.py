import pickle
import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io
from io import BytesIO
from ast import literal_eval as make_tuple
import click
import os

@click.command()
@click.option('--dataset-path', default="/mnt/ms-coco")
@click.option('--top-k', default=5000)

def tokenize_captions(dataset_path, top_k):
    
    # Convert output from string to tuple and unpack
    # preprocess_output = make_tuple(preprocess_output)
    # train_caption_path = preprocess_output[0]
    train_caption_path = dataset_path + '/preprocess/train_captions.npy'
    OUTPUT_DIR = dataset_path + '/tokenize/'
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    # f = BytesIO(file_io.read_file_to_string(train_caption_path, 
    #                                         binary_mode=True))
    train_captions = np.load(train_caption_path)
    print(len(train_captions))
    
    # Tokenize captions
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    
    # Find the maximum length of any caption in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)
    
    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)
    print("max_length: ", max_length)
    
    if not (os.path.isdir(OUTPUT_DIR)):
        os.mkdir(OUTPUT_DIR)

    # Save tokenizer
    tokenizer_file_path = OUTPUT_DIR + 'tokenizer.pickle'
    with file_io.FileIO(tokenizer_file_path, 'wb') as output:
        pickle.dump(tokenizer, output, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Save train_seqs
    cap_vector_file_path = OUTPUT_DIR + 'cap_vector.npy'
    np.save(cap_vector_file_path, cap_vector)

    return str(max_length)#, tokenizer_file_path, cap_vector_file_path

if __name__ == "__main__":
    tokenize_captions()