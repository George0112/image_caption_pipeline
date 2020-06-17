import click
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from sklearn.utils import shuffle
import os

@click.command()
@click.option('--dataset-path', default="/mnt/ms-coco")
@click.option('--images-path', default="/mnt/ms-coco/train2014/train2014")
@click.option('--annotation-path', default="/mnt/ms-coco/annotations_trainval2014/annotations/captions_train2014.json")
@click.option('--num-examples', default=30000)
@click.option('--batch-size', default=8)

def preprocess(dataset_path, images_path, annotation_path, num_examples, batch_size: int):

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                           [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

    OUTPUT_DIR = dataset_path + '/preprocess/'
    
    annotation_file = dataset_path + '/annotations_trainval2014/annotations/captions_train2014.json'
    PATH = dataset_path + '/train2014/train2014/'
    files_downloaded = tf.io.gfile.listdir(PATH)
    
    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + '/COCO_train2014_' + '%012d.jpg' % (image_id)
        img_name = 'COCO_train2014_' + '%012d.jpg' % (image_id)

        if img_name in files_downloaded: # Only have subset
            all_img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                            all_img_name_vector,
                                            random_state=1)

    # Select the first 30000 captions from the shuffled set
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]
    len(train_captions), len(all_captions)
    
    print(len(train_captions), len(all_captions))
    
    # Preprocess the images before feeding into inceptionV3
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path
    
    # Create model for processing images 
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
    # Save extracted features in PVC
    print('Extracting features from images...')
        
    if not (os.path.isdir(OUTPUT_DIR)):
        os.mkdir(OUTPUT_DIR)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
    
    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            # Save to a different location and as numpy array
            path_of_feature = path_of_feature.replace('.jpg', '.npy')
            path_of_feature = path_of_feature.replace(PATH, OUTPUT_DIR)
            np.save(path_of_feature, bf.numpy())
    
    # Create array for locations of preprocessed images
    preprocessed_imgs = [img.replace('.jpg', '.npy') for img in img_name_vector]
    preprocessed_imgs = [img.replace(PATH, OUTPUT_DIR) for img in preprocessed_imgs]

    # Save train_captions and preprocessed_imgs to file
    train_cap_path = OUTPUT_DIR + 'train_captions.npy' # array of captions
    preprocessed_imgs_path = OUTPUT_DIR + 'preprocessed_imgs'# array of paths to preprocessed images
    
    train_captions = np.array(train_captions)
    np.save(train_cap_path, train_captions)
    
    preprocessed_imgs = np.array(preprocessed_imgs)
    np.save(preprocessed_imgs_path, preprocessed_imgs)

    return (train_cap_path, preprocessed_imgs_path)

if __name__ == "__main__":
    preprocess()