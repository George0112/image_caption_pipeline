import click
import numpy as np
import dill
import pandas as pd
import zipfile
import wget

@click.command()
@click.option('--images-path', default="/mnt/ms-coco/train2014/train2014")
@click.option('--annotation-path', default="/mnt/ms-coco/annotations_trainval2014/annotations/captions_train2014.json")
@click.option('--zip-url', default="http://140.114.79.84/ms-coco.zip")
# @click.option('--csv-encoding', default="ISO-8859-1")
# @click.option('--features-column', default="BODY")
# @click.option('--labels-column', default="REMOVED")
def run_pipeline(
        images_path, 
        annotation_path,
        zip_url):
    wget.download(zip_url, out="/mnt/ms-coco.zip")
    
    with zipfile.ZipFile("/mnt/ms-coco.zip", 'r') as zip_ref:
        zip_ref.extractall("/mnt/")

#     df = pd.read_csv(csv_url, encoding=csv_encoding)

#     x = df[features_column].values

#     with open(features_path, "wb") as out_f:
#         dill.dump(x, out_f)

#     y = df[labels_column].values

#     with open(labels_path, "wb") as out_f:
#         dill.dump(y, out_f)

if __name__ == "__main__":
    run_pipeline()