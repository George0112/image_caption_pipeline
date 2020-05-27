import click
import numpy as np
import dill
import pandas as pd
import zipfile
import wget

@click.command()
@click.option('--zip-url', default="http://140.114.79.84/ms-coco.zip")

def run_pipeline(zip_url):
    wget.download(zip_url, out="/mnt/ms-coco.zip")
    
    with zipfile.ZipFile("/mnt/ms-coco.zip", 'r') as zip_ref:
        zip_ref.extractall("/mnt/")
if __name__ == "__main__":
    run_pipeline()