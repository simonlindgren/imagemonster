#!/usr/bin/env python3

'''
IMAGEMONSTER

https://github.com/simonlindgren/imagemonster

'''

import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import xception
from tensorflow import keras
import pickle
import umap
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2 # installed with `pip install opencv-contrib-python`

def main():
    print("python main function")
    
    
def img2vec_resnet50(imagedir):
    
    image_paths = glob.glob(str(imagedir) + '/*.jpg')
    
    _IMAGE_NET_TARGET_SIZE = (224, 224)
    
    model = resnet50.ResNet50(weights='imagenet')
    layer_name = 'avg_pool'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    image_vectors = {}

    for c,image_path in enumerate(image_paths):
        
        print("\r" + str(c+1) + "/" + str(len(image_paths)), end="")
        img = image.load_img(image_path, target_size=_IMAGE_NET_TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        intermediate_output = intermediate_layer_model.predict(x)
        vector = intermediate_output[0]
        image_vectors[image_path] = vector
    
    embeddings = np.stack(list(image_vectors.values()))
    with open('res50.pkl','wb') as f:
        pickle.dump(embeddings, f)
    
def img2vec_xception(imagedir):
    
    image_paths = glob.glob(str(imagedir) + '/*.jpg')
    image_paths = image_paths[:400]
    
    _IMAGE_NET_TARGET_SIZE = (299, 299)
    model = xception.Xception(weights='imagenet')
    layer_name = 'avg_pool'
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

    image_vectors = {}
    global image_path
    for c,image_path in enumerate(image_paths):
        print("\r" + str(c+1) + "/" + str(len(image_paths)), end="")
        img = image.load_img(image_path, target_size=_IMAGE_NET_TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = xception.preprocess_input(x)
        intermediate_output = intermediate_layer_model.predict(x)
        vector = intermediate_output[0]
        image_vectors[image_path] = vector
        
    
    embeddings = np.stack(list(image_vectors.values()))
    with open('xception.pkl','wb') as f:
        pickle.dump(embeddings, f)


def run_tsne(embeddings_file, n_iter, perplexity):
    with open(embeddings_file, 'rb') as pkl:
        embeddings = pickle.load(pkl)
    tsne = TSNE(n_components=2, n_iter=n_iter, perplexity=perplexity, verbose = 1)
    tsne_result = tsne.fit_transform(embeddings)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
    plt.scatter(tsne_result_scaled[:,0], tsne_result_scaled[:,1])
    
    outfilename = str(embeddings_file.split(".")[0] + "_tsne.pkl")
    with open(outfilename, 'wb') as pkl:
        tsne_result_scaled = pickle.dump(tsne_result, pkl)
    

def run_umap(embeddings_file,n_neighbours,min_dist,metric):
    with open(embeddings_file, 'rb') as pkl:
        embeddings = pickle.load(pkl)
    umap_r = umap.UMAP(n_neighbors=n_neighbours,min_dist=min_dist,metric=metric,n_components=2)
    umap_result = umap_r.fit_transform(embeddings)
    umap_result_scaled = StandardScaler().fit_transform(umap_result)
    plt.scatter(umap_result_scaled[:,0], umap_result_scaled[:,1])

    outfilename = str(embeddings_file.split(".")[0] + "_umap.pkl")
    with open(outfilename, 'wb') as pkl:
        umap_result_scaled = pickle.dump(umap_result_scaled, pkl)
    
def image_map(imagedir, final_data_file):
    
    image_paths = glob.glob(str(imagedir) + '/*.jpg')
    image_paths = image_paths[:400]

    with open(final_data_file, 'rb') as pkl:
        final_result_scaled = pickle.load(pkl)

    images = []
    for image_path in image_paths:
      image = cv2.imread(image_path, 3)
      b,g,r = cv2.split(image)           # get b, g, r
      image = cv2.merge([r,g,b])         # switch it to r, g, b
      image = cv2.resize(image, (50,50))
      images.append(image)

    fig, ax = plt.subplots(figsize=(20,15))
    artists = []

    for xy, i in zip(final_result_scaled, images):
      x0, y0 = xy
      img = OffsetImage(i, zoom=.7)
      ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
      artists.append(ax.add_artist(ab))
    ax.update_datalim(final_result_scaled)
    ax.autoscale(enable=True, axis='both', tight=True)
    plt.savefig("imagemonster_plot.pdf")
    plt.show()
    
if __name__ == '__main__':
    main()
