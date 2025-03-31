import streamlit as st
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans
st.title('Image Segmentation with Kmeans')
url = st.text_input('URL: ')
if url:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    K = st.slider('K', 1,10,1)

    img = img.reshape(-1,img.shape[-1])
    kmeanModel = KMeans(n_cluster = K, init = 'k-means++')
    kmeanModel.fit(img)
    kmeanModel.predict(img)

    centers = kmeanModel.cluster_centers_
    labels = kmeanModel.labels_
    new_img = centers[labels]
    new_img = new_img.astype(np.uint8)
    new_img = Image.fromarray(new_img)
    
    st.img(new_img)

