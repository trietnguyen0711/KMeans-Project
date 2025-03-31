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
    img = np.array(Image.open(BytesIO(response.content)))
    K = st.slider('K', 1,10,1)
    
    h, w , c = img.shape
    img = img.reshape(-1,c)
    kmeanModel = KMeans(n_clusters = K, init = 'k-means++')
    kmeanModel.fit(img)
    kmeanModel.predict(img)

    centers = kmeanModel.cluster_centers_
    labels = kmeanModel.labels_
    new_img = centers[labels]
    new_img = new_img.reshape(h,w,c).astype(np.uint8)
    new_img = Image.fromarray(new_img)
    
    st.image(new_img)

