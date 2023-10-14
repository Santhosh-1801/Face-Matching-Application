from keras_vggface.utils import preprocess_input 
from keras_vggface.vggface import VGGFace 
from src.utils.all_utils import read_yaml,create_directory 
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np 



config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#upload

upload_image_dir = artifacts['upload_image_dir']
upload_path = os.path.join(artifacts_dir, upload_image_dir)

#pickle format data dir 
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name) 

#Feature Path 
feature_extractor_dir=artifacts['feature_extraction_dir']
extracted_features_name=artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extractor_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)


model_name=params['base']['BASE_MODEL']
include_tops=params['base']['include_top']
poolings=params['base']['pooling']

detector=MTCNN()
model=VGGFace(model=model_name,include_top=include_tops,input_shape=(224,224,3),pooling=poolings) 
filenames = pickle.load(open(pickle_file,'rb'))
feature_list=pickle.load(open(features_name,'rb'))






def save_uploaded_image(uploaded_image):
    try:
        create_directory(dirs=[upload_path])

        with open(os.path.join(upload_path,uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False
    

def recommend(feature_list,features):
    similarity=[]
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])
    
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


    
def extract_features(img_path,model,detector):
    img=cv2.imread(img_path)
    results=detector.detect_faces(img)

    x,y,width,height=results[0]['box']

    face=img[y:y+height,x:x+width]

    #extract its features 

    image=Image.fromarray(face)
    image=image.resize((224,224))

    face_array=np.asarray(image)

    face_array=face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

st.set_page_config(page_title = "Face Matching Application")
st.header('Face Matching Application')
st.title("Which actor or actress does your face look like") 


uploaded_image=st.file_uploader('Choose an Image') 


if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image=Image.open(uploaded_image)
        features=extract_features(os.path.join(upload_path,uploaded_image.name),model,detector)
        index_pos=recommend(feature_list,features) 
        predicted_actor=" ".join(filenames[index_pos].split("\\")[1].split('_')) 

        col1,col2=st.columns(2)

        with col1:
            st.header("Your Uploaded Image")
            st.image(display_image)

        with col2:
            st.header("Seems like "+predicted_actor)
            st.image(filenames[index_pos],width=300)
