from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import streamlit as st
from PIL import Image
import heapq as hq

@st.cache(allow_output_mutation=True)
def get_model():
        model = load_model('inception.hdf5')
        print('Model Loaded')
        return model 

        
def predict(image):
        loaded_model = get_model()

        image = image.resize((224, 224))
        image = np.asarray(image)
        image = image/255.0
        image = np.reshape(image,[1,224,224,3])
        #st.write("{}".format(image.shape))
        classes = loaded_model.predict(image)#.argmax()

        ans = []
        for i, n in enumerate(classes[0]):
          hq.heappush(ans, (n, i))
          if len(ans)>5:
            hq.heappop(ans)
        return ans
sign_names = ('adonis', 'american snoot','an 88', 'banded peacock', 'beckers white', 'black hairstreak', 'cabbage white', 'chestnut', 'clodius parnassian', 'clouded sulphur',
                  'copper tail', 'crecent', 'crimson patch', 'eastern coma', 'gold banded', 'great eggfly', 'grey hairstreak', 'indra swallow', 'julia', 'large marble',
                  'malachite', 'mangrove skipper', 'metalmark', 'monarch', 'morning cloak', 'orange oakleaf', 'orange tip', 'orchard swallow', 'painted lady', 'paper kite', 
                  'peacock', 'pine white', 'pipevine swallow', 'purple hairstreak', 'question mark', 'red admiral', 'red spotted purple', 'scarce swallow', 'silver spot skipper', 'sixspot burnet',
                   'skipper', 'sootywing', 'southern dogface', 'straited queen', 'two barred flasher', 'ulyses', 'viceroy', 'wood satyr', 'yellow swallow tail', 'zebra long wing')

st.title("Butterfly Species Classifier")
st.write("Dataset from https://www.kaggle.com/pnkjgpt/butterfly-classification-dataset")

uploaded_file = st.file_uploader("Choose an image of a butterfly", type=["jpg", "png", "jpeg"])
if uploaded_file:
        image = Image.open(uploaded_file)
        st.write("")
        st.image(image, caption='Image', use_column_width=False,)
        st.write("")
        if st.button('Predict'):
                st.write("The given image has been classified as:")
                label = predict(image)
                #res = sign_names.get(label)
                for p, c in sorted(label, reverse = True):
                  st.write('{0} : {1}%'.format(sign_names[c], round(p*100, 3)))
