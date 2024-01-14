import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Function to classify the bird species in an image
def classify_bird_species(image):
    image = load_img(img_path,target_size=(224,224,3))
    image = Image.open(image_path)
    image = image.resize((224, 224))  # ResNet50 input size
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Make predictions
    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Print the top predicted bird species
    print("Predictions:")
    for _, species, confidence in decoded_predictions:
        print(f"{species}: {confidence * 100:.2f}%")



def run():
    img1 = Image.open('C:/Users/Administrator/Downloads/logob.jpg')
    img1 = img1.resize((350,350))
    st.image(img1,use_column_width=False)
    st.title("Birds Species Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Birds species classification"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path ='C:/Users/Administrator/Downloads/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = classify_bird_species(image)
            st.success("Predicted Bird is: "+result)

run()