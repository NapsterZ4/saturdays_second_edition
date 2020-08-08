import streamlit as st

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

threshold = 0.95
width = 700

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model1.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


@st.cache(allow_output_mutation=True)
def clasify(image):
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    if image_array.shape == (224, 224, 3):
        # display the resized image
        # image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        return prediction
    else:
        pass


def main():
    # Desabilita el FileUploaderEncodingWarning This change will go in effect after August 15, 2020.
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Detector de Cataratas ")

    # agrega la barra de al lado para seleccionar el modelo que deseamos utilizar
    activities = ["Modelo Teachable Machine", "Modelo CNN", "Modelo ML"]
    choice = st.sidebar.selectbox("Seleccione el modelo a utilizar:", activities)

    if choice == "Modelo Teachable Machine":
        st.subheader("Validación con modelo generado con Teachable Machine")
        # Upload image
        uploaded_file = st.file_uploader("Ingrese una imagen para analizar", type='png')

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="El archivo ha sido cargado exitosamente", width=width)
            pred = clasify(image)

            if pred is not None:
                # Cataract
                if pred[0][0] >= threshold:
                    st.success("Catarata with probability: {}".format(str(pred[0][0])))
                # Normal
                elif pred[0][1] >= threshold:
                    st.success("Normal with probability: {}".format(str(pred[0][1])))
                else:
                    st.warning(
                        "La probabilidad no es confiable en la validacion de la imagen con las siguientes "
                        "valoraciones: \n "
                        "Catarata: {0} \n"
                        "Normal: {1}".format(pred[0][0], pred[0][1]))
            else:
                st.warning("ERROR image shape is not support")

            image.close()

    if choice == "Modelo CNN":
        st.subheader("Clasificación mediante el modelo de red neuronal convolucional")
        uploaded_file = st.file_uploader("Ingrese una imagen para analizar", type=None)
        image = Image.open(uploaded_file)

        st.image(image, caption="El archivo ha sido cargado exitosamente pero aún no está lista la clasificación CNN",
                 width=width)

    if choice == "Modelo ML":
        st.subheader("Clasificación mediante el modelo de aprendizaje de máquina")
        uploaded_file = st.file_uploader("Ingrese una imagen para analizar", type=None)
        image = Image.open(uploaded_file)
        st.image(image, caption="El archivo ha sido cargado exitosamente pero aún no está lista la clasificación ML",
                 width=width)


if __name__ == "__main__":
    main()
