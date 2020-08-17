import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from pycaret.classification import load_model, predict_model
import pandas as pd
import cv2
import os

threshold = 0.95
width = 700

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model_teachable = tensorflow.keras.models.load_model('keras_model1.h5')
model_ml = load_model('pycaret_model')

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
        prediction = model_teachable.predict(data)
        return prediction
    else:
        pass


def normalize_histograms(im):  # normalizes the histogram of images
    im1 = im.copy()
    for i in range(3):
        imi = im[:, :, i]
        # print(imi.shape)
        minval = np.min(imi)
        maxval = np.max(imi)
        # print(minval,maxval)
        imrange = maxval - minval
        im1[:, :, i] = (255 / (imrange + 0.0001) * (
                imi - minval))  # imi-minval will turn the color range between 0-imrange, and the scaleing will stretch the range between 0-255
    return im1


######################################################################
# This following function reads the images from file,
# auto crops the image to its relevant content, then normalizes
# the histograms of the cropped images
######################################################################

def read_and_process_image(filename):
    path = "analizer.png"
    im_array = np.array(filename)
    cv2.imwrite(path, cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))
    im = cv2.imread(path)

    # The following steps re needed for auto cropping the black paddings in the images

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert 2 grayscale
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # turn it into a binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    if len(contours) != 0:
        # find the biggest area
        cnt = max(contours, key=cv2.contourArea)

        # find the bounding rect
        x, y, w, h = cv2.boundingRect(cnt)

        crop = im[y:y + h, x:x + w]  # crop image
        # crop1=cv2.resize(crop,(im_size,im_size)) # resize to im_size X im_size size
        crop = normalize_histograms(crop)
        return crop
    else:
        return (normalize_histograms(im))


##################################################################################
#### The following functions are for extracting features from the images #########
##################################################################################

# histogram statistics (mean, standard deviations, energy, entropy, log-kurtosis)

def histogram_statistics(hist):
    # hist= cv2.calcHist([gr],[0],None,[256],[0,256])
    hist = hist / np.sum(hist)  # probabilities
    hist = hist.reshape(-1)
    hist[hist == 0] = 10 ** -20  # replace zeros with a small number
    mn = np.sum([i * hist[i] for i in range(len(hist))])  # mean
    std_dev = np.sqrt(np.sum([((i - mn) ** 2) * hist[i] for i in range(len(hist))]))  # standard deviation
    energy = np.sum([hist[i] ** 2 for i in range(len(hist))])  # energy
    entropy = np.sum([hist[i] * np.log(hist[i]) for i in range(len(hist))])  # entropy
    kurtosis = np.log(np.sum([(std_dev ** -4) * ((i - mn) ** -4) * hist[i] for i in range(len(hist))]))  # kurtosis
    return [mn, std_dev, energy, entropy, kurtosis]


#################################################################
# create thresholding based features, the idea is to hand engineer some features based on adaptive thresholding.
# After looking at the images it appeared  that adaptive thresholding may
# leave different artifacts in the processed images, we can extract several features from these artifacts            
##################################################################

def thresholding_based_features(im, imsize, quartiles):
    im = cv2.resize(im, (imsize, imsize))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    w = 11  # window
    t = 5  # threshold
    counts = []
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, w,
                               t)  # adaptive gaussian threshold the image
    th = cv2.bitwise_not(
        th)  # invert the image (the black pixels will turn white and the white pixels will turn black)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # find cntours in the image
    # print(len(contours))

    q = np.zeros(len(quartiles))  # quartiles of contours will be stored here

    for cnt in contours:
        area = cv2.contourArea(cnt)  # calculate the area of the contours
        if area < 40000:  # Exclude contours that are too big, generally these are the image outlines
            counts.append(area)
    if len(counts) > 1:
        q = np.quantile(np.array(counts), quartiles)  # contour quartiles

    return (q, len(counts), np.sum(th) / (255 * th.shape[0] * th.shape[
        1]))  # return the contour quartiles, number of contours, proportion of white pixels in the thresholded images
    # counts.append(np.sum(th)/(normalizing_factor*(th.shape[0]*th.shape[1])))


##########################################################################
############ The following code creates the various features #############
##########################################################################

# color averages
def various_features(image_ml):
    B = []
    G = []
    R = []

    # mini 16 bin histograms
    hist_B = []
    hist_G = []
    hist_R = []

    # statistics fom full 256 bin shitogram
    hist_feat_B = []
    hist_feat_G = []
    hist_feat_R = []
    hist_feat_GS = []

    # thresholding based features
    mean_pixels = []  # proportion of white pixels
    contour_quartiles = []  # contour area quartiles
    no_of_contours = []  # total number of contours

    quartiles = np.arange(0.1, 1, 0.1)  # contour area quartiles
    bins = 16  # mini histogram bins

    im = read_and_process_image(image_ml)
    # im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    # im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])

    # convert the YUV image back to RGB format
    # im = cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)

    # median color
    B.append(np.median(im[:, :, 0]))
    G.append(np.median(im[:, :, 1]))
    R.append(np.median(im[:, :, 2]))

    # histograms
    hist_B.append(cv2.calcHist([im], [0], None, [bins], [0, 256]) / (im.size / 3))
    hist_G.append(cv2.calcHist([im], [1], None, [bins], [0, 256]) / (im.size / 3))
    hist_R.append(cv2.calcHist([im], [2], None, [bins], [0, 256]) / (im.size / 3))

    # more histogram features

    hist_feat_B.append(histogram_statistics(cv2.calcHist([im], [0], None, [256], [0, 256])))
    hist_feat_G.append(histogram_statistics(cv2.calcHist([im], [1], None, [256], [0, 256])))
    hist_feat_R.append(histogram_statistics(cv2.calcHist([im], [2], None, [256], [0, 256])))

    gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gr = cv2.equalizeHist(gr)
    hist_feat_GS.append(histogram_statistics(cv2.calcHist([gr], [0], None, [256], [0, 256])))

    # threshold featues
    q, nc, m = thresholding_based_features(im, 256, quartiles)
    mean_pixels.append(m)
    contour_quartiles.append(q)
    no_of_contours.append(nc)

    # create feature vectors
    width_of_features = 3 * bins + len(quartiles) + 2 + 20  # 20 features are histogram statistics

    X = np.zeros((1, width_of_features))   # this is where all features will be stored

    X[0, 0:bins] = hist_B[0].reshape(-1)
    X[0, bins:2 * bins] = hist_G[0].reshape(-1)
    X[0, 2 * bins:3 * bins] = hist_R[0].reshape(-1)
    X[0, 3 * bins:3 * bins + len(quartiles)] = contour_quartiles[0].reshape(-1)
    X[0, 3 * bins + len(quartiles)] = mean_pixels[0]
    X[0, 3 * bins + len(quartiles) + 1] = no_of_contours[0]
    start = 3 * bins + len(quartiles) + 2
    X[0, start:start + 5] = hist_feat_B[0]
    X[0, start + 5:start + 10] = hist_feat_G[0]
    X[0, start + 10:start + 15] = hist_feat_R[0]
    X[0, start + 15:start + 20] = hist_feat_B[0]
    print(X)

    return X


#######################################################################
########### Divide the dataset into 70%train and 30% test data########
######################################################################

# Parameters to pycaret
# index = int(len(labels1) * 0.7)
#
# X_train = X[:index, :]
# Y_train1 = labels1[:index]
#
# X_test = X[index:, :]
# Y_test1 = labels1[index:]


def teachable_machine_model():
    st.subheader("Validación con modelo generado con Teachable Machine")
    # Upload image
    uploaded_file = st.file_uploader("Ingrese una imagen para analizar", type='png')

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="El archivo ha sido cargado exitosamente...", width=width)
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


def machine_learning_model():
    st.subheader("Clasificación mediante el modelo de aprendizaje de máquina")
    uploaded_file = st.file_uploader("Ingrese una imagen para analizar", type=None)

    if uploaded_file is not None:
        image_ml = Image.open(uploaded_file)
        st.image(image_ml, caption="El archivo ha sido cargado exitosamente...",
                 width=width)


        # Send to extract image caracters in dataset
        image_processing = various_features(image_ml)
        df_image_processing = pd.DataFrame(image_processing)
        predictions = predict_model(model_ml, df_image_processing)
        os.remove("/mnt/napster_disk/saturdays_ai/segunda_edicion/Proyecto/saturdays_second_edition/analizer.png")
        result = predictions['Label']
        score = predictions['score']

        if result == 0:
            st.success("Es catarata")
        elif result == 1:
            st.success("Result")
        elif result == 2:
            st.success("Result")
        elif result == 3:
            st.success("Result")



def main():
    # Desabilita el FileUploaderEncodingWarning This change will go in effect after August 15, 2020.
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Detector de Cataratas ")

    # agrega la barra de al lado para seleccionar el modelo que deseamos utilizar
    activities = ["Modelo Teachable Machine", "Modelo ML"]
    choice = st.sidebar.selectbox("Seleccione el modelo a utilizar:", activities)

    if choice == "Modelo Teachable Machine":
        teachable_machine_model()
    elif choice == "Modelo ML":
        machine_learning_model()


if __name__ == "__main__":
    main()
