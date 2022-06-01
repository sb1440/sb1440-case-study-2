#Image processing libraries
import cv2
from PIL import Image
from PIL import ImageOps

import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Activation, Dropout, BatchNormalization, ReLU, Concatenate, Conv2DTranspose
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.models import Model

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pytesseract
from io import StringIO

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

#defining custom architecture
def DenseNet121(input_shape):
    
    dense = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    dense.trainable = False
    layer_names = ['conv3_block12_concat', 'conv4_block24_concat',  'relu']
    outputs = [dense.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([dense.input], outputs)
    
    return model 

class TableDecoder(tf.keras.layers.Layer):  #kernels = [(1,1), (2,2)], #strides = [(1,1), (2,2)]

    def __init__(self, kernels, strides):
        super().__init__()
        self.conv_7 = Conv2D(kernel_size=kernels[0], filters=64, strides=strides[0], kernel_regularizer=tf.keras.regularizers.l2(0.002)) #First convolutional layer in Table_Decoder

        self.upsample_1_table = Conv2DTranspose(filters=64, kernel_size = kernels[1], strides = strides[1], padding='same')
        self.upsample_2_table = Conv2DTranspose(filters = 64, kernel_size = kernels[1], strides = strides[1], padding='same')
        self.upsample_3_table = Conv2DTranspose(filters = 64, kernel_size = kernels[1], strides = strides[1], padding='same')
        self.upsample_4_table = Conv2DTranspose(filters = 64, kernel_size = kernels[1], strides = strides[1], padding='same')
        self.upsample_5_table = Conv2DTranspose(filters=1, kernel_size=kernels[1], strides=strides[1], padding='same', activation='sigmoid')

    def call(self, input_, pool3, pool4):
        
        x = self.conv_7(input_)  #input.shape = (None, 32, 32, 256) and output.shape = (None, 32, 32, 256)
        
        x = self.upsample_1_table(x)  #after upsampling output.shape = (None, 64, 64, 256)
        x = Concatenate()([x, pool4]) 
       
        x = self.upsample_2_table(x) #after upsampling output.shape = (None, 128, 128, 256)
        x = Concatenate()([x, pool3]) 
        
        x = self.upsample_3_table(x)
        x = self.upsample_4_table(x)
        x = self.upsample_5_table(x) #after upsampling output.shape = (None, 1024, 1024, 1)
        
        return x
    
class ColumnDecoder(tf.keras.layers.Layer):    #kernels = [(1,1), (2,2)], #strides = [(1,1), (2,2)]
    
    def __init__(self, kernels, strides):
        super().__init__()
        self.conv_7 = Conv2D(kernel_size=kernels[0], filters=64, strides=strides[0], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.004), kernel_initializer='he_normal') #first conv in column_decoder
        self.drop = Dropout(0.8)
        self.conv_8 = Conv2D(kernel_size=kernels[0], filters=64, strides=strides[0], kernel_regularizer=tf.keras.regularizers.l2(0.004), kernel_initializer='he_normal',) #second conv in column_decoder
        
        self.upsample_1_column = Conv2DTranspose(filters = 64, kernel_size = kernels[1], strides = strides[1], padding='same')
        self.upsample_2_column = Conv2DTranspose(filters = 64, kernel_size = kernels[1], strides = strides[1], padding='same')
        self.upsample_3_column = Conv2DTranspose(filters = 64, kernel_size = kernels[1], strides = strides[1], padding='same')
        self.upsample_4_column = Conv2DTranspose(filters = 64, kernel_size = kernels[1], strides = strides[1], padding='same')
        self.upsample_5_column = Conv2DTranspose(filters = 1, kernel_size = kernels[1], strides = strides[1], padding='same', activation='sigmoid')

    def call(self, input_, pool3, pool4):
        
        x = self.conv_7(input_) #input.shape = (None, 32, 32, 256) and output.shape = (None, 32, 32, 256)
        x = self.drop(x)  
        x = self.conv_8(x) #after second convultion output.shape = (None, 32, 32, 256)

        x = self.upsample_1_column(x) #after upsampling output.shape = (None, 64, 64, 256)
        x = Concatenate()([x, pool4]) 
       
        x = self.upsample_2_column(x) #after upsampling output.shape = (None, 128, 128, 256)
        x = Concatenate()([x, pool3]) 
        
        x = self.upsample_3_column(x)
        x = self.upsample_4_column(x)
        x = self.upsample_5_column(x) #after upsampling output.shape = (None, 1024, 1024, 1)

        return x
    
class TableNet(tf.keras.Model):
    def __init__(self, branch_kernels=[(1,1), (2,2)], branch_strides=[(1,1), (2,2)], input_shape=(1024, 1024, 3)):
        super(TableNet, self).__init__()

        self.feature_extractor = DenseNet121(input_shape)
            
        self.conv_1= Conv2D(filters=64, kernel_size=(1,1), activation='relu', name='common_conv_1', kernel_regularizer=l2(0.04))
        self.drop_1= Dropout(0.8)
        self.conv_2= Conv2D(filters=64, kernel_size=(1,1), activation='relu', name='common_conv_2', kernel_regularizer=l2(0.04))
        self.drop_2= Dropout(0.8)
        
        self.table_branch = TableDecoder(branch_kernels, branch_strides)
        self.column_branch = ColumnDecoder(branch_kernels, branch_strides)
        
    def call(self, x):
        pool3, pool4, pool5 = self.feature_extractor(x)
        
        out = self.conv_1(pool5)
        out = self.drop_1(out)
        out = self.conv_2(out)
        out = self.drop_2(out)
        
        table_output = self.table_branch(out, pool3, pool4)
        column_output = self.column_branch(out, pool3, pool4)
        
        return table_output, column_output
   

def get_pred_masks(model, input_image_path):
    '''predicting the table and column masks for given image using the given model'''
    
  
    image_original = Image.open(input_image_path)
    image = image_original.convert('L').convert('RGB')                   #converting the input image into greyscale and again into RGB to                                                                             remove the different colors of the image.
    image = ImageOps.equalize(image, mask=None)                     #histogram equilization
    input_image_processed = np.array(image.resize((1024,1024)))    #resizing Image to new shape

        
    input_image = np.expand_dims(input_image_processed/255.0, axis=0) #expanding the dimensions to add batch_size such that it is compatible
    
    table_mask_pred, column_mask_pred = model(input_image) #predicting the masks using trained model 
    
    
    table_mask = np.squeeze(tf.cast((table_mask_pred>0.5), dtype=tf.int32).numpy(), axis=0).astype(np.uint8)
    column_mask = np.squeeze(tf.cast((column_mask_pred>0.5), dtype=tf.int32).numpy(), axis=0).astype(np.uint8)
    
    return input_image_processed, table_mask, column_mask

#define approximation using cv2 commands



    
def fixMasks(table_mask):
    '''Approximating the table surface such that it is a 4 sided polygon either rectangle or square'''
    
    
    #contours is a list where it contains boundary the co-ordinates (x,y) which are approximated to save the memory
    contours, table_heirarchy = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    table_contours = []

    for c in contours:
        
        if cv2.contourArea(c)>9000:
            table_contours.append(c)
    
    if len(table_contours) == 0:
        return None
    
    table_boundRect = [None]*len(table_contours)

    for i, c in enumerate(table_contours):     #approximating the table boundaries
        polygon = cv2.approxPolyDP(c, 1, True)
        table_boundRect[i] = cv2.boundingRect(polygon)
    
    #table bounding Box
    table_boundRect.sort()
    
    return  table_boundRect



@st.cache(allow_output_mutation=True)
def load_model():

    model = TableNet()
    model.load_weights(r'C:\Users\ADMIN\Documents\Case study 2\saved_models\model_dense_diceloss\cp-0020.ckpt')
    return model


def predict(image_path):
    '''For a given input processed image and its table boundaries, it returns the text'''
    
    with st.spinner('Processing..'):
        image, table_mask, column_mask = get_pred_masks(dense_loaded, image_path)
        table_boundRect = fixMasks(table_mask)
        
        #draw bounding boxes
        color = (0,2,255)
        thickness = 4
        for x,y,w,h in table_boundRect:
            cv2.rectangle(image, (x,y),(x+w,y+h), color, thickness)

        st.image(image)
        
        for i,(x,y,w,h) in enumerate(table_boundRect): #for each table in the givne image
            image_crop = image[y:y+h,x:x+w] #cropping the image with respect to the table boundaries
            data = pytesseract.image_to_string(image_crop) #extracting the text using ocr
            if data:
                try:
                    df = pd.read_csv(StringIO(data),sep=r'\|',engine='python')
                    st.write(f" ## Table {i+1}")
                    st.write(df)
                except pd.errors.ParserError:
                    try:
                        df = pd.read_csv(StringIO(data),delim_whitespace=True,engine='python')
                        st.write(f" ## Table {i+1}")
                        st.write(df)
                    except pd.errors.ParserError:
                        st.write(f" ## Table {i+1}")
                        st.write(data)
            else:
                st.write('No data is identified by tesseract OCR')
  
    
    
                    
with st.spinner("Loading Last Checkpoint"):
    dense_loaded = load_model()
  
st.header("Data Extraction from Tables")
#upload files
file = st.file_uploader("Please upload an Image file")

if file is not None:
    predict(file)