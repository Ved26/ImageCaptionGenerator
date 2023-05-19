#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
import matplotlib.pyplot as plt
from keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from keras.models import Model, load_model
from keras.utils.data_utils import pad_sequences
from DataExtraction import word_to_idx,idx_to_word,max_len,vocab_size

model = load_model('models/model_30.h5')



resnet = ResNet50(weights="imagenet",input_shape=(224,224,3))
feature_extractor = Model(resnet.input,resnet.layers[-2].output)




def preprocess_image(img):
    img = image.load_img(img,target_size=(224,224))  #loads image
    img = image.img_to_array(img) #converts the image into n-dim array
    img = np.expand_dims(img,axis=0)   #adds extra dimension requried for the model
    img = preprocess_input(img)  #preprocessing the input according the format on which model has been trained 
    return img

def encode_image(img):
    img = preprocess_image(img) 
    feature_vector = feature_extractor.predict(img)  #predict to get feautre vector from our model
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])  #convert into 1-D array
    return feature_vector




def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += ' '+word
    
        if word == 'endseq':
            break

    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption


def caption_this_image(image):
    encodings = encode_image(image)
    caption = predict_caption(encodings)
    return caption 

'''
test_path = '../datasets/test_imgs'
n = random.randint(1, 6)
img = '../datasets/test_imgs/' + "{}.jpeg".format(n)
encodings = encode_image(img)

i = plt.imread(img)
plt.imshow(i)
plt.axis("off")
plt.show()
print(predict_caption(encodings))
'''