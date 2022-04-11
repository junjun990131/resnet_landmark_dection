
import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf

from keras.models import Model, model_from_json, Sequential
from keras.models import load_model

from keras.layers import Input, add, Flatten, Dense, Concatenate, Embedding, Reshape, Lambda,MaxPooling2D,LeakyReLU
from keras.layers.core import Activation
# from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import adam_v2
# from keras.utils import multi_gpu_model
from keras.layers import LeakyReLU, Dropout
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import rmsprop_v2
from keras import backend as K
# from libs.pconv_layer import PConv2D


class CNN(object):

    def __init__(self, img_rows=256, img_cols=256, vgg_weights="imagenet", net_name='default', gpus=1):
        # 
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.net_name = net_name
        self.gpus = gpus

        # assert self.img_rows >= 128, 'Height must be >128 pixels'
        # assert self.img_cols >= 128, 'Width must be >128 pixels'

        self.current_epoch = 0
        self.mask_dim = 5
        self.depth = 64 + 64 + 64 + 64
        # self.image_shape(28,28)
        self.d_depth = 64
        self.dropout = 0.4
        self.latent_dim =100
        self.channel = 1
        self.cnn = self.build_cnn_model()
        # self.gen = self.build_gen_model()
        # self.dcrm = self.build_dcrm_model()
        # self.dcgan = self.build_dcgan_model(gen_model = self.gen, dcrm_model = self.dcrm)
        # self.img_rows = 28
        # self.img_cols = 28
        # if self.gpus > 1:
        #     self.gen = multi_gpu_model(gen, gpus = self.gpus)
        #     self.dcrm = multi_gpu_model(dcrm, gpus = self.gpus)
        #     self.dcgan = multi_gpu_model(dcgan, gpus = self.gpus)

        self.compile_model()     


    def build_cnn_model(self):
        kernel=(3,3)
        model = Sequential()
        model.add(Conv2D(filters=128,kernel_size=kernel,padding='same',input_shape=(256,256,3),activation='relu'))
        model.add(Conv2D(filters=128,kernel_size=kernel,padding='same',input_shape=(256,256,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=128,kernel_size=kernel,padding='same',activation='relu'))
        model.add(Conv2D(filters=128,kernel_size=kernel,padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=256,kernel_size=kernel,padding='same',activation='relu'))
        model.add(Conv2D(filters=256,kernel_size=kernel,padding='same',activation='relu'))
        model.add(Conv2D(filters=256,kernel_size=kernel,padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=512,kernel_size=kernel,padding='same',activation='relu'))
        model.add(Conv2D(filters=512,kernel_size=kernel,padding='same',activation='relu'))
        model.add(Conv2D(filters=512,kernel_size=kernel,padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=512,kernel_size=kernel,padding='same',activation='relu'))
        model.add(Conv2D(filters=512,kernel_size=kernel,padding='same',activation='relu'))
        model.add(Conv2D(filters=512,kernel_size=kernel,padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.25))
        # model.add(Dense(4096, activation='relu'))
        # model.add(Dropout(0.5))                                       
        # output layer
        model.add(Dense(1000,activation='softmax'))
        return model

    def compile_model(self):
        # rmsprop_v2.RMSprop(lr=0.01, decay=1e-8)
        # self.dcgan.compile(loss = 'mse', optimizer = rmsprop_v2.RMSprop(lr=0.0005, decay=1e-8))#,0.5))
        # self.dcrm.compile(loss = 'mse', optimizer = rmsprop_v2.RMSprop(lr=0.0005, decay=1e-8))#,0.5))binary_crossentropy
        # self.gen.compile(loss = 'mse', optimizer = rmsprop_v2.RMSprop(lr=0.0008, decay=1e-8))
        self.cnn.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy']) 
        print('cnn')
        self.cnn.summary()
        # print('gen')
        # self.gen.summary()
        # print('dcrm')
        # self.dcrm.summary()
        # print('gan')
        # self.dcrm.trainable = False
        # self.dcgan.summary()  

    def cnn_evaluate(self, sample, **kwargs):
        self.predict_x=self.cnn.predict(sample, **kwargs)
        return np.argmax(self.predict_x,axis=1)
        
    def train_on_batch_CNN(self, *args, **kwargs ):
        res = self.cnn.train_on_batch(*args, **kwargs)
        return res

    def set_G2_trainable(self, train_able):
        self.dcrm.trainable = train_able
    
    def set_G1_trainable(self, train_able):
        self.gen.trainable = train_able

    def to_json(self, *args, **kwargs):

        return self.model.to_json(*args, **kwargs)
        
    def save_weights(self, *args, **kwargs):
        return self.cnn.save_weights(*args, **kwargs)

    # def summary(self):
    #     """Get summary of the model"""
    #     print("生成模型")
    #     print(self.gen.summary())
    #     print("判别模型")
    #     print(self.dcrm.summary())
    #     print("整体模型")
    #     print(self.dcgan.summary())

    def load(self, c_path, lr=0.0002):

        # Create model
        self.cnn = self.build_cnn_model()

        self.compile_model(self.cnn)  

        # Load weights into model
        print("model loaded,start to load paramerter")
        # self.gen.load_weights(g_path) 
        # self.dcrm.load_weights(d_path)
        self.cnn.load_weights(c_path)
        print('complate')

    

    @staticmethod
    def PSNR(y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        
        Our input is scaled with be within the range -2.11 to 2.64 (imagenet value scaling). We use the difference between these
        two values (4.75) as MAX_I        
        """        
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0) 
        return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0) 

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
    
    @staticmethod
    def gram_matrix(x, norm_by_channels=False):
        """Calculate gram matrix used in style loss"""
        
        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
        
        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        
        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        
        # Normalize with channels, height and width
        gram = gram /  K.cast(C * H * W, x.dtype)
        
        return gram
    
    # Prediction functions
    ######################
    def cnn_predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.cnn.predict(sample, **kwargs)
