
import os
import sys
import numpy as np
from datetime import datetime
import keras
import tensorflow as tf

from keras.models import Model, model_from_json, Sequential
from keras.models import load_model

from keras.layers import Input, add, Flatten, Dense, Concatenate, Embedding, Reshape, Lambda,MaxPooling2D,LeakyReLU,GlobalMaxPooling2D
from keras.layers.core import Activation
# from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import adam_v2
# from keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU, Dropout
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import rmsprop_v2
from keras import backend as K
from tensorflow.keras.applications import resnet
from keras.metrics import top_k_categorical_accuracy

# from libs.pconv_layer import PConv2D


class CNN(object):

    def __init__(self, img_rows=256, img_cols=256, vgg_weights="imagenet", net_name='default', gpus=1):
        # 
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.net_name = net_name
        self.gpus = gpus
        self.vgg = 19


        self.current_epoch = 0
        self.mask_dim = 5
        self.depth = 64 + 64 + 64 + 64
        # self.image_shape(28,28)
        self.d_depth = 64
        self.dropout = 0.4
        self.latent_dim =100
        self.channel = 1
        self.vgg_model = self.build_vgg_model()
        # self.dcgan = self.build_dcgan_model(gen_model = self.gen, dcrm_model = self.dcrm)
        # self.img_rows = 28
        # self.img_cols = 28
        if self.gpus > 1:
            self.vgg_model = multi_gpu_model(self.vgg_model, gpus = self.gpus)
            # self.dcrm = multi_gpu_model(dcrm, gpus = self.gpus)
            # self.dcgan = multi_gpu_model(dcgan, gpus = self.gpus)

        self.compile_model()     


    
    def build_vgg_model(self):
        # Img = Input(shape = (self.img_rows, self.img_cols, 1))
        # model=resnet(weights="imagenet",include_top=False,input_shape=(256, 256, 3))
        model=resnet.ResNet50(include_top=False,weights='imagenet',input_shape=(224, 224, 3)) #,pooling = 'avg' weights='imagenet',
        model.trainable=True
        x = model.output
        x = GlobalMaxPooling2D()(x)
        # x = BatchNormalization()(x)
        # x = Conv2D(filters = 512, kernel_size=(1,1),padding='same',activation='relu')(x)
        
        x = Dense(2048, activation='relu', name='fc1')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(1000, activation='softmax')(x)
        
        res_model = Model(inputs=model.input, outputs=predictions) 

        return res_model
    def acc_top5(self,y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)

    def compile_model(self):


        self.vgg_model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.00001, beta_1=0.9,beta_2=0.99,epsilon=1e-08,decay=1e-6), metrics=['acc',self.acc_top5])
        self.vgg_model.summary()




    def cnn_evaluate(self, sample, **kwargs):
        self.predict_x=self.vgg_model.predict(sample, **kwargs)
        return np.argmax(self.predict_x,axis=1)
        
    def train_on_batch_vgg(self, *args, **kwargs ):
        res = self.vgg_model.train_on_batch(*args, **kwargs)
        return res
    
    def set_G1_trainable(self, train_able):
        self.gen.trainable = train_able

    def to_json(self, *args, **kwargs):

        return self.model.to_json(*args, **kwargs)
        
    def save_weights(self, *args, **kwargs):
        return self.vgg_model.save_weights(*args, **kwargs)



    def load(self, c_path, lr=0.0002):

        # Create model
        self.vgg_model = self.build_vgg_model()

        self.compile_model()  

        # Load weights into model
        print("model loaded,start to load paramerter")

        self.vgg_model.load_weights(c_path)
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
