# from curses import A_RIGHT, ACS_GEQUAL
from ctypes import sizeof
from datetime import datetime
import numpy as np
import csv
import math
import cv2
import os
from tqdm import tqdm 
from tqdm._tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import random
from collections import Counter
# from utils.dataloader import Data, TestData
from libs.vgg import CNN
import tensorflow as tf
from tensorflow import config
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
 
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MUTI_GPU = 1 
Total_Epoch = 15000
BATCH = 20
# data = Data() 			
# test_data = TestData()	

def build_model():
	global cnn_model
	cnn_model = CNN(img_rows = 256, img_cols = 256, gpus = MUTI_GPU)

def save_model():
	cnn_weights_path = '/data/landmark/script/checkpoints/vgg_ttt.hdf5'
	cnn_model.save_weights(cnn_weights_path)
	# cnn_weights_path = '/data/landmark/script/checkpoints/vgg19.hdf5'
	# cnn_weights_path = '/data/landmark/script/checkpoints/resnet.hdf5'


def load_model(d_size='ttt'):
	global cnn_model
	if d_size=='vgg16-500':
		cnn_weights_path = '/data/landmark/script/checkpoints/vgg_16_500.hdf5'
		
	elif d_size=='vgg16':
		cnn_weights_path = '/data/landmark/script/checkpoints/cnn.hdf5'
	elif d_size=='ttt':
		cnn_weights_path = '/data/landmark/script/checkpoints/vgg_ttt.hdf5'


	cnn_model = CNN(img_rows = 256, img_cols = 256)
	cnn_model.load(cnn_weights_path,lr = 0.0002)

def save_csv(loss_list,acc_list,t5_list,evl_acc):
	dataframe = pd.DataFrame({'loss':loss_list,'acc':acc_list,'t5':t5_list,"evl_acc":evl_acc})
	dataframe.to_csv("/data/landmark/script/checkpoints/vgg_train_13000.csv",index=False,sep=',')

def save_img(epoch, steps,img,contour_img,contour_train):
	fig, axs = plt.subplots(1,3) 

	axs[0].imshow(img)
	axs[0].set_title("img")
	axs[0].axis("off")
	axs[1].imshow(contour_img)
	axs[1].set_title("contour_img")
	axs[1].axis("off")
	axs[2].imshow(contour_train,cmap='gray')
	axs[2].set_title("contour_train")
	axs[2].axis("off")

	fig.savefig(r"C:\Users\13575\Desktop\mcm_proj\data\try2\asd%d_%d.png"% (epoch+240,steps))
	plt.close()

def normalize(imgs,size):
	a=np.ndarray((imgs.shape[0],size[0],size[1],3)) 
	i=0
	for img in imgs:

		# print(img.shape)
		img = cv2.resize(img, size, interpolation=4)
		a[i]=img
		i+=1
	return a

def evaluate(inv_dict,data_name,data_label,batch_size):
	train_data,label=random_select(inv_dict,data_name,data_label,batch_size=batch_size) #0-1
	test_label = to_categorical(label,1000)
	pre_label = cnn_model.cnn_evaluate(train_data)
	label=np.array(label)
	# pre_label[:10]
	a = label - pre_label
	zero_conunter=0
	for i in a:
		if i == 0:
			zero_conunter+=1
	# print(label.shape)
	# print(pre_label.shape)
	b = pd.crosstab(label,pre_label,rownames=['label'],colnames=['predict'])
	print(b)
	acc=round(zero_conunter*100/batch_size,2)
	print("accuracy : "+ str(acc)+'%')
	return acc

def random_select(inv_dict,data_name,data_label,batch_size=BATCH):
# inv_dict:0-999 to label
# data_name:list of image name
# data_label:list of image label
# batch_size:volume of images
	data_dir = "/data/landmark/dataset"
	rand_index = random.sample(range(0, 999), batch_size)
	index_list=[]
	for index in rand_index:
		index_list.append(inv_dict[index])
	img_num_dict=Counter(data_label)
	img_num_list=[]
	for index in index_list:
		img_num_list.append(img_num_dict[index])
	img_index_list=[]
	img_name_list=[]
	for num in img_num_list:
		img_index_list.append(np.random.randint(0,num))
	for i in range(len(img_index_list)):
		img_name_list.append(data_name[data_label.index(index_list[i])+img_index_list[i]])
	train_data=np.ndarray((batch_size,256,256,3))
	i=0
	for key in img_name_list:
		filename = os.path.join(data_dir, '%s.jpg' % key)
		train_data[i] = cv2.imread(filename)
		i+=1
	return train_data/255.,rand_index

def csv_loader(data_file):

	# data_dir = r"C:\Users\13575\Desktop\Landmark\dataset"
	csvfile = open(data_file, 'r')
	new_name = []
	new_label = []
	csvreader = csv.reader(csvfile)
	key_label_list = [line[:2] for line in csvreader]
	key_label_list = key_label_list[1:]  # Chop off header
	for key_label in key_label_list:
		(key, label) = key_label
		new_name.append(key)
		new_label.append(label) 
	return new_name,new_label
		# filename = os.path.join(data_dir, '%s.jpg' % key)

def label_mapping(label_list):
	i=0
	last_j=label_list[0]
	dict={}             # label to 0-999
	inv_dict={}			# 0-999 to label
	for j in label_list:
		if j!=last_j:
			last_j=j
			i+=1
		dict[j]=i
		inv_dict[i]=j
	return dict,inv_dict

def train():
	loss_list=[]
	acc_list=[]
	t5_list=[]
	evl_acc=[]
	train_data_file="/data/landmark/dataset_csv/image_label_train.csv"
	test_data_file="/data/landmark/dataset_csv/image_label_test.csv"
	data_name,data_label=csv_loader(train_data_file)
	t_data_name,t_data_label=csv_loader(test_data_file)
	dict,inv_dict=label_mapping(data_label) #label to one-hot
	t_data_name,t_data_label=csv_loader(test_data_file)
	t_dict,t_inv_dict=label_mapping(t_data_label) #label to one-hot
	start_time = datetime.now()
	total_time = start_time

	step=1
	for epoch in range(1,Total_Epoch+1):
		batch_size=60
		train_data,label=random_select(inv_dict,data_name,data_label,batch_size=batch_size) #0-1
		train_label = to_categorical(label,1000)
		# images_train = train_x[np.random.randint(0, train_x.shape[0], size=BATCH), :, :, :]
		print("\n*******************************************************")
		print('|**************|')
		print('  epoch:  %s  '%str(epoch))
		print('|**************|')
		for steps in tqdm(range(step), ncols = 50):
			steps += 1
			d_loss_G1 = cnn_model.train_on_batch_vgg(train_data, train_label)
			# d_loss_G1 = cnn_model.train_on_batch_G(img_train, ground_truth)

		print("d_loss_G1 = "+str(d_loss_G1))
		save_model()
		loss_list.append(d_loss_G1[0])
		acc_list.append(d_loss_G1[1])
		t5_list.append(d_loss_G1[2])
		current_time = datetime.now()
		print('using time:', current_time - start_time)
		evl_acc.append(evaluate(t_inv_dict,t_data_name,t_data_label,100))
		save_csv(loss_list,acc_list,t5_list,evl_acc)


def main():
	g = gen_model_res(INPUT_SHAPE)
	# g.summary()

	d = dcrm_model(INPUT_SHAPE)
	# d.summary()

	dc = cnn_model(g,d,INPUT_SHAPE)
	# dc.summary()

if __name__ == '__main__':
	# main()
	# input_shape = (128,128,3)
	# evaluate('test',10000)
	load_model('ttt')
	# # perdict()
	# build_model()
	train()






