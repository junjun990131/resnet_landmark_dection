
# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from PIL import Image
from io import BytesIO
from urllib.request import urlopen
import cv2
import numpy as np
import pandas as pd
from collections import Counter
def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  key_url_list = [line[:3] for line in csvreader]
  # key_url_list=np.array(key_url_list)
  # print(key_url_list.shape)
  return key_url_list[1:]  # Chop off header


def DownloadImage(key_urls):
  out_dir = r"C:\Users\13575\Desktop\Landmark\dataset"
  i = 0
  new_name=[]
  new_label=[]
  for key_url in key_urls:
    (key, url,label) = key_url
    filename = os.path.join(out_dir, '%s.jpg' % key)
    if os.path.exists(filename):
      if label!='None':
        # os.system('cls')
        i+=1
        new_name.append(key)
        new_label.append(label) 
        print('Image %s already exists. NO %d img has been processed' % (key,i))
  data_name=[]
  data_label=[]
  dict=Counter(new_label)
  dict2 = sorted(dict.items(), key=lambda dict: dict[1], reverse=True)
  # print(dict)
  # return
  i=0
  for item in dict2:
    (key,values)=item
    if values>=2000:
      values=2000
    # print(str(key)+"  "+str(values))
    print(str(values))

    for value in range(values):
      index=new_label.index(key)
      data_name.append(new_name.pop(index))
      data_label.append(key)
      new_label.pop(index)
    i+=1
    if i==1000:
      break
  dataframe = pd.DataFrame({'Keys':data_name,'labels':data_label})
  dataframe.to_csv(r"C:\Users\13575\Desktop\Landmark\dataset_csv\image_label.csv",index=False,sep=',')
  print(dict)

  return
  
def Run():
  # if len(sys.argv) != 3:
  #   print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
  #   sys.exit(0)
  # (data_file, out_dir) = sys.argv[1:]
  data_file=r"C:\Users\13575\Desktop\Landmark\dataset_csv\train_sorted.csv"
  key_url_list = ParseData(data_file)
  # pool = multiprocessing.Pool(processes=50)
  # pool.map(DownloadImage, key_url_list)
  DownloadImage(key_url_list)

if __name__ == '__main__':
  Run()