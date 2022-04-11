import matplotlib.pyplot as plt 
import csv
import numpy as np
data_file= r"D:\pi\resnet_train.csv"
csvfile = open(data_file, 'r')
loss_list = []
acc_list = []
t5_list = []
evl_list = []
x_list=[]
csvreader = csv.reader(csvfile)
key_label_list = [line[:4] for line in csvreader]
key_label_list = key_label_list[1:]  # Chop off header
(last_loss, last_acc,last_t5,last_evl )=key_label_list[0]
last_loss=float(last_loss)
last_acc=float(last_acc)
last_t5=float(last_t5)
last_evl=float(last_evl)/100.
print(last_loss)
p = 0.01
for key_label in key_label_list:
    (loss, acc,t5,evl ) = key_label
    loss_list.append(float(loss)*p+last_loss*(1-p))
    acc_list.append(float(acc)*p+last_acc*(1-p)) 
    t5_list.append(float(t5)*p+last_t5*(1-p)) 
    evl_list.append(float(evl)*p/100.+last_evl*(1-p))
    last_loss=float(loss)*p+last_loss*(1-p)
    last_acc=float(acc)*p+last_acc*(1-p)
    last_t5=float(t5)*p+last_t5*(1-p)
    last_evl=float(evl)*p/100+last_evl*(1-p)

# loss_li=loss_arr.tolist()

for i in range(len(acc_list)):
    x_list.append(i)
# print(loss_list)
plt.figure(figsize=(20, 10), dpi=100)
plt.title("loss")
plt.xlabel("steps")
plt.ylabel("losss")#
plt.xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
# plt.yticks([0, 2, 4, 6, 8, 10])
plt.plot(loss_list,label = "loss")
plt.legend(fontsize=18) 
# plt.show()
plt.savefig(r"D:\pi\result_img\loss.png")  #savefig, don't show


plt.figure(figsize=(20, 10), dpi=100)
plt.title("top_1_acc")
plt.xlabel("steps")
plt.ylabel("top_1_acc")#
plt.xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
# plt.yticks([0, 2, 4, 6, 8, 10])
plt.plot(acc_list,label = "top_1_acc")
plt.legend(fontsize=18) 
# plt.show()
plt.savefig(r"D:\pi\result_img\acc.png")  #savefig, don't show


plt.figure(figsize=(20, 10), dpi=100)
plt.title("top_5_acc")
plt.xlabel("steps")
plt.ylabel("top_5_acc")#
plt.xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
# plt.yticks([0, 2, 4, 6, 8, 10])
plt.plot(t5_list,label = "top_5_acc")
plt.legend(fontsize=18) 
# plt.show()
plt.savefig(r"D:\pi\result_img\top_5_acc.png")  #savefig, don't show


plt.figure(figsize=(20, 10), dpi=100)
plt.title("evl_acc")
plt.xlabel("steps")
plt.ylabel("evl_acc")#
plt.xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
# plt.yticks([0, 2, 4, 6, 8, 10])
plt.plot(evl_list,label = "evl_acc")
plt.legend(fontsize=18) 
# plt.show()
plt.savefig(r"D:\pi\result_img\evl_acc.png")  #savefig, don't show


fig, ax1 = plt.subplots(figsize=(20, 10), dpi=60)

color = 'tab:blue'
ax1.set_xlabel('steps',fontsize=18)
ax1.set_ylabel('acc', color=color,fontsize=18)
ax1.plot(evl_list,color = "g" , label = "test_acc")
ax1.plot(t5_list,color = "b" , label = "top_5_acc")
ax1.plot(acc_list,color = "y" , label = "train_acc")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() 

color = 'tab:red'
ax2.set_ylabel('loss', color=color,fontsize=18)
ax2.plot(loss_list,color = "r" , label = "loss")
ax2.tick_params(axis='y', labelcolor=color)
ax1.legend(fontsize=18,loc = "best") 
ax2.legend(fontsize=18,loc = "best") 
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
fig.tight_layout()
plt.savefig(r"D:\pi\result_img\all.png")  #savefig, don't show
