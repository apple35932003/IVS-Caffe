#!/usr/bin/python
import os
os.environ["GLOG_minloglevel"] = '2'
import sys
sys.path.append("python")
import caffe
import numpy as np
import random

#Define Use GPU or CPU
USE_GPU = True
if USE_GPU:
    caffe.set_device(0) # Or the index of the GPU you want to use
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
print("Initialized caffe")

#load network
#net = caffe.Net("models/cnn/cnn.prototxt","models/cnn/cnn_ori_iter_1000.caffemodel", caffe.TEST)
net = caffe.Net("models/test_BAQ/cnn_JLDLA2.prototxt", caffe.TEST)
#net_ori = caffe.Net("models/test_BAQ/cnn_ori.prototxt", caffe.TEST)

#Creating input and label
#np.random.seed(1)
#print net.blobs["data"].data.shape
batch = np.ones((net.blobs["data"].data.shape))
#a = np.ones((1,3))
#b = np.ones((3,1))
batch[0][0][0][0]=0 /128.0
batch[0][0][0][1]=0 /128.0
batch[0][0][0][2]=0/128.0
batch[0][0][1][0]=67/128.0
batch[0][0][1][1]=68/128.0
batch[0][0][1][2]=68/128.0
batch[0][0][2][0]=67/128.0
batch[0][0][2][1]=68/128.0
batch[0][0][2][2]=68/128.0
batch[0][1][0][0]=0/128.0
batch[0][1][0][1]=0/128.0
batch[0][1][0][2]=0/128.0
batch[0][1][1][0]=67/128.0
batch[0][1][1][1]=67/128.0
batch[0][1][1][2]=67/128.0
batch[0][1][2][0]=67/128.0
batch[0][1][2][1]=67/128.0
batch[0][1][2][2]=67/128.0
batch[0][2][0][0]=0/128.0
batch[0][2][0][1]=0/128.0
batch[0][2][0][2]=0/128.0
batch[0][2][1][0]=66/128.0
batch[0][2][1][1]=66/128.0
batch[0][2][1][2]=66/128.0
batch[0][2][2][0]=66/128.0
batch[0][2][2][1]=66/128.0
batch[0][2][2][2]=66/128.0
#for batch_n in range(1):
#    for dim_n in range(1):
#        for width_n in range(3):
#            for height_n in range(3):
#                #batch[batch_n][dim_n][width_n][height_n]=batch_n*1+dim_n*1+width_n*1+height_n*1
#                #batch[batch_n][dim_n][width_n][height_n]=random.randint(1,9)
#print net.params["conv1"][0].data.shape
#for kernel_n in range(2):
#    for dim_n in range(10):
#        for width_n in range(2):
#            for height_n in range(2):
#                #net.params["conv1"][0].data[kernel_n][dim_n][width_n][height_n]=kernel_n*1+dim_n*1+width_n*1+height_n
#                net_ori.params["conv1"][0].data[kernel_n][dim_n][width_n][height_n]=random.randint(1,9)
#batch = np.ones((1,4,4,3))
#batch = np.random.randn(*net.blobs["data"].shape) * 50 # normal distribution(0, 50), in the shape of the input batch
#labels = np.random.randint(0, 10, net.blobs["label"].shape) # random labels
print net.params["conv1"][0].data.shape
net.params["conv1"][0].data[0][0][0][0]=-29/8.0
net.params["conv1"][0].data[0][0][0][1]=-54/8.0
net.params["conv1"][0].data[0][0][0][2]=-38/8.0
net.params["conv1"][0].data[0][0][1][0]=4/8.0
net.params["conv1"][0].data[0][0][1][1]=4/8.0
net.params["conv1"][0].data[0][0][1][2]=0/8.0
net.params["conv1"][0].data[0][0][2][0]=34/8.0
net.params["conv1"][0].data[0][0][2][1]=56/8.0
net.params["conv1"][0].data[0][0][2][2]=24/8.0
net.params["conv1"][0].data[0][1][0][0]=-60/8.0
net.params["conv1"][0].data[0][1][0][1]=-74/8.0
net.params["conv1"][0].data[0][1][0][2]=-57/8.0
net.params["conv1"][0].data[0][1][1][0]=2/8.0
net.params["conv1"][0].data[0][1][1][1]=15/8.0
net.params["conv1"][0].data[0][1][1][2]=7/8.0
net.params["conv1"][0].data[0][1][2][0]=51/8.0
net.params["conv1"][0].data[0][1][2][1]=74/8.0
net.params["conv1"][0].data[0][1][2][2]=41/8.0
net.params["conv1"][0].data[0][2][0][0]=-44/8.0
net.params["conv1"][0].data[0][2][0][1]=-59/8.0
net.params["conv1"][0].data[0][2][0][2]=-36/8.0
net.params["conv1"][0].data[0][2][1][0]=-4/8.0
net.params["conv1"][0].data[0][2][1][1]=6/8.0
net.params["conv1"][0].data[0][2][1][2]=9/8.0
net.params["conv1"][0].data[0][2][2][0]=38/8.0
net.params["conv1"][0].data[0][2][2][1]=52/8.0
net.params["conv1"][0].data[0][2][2][2]=39/8.0
net.params["conv1"][1].data[0]=-3.75
#Assign input and label to network
net.blobs["data"].data[...] = batch
#net_ori.blobs["data"].data[...] = batch
#net.params["conv1"][0].data[...]=net_ori.params["conv1"][0].data[...]
#net.params["conv1"][1].data[...]=net_ori.params["conv1"][1].data[...]
#net.blobs["label"].data[...] = labels

print batch
print net.params["conv1"][0].data
res = net.forward(start="conv1", end="conv1")
print'con done net'
#res = net_ori.forward(start="conv1", end="conv1")
#print'con done net_ori'

#print net.blobs["data"].data[...]

#res = net.forward(start="conv1", end="conv2")

#print net.blobs["data"].data[...]
#print 'net param'
#print 'test\n'
#print '\n'
#print net.params["conv1"][0].data[...]
#print '\n'
#print 'net_ori param'
#print net_ori.params["conv1"][0].data[...]
#print '\n'
print 'IVS output'
print net.blobs["conv1"].data[...]
#print "original output\n"
#print net_ori.blobs["conv1"].data[...]
#print '\n'
#print net.blobs["conv1"].data[...] - net_ori.blobs["conv1"].data[...]
#Do a full forward propagation
#loss = net.forward()

#net.params["conv1"][0].data[0][0][0][0]=2610214
#for batch_n in range(3):
#    for dim_n in range(3):
#        for width_n in range(3):
#            for height_n in range(3):
#                batch[batch_n][dim_n][width_n][height_n]=2610214

#net.blobs["data"].data[...] = batch
#net_ori.blobs["data"].data[...] = batch
#res = net.forward(start="conv1", end="conv1")
#res = net_ori.forward(start="conv1", end="conv1")
#print 'IVS'
#print net.blobs["conv1"].data[...]
#print net.blobs["conv1"].data.reshape(net.blobs["conv1"].data.size)[0]
#print net.blobs["conv1"].data.reshape(net.blobs["conv1"].data.size)[1]
#print net.blobs["conv1"].data.reshape(net.blobs["conv1"].data.size)[2]
#print net.blobs["conv1"].data.reshape(net.blobs["conv1"].data.size)[3]
#print 'ori'
#print net_ori.blobs["conv1"].data[...]



#Display Result
#print loss

#Do partial forward propagation
#res = net.forward(start="mnist", end="conv1")
#Display Result
#print res

#Display result in ip2 blob
#print net.blobs["ip2"].data[0]
#Display result in loss blob
#print net.blobs["loss"].data
#

#Do Backward propagation
#net.backward()

#Display gradient for ip2
#print net.layers[list(net._layer_names).index("ip2")].blobs[0].diff



#Update Weights
#lr = 0.01
#for l in net.layers:
#    for b in l.blobs:
#        b.data[...] -= lr * b.diff


