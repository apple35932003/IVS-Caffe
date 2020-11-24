#!/usr/bin/python
import os
#os.environ["GLOG_minloglevel"] = '2'
import sys
sys.path.append("python")
import caffe
import numpy as np
import random

#Define Use GPU or CPU
USE_GPU = False
if USE_GPU:
    caffe.set_device(0) # Or the index of the GPU you want to use
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
print("Initialized caffe")

#load network
#net = caffe.Net("models/cnn/cnn.prototxt","models/cnn/cnn_ori_iter_1000.caffemodel", caffe.TEST)
net = caffe.Net("models/test_BAQ/cnn_XNOR.prototxt", caffe.TEST)
#net_ori = caffe.Net("models/test_BAQ/cnn_ori_.prototxt", caffe.TEST)

#Creating input and label
#np.random.seed(1)
#print net.blobs["data"].data.shape
batch = np.ones((net.blobs["data"].data.shape))
#a = np.ones((1,3))
#b = np.ones((3,1))
for batch_n in range(1):
    for dim_n in range(3):
        for width_n in range(1):
            for height_n in range(1):
                #batch[batch_n][dim_n][width_n][height_n]=batch_n*1+dim_n*1+width_n*1+height_n*
                batch[batch_n][dim_n][width_n][height_n]=random.randint(0,1)
#print net.params["conv1"][0].data.shape
#for kernel_n in range(6):
#    for dim_n in range(3):
#        for width_n in range(1):
#            for height_n in range(1):
#                net.params["conv1"][0].data[kernel_n][dim_n][width_n][height_n]=kernel_n*1+dim_n*1+width_n*1+height_n
#batch = np.ones((1,4,4,3))
#batch = np.random.randn(*net.blobs["data"].shape) * 50 # normal distribution(0, 50), in the shape of the input batch
#labels = np.random.randint(0, 10, net.blobs["label"].shape) # random labels


#Assign input and label to network
net.blobs["data"].data[...] = batch
#net_ori.blobs["data"].data[...] = batch
#net.params["conv1"][0].data[...]=net_ori.params["conv1"][0].data[...]
#net.blobs["label"].data[...] = labels
print 'Input'
print batch
print 'weight'
print net.params["conv1"][0].data[...]
res = net.forward(start="conv1", end="conv1")
#res = net_ori.forward(start="conv1", end="conv1")

#print net.blobs["data"].data[...]

#res = net.forward(start="conv1", end="conv2")

#print net.blobs["data"].data[...]
#print 'net param'
print '\n'
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


#Do Backward propagation
#net.backward()

#Display gradient for ip2
#print net.layers[list(net._layer_names).index("ip2")].blobs[0].diff



#Update Weights
#lr = 0.01
#for l in net.layers:
#    for b in l.blobs:
#        b.data[...] -= lr * b.diff


