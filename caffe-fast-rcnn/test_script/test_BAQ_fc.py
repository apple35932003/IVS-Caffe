#!/usr/bin/python
import os
#os.environ["GLOG_minloglevel"] = '2'
import sys
sys.path.append("python")
import caffe
import numpy as np
import random
from datetime import datetime
#Define Use GPU or CPU
USE_GPU = True
if USE_GPU:
    caffe.set_device(2) # Or the index of the GPU you want to use
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
print("Initialized caffe")

#load network
#net = caffe.Net("models/cnn/cnn.prototxt","models/cnn/cnn_ori_iter_1000.caffemodel", caffe.TEST)
net_ori = caffe.Net("models/test_BAQ/nn__.prototxt", caffe.TEST)
print 'loading IVS net'
net = caffe.Net("models/test_BAQ/nn_IVS.prototxt", caffe.TEST)
print 'loading done'
#Creating input and label
#np.random.seed(1)
print net.blobs["data"].data.shape
print net_ori.blobs["data"].data.shape
batch = np.ones((net.blobs["data"].data.shape))
#a = np.ones((1,3))
#b = np.ones((3,1))
for batch_n in range(3):
    for dim_n in range(1):
        for width_n in range(28):
            for height_n in range(28):
                batch[batch_n][dim_n][width_n][height_n]=np.random.randint(0,10)
                #batch[batch_n][dim_n][width_n][height_n]=1
#print net.params["ip1"][0].data.shape
#for num_output in range(100):
#    for index in range(783):
#        net.params["ip1"][0].data[num_output][index] = num_output * 1000 + index
#for kernel_n in range(6):
#    for dim_n in range(1):
#        for width_n in range(2):
#            for height_n in range(2):
#                net.params["conv1"][0].data[kernel_n][dim_n][width_n][height_n]=kernel_n*1+dim_n*1+width_n*1+height_n
#batch = np.ones((1,4,4,3))
#batch = np.random.randn(*net.blobs["data"].shape) * 50 # normal distribution(0, 50), in the shape of the input batch
#labels = np.random.randint(0, 10, net.blobs["label"].shape) # random labels


#Assign input and label to network
net.blobs["data"].data[...] = batch
net_ori.blobs["data"].data[...] = batch

net.params["ip1"][0].data[...]=net_ori.params["ip1"][0].data[...]
#net.blobs["label"].data[...] = labels

#print '-------------'
#print net.blobs["data"].data[...]
#print batch
res = net.forward(start="ip1", end="ip1")
#print 'Done'
#print batch
res = net_ori.forward(start="ip1", end="ip1")

#res = net.forward(start="conv2/", end="conv2")

print "Data"
print net.blobs["data"].data[...]
#print 'net param'
#print '\n'
#print '-----------'
#print net.params["ip1"][0].data[1][50]
#print '-------------'
print "Q output"
print net.blobs["ip1"].data[...]
print "ORI output"
print net_ori.blobs["ip1"].data[...]
#print '--------------'
#print '\n'
#print 'net_ori param'
#print net_ori.params["conv1"][0].data[...]
#print '\n'
#print net.blobs["conv1"].data[...]
#print "\n"
#print net_ori.blobs["conv1"].data[...]
#print '\n'
print "diff"
print net.blobs["ip1"].data[...] - net_ori.blobs["ip1"].data[...]
#print np.mean(net.blobs["ip1"].data[...] - net_ori.blobs["ip1"].data[...])

sys.exit(0)
start = datetime.now()
for i in range(100):
    net.blobs["data"].data[...] = batch
    res = net.forward(start="ip1", end="ip1")
end = datetime.now()
elapsedTime = end-start
print 'total time is " milliseconds', elapsedTime.total_seconds()*1000
start = datetime.now()
for i in range(100):
    net_ori.blobs["data"].data[...] = batch
    res = net_ori.forward(start="ip1", end="ip1")
end = datetime.now()
elapsedTime = end-start
print 'ori total time is " milliseconds', elapsedTime.total_seconds()*1000


#Do a full forward propagation
#loss = net.forward()

#print 'loading ori net'
#net_ori = caffe.Net("models/test_BAQ/nn.prototxt", caffe.TEST)
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

#
