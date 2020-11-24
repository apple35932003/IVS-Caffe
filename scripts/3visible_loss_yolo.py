#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i1", "--input1", help="input_file1", dest="input1", default="")
parser.add_argument("-i2", "--input2", help="input_file2", dest="input2", default="")
parser.add_argument("-i3", "--input3", help="input_file3", dest="input3", default="")
args = parser.parse_args()
 
# read the log file
fp = open(args.input1, 'r')
 
train_iterations1 = []
train_loss1 = []
#mbox_loss1 = []
#test_accuracy = []
 
for ln in fp:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b,',ln)
        train_iterations1.append(int(arr[0].strip(',')[4:]))
        train_loss1.append(float(ln.strip().split(' = ')[-1]))
    if 'Train net output #0: det_loss1 = ' in ln:
        t = ln.split()
        #mbox_loss1.append(float(t[10]))
        

train_iterations1_ = []
train_loss1_ = []
#mbox_loss1_ = []

count = 0
iteration = 0
loss = 0
#mbox = 0
average_ = 50
for i in range(len(train_iterations1)):
    if count == average_:
        train_iterations1_.append(iteration/average_)
        train_loss1_.append(loss/average_)
        #mbox_loss1_.append(#mbox/average_)

        count = 0
        iteration = 0
        loss = 0
        #mbox = 0
    count += 1
    iteration += train_iterations1[i]
    loss += train_loss1[i]
    #mbox += #mbox_loss1[i]
        
    
        
fp.close()
 
fp = open(args.input2, 'r')
 
train_iterations2 = []
train_loss2 = []
#mbox_loss2 = []
#test_accuracy = []
 
for ln in fp:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b,',ln)
        train_iterations2.append(int(arr[0].strip(',')[4:]))
        train_loss2.append(float(ln.strip().split(' = ')[-1]))
    if 'Train net output #0: det_loss1 = ' in ln:
        t = ln.split()
        #mbox_loss2.append(float(t[10]))
        

train_iterations2_ = []
train_loss2_ = []
#mbox_loss2_ = []

count = 0
iteration = 0
loss = 0
#mbox = 0
average_ = 100
for i in range(len(train_iterations2)):
    if count == average_:
        train_iterations2_.append(iteration/average_)
        train_loss2_.append(loss/average_)
        #mbox_loss2_.append(#mbox/average_)

        count = 0
        iteration = 0
        loss = 0
        #mbox = 0
    count += 1
    iteration += train_iterations2[i]
    loss += train_loss2[i]
    #mbox += #mbox_loss2[i]
        
    
        
fp.close()
fp = open(args.input3, 'r')
 
train_iterations3 = []
train_loss3 = []
#mbox_loss3 = []
#test_accuracy = []
 
for ln in fp:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b,',ln)
        train_iterations3.append(int(arr[0].strip(',')[4:]))
        train_loss3.append(float(ln.strip().split(' = ')[-1]))
    if 'Train net output #0: det_loss1 = ' in ln:
        t = ln.split()
        #mbox_loss3.append(float(t[10]))
        

train_iterations3_ = []
train_loss3_ = []
#mbox_loss3_ = []

count = 0
iteration = 0
loss = 0
#mbox = 0
average_ = 100
for i in range(len(train_iterations3)):
    if count == average_:
        train_iterations3_.append(iteration/average_)
        train_loss3_.append(loss/average_)
        #mbox_loss3_.append(#mbox/average_)

        count = 0
        iteration = 0
        loss = 0
        #mbox = 0
    count += 1
    iteration += train_iterations3[i]
    loss += train_loss3[i]
    #mbox += #mbox_loss3[i]
        
    
        
fp.close()
host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
#par1 = host.twinx()
host.set_title(args.input1 + ' v.s. ' + args.input2 + ' v.s. ' + args.input3)
# set labels
host.set_xlabel("iterations")
host.set_ylabel("loss")
#par1.set_ylabel("validation accuracy")
 
# plot curves
p1, = host.plot(train_iterations1_, train_loss1_, label=" loss " + args.input1)
#p2, = host.plot(train_iterations_, #mbox_loss1_, label="#mbox loss ")
p3, = host.plot(train_iterations2_, train_loss2_, label=" loss " + args.input2)
#p4, = host.plot(train_iterations_, #mbox_loss2_, label="#mbox loss ")
#p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")
p4, = host.plot(train_iterations3_, train_loss3_, label=" loss " + args.input3)
 
# set location of the legend, 
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)
 
# set label color
host.axis["left"].label.set_color(p1.get_color())
#par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([-100, train_iterations1_[-1]+1000])
#host.set_xlim([-100, 6000])
host.set_ylim([0., max(train_loss2_)*1.5])
#host.set_ylim([0., 2])
 
plt.draw()
plt.savefig('out.png')

plt.show()
