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
parser.add_argument("-i4", "--input4", help="input_file4", dest="input4", default="")
args = parser.parse_args()
 
# read the log file
fp = open(args.input1, 'r')
 
train_iterations = []
train_loss = []
 
for ln in fp:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b,',ln)
        train_iterations.append(int(arr[0].strip(',')[4:]))
        train_loss.append(float(ln.strip().split(' = ')[-1]))
        

train_iterations_ = []
train_loss_ = []

count = 0
iteration = 0
loss = 0
average_ = 200
for i in range(len(train_iterations)):
    if count == average_:
        train_iterations_.append(iteration/average_)
        train_loss_.append(loss/average_)

        count = 0
        iteration = 0
        loss = 0
    count += 1
    iteration += train_iterations[i]
    loss += train_loss[i]
        
    
        
fp.close()
 
fp2 = open(args.input2, 'r')
 
train_iterations2 = []
train_loss2 = []
 
for ln in fp2:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b,',ln)
        train_iterations2.append(int(arr[0].strip(',')[4:]))
        train_loss2.append(float(ln.strip().split(' = ')[-1]))
        

train_iterations2_ = []
train_loss2_ = []

count2 = 0
iteration2 = 0
loss2 = 0
for i in range(len(train_iterations2)):
    if count2 == average_:
        train_iterations2_.append(iteration2/average_)
        train_loss2_.append(loss2/average_)

        count2 = 0
        iteration2 = 0
        loss2 = 0
    count2 += 1
    iteration2 += train_iterations2[i]
    loss2 += train_loss2[i]
        
    
        
fp2.close()
fp3 = open(args.input3, 'r')
 
train_iterations3 = []
train_loss3 = []
 
for ln in fp3:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b,',ln)
        train_iterations3.append(int(arr[0].strip(',')[4:]))
        train_loss3.append(float(ln.strip().split(' = ')[-1]))
        

train_iterations3_ = []
train_loss3_ = []

count3 = 0
iteration3 = 0
loss3 = 0
for i in range(len(train_iterations3)):
    if count3 == average_:
        train_iterations3_.append(iteration3/average_)
        train_loss3_.append(loss3/average_)

        count3 = 0
        iteration3 = 0
        loss3 = 0
    count3 += 1
    iteration3 += train_iterations3[i]
    loss3 += train_loss3[i]      
fp3.close()

fp4 = open(args.input4, 'r')
 
train_iterations4 = []
train_loss4 = []
 
for ln in fp4:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b,',ln)
        train_iterations4.append(int(arr[0].strip(',')[4:]))
        train_loss4.append(float(ln.strip().split(' = ')[-1]))
        

train_iterations4_ = []
train_loss4_ = []

count4 = 0
iteration4 = 0
loss4 = 0
for i in range(len(train_iterations4)):
    if count4 == average_:
        train_iterations4_.append(iteration4/average_)
        train_loss4_.append(loss4/average_)

        count4 = 0
        iteration4 = 0
        loss4 = 0
    count4 += 1
    iteration4 += train_iterations4[i]
    loss4 += train_loss4[i]      
fp4.close()

host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
#par1 = host.twinx()
host.set_title(args.input1+" v.s. "+args.input2+ " v.s. " + args.input3)
# set labels
host.set_xlabel("iterations")
host.set_ylabel("RPN loss")
#par1.set_ylabel("validation accuracy")
 
# plot curves
#p1, = host.plot(train_iterations_, train_loss_, label=" loss " + args.input1)
#p1, = host.plot(train_iterations2_, train_loss2_, label=" loss " + args.input2)
#p1, = host.plot(train_iterations3_, train_loss3_, label=" loss " + args.input3)
#p1, = host.plot(train_iterations4_, train_loss4_, label=" loss " + args.input4)
p1, = host.plot(train_iterations_, train_loss_, label=" loss Set1")
p1, = host.plot(train_iterations2_, train_loss2_, label=" loss Set2")
p1, = host.plot(train_iterations3_, train_loss3_, label=" loss Set3")
p1, = host.plot(train_iterations4_, train_loss4_, label=" loss Set4")
#p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")
 
# set location of the legend, 
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)
 
# set label color
host.axis["left"].label.set_color(p1.get_color())
#par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
#host.set_xlim([-100, train_iterations_[-1]+1000])
host.set_xlim([-100, 210000])
#host.set_xlim([-100, 6000])
host.set_ylim([0., max(train_loss_)*1.5])
#host.set_ylim([0., 2])
 
plt.draw()
plt.savefig('out.png')

plt.show()
