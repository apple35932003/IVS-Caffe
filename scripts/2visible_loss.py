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
average_ = 50
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
host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
#par1 = host.twinx()
host.set_title(args.input1+" "+args.input2)
# set labels
host.set_xlabel("iterations")
host.set_ylabel("RPN loss")
#par1.set_ylabel("validation accuracy")
 
# plot curves
p1, = host.plot(train_iterations_, train_loss_, label=" loss " + args.input1)
p1, = host.plot(train_iterations2_, train_loss2_, label=" loss " + args.input2)
#p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")
 
# set location of the legend, 
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)
 
# set label color
host.axis["left"].label.set_color(p1.get_color())
#par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([-100, train_iterations_[-1]+1000])
#host.set_xlim([-100, 6000])
host.set_ylim([0., max(train_loss_)*1.5])
#host.set_ylim([0., 2])
 
plt.draw()
plt.savefig('out.png')

plt.show()
