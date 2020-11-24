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
parser.add_argument("-i", "--input", help="input_file", dest="input", default="")
args = parser.parse_args()
 
# read the log file
fp = open(args.input, 'r')
 
train_iterations = []
train_loss = []
test_iterations = []
mbox_loss = []
#test_accuracy = []
 
for ln in fp:
    # get train_iterations and train_loss
    if '] Iteration ' in ln and 'loss = ' in ln:
        arr = re.findall(r'ion \b\d+\b',ln)
        train_iterations.append(int(arr[0].strip(',')[4:]))
        train_loss.append(float(ln.strip().split(' = ')[-1]))
    if 'Train net output #0: det_loss1 = ' in ln:
        t = ln.split()
        mbox_loss.append(float(t[10]))
        

train_iterations_ = []
train_loss_ = []
mbox_loss_ = []

count = 0
iteration = 0
loss = 0
mbox = 0
average_ = 10
for i in range(len(mbox_loss)):
    if count == average_:
        train_iterations_.append(iteration/average_)
        train_loss_.append(loss/average_)
        mbox_loss_.append(mbox/average_)

        count = 0
        iteration = 0
        loss = 0
        mbox = 0
    count += 1
    iteration += train_iterations[i]
    loss += train_loss[i]
    mbox += mbox_loss[i]
        
    
        
fp.close()
 
host = host_subplot(111)
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
#par1 = host.twinx()
host.set_title(args.input)
# set labels
host.set_xlabel("iterations")
host.set_ylabel("RPN loss")
#par1.set_ylabel("validation accuracy")
 
# plot curves
p1, = host.plot(train_iterations_, train_loss_, label=" loss")
#p2, = host.plot(train_iterations_, mbox_loss_, label="det_loss")
#p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")
 
# set location of the legend, 
# 1->rightup corner, 2->leftup corner, 3->leftdown corner
# 4->rightdown corner, 5->rightmid ...
host.legend(loc=1)
 
# set label color
host.axis["left"].label.set_color(p1.get_color())
#par1.axis["right"].label.set_color(p2.get_color())
# set the range of x axis of host and y axis of par1
host.set_xlim([-100, train_iterations_[-1]+2000])
#host.set_xlim([-100, 6000])
host.set_ylim([0., max(train_loss_)*1.5])
#host.set_ylim([0., 2])
 
plt.draw()
plt.savefig('out.png')

plt.show()
