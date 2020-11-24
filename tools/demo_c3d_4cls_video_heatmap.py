#!/usr/bin/env python
import _init_paths
import numpy as np
import cv2
import os
import sys
import time
import argparse
import heapq
import random
import csv
import glob 
from utils.timer import Timer
from shutil import copyfile

#def add_path(path):
#    if path not in sys.path:
#        sys.path.insert(0, path)
#caffe_dir = '/home/ivskaikai/Deep_Learning_Framework/caffe'
#python_path=os.path.join(caffe_dir,'python')
#add_path(python_path)

import caffe
import numpy as np
import matplotlib.pyplot as plt
import time
caffe.set_mode_gpu()


 
def video_list_to_blob(videos):
    """Convert a list of videos into a network input.

    Assumes videos are already prepared (means subtracted, BGR order, ...).
    """
    shape = videos[0].shape
    num_videos = len(videos)
    blob = np.zeros((num_videos, shape[0], shape[1], shape[2], shape[3]),
                    dtype=np.float32)
    for i in xrange(num_videos):
        blob[i] = videos[i]
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, length, height, width)
    channel_swap = (0, 3, 4, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob
def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    for i in range(im_orig.shape[-1]):
        img = im_orig[:,:,:,i]
        img -= [ 90, 98, 102]
        im_orig[:,:,:,i] = img

    processed_ims = []

    processed_ims.append(im_orig)
    # Create a blob to hold the input images
    blob = video_list_to_blob(processed_ims)

    return blob

def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data']= _get_image_blob(im)
    return blobs
def demo(net, image_name,iteration):
    """Detect object classes in an image using pre-computed object proposals."""
    
    
    cap = cv2.VideoCapture(image_name)
    CONF_THRESH = args.heat
    
    count = 0
    total_time = 0

    width = 720
    height = 480
    
    CNNwidth = 112
    CNNheight = 112
    length = 16
    image_queue = np.zeros((CNNheight, CNNwidth,3,length), dtype=np.float32)
    
    
    
    for idx,path in enumerate(task.split('/')[1:-1]):
        if idx == 0:
            out_name = path
        else:
            out_name += '-'+path
        
    out_path = 'test_video'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for path in image_name.split('/')[1:-1]:
        out_path += '/'+path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_name = image_name.split('/')[-1]
    capSize = (width, height)

    out = cv2.VideoWriter(os.path.join(out_path,video_name+'_'+out_name+'_'+iteration+'_'+dataset+'_'+str(CONF_THRESH)+'.mov'),fourcc, cap.get(5), capSize)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_LINEAR)
            #frame = cv2.flip(frame,-1)
            _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
            timer = Timer()
            timer.tic()
            
            
            _frame = cv2.resize(frame, (CNNwidth, CNNheight), interpolation = cv2.INTER_LINEAR)
            if count == 0:
                for i in range(length):
                    image_queue[:,:,:,i] = _frame
            else:
                image_queue[:,:,:,:length-1] = image_queue[:,:,:,1:]
                image_queue[:,:,:,length-1] = _frame

            _t['im_preproc'].tic()
            blobs = _get_blobs(image_queue)
            
            
            net.blobs['data'].reshape(*(blobs['data'].shape))
            net.blobs['data'].data[...] = blobs['data']
            _t['im_preproc'].toc()
            
            
            _t['im_net'].tic()
            blobs_out = net.forward()
            _t['im_net'].toc()
            
            #prob = blobs_out['prob']
            #################################
            #if 'convcls' in net.blobs:
            #    _t['im_postproc'].tic()
            #    top1_map = net.blobs['convcls'].data[0,prob[0].argmax(),0,:,:]
            #    #fcact = net.params['fcact'][0].data
            #    
            #    #top1_map = fcact[1,:].reshape((-1,1,1)) * top1_map
            #    
            #    #top1_map = np.sum(top1_map,axis=0)
            #    top1_map = top1_map.reshape((1,top1_map.shape[0],top1_map.shape[1]))
            #    
            #    top1_map = top1_map.transpose((1,2,0))
            #    top1_map = cv2.resize(top1_map,(width, height),interpolation=cv2.INTER_CUBIC)
            #    max_n = np.amax(top1_map)
            #    min_n = np.amin(top1_map)
            #    top1_map = top1_map*(-120/(max_n-min_n))+(120*max_n/(max_n-min_n))
            #    matrix = np.zeros((height,width,3),dtype=np.uint8)
            #    #for index_y in range(CNNheight):
            #    #    for index_x in range(CNNwidth):
            #    #        matrix[index_y][index_x][0] = int(top1_map[index_y][index_x])
            #    #        matrix[index_y][index_x][1] = 255*0.7
            #    #        matrix[index_y][index_x][2] = 255*0.7
            #    matrix[:,:,0] = top1_map[:,:]
            #    matrix[:,:,1] = 255
            #    matrix[:,:,2] = 255                
            #    
            #    bgrimg = cv2.cvtColor(matrix, cv2.COLOR_HSV2BGR)
            #    #cv2.imshow('heat map',bgrimg)
            #    frame = cv2.addWeighted(frame,0.7,bgrimg,0.3,0)
            #    _t['im_postproc'].toc()
            if 'convheatmap' in net.blobs:
                _t['im_postproc'].tic()
                top1_map = net.blobs['convheatmap'].data[0,0,0,:,:]
                if np.max(top1_map) > args.heat:
                    top1_map = top1_map.reshape((1,top1_map.shape[0],top1_map.shape[1]))
                    
                    top1_map = top1_map.transpose((1,2,0))
                    top1_map = cv2.resize(top1_map,(width, height),interpolation=cv2.INTER_CUBIC)
                    max_n = np.amax(top1_map)
                    min_n = np.amin(top1_map)
                    top1_map = top1_map*(-120/(max_n-min_n))+(120*max_n/(max_n-min_n))
                    matrix = np.zeros((height,width,3),dtype=np.uint8)
                    matrix[:,:,0] = top1_map[:,:]
                    matrix[:,:,1] = 255
                    matrix[:,:,2] = 255
                else:
                    matrix = np.zeros((height,width,3),dtype=np.uint8)
                    matrix[:,:,0] = 120
                    matrix[:,:,1] = 255
                    matrix[:,:,2] = 255 
                #for index_y in range(CNNheight):
                #    for index_x in range(CNNwidth):
                #        matrix[index_y][index_x][0] = int(top1_map[index_y][index_x])
                #        matrix[index_y][index_x][1] = 255*0.7
                #        matrix[index_y][index_x][2] = 255*0.7
                #if prob[0].argmax()!=0:
                #    matrix[:,:,0] = top1_map[:,:]
                #    matrix[:,:,1] = 255
                #    matrix[:,:,2] = 255                
                #else:
                #    matrix[:,:,0] = 120
                #    matrix[:,:,1] = 255
                #    matrix[:,:,2] = 255  
                
                bgrimg = cv2.cvtColor(matrix, cv2.COLOR_HSV2BGR)
                #cv2.imshow('heat map',bgrimg)
                frame = cv2.addWeighted(frame,0.7,bgrimg,0.3,0)
                _t['im_postproc'].toc()
            #######################################
            frameshape = frame.shape
            #print prob[0].argmax()
            #print prob[0].max()
            print 'im_detect: net {:.3f}s  preproc {:.3f}s  postproc {:.3f}s  misc {:.3f}s' \
              .format(_t['im_net'].average_time,
                      _t['im_preproc'].average_time, _t['im_postproc'].average_time,
                      _t['misc'].average_time)
            #if prob[0].argmax() == 1:
            #    cv2.circle(frame,(int(frameshape[1]*0.7),int(frameshape[0]*0.2)), 10, (0,0,255), -1)
            #elif prob[0].argmax() == 2:
            #    cv2.circle(frame,(int(frameshape[1]*0.3),int(frameshape[0]*0.2)), 10, (0,0,255), -1)
            #else:
            #    cv2.circle(frame,(int(frameshape[1]*0.5),int(frameshape[0]*0.2)), 10, (0,255,0), -1)
            timer.toc()
            total_time += timer.total_time
            out.write(frame)
            if args.show == 1:
                cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        count += 1    
    #Calculate frame per second    
    print 1 / (total_time / count)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--video', dest='video', help='video',
                        default='video/FILE2171.mp4', type=str)
    parser.add_argument('--heat', dest='heat', help='heat',
                        default=5, type=float)
    parser.add_argument('--show', dest='show', help='show',
                        default=1, type=int)
    args = parser.parse_args()

    return args



if __name__ == '__main__': 
    

    args = parse_args()
    
    iteration = args.caffemodel.split('/')[-1].split('_')[-1].split('.')[0]

    dataset = args.caffemodel.split('/')[-2]

    task=args.prototxt
        
    prototxt = args.prototxt
    caffemodel = args.caffemodel
    
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    video = args.video
    demo(net,video,iteration)
    

    
