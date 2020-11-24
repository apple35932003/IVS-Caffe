#!/usr/bin/env python

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

    
CLASSES = ('__background__', 'vehicle','pedestrian', 'bike')
#CLASSES = ('__background__', 'person', 'car')
#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
width  = 416
height = 416

y_offset = 0 


camera_focal = 550
camera_height = 1.6
global_h_position = 130
def vis_detections(im, class_name, dets, cls, frame_id,res_out=None,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    #cv2.line(im, (0,480*3/16), (720,480*3/16), (0,0,255), 2)
    #cv2.line(im, (0,480*7/16), (720,480*7/16), (0,0,255), 2)
    #cv2.line(im, (720*1/3,0), (720*1/3,480), (0,0,255), 2)
    #cv2.line(im, (720*2/3,0), (720*2/3,480), (0,0,255), 2)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]    
        width = bbox[2]-bbox[0]
        height = bbox[3]-bbox[1]
        if cls == 'pedestrian':
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,116,21), 2)
            if(args.dis==0):
                if(args.pr==1):
                    cv2.putText(im,str(score),(bbox[0],bbox[3]), cv2.FONT_HERSHEY_DUPLEX ,0.3,(255,116,21))
            else:
                if bbox[3] != global_h_position:
                    distance = camera_height * camera_focal / float(bbox[3] - global_h_position)
                    distance_str = str(int(np.floor(distance)))
                    cv2.putText(im,distance_str,(bbox[0],bbox[3]),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                    cv2.putText(im,distance_str,(bbox[0],bbox[3]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1)
        elif cls == 'vehicle':
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            if(args.dis==0):
                if(args.pr==1):
                    cv2.putText(im,str(score),(bbox[0],bbox[3]), cv2.FONT_HERSHEY_DUPLEX ,0.3,(0,255,0))
            else:
                if bbox[3] != global_h_position:
                    distance = camera_height * camera_focal / float(bbox[3] - global_h_position)
                    distance_str = str(int(np.floor(distance)))
                    cv2.putText(im,distance_str,(bbox[0],bbox[3]),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                    cv2.putText(im,distance_str,(bbox[0],bbox[3]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1)
            if args.txt==1:
                print >> res_out, frame_id, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], score
        elif cls == 'bike':
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
            if(args.dis==0):
                if(args.pr==1):
                    cv2.putText(im,str(score),(bbox[0],bbox[3]), cv2.FONT_HERSHEY_DUPLEX ,0.3,(0,0,255))
            else:
                if bbox[3] != global_h_position:
                    distance = camera_height * camera_focal / float(bbox[3] - global_h_position)
                    distance_str = str(int(np.floor(distance)))
                    cv2.putText(im,distance_str,(bbox[0],bbox[3]),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
                    cv2.putText(im,distance_str,(bbox[0],bbox[3]),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1)

def demo(net, image_name, task, iteration, dataset):
    """Detect object classes in an image using pre-computed object proposals."""
    
    CONF_THRESH = args.conf
    NMS_THRESH = args.nms
    
    cap = cv2.VideoCapture(image_name)
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
    
    if args.txt==1:
        if not os.path.exists('caltech/'+_image_name+'/'):
            os.makedirs('caltech/'+_image_name+'/')
        res_out = open('caltech/'+_image_name+'/'+_image_name+'_'+task+'_'+iteration+'_'+dataset+'_'+str(CONF_THRESH)+'_'+str(NMS_THRESH)+'.txt', 'w')
    
    count = 0
    total_time = 0
    total_obj_nms = 0
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_LINEAR)
            _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
            timer = Timer()
            
            _frame = frame[y_offset:, :]
            timer.tic()
            rpn_scores ,rpn_boxes = im_detect(net, _frame,_t)
            timer.toc()

            
            #print rpn_scores.shape
            #print rpn_boxes.shape
            print 'net {:.3f}s  preproc {:.3f}s  postproc {:.3f}s  ' \
              .format(_t['im_net'].average_time,
                      _t['im_preproc'].average_time, _t['im_postproc'].average_time)

            for cls_ind in range(1,rpn_scores.shape[1]):
                
                cls = CLASSES[cls_ind]
                cls_boxes = rpn_boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = rpn_scores[:, cls_ind]
                cls_scores = cls_scores.reshape((-1,1))
                dets = np.hstack((cls_boxes,cls_scores)).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                if args.txt==1:
                    vis_detections(frame, cls, dets, cls,count,res_out=res_out, thresh=CONF_THRESH)
                else:
                    vis_detections(frame, cls, dets, cls,count, thresh=CONF_THRESH)
		    

            
            total_obj_nms += len(keep)
            print ('Detection took {:.3f}FPS for '
               '{:d} object proposals and {:d} object proposals after nms').format(1/timer.total_time, rpn_boxes.shape[0], len(keep))
            total_time += timer.total_time
            #frame = cv2.resize(frame, (720, 480), interpolation = cv2.INTER_LINEAR)
            if args.show == 1:
                cv2.imshow('frame',frame)
            out.write(frame)
            #cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        count += 1    
    #Calculate frame per second    
    print 1 / (total_time / count), total_obj_nms

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--video', dest='video', help='video',
                        default='video/FILE2171.mp4', type=str)
    parser.add_argument('--conf', dest='conf', help='conf',
                        default=0.85, type=float)
    parser.add_argument('--nms', dest='nms', help='nms',
                        default=0.3, type=float)
    parser.add_argument('--show', dest='show', help='show',
                        default=1, type=int)
    parser.add_argument('--txt', dest='txt', help='txt',
                        default=0, type=int)
    parser.add_argument('--dis', dest='dis', help='distance',
                        default=0, type=int)
    parser.add_argument('--pr', dest='pr', help='probability',
                        default=0, type=int)
    parser.add_argument('--l_res', dest='low_resolution', help='low_resolution',
                        default=0, type=int)
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    cfg_from_file(args.cfg_file)

    '''ZF or VGG16'''
    arch = 0
    iteration = args.caffemodel.split('/')[-1].split('_')[-1].split('.')[0]
    '''Which dataset'''
    #dataset = 'IVS_dataset_compcar_Compcar_random'
    #dataset = 'voc_2007_PD_CAR_SCOOTER' 
    dataset = args.caffemodel.split('/')[-2]
    #dataset = 'IVS_dataset_281_All'
    task=args.prototxt
        
    prototxt = args.prototxt
    caffemodel = args.caffemodel
    
    '''Check Model exist or not'''
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    '''Set GPU Mode gpu_id:0~3'''
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id

    '''Parsing Network'''
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    '''Main Function'''
    '''Input Video'''
    video = args.video

    demo(net, video, task, iteration, dataset)
