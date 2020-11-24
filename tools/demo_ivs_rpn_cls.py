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
Models = ['pvalite_3cls_rpn_merge']
Models_list = ''
for index,i in enumerate(Models):
    Models_list = Models_list + '('+str(index)+',' + i.split('/')[-1]+')'+'\n'
    
Datasets = ['IVS_PASCAL_All']   
Datasets_list = ''
for index,i in enumerate(Datasets):
    Datasets_list = Datasets_list + '('+str(index)+',' + i+')'+'\n'

    
CLASSES = ('__background__', 'person', 'car','scooter')
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
width  = 720
height = 480

y_offset = 0 
def vis_detections(im, class_name, dets, cls, frame_id,res_out=None,thresh=0.5):
    """Draw detected bounding boxes."""
    person_bbx = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return person_bbx
    tmp = im
    im = im[:, :, (2, 1, 0)]

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]    
        width = bbox[2]-bbox[0]
        height = bbox[3]-bbox[1]
        if cls == 'person':
            cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,116,21), 2)
            cv2.putText(tmp,str(score),(bbox[0],int(bbox[3]-height*0.1)), cv2.FONT_HERSHEY_DUPLEX ,0.5,(255,116,21))
            person_bbx.append([bbox[0], bbox[1], bbox[2], bbox[3]])
        elif cls == 'car':
            cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(tmp,str(score),(bbox[0],int(bbox[3]-height*0.1)), cv2.FONT_HERSHEY_DUPLEX ,0.5,(0,255,0))
            if args.txt==1:
                print >> res_out, frame_id, bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], score
        elif cls == 'scooter':
            cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
            cv2.putText(tmp,str(score),(bbox[0],int(bbox[3]-height*0.1)), cv2.FONT_HERSHEY_DUPLEX ,0.5,(0,0,255))
    return person_bbx

def vis_detections_rpn(im, dets,rpn_cls_scores, thresh=0.5):
    """Draw detected bounding boxes."""
    person_bbx = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return person_bbx
    tmp = im
    im = im[:, :, (2, 1, 0)]

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        class_type =  np.argmax(rpn_cls_scores[i,:]) 
        width = bbox[2]-bbox[0]
        height = bbox[3]-bbox[1]
        if class_type == 1:
            cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,116,21), 2)
        elif class_type == 2:
            cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
        else:
            cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,255), 2)
    return person_bbx

def demo(net, image_name, task, iteration, dataset):
    """Detect object classes in an image using pre-computed object proposals."""
    
    CONF_THRESH = args.conf
    NMS_THRESH = args.nms
    
    cap = cv2.VideoCapture(image_name)
    capSize = (width, height)
    _image_name = image_name.split('.')[0]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('result_videos/'+_image_name+'_'+task+'_'+iteration+'_'+dataset+'.mov',fourcc, 29.97, capSize)
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
                    tmp = vis_detections(frame, cls, dets, cls,count,res_out=res_out, thresh=CONF_THRESH)
                else:
                    tmp = vis_detections(frame, cls, dets, cls,count, thresh=CONF_THRESH)
		    
            #dets = np.hstack((rpn_boxes[:,1:],rpn_scores)).astype(np.float32)
            #keep = nms(dets, NMS_THRESH)
            #dets = dets[keep, :]
            #print np.amax(rpn_cls_scores,axis = 0)
            #rpn_cls_scores = rpn_cls_scores[keep,:]
            #vis_detections_rpn(frame, dets,rpn_cls_scores, thresh=CONF_THRESH)

            
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
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--iters', dest='iters', help='iters',
                        default=10000, type=int)
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
    parser.add_argument('--models', dest='models', help=Models_list,
                        default=0, type=int)
    parser.add_argument('--data', dest='dataset', help=Datasets_list,
                        default=0, type=int)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    if width < 1000:
        cfg_from_file('./models/pvanet/cfgs/300prop.yml')
    else:
        cfg_from_file('./models/pvanet/cfgs/submit_0716.yml')

    '''ZF or VGG16'''
    arch = 0
    iteration = str(args.iters)
    '''Which dataset'''
    #dataset = 'IVS_3cls_PD_Scooter_Vehicle_random'
    dataset = Datasets[args.dataset]
    if arch == 0:
        #task = 'pvanet_lite_3cls'
        task=Models[args.models].split('/')[-1]
        
    	prototxt = './models/'+Models[args.models]+'/test.prototxt'
    	caffemodel = './models/'+Models[args.models]+'/'+dataset+'/kai_train_iter_'+iteration+'.caffemodel'
        #caffemodel = '/home/ivskaikai/pva-faster-rcnn/output/'+task+'/'+dataset+'/zf_faster_rcnn_iter_'+iteration+'.caffemodel'
    elif arch ==1:
        task='pva_lite'
        prototxt = './models/pvanet/lite/test.pt'
    	caffemodel = './models/pvanet/lite/test.model'
    else: 
    	#task = 'pvanet_lite_3cls'
        task='pvalite_v3'
    	
    	prototxt = './models/MobileNet/MobileNet_2cls/test.prototxt'
    	caffemodel = './output/MobileNet_2cls/'+dataset+'/kai_train_iter_'+iteration+'.caffemodel'
    
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
    video = args.video
    print '\n\nLoaded network {:s}'.format(caffemodel)
    '''Main Function'''
    '''Input Video'''
    demo(net,video, task, iteration, dataset)
    #demo(net, 'video/IMG_6023.mp4', task, iteration, dataset)
