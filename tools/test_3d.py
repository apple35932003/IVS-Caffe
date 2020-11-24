#!/usr/bin/env python

import _init_paths_yolo
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

y_offset = 0 
def vis_detections(im, class_name, dets, cls, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    tmp = im
    im = im[:, :, (2, 1, 0)]
    person_bbx = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]    
        width = bbox[2] - bbox[0]
        cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
        cv2.rectangle(tmp, (bbox[0], int(bbox[1]-20)), (int(bbox[0]+width), int(bbox[1])), (0,0,255), cv2.FILLED)
        cv2.putText(tmp,cls + ' '+ str(score),(bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(0,0,0))

        #if bbox[0]>x_offset:
        #    width = bbox[2] - bbox[0]
        #    if cls == 'scooter':
        #        bbox[1] += y_offset
        #        bbox[3] += y_offset
        #        cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
        #        cv2.rectangle(tmp, (bbox[0], int(bbox[1]-20)), (int(bbox[0]+width), int(bbox[1])), (0,0,255), cv2.FILLED)
        #        cv2.putText(tmp,"motorcycle",(bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(0,0,0))
        #        if(args.pr==1):
        #            cv2.putText(tmp,str(score),(bbox[0],bbox[3]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(0,0,255))
        #    elif cls =='Pedestrian':
        #        bbox[1] += y_offset
        #        bbox[3] += y_offset
        #        cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,116,21), 2)
        #        cv2.rectangle(tmp, (bbox[0], int(bbox[1]-20)), (int(bbox[0]+width), int(bbox[1])), (255,116,21), cv2.FILLED)
        #        cv2.putText(tmp,"person",(bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(0,0,0))
        #        if(args.pr==1):
        #            cv2.putText(tmp,str(score),(bbox[0],bbox[3]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(255,116,21))
        #    elif cls == 'vehicle':
        #        bbox[1] += y_offset
        #        bbox[3] += y_offset
        #        cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,116,21), 2)
        #        cv2.rectangle(tmp, (bbox[0], int(bbox[1]-20)), (int(bbox[0]+width), int(bbox[1])), (255,116,21), cv2.FILLED)
        #        cv2.putText(tmp,"vehicle",(bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(0,0,0))
        #        if(args.pr==1):
        #            cv2.putText(tmp,str(score),(bbox[0],bbox[3]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(255,116,21))
        #    
        #    # elif cls == 'car': 
        #        # bbox[1] += y_offset
        #        # bbox[3] += y_offset
        #        # cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,116,21), 2)
        #        # cv2.rectangle(tmp, (bbox[0], int(bbox[1]-20)), (int(bbox[0]+width), int(bbox[1])), (255,116,21), cv2.FILLED)
        #        # cv2.putText(tmp,"vehicle",(bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(0,0,0))
        #        # if(args.pr==1):
        #            # cv2.putText(tmp,str(score),(bbox[0],bbox[3]), cv2.FONT_HERSHEY_DUPLEX ,0.5,(255,116,21))
    return person_bbx

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    CONF_THRESH = args.conf
    NMS_THRESH = 0.3
    
    cap = cv2.VideoCapture(image_name)
    capSize = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('out2.mov',fourcc, 29.97, capSize)
    count = 0
    total_time = 0
    total_obj_nms = 0
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    while(cap.isOpened()):
        print count
        count += 1
        ret, frame = cap.read()
        #if count==2000:
        #    break
        if ret:
            frame = cv2.resize(frame, capSize, interpolation = cv2.INTER_LINEAR)
            #frame = cv2.resize(frame, (720,480), interpolation = cv2.INTER_LINEAR)
            timer = Timer()
            timer.tic()
            scores, boxes = im_detect(net, frame,_t)
            
            print net.blobs['fc6/t/t'].data[...]
            sys.exit(0)
            for cls_ind, cls in enumerate(CLASSES[1:]):
                cls_ind += 1 # because we skipped background
                #if cls == 'person' or cls == 'car' or cls == 'scooter':
                    
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                           cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]

                tmp = vis_detections(frame, cls, dets, cls, thresh=CONF_THRESH)

            timer.toc()
            total_obj_nms += len(keep)
            print ('Detection took {:.3f}s for '
               '{:d} object proposals and {:d} object proposals after nms //// {:d}s ').format(timer.total_time, boxes.shape[0], len(keep),count/29)
            total_time += timer.total_time
            frame = cv2.resize(frame, capSize, interpolation = cv2.INTER_LINEAR)
            frame = cv2.resize(frame, (720,480), interpolation = cv2.INTER_LINEAR)
            out.write(frame)
            #frame = cv2.resize(frame, (720, 480), interpolation = cv2.INTER_LINEAR)
            if args.show == 1:
                cv2.imshow('frame',frame)
            #if cv2.waitKey(0) & 0xFF == ord('q'):
            #    break
            if cv2.waitKey(1) & 0xFF == 83:
                continueSS
                
        else:
            break

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
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--video', dest='video', help='video',
                        default='video/iron_video/HighwayDay_Science\ park20170504-3_NCTU.mp4', type=str)
    parser.add_argument('--conf', dest='conf', help='conf',
                        default=0.85, type=float)
    parser.add_argument('--show', dest='show', help='show',
                        default=1, type=int)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    cfg_from_file('./experiments/cfgs/faster_rcnn_end2end.yml')

    '''ZF or VGG16'''
    #prototxt = './models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC12/train_test_quant_BAC.prototxt'
    #caffemodel = './output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_q6_BAC12_iter_160000.caffemodel'
    prototxt = './models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC12/train_test.prototxt'
    caffemodel = './models/pascal_voc/ZF/faster_rcnn_end2end_q6_BAC12/zf_faster_rcnn_iter_70000.caffemodel'
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
    #demo(net, 'coretronic_video/005.avi', task, iteration, dataset)
    #demo(net, 'video/FILE2171.mp4', task, iteration, dataset)
    demo(net, video) 
