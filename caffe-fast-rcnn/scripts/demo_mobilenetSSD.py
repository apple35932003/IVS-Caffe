#!/usr/bin/python
import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/chiachi/IVS-Caffe/caffe-fast-rcnn-c3d/caffe-fast-rcnn-2/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


#net_file= '/home/chiachi/IVS-Caffe/caffe-fast-rcnn-c3d/caffe-fast-rcnn-2/models/MobileNet-SSD/q8_BAC3/train_test_quant_BAC_deploy.prototxt'  
#caffe_model='/home/chiachi/IVS-Caffe/caffe-fast-rcnn-c3d/caffe-fast-rcnn-2/models/MobileNet-SSD/q8_BAC3/mobilenet_BAC3_lr000001_iter_5000.caffemodel'  
net_file= '/home/chiachi/IVS-Caffe/caffe-fast-rcnn-c3d/caffe-fast-rcnn-2/models/MobileNet-SSD/ori_/train_test_deploy.prototxt'
caffe_model='/home/chiachi/IVS-Caffe/caffe-fast-rcnn-c3d/caffe-fast-rcnn-2/models/MobileNet-SSD/ori_/mobilenet_iter_90000.caffemodel'  
test_video = "16_cut.mp4"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(frame):
    origimg = frame
    img = preprocess(origimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)
    inds = np.where(conf[:] >= 0.4)[0]
    for i in inds:
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(origimg, p1, p2, (0,255,0),2)
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg, title, p3, cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 0))
    #cv2.imshow("SSD", origimg)
 
    #k = cv2.waitKey(0) & 0xff
    #    #Exit if ESC pressed
    #if k == 27 : return False
    return origimg
if __name__ == '__main__':
    caffe.set_mode_gpu()
    caffe.set_device(1)
    cap = cv2.VideoCapture(test_video)

    capSize = (300, 300)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('out.MOV',fourcc, 29.97, capSize)
    counter = 0
    while(cap.isOpened()):
        print counter
        counter+=1
        ret, frame = cap.read()
        if ret:
            out_frame = detect(frame)
            out.write(out_frame)
    #for f in os.listdir(test_dir):
    #    if detect(test_dir + "/" + f) == False:
    #        break
