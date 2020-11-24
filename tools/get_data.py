#!/usr/bin/env python

# --------------------------------------------------------
# Quantize Fast R-CNN based Network
# Written by Chia-Chi Tsai
# --------------------------------------------------------

"""Quantize a Fast R-CNN network on an image database."""

import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import math
import cv2
import shutil
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Quantize a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--image', dest='image',
                        help='image path to test',
                        default=None, type=str)
    parser.add_argument('--layer_name', dest='layer_name',
                        help='layer',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    img = caffe.io.load_image(args.image)
    inputs = img
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
    print net.blobs[args.layer_name].data[...]
    #for data in net.blobs:
    #   print data[...] 
