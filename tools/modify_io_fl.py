#!/usr/bin/env python

# --------------------------------------------------------
# Quantize Fast R-CNN based Network
# Written by Chia-Chi Tsai
# --------------------------------------------------------

"""Quantize a Fast R-CNN network on an image database."""

import os
os.environ['GLOG_minloglevel'] = '2'
import _init_paths
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

def read_from_prototxt(ori_net_path):
    net_proto = caffe_pb2.NetParameter()
    fn = ori_net_path;
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net_proto)
    return net_proto

def write_to_prototxt(net_proto, out_net_path):
    outf = out_net_path
    #print 'writing', outf
    with open(outf, 'w') as f:
        f.write(str(net_proto))

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Modify network according to txt file')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--def_out', dest='prototxt_out',
                        help='output prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--modify_txt', dest='modify_txt',
                        help='txt file containing modification of network',
                        default='None', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.modify_txt) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    net_proto = read_from_prototxt(args.prototxt)
    for line in content:
        t = line.split()
        if t[0] != "Layer:":
            continue
        layer_name = t[1]
        type_param = t[-5]
        fl = t[-1]
        for l in net_proto.layer:
            if l.name == layer_name:
                if type_param == "layer_in_fl":
                    print 'change Layer {} fl_layer_in from {} to {}'.format(layer_name, l.quantization_param.fl_layer_in, fl)
                    l.quantization_param.fl_layer_in = int(fl)
                elif type_param == "layer_out_fl":
                    print 'change Layer {} fl_layer_out from {} to {}'.format(layer_name, l.quantization_param.fl_layer_out, fl)
                    l.quantization_param.fl_layer_out = int(fl)
                elif type_param == "fl_params":
                    print 'change Layer {} param_fl from {} to {}'.format(layer_name, l.quantization_param.fl_params, fl)
                    l.quantization_param.fl_params = int(fl)
    write_to_prototxt(net_proto, args.prototxt_out)
    sys.exit(0)
