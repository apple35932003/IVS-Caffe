#!/usr/bin/env python

# --------------------------------------------------------
# Quantize Fast R-CNN based Network
# Written by Chia-Chi Tsai
# --------------------------------------------------------

"""Quantize a Fast R-CNN network on an image database."""

import os
os.environ['GLOG_minloglevel'] = '2' 
import _init_paths
from fast_rcnn.test import test_net, test_net_silent, im_detect
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import math
import cv2
from utils.timer import Timer
import multiprocessing
import json
import shutil
import warnings
warnings.filterwarnings("ignore")
from utils.timer import Timer


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
    parser.add_argument('--def_quant', dest='prototxt_quantized',
                        help='quantized prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--def_quant_BAC', dest='prototxt_quantized_BAC',
                        help='quantized prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--act_analysis', dest='act_analysis',
                        help='input and output analysis file',
                        default=None, type=str)
    parser.add_argument('--accumulator_analysis', dest='accumulator_analysis',
                        help='adder and multiplier analysis file',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--error_margin', dest='error_margin',
                        help='tolerance error of quantized network',
                        default=0.1, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def analyze_network(net_proto):
    has_fc = False
    has_deconv = False
    has_conv = False
    for l in net_proto.layer:
        if l.type == 'Convolution':
            has_conv = True
        elif l.type == 'Deconvolution':
            has_deconv = True
        elif l.type =='InnerProduct':
            has_fc = True
    return has_conv, has_deconv, has_fc

# convert network to quantized network with 32 bit width
def convert_net_to_qnet(ori_net_path, q_net_path):
    net_proto = read_from_prototxt(ori_net_path)
    new_blob_name = {}
    for l in net_proto.layer:
        for i in range(len(l.top)):
            for j in range(len(l.bottom)):
                if l.top[i] == l.bottom[j]:
                    if not l.top[i] in new_blob_name.keys():
                        new_blob_name[l.top[i]]=l.top[i]+'/t'
                    else:
                        l.bottom[j] = new_blob_name[l.bottom[j]]
                        new_blob_name[l.top[i]]=new_blob_name[l.top[i]]+'/t'
                    l.top[i] = new_blob_name[l.top[i]]
                else:
                    for k in range(len(l.bottom)):
                        if l.bottom[k] in new_blob_name.keys():
                            l.bottom[k] = new_blob_name[l.bottom[k]]
        if l.type == 'Convolution':
            l.type = 'ConvolutionIVS'
            l.quantization_param.precision =  0 #DYNAMIC_FIXED_POINT
            l.quantization_param.bw_layer_in = 32
            l.quantization_param.bw_layer_out = 32
            l.quantization_param.bw_params = 32
            l.quantization_param.fl_layer_in = 16
            l.quantization_param.fl_layer_out= 16
            l.quantization_param.fl_params = 16
            l.quantization_param.rounding_time = 0
        elif l.type =='InnerProduct':
            l.type = 'FcIVS'
            l.quantization_param.precision =  0 #DYNAMIC_FIXED_POINT
            l.quantization_param.bw_layer_in = 32
            l.quantization_param.bw_layer_out = 32
            l.quantization_param.bw_params = 32
            l.quantization_param.fl_layer_in = 16
            l.quantization_param.fl_layer_out= 16
            l.quantization_param.fl_params = 16
            l.quantization_param.rounding_time = 0
        elif l.type =='Deconvolution':
            l.type = 'DeconvolutionRistretto'
            l.quantization_param.precision =  0 #DYNAMIC_FIXED_POINT
            l.quantization_param.bw_layer_in = 32
            l.quantization_param.bw_layer_out = 32
            l.quantization_param.bw_params = 32
            l.quantization_param.fl_layer_in = 16
            l.quantization_param.fl_layer_out= 16
            l.quantization_param.fl_params = 16
            l.quantization_param.rounding_time = 0
        
    write_to_prototxt(net_proto, q_net_path)

# convert network to quantized network with 32 bit width
def convert_net_to_qnet_BAC_analysis(ori_net_path, q_net_path):
    net_proto = read_from_prototxt(ori_net_path)
    new_blob_name = {}
    for l in net_proto.layer:
        for i in range(len(l.top)):
            for j in range(len(l.bottom)):
                if l.top[i] == l.bottom[j]:
                    if not l.top[i] in new_blob_name.keys():
                        new_blob_name[l.top[i]]=l.top[i]+'/t'
                    else:
                        l.bottom[j] = new_blob_name[l.bottom[j]]
                        new_blob_name[l.top[i]]=new_blob_name[l.top[i]]+'/t'
                    l.top[i] = new_blob_name[l.top[i]]
                else:
                    for k in range(len(l.bottom)):
                        if l.bottom[k] in new_blob_name.keys():
                            l.bottom[k] = new_blob_name[l.bottom[k]]
        if l.type == 'Convolution' or l.type == 'ConvolutionIVS':
            l.type = 'ConvolutionIVS'
            l.quantization_param.precision =  0 #DYNAMIC_FIXED_POINT
            l.quantization_param.bw_add = 32
            l.quantization_param.bw_multiply = 32
            l.quantization_param.fl_add = 16
            l.quantization_param.fl_multiply = 16
            l.quantization_param.rounding_time = 1
            l.quantization_param.analyze_mode = 3
        if l.type == 'InnerProduct' or l.type == 'FcIVS':
            l.type = 'FcIVS'
            l.quantization_param.precision =  0 #DYNAMIC_FIXED_POINT
            l.quantization_param.bw_add = 32
            l.quantization_param.bw_multiply = 32
            l.quantization_param.fl_add = 16
            l.quantization_param.fl_multiply = 16
            l.quantization_param.rounding_time = 1
            l.quantization_param.analyze_mode = 3
    write_to_prototxt(net_proto, q_net_path)

def convert_net_to_qnet_BAC(ori_net_path, q_net_path):
    net_proto = read_from_prototxt(ori_net_path)
    new_blob_name = {}
    for l in net_proto.layer:
        for i in range(len(l.top)):
            for j in range(len(l.bottom)):
                if l.top[i] == l.bottom[j]:
                    if not l.top[i] in new_blob_name.keys():
                        new_blob_name[l.top[i]]=l.top[i]+'/t'
                    else:
                        l.bottom[j] = new_blob_name[l.bottom[j]]
                        new_blob_name[l.top[i]]=new_blob_name[l.top[i]]+'/t'
                    l.top[i] = new_blob_name[l.top[i]]
                else:
                    for k in range(len(l.bottom)):
                        if l.bottom[k] in new_blob_name.keys():
                            l.bottom[k] = new_blob_name[l.bottom[k]]
        if l.type == 'Convolution' or l.type == 'ConvolutionIVS':
            l.type = 'ConvolutionIVS'
            l.quantization_param.analyze_mode = 0
            l.quantization_param.rounding_time = 1
        if l.type == 'InnerProduct' or l.type == 'FcIVS':
            l.type = 'FcIVS'
            l.quantization_param.analyze_mode = 0
            l.quantization_param.rounding_time = 1

    write_to_prototxt(net_proto, q_net_path)

#change single layer bit width
def change_layer_bw(net_proto, layer_name, 
                    bw_layer_in, fl_layer_in, 
                    bw_layer_out, fl_layer_out, 
                    bw_params, fl_params,
                    bw_add, fl_add,
                    bw_multiply, fl_multiply):
    for l in net_proto.layer:
        if l.name == layer_name:
            l.quantization_param.precision =  0
            l.quantization_param.bw_layer_in = int(bw_layer_in)
            l.quantization_param.bw_layer_out = int(bw_layer_out)
            l.quantization_param.bw_params = int(bw_params)
            l.quantization_param.bw_add = int(bw_add)
            l.quantization_param.bw_multiply = int(bw_multiply)
            l.quantization_param.fl_layer_in = int(fl_layer_in)
            l.quantization_param.fl_layer_out= int(fl_layer_out)
            l.quantization_param.fl_params = int(fl_params)
            l.quantization_param.fl_add = int(fl_add)
            l.quantization_param.fl_multiply = int(fl_multiply)
    return net_proto

def change_layer_BAC_bw(net_proto, lVayer_name, 
                        bw_add, fl_add, 
                        bw_multiply, fl_multiply):
    for l in net_proto.layer:
        if l.name == layer_name:
            l.quantization_param.bw_add = bw_add
            l.quantization_param.fl_add = fl_add
            l.quantization_param.bw_multiply = bw_multiply
            l.quantization_param.fl_multiply = fw_multiply
    return net_proto

def change_layer_bottom_name(net_proto, layer_name,
                    layer_bottom_name):
    for l in net_proto.layer:
        if l.name == layer_name:
            l.bottom = layer_bottom_name
    return net_proto

def change_layer_top_name(net_proto, layer_name,
                    layer_top_name):
    for l in net_proto.layer:
        if l.name == layer_name:
            l.top = layer_top_name
    return net_proto

#calculate needed Integer Length of layer parameters
def calc_layer_param_IL(net,layer):
    layer_param = net.params[layer.name] 
    max_weight = max(layer_param[0].data[...].max(), layer_param[0].data[...].min(), key=abs)
    if layer.convolution_param.bias_term:
        max_bias = max(layer_param[1].data[...].max(), layer_param[1].data[...].min(), key=abs)
    else:
        max_bias = 0
    max_param = max(max_weight, max_bias, key=abs)
    return math.ceil(math.log(abs(max_param), 2)) + 1


def analyze_net_param_IL(net, net_proto):
    net_param_IL = dict()
    for layer in net_proto.layer:
        if layer.type == 'ConvolutionIVS' \
            or layer.type == 'FcIVS' \
            or layer.type == 'DeconvolutionRistretto':
            net_param_IL[layer.name] = calc_layer_param_IL(net, layer)
    return net_param_IL


#calculate needed Integer Length of layer output
def calc_layer_inout_IL(net, layer_bottom_name):
    layer_output = net.blobs[layer_bottom_name].data
    layer_output_max = abs(max(layer_output.max(), layer_output.min(), key=abs))
    #if layer_bottom_name == 'data':
    #    print net.blobs[layer_bottom_name].data 
    #    print math.ceil(math.log(layer_output_max, 2)) + 1
    return math.ceil(math.log(layer_output_max, 2)) + 1


def analyze_net_output_IL(net, net_proto, imdb, max_per_image=100, thresh=0.05, vis=False):
    num_images = len(imdb.image_index)
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb
    net_output_IL = dict()
    net_input_IL = dict()
    for layer in net_proto.layer:
        #if layer.top[0] == layer.bottom[0]:
        #    print layer.name, layer.type
        if layer.type == 'ConvolutionIVS' \
            or layer.type == 'FcIVS' \
            or layer.type == 'DeconvolutionRistretto':
            assert layer.top[0] != layer.bottom[0],"bottom name cannot be the same as top name in the same layer, at layer:{} top:{} bottom:{}".format(layer.name,layer.top[0],layer.bottom[0])
            net_output_IL[layer.name] = -sys.maxint - 1
            net_input_IL[layer.name] = -sys.maxint - 1
                
    for i in xrange(num_images):
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        scores, boxes = im_detect(net, im, _t, box_proposals)
        for layer in net_proto.layer:
            if layer.type == 'ConvolutionIVS' \
                or layer.type == 'FcIVS' \
                or layer.type == 'DeconvolutionRistretto':
                net_output_IL[layer.name] = max(calc_layer_inout_IL(net, layer.top[0]), net_output_IL[layer.name])
                net_input_IL[layer.name] = max(calc_layer_inout_IL(net, layer.bottom[0]), net_input_IL[layer.name])
                #print layer.type, layer.name, net_output_IL[layer.name],net_input_IL[layer.name]
    return net_output_IL, net_input_IL
    
#calculate needed Integer Length of layer adder
def calc_layer_adder_IL(net, layer_top_name):
    layer_adder_max = abs(max(
            net.blobs[layer_top_name].data.reshape(net.blobs[layer_top_name].data.size)[0],
            net.blobs[layer_top_name].data.reshape(net.blobs[layer_top_name].data.size)[1],
            key=abs))
    return math.ceil(math.log(layer_adder_max, 2)) + 1
#calculate needed Integer Length of layer multiplier
def calc_layer_multiplier_IL(net, layer_top_name):
    layer_multiplier_max = abs(max(
            net.blobs[layer_top_name].data.reshape(net.blobs[layer_top_name].data.size)[2],
            net.blobs[layer_top_name].data.reshape(net.blobs[layer_top_name].data.size)[3],
            key=abs))
    return math.ceil(math.log(layer_multiplier_max, 2)) + 1
#analyze adder and multiplier of each layer in network        
def analyze_net_adder_multiplier_IL(net, net_proto, imdb, max_per_image=100, thresh=0.05, vis=False):
    num_images = len(imdb.image_index)
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb
    net_adder_IL = dict()
    net_multiplier_IL = dict()
    for layer in net_proto.layer:
        #if layer.top[0] == layer.bottom[0]:
        #    print layer.name, layer.type
        if layer.type == 'ConvolutionIVS' \
            or layer.type == 'FcIVS' :
            assert layer.top[0] != layer.bottom[0],"bottom name cannot be the same as top name in the same layer, at layer:{} top:{} bottom:{}".format(layer.name,layer.top[0],layer.bottom[0])
            net_adder_IL[layer.name] = -sys.maxint - 1
            net_multiplier_IL[layer.name] = -sys.maxint - 1
                
    for i in xrange(num_images):
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        scores, boxes = im_detect(net, im, _t, box_proposals)

    for layer in net_proto.layer:
        if layer.type == 'ConvolutionIVS':
            net.params[layer.name][0].data[0][0][0][0]=2610214
        elif layer.type == 'FcIVS':
            net.params[layer.name][0].data[0][0]=2610214
    net.forward()

    for layer in net_proto.layer:
        if layer.type == 'ConvolutionIVS' \
            or layer.type == 'FcIVS':
            net_adder_IL[layer.name] = max(calc_layer_adder_IL(net, layer.top[0]), 
                                            net_adder_IL[layer.name])
            net_multiplier_IL[layer.name] = max(calc_layer_multiplier_IL(net, layer.top[0]), 
                                                net_multiplier_IL[layer.name])
    return net_adder_IL, net_multiplier_IL

#quantize adder in network
def quantize_net_adder(net_proto, net_adder_IL, adder_bw, extra_IL):
    for layer in net_proto.layer:
        if layer.type == 'ConvolutionIVS' \
            or layer.type == 'FcIVS':
            adder_IL = net_adder_IL[layer.name] + extra_IL
            adder_FL = adder_bw - adder_IL
            change_layer_bw(net_proto, layer.name, \
                            layer.quantization_param.bw_layer_in, \
                            layer.quantization_param.fl_layer_in, \
                            layer.quantization_param.bw_layer_out, \
                            layer.quantization_param.fl_layer_out, \
                            layer.quantization_param.bw_params, \
                            layer.quantization_param.fl_params, \
                            adder_bw, adder_FL, \
                            layer.quantization_param.bw_multiply, \
                            layer.quantization_param.fl_multiply, \
                            )

#quantize multiplier in network
def quantize_net_multiplier(net_proto, net_multiplier_IL, multiplier_bw, extra_IL):
    for layer in net_proto.layer:
        if layer.type == 'ConvolutionIVS' \
            or layer.type == 'FcIVS':
            multiplier_IL = net_multiplier_IL[layer.name] + extra_IL
            multiplier_FL = multiplier_bw - multiplier_IL
            change_layer_bw(net_proto, layer.name, \
                            layer.quantization_param.bw_layer_in, \
                            layer.quantization_param.fl_layer_in, \
                            layer.quantization_param.bw_layer_out, \
                            layer.quantization_param.fl_layer_out, \
                            layer.quantization_param.bw_params, \
                            layer.quantization_param.fl_params, \
                            layer.quantization_param.bw_add, \
                            layer.quantization_param.fl_add, \
                            multiplier_bw, multiplier_FL, \
                            )

#quantize input and output of each layer in network
def quantize_net_output(net_proto, net_output_IL, net_input_IL, output_bw, extra_IL):
    input_bw = output_bw;
    #input_FL = 0;
    for layer in net_proto.layer:
        if layer.type == 'ConvolutionIVS' \
             or layer.type == 'FcIVS' \
                or layer.type == 'DeconvolutionRistretto':
            output_IL = net_output_IL[layer.name] + extra_IL 
            output_FL = output_bw - output_IL
            input_IL = net_input_IL[layer.name] + extra_IL
            input_FL = input_bw - input_IL
            #if layer.name=='conv1_1/conv':
            #    print input_IL,output_IL
            #print layer.name
            #if layer.name == 'conv1_1/conv':
            #    print output_IL
            #    continue
            change_layer_bw(net_proto, layer.name, \
                            input_bw, input_FL, \
                            output_bw, output_FL, \
                            layer.quantization_param.bw_params, \
                            layer.quantization_param.fl_params, \
                            layer.quantization_param.bw_add, \
                            layer.quantization_param.fl_add, \
                            layer.quantization_param.bw_multiply, \
                            layer.quantization_param.fl_multiply, \
                            )
                            
            #input_FL = output_FL


#quantize convolution layers in network
def quantize_net_conv(net_proto, net_param_IL, weighting_bw, extra_IL):
    for layer in net_proto.layer:
        if layer.type == 'ConvolutionIVS':
            weighting_IL = net_param_IL[layer.name] + extra_IL
            weighting_FL = weighting_bw - weighting_IL
            change_layer_bw(net_proto, layer.name, \
                            layer.quantization_param.bw_layer_in, \
                            layer.quantization_param.fl_layer_in, \
                            layer.quantization_param.bw_layer_out, \
                            layer.quantization_param.fl_layer_out, \
                            weighting_bw, weighting_FL, \
                            layer.quantization_param.bw_add, \
                            layer.quantization_param.fl_add, \
                            layer.quantization_param.bw_multiply, \
                            layer.quantization_param.fl_multiply, \
                            )

#quantize fully connected layer in network
def quantize_net_fc(net_proto, net_param_IL, weighting_bw, extra_IL):
    for layer in net_proto.layer:
        if layer.type == 'FcIVS':
            weighting_IL = net_param_IL[layer.name] + extra_IL
            weighting_FL = weighting_bw - weighting_IL
            change_layer_bw(net_proto, layer.name, \
                            layer.quantization_param.bw_layer_in, \
                            layer.quantization_param.fl_layer_in, \
                            layer.quantization_param.bw_layer_out, \
                            layer.quantization_param.fl_layer_out, \
                            weighting_bw, weighting_FL, \
                            layer.quantization_param.bw_add, \
                            layer.quantization_param.fl_add, \
                            layer.quantization_param.bw_multiply, \
                            layer.quantization_param.fl_multiply, \
                            )

#quantize deconvolution layer in network
def quantize_net_deconv(net_proto, net_param_IL, weighting_bw, extra_IL):
    for layer in net_proto.layer:
        if layer.type == 'DeconvolutionRistretto':
            weighting_IL = net_param_IL[layer.name] + extra_IL
            weighting_FL = weighting_bw - weighting_IL
            change_layer_bw(net_proto, layer.name, \
                            layer.quantization_param.bw_layer_in, \
                            layer.quantization_param.fl_layer_in, \
                            layer.quantization_param.bw_layer_out, \
                            layer.quantization_param.fl_layer_out, \
                            weighting_bw, weighting_FL, \
                            layer.quantization_param.bw_add, \
                            layer.quantization_param.fl_add, \
                            layer.quantization_param.bw_multiply, \
                            layer.quantization_param.fl_multiply, \
                            )

#read network spec in prototxt
def read_from_prototxt(ori_net_path):
    net_proto = caffe_pb2.NetParameter()
    fn = ori_net_path;
    with open(fn) as f:
        s = f.read()
        txtf.Merge(s, net_proto)
    return net_proto

#write network spec to prototxt
def write_to_prototxt(net_proto, out_net_path):
    outf = out_net_path
    #print 'writing', outf
    with open(outf, 'w') as f:
        f.write(str(net_proto))

#test network with no string printed
def test_qnet(net_path, caffemodel_path, imdb):
    net = caffe.Net(net_path, caffemodel_path, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel_path))[0]
    ap = test_net_silent(net, imdb, max_per_image=args.max_per_image, vis=args.vis)
    return ap

#print each layer name and spec
def print_net_layer_names(net): 
    print("Network layers:")
    
    for name, layer in zip(net._layer_names, net.layers):
        if layer.type  == 'ConvolutionIVS' or layer.type == 'Convolution':
            print("{:<30}: {:22s}({} blobs)".format(name, layer.type, len(layer.blobs)))
            print dir(layer)
            print layer.reshape
            print layer.convolution_param
    print net.layer[1].name

def mAP_worker(i, net_path, shared_dict, GPU_ID):
    #caffe.set_mode_cpu()
    #GPU_ID = 2 # Switch between 0 and 1 depending on the GPU you want to use.
    cfg.GPU_ID = GPU_ID
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    ap = test_qnet(net_path, args.caffemodel, imdb)
    shared_dict[i] = ap


def analyze_net_output_IL_worker(net_output_IL, net_input_IL, GPU_ID):
    cfg.GPU_ID = GPU_ID

    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()
    net_proto = read_from_prototxt(args.prototxt_quantized)
    net = caffe.Net(args.prototxt_quantized, args.caffemodel, caffe.TEST)
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    net_output_IL_, net_input_IL_ = analyze_net_output_IL(net, net_proto, imdb, max_per_image=args.max_per_image, vis=args.vis)
    for t in net_output_IL_.keys():
        net_output_IL[t] = net_output_IL_[t]
    for t in net_input_IL_.keys():
        net_input_IL[t] = net_input_IL_[t]

def analyze_net_adder_multiplier_IL_worker(net_adder_IL, net_multiplier_IL, GPU_ID):
    cfg.GPU_ID = GPU_ID

    #caffe.set_mode_cpu()
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()

    net_proto_BAC = read_from_prototxt(args.prototxt_quantized_BAC)
    net_BAC = caffe.Net(args.prototxt_quantized_BAC, args.caffemodel, caffe.TEST)
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    net_adder_IL_, net_multiplier_IL_ = analyze_net_adder_multiplier_IL(net_BAC, net_proto_BAC, imdb,
                                             max_per_image=args.max_per_image, vis=args.vis)

    for t in net_adder_IL_.keys():
        net_adder_IL[t] = net_adder_IL_[t]
    for t in net_multiplier_IL_.keys():
        net_multiplier_IL[t] = net_multiplier_IL_[t]
def analyze_net_param_IL_worker(net_param_IL, GPU_ID):
    cfg.GPU_ID = GPU_ID

    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    net_proto = read_from_prototxt(args.prototxt_quantized)
    net = caffe.Net(args.prototxt_quantized, args.caffemodel, caffe.TEST)
    net_param_IL_ = analyze_net_param_IL(net, net_proto)
    for t in net_param_IL_.keys():
        net_param_IL[t] = net_param_IL_[t]



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

    print 'Create quantized prototxt'
    print 'Testing Full Precision Accuracy'
    manager = multiprocessing.Manager()

    shared_dict = manager.dict()
    #p = multiprocessing.Process(target=mAP_worker, args=('FP-FP-FP-FP-FP', args.prototxt, shared_dict, 2))
    timer = Timer()
    timer.tic()
    #p.start()
    #p.join()
    timer.toc()
    print ('Took {:.3f}s').format(timer.total_time)
    #full_ap = shared_dict['FP-FP-FP-FP-FP']
    full_ap = 0.540425
    print 'Full precision accuracy : {}'.format(full_ap)
    
    # Bit Width for Analyze
    bw_range_conv = [8, 4] #bit width for convolution layers
    bw_range_deconv = [32, 16, 8, 4, 2] #bit width for deconvolution layers
    bw_range_fc = [32, 16, 8, 7, 6, 5, 4, 2] #bit width for fully connected layers
    bw_range_output = [32, 16, 8, 4, 2] #bit width for layer input and output
    bw_conv = 6 #just initial
    bw_deconv = 6 #just initial
    bw_fc = 6 #just initial
    bw_output = 6 #just initial
    bw_adder = 14 #just initial
    bw_multiplier = 10 #just initial
    convIL_reduction = 0
    deconvIL_reduction = 0
    fcIL_reduction = -1
    actIL_reduction = 0
    adderIL_reduction = 0
    multIL_reduction = 0
    GPU1 = 1
    GPU2 = 1
    
        
    #Make Final Quantized Prototxt
    print 'Final Quantization Testing'
    #net_proto = read_from_prototxt(args.prototxt_quantized)
    #write_to_prototxt(net_proto, './temp.prototxt')
    #p = multiprocessing.Process(target=mAP_worker, args=('DQ-DQ-DQ-32-32', './temp.prototxt',
    #                             shared_dict, GPU1))
    #p.start()
    #p.join()
    #ap = shared_dict['DQ-DQ-DQ-32-32']
    #layer_ap = ap
    ap = 0.544
    #ap = test_qnet('./temp.prototxt', args.caffemodel, imdb)
    print '----------------------------------------'
    print '{}bit CONV, {}bit FC, {}bit layer output'.format(bw_conv, bw_fc, bw_output)
    print 'Accuracy {}'.format(ap)
        
    print 'Dynamic fixed point net:'
    print '{}bit CONV and DECONV weights'.format(bw_conv)
    print '{}bit FC weights'.format(bw_fc)
    print '{}bit layer activations'.format(bw_output)
    print 'Please fine-tune'
    
    print 'Quantized Model saved to', args.prototxt_quantized
    


    #print 'Create Bit-Accurate quantized prototxt'
    #convert_net_to_qnet_BAC_analysis(args.prototxt_quantized, args.prototxt_quantized_BAC)
    #net_proto_BAC = read_from_prototxt(args.prototxt_quantized_BAC)
    #print 'Loading Bit-Accurate quantized prototxt'



    #print 'Analyzing network adder and multiplier'
    net_adder_IL = manager.dict()
    net_multiplier_IL = manager.dict()
    if args.accumulator_analysis == None:
        print 'Analyzing network adder and multiplier'
        p = multiprocessing.Process(target=analyze_net_adder_multiplier_IL_worker, 
                                args=(net_adder_IL, net_multiplier_IL, GPU1))
        p.start()
        p.join()
        with open('accumulator_analysis.json', 'w') as outfile:
            accumulator_analysis = dict()
            accumulator_analysis['net_adder_IL'] = dict()
            accumulator_analysis['net_multiplier_IL'] = dict()
            for t in net_adder_IL.keys():
                accumulator_analysis['net_adder_IL'][t] = net_adder_IL[t]
            for t in net_multiplier_IL.keys():
                accumulator_analysis['net_multiplier_IL'][t] = net_multiplier_IL[t]
            json.dump(accumulator_analysis, outfile)
    else:
        print 'Loading network adder and multiplier analysis file'
        with open(args.accumulator_analysis) as json_data:
            accumulator_analysis = json.load(json_data)
            for t in accumulator_analysis['net_adder_IL'].keys():
                net_adder_IL[t] = accumulator_analysis['net_adder_IL'][t]
            for t in accumulator_analysis['net_multiplier_IL'].keys():
                net_multiplier_IL[t] = accumulator_analysis['net_multiplier_IL'][t]



    print 'Create Final Bit-Accurate quantized prototxt'
    convert_net_to_qnet_BAC(args.prototxt_quantized, args.prototxt_quantized_BAC)
    net_proto_final = read_from_prototxt(args.prototxt_quantized_BAC)
    print 'Loading Final Bit-Accurate quantized prototxt'


    quantize_net_multiplier(net_proto_final, net_multiplier_IL, bw_multiplier, multIL_reduction)
    quantize_net_adder(net_proto_final, net_adder_IL, bw_adder, adderIL_reduction)
    write_to_prototxt(net_proto_final, args.prototxt_quantized_BAC)
    write_to_prototxt(net_proto_final, './temp_f.prototxt')
    print 'Testing Final Bit-Accurate quantized prototxt'
    p = multiprocessing.Process(target=mAP_worker, args=('DQ-DQ-DQ-DQ-DQ', './temp_f.prototxt',
                                 shared_dict,GPU1))
    p.start()
    p.join()
    ap = shared_dict['DQ-DQ-DQ-DQ-DQ']
    #ap = test_qnet('./temp_f.prototxt', args.caffemodel, imdb)
    print '----------------------------------------'
    print '{}bit adder, {}bit multiplier,'.format(bw_adder, bw_multiplier)
    print 'Accuracy {}'.format(ap)
        
    print 'Dynamic fixed point net:'
    print '{}bit CONV and DECONV weights'.format(bw_conv)
    print '{}bit FC weights'.format(bw_fc)
    print '{}bit layer activations'.format(bw_output)
    print '{}bit adder'.format(bw_adder)
    print '{}bit multiplier'.format(bw_multiplier)
    print 'Please fine-tune'
    
    write_to_prototxt(net_proto_final, args.prototxt_quantized_BAC)
    print 'Bit-Accurate Quantized Model saved to', args.prototxt_quantized_BAC


