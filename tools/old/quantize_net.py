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
            l.type = 'ConvolutionRistretto'
            l.quantization_param.precision =  0 #DYNAMIC_FIXED_POINT
            l.quantization_param.bw_layer_in = 32
            l.quantization_param.bw_layer_out = 32
            l.quantization_param.bw_params = 32
            l.quantization_param.fl_layer_in = 16
            l.quantization_param.fl_layer_out= 16
            l.quantization_param.fl_params = 16
        elif l.type =='InnerProduct':
            l.type = 'FcRistretto'
            l.quantization_param.precision =  0 #DYNAMIC_FIXED_POINT
            l.quantization_param.bw_layer_in = 32
            l.quantization_param.bw_layer_out = 32
            l.quantization_param.bw_params = 32
            l.quantization_param.fl_layer_in = 16
            l.quantization_param.fl_layer_out= 16
            l.quantization_param.fl_params = 16
        elif l.type =='Deconvolution':
            l.type = 'DeconvolutionRistretto'
            l.quantization_param.precision =  0 #DYNAMIC_FIXED_POINT
            l.quantization_param.bw_layer_in = 32
            l.quantization_param.bw_layer_out = 32
            l.quantization_param.bw_params = 32
            l.quantization_param.fl_layer_in = 16
            l.quantization_param.fl_layer_out= 16
            l.quantization_param.fl_params = 16
        
    write_to_prototxt(net_proto, q_net_path)

#change single layer bit width
def change_layer_bw(net_proto, layer_name, 
                    bw_layer_in, fl_layer_in, 
                    bw_layer_out, fl_layer_out, 
                    bw_params, fl_params):
    for l in net_proto.layer:
        if l.name == layer_name:
            l.quantization_param.precision =  0
            l.quantization_param.bw_layer_in = int(bw_layer_in)
            l.quantization_param.bw_layer_out = int(bw_layer_out)
            l.quantization_param.bw_params = int(bw_params)
            l.quantization_param.fl_layer_in = int(fl_layer_in)
            l.quantization_param.fl_layer_out= int(fl_layer_out)
            l.quantization_param.fl_params = int(fl_params)
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
        if layer.type == 'ConvolutionRistretto' \
            or layer.type == 'FcRistretto' \
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
        assert layer.top[0] != layer.bottom[0],"bottom name cannot be the same as top name in the same layer, at layer:{} top:{} bottom:{}".format(layer.name,layer.top[0],layer.bottom[0])
        #if layer.top[0] == layer.bottom[0]:
        #    print layer.name, layer.type
        if layer.type == 'ConvolutionRistretto' \
            or layer.type == 'FcRistretto' \
            or layer.type == 'DeconvolutionRistretto':
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
            if layer.type == 'ConvolutionRistretto' \
                or layer.type == 'FcRistretto' \
                or layer.type == 'DeconvolutionRistretto':
                net_output_IL[layer.name] = max(calc_layer_inout_IL(net, layer.top[0]), net_output_IL[layer.name])
                net_input_IL[layer.name] = max(calc_layer_inout_IL(net, layer.bottom[0]), net_input_IL[layer.name])
                #print layer.type, layer.name, net_output_IL[layer.name],net_input_IL[layer.name]
    return net_output_IL, net_input_IL
    
        
#quantize input and output of each layer in network
def quantize_net_output(net, net_proto, net_output_IL, net_input_IL, output_bw):
    input_bw = output_bw;
    #input_FL = 0;
    for layer in net_proto.layer:
        if layer.type == 'ConvolutionRistretto' \
             or layer.type == 'FcRistretto' \
                or layer.type == 'DeconvolutionRistretto':
            output_IL = net_output_IL[layer.name]  
            output_FL = output_bw - output_IL
            input_IL = net_input_IL[layer.name] #-1
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
                            layer.quantization_param.fl_params)
            #input_FL = output_FL


#quantize convolution layers in network
def quantize_net_conv(net, net_proto, net_param_IL, weighting_bw):
    for layer in net_proto.layer:
        if layer.type == 'ConvolutionRistretto':
            weighting_IL = net_param_IL[layer.name] - 1
            weighting_FL = weighting_bw - weighting_IL
            change_layer_bw(net_proto, layer.name, \
                            layer.quantization_param.bw_layer_in, \
                            layer.quantization_param.fl_layer_in, \
                            layer.quantization_param.bw_layer_out, \
                            layer.quantization_param.fl_layer_out, \
                            weighting_bw, weighting_FL)

#quantize fully connected layer in network
def quantize_net_fc(net, net_proto, net_param_IL, weighting_bw):
    for layer in net_proto.layer:
        if layer.type == 'FcRistretto':
            weighting_IL = net_param_IL[layer.name] - 1
            weighting_FL = weighting_bw - weighting_IL
            change_layer_bw(net_proto, layer.name, \
                            layer.quantization_param.bw_layer_in, \
                            layer.quantization_param.fl_layer_in, \
                            layer.quantization_param.bw_layer_out, \
                            layer.quantization_param.fl_layer_out, \
                            weighting_bw, weighting_FL)

#quantize deconvolution layer in network
def quantize_net_deconv(net, net_proto, net_param_IL, weighting_bw):
    for layer in net_proto.layer:
        if layer.type == 'DeconvolutionRistretto':
            weighting_IL = net_param_IL[layer.name] - 1
            weighting_FL = weighting_bw - weighting_IL
            change_layer_bw(net_proto, layer.name, \
                            layer.quantization_param.bw_layer_in, \
                            layer.quantization_param.fl_layer_in, \
                            layer.quantization_param.bw_layer_out, \
                            layer.quantization_param.fl_layer_out, \
                            weighting_bw, weighting_FL)

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
        if layer.type  == 'ConvolutionRistretto' or layer.type == 'Convolution':
            print("{:<30}: {:22s}({} blobs)".format(name, layer.type, len(layer.blobs)))
            print dir(layer)
            print layer.reshape
            print layer.convolution_param
    print net.layer[1].name

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

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    
    #net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    convert_net_to_qnet(args.prototxt, args.prototxt_quantized)
    print 'Create quantized prototxt'
    net = caffe.Net(args.prototxt_quantized, args.caffemodel, caffe.TEST)
    print 'Testing Full Precision Accuracy'
    full_ap = test_qnet(args.prototxt, args.caffemodel, imdb)
    #full_ap = 0.540425
    print 'Full precision accuracy : {}'.format(full_ap)
    
    # Bit Width for Analyze
    bw_range_conv = [8, 4] #bit width for convolution layers
    bw_range_deconv = [32, 16, 8, 4, 2] #bit width for deconvolution layers
    bw_range_fc = [32, 16, 8, 7, 6, 5, 4, 2] #bit width for fully connected layers
    bw_range_output = [32, 16, 8, 4, 2] #bit width for layer input and output
    bw_conv = 0 #just initial
    bw_deconv = 0 #just initial
    bw_fc = 0 #just initial
    bw_output = 0 #just initial
    
    print 'Analyzing network'
    net_proto = read_from_prototxt(args.prototxt)
    has_conv, has_deconv, has_fc = analyze_network(net_proto)
    print 'Network Structure'
    print 'CONV:{}, DECONV:{}, FC:{}'.format(has_conv, has_deconv, has_fc)    
    
    print '-----------------------------------'
    net_proto = read_from_prototxt(args.prototxt_quantized)
    print 'Analyzing network parameter IL'
    net_param_IL = analyze_net_param_IL(net, net_proto)
    print 'Analyzing network output IL'
    net_output_IL, net_input_IL = analyze_net_output_IL(net, net_proto, imdb, max_per_image=args.max_per_image, vis=args.vis)
 
    #print 'Analyzing layer output'
    #print '\tbit width\t accuracy'
    #for bw in bw_range_output:
    #    net_proto = read_from_prototxt(args.prototxt_quantized)
    #    quantize_net_output(net, net_proto, net_output_IL, net_input_IL, bw)
    #    write_to_prototxt(net_proto, './temp.prototxt')
    #    ap = test_qnet('./temp.prototxt', args.caffemodel, imdb)
    #    print '\t{}bit:\t\t{}'.format(bw,ap)
    #    if ap < (full_ap - args.error_margin):
    #        break;
    #    bw_output = bw
    
    # Analyze Convolution and DeConvolution Layers
    if has_conv:
        print 'Analyzing CONV and DECONV'
        print '\tbit width\t accuracy'
        bw_h = 32
        bw_l = 0
        bw = 32
        while True:
            net_proto = read_from_prototxt(args.prototxt_quantized)
            quantize_net_conv(net, net_proto, net_param_IL,  bw)
            quantize_net_deconv(net, net_proto, net_param_IL, bw)
            write_to_prototxt(net_proto, './temp.prototxt')
            ap = test_qnet('./temp.prototxt', args.caffemodel, imdb)
            print '\t{}bit:\t\t{}'.format(bw,ap)
            if ap < (full_ap - args.error_margin):
                bw_l = bw
            else:
                bw_h = bw
                bw_conv = bw
            if bw_h - bw_l <= 1:
                break
            bw = bw_l + (bw_h-bw_l)/2


    #if has_conv:
    #    print 'Analyzing CONV and DECONV'
    #    print '\tbit width\t accuracy'
    #    for bw in bw_range_conv:
    #        net_proto = read_from_prototxt(args.prototxt_quantized)
    #        quantize_net_conv(net, net_proto, net_param_IL,  bw)
    #        quantize_net_deconv(net, net_proto, net_param_IL, bw)
    #        write_to_prototxt(net_proto, './temp.prototxt')
    #        ap = test_qnet('./temp.prototxt', args.caffemodel, imdb)
    #        print '\t{}bit:\t\t{}'.format(bw,ap)
    #        if ap < (full_ap - args.error_margin):
    #            break;
    #        bw_conv = bw

    # Analyze Fully Connected Layers

    if has_fc:
        print 'Analyzing FC'
        print '\tbit width\t accuracy'
        bw_h = 32
        bw_l = 0
        bw = 32
        while True:
            net_proto = read_from_prototxt(args.prototxt_quantized)
            quantize_net_fc(net, net_proto, net_param_IL, bw)
            write_to_prototxt(net_proto, './temp.prototxt')
            ap = test_qnet('./temp.prototxt', args.caffemodel, imdb)
            print '\t{}bit:\t\t{}'.format(bw,ap)
            if ap < (full_ap - args.error_margin):
                bw_l = bw
            else:
                bw_h = bw
                bw_fc = bw
            if bw_h - bw_l <=1:
                break
            bw = bw_l + (bw_h-bw_l)/2

    # Analyze input and output of layers
    bw_h = 32
    bw_l = 0
    bw = 32
    print 'Analyzing layer output'
    print '\tbit width\t accuracy'
    while True:
        net_proto = read_from_prototxt(args.prototxt_quantized)
        quantize_net_output(net, net_proto, net_output_IL, net_input_IL, bw)
        write_to_prototxt(net_proto, './temp.prototxt')
        ap = test_qnet('./temp.prototxt', args.caffemodel, imdb)
        print '\t{}bit:\t\t{}'.format(bw,ap)
        if ap < (full_ap - args.error_margin):
            bw_l = bw
        else:
            bw_h = bw
            bw_output = bw
        if bw_h - bw_l <=1:
            break
        bw = bw_l + (bw_h-bw_l)/2

    #Make Final Quantized Prototxt
    print 'Final Quantization Testing'
    net_proto = read_from_prototxt(args.prototxt_quantized)
    quantize_net_conv(net, net_proto, net_param_IL, bw_conv)
    quantize_net_deconv(net, net_proto, net_param_IL, bw_conv)
    quantize_net_fc(net, net_proto, net_param_IL, bw_fc)
    quantize_net_output(net, net_proto, net_output_IL, net_input_IL, bw_output)
    write_to_prototxt(net_proto, './temp.prototxt')
    ap = test_qnet('./temp.prototxt', args.caffemodel, imdb)
    print '----------------------------------------'
    print '{}bit CONV, {}bit FC, {}bit layer output'.format(bw_conv, bw_fc, bw_output)
    print 'Accuracy {}'.format(ap)
        
    print 'Dynamic fixed point net:'
    print '{}bit CONV and DECONV weights'.format(bw_conv)
    print '{}bit FC weights'.format(bw_fc)
    print '{}bit layer activations'.format(bw_output)
    print 'Please fine-tune'
    
    write_to_prototxt(net_proto, args.prototxt_quantized)
    print 'Quantized Model saved to', args.prototxt_quantized
    










