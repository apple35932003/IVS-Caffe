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
from subprocess import check_output


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
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
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
    percentile = 0.
    layer_param = net.params[layer.name] 
    #max_weight = max(layer_param[0].data[...].max(), layer_param[0].data[...].min(), key=abs)
    weight_sorted = np.sort(layer_param[0].data[...], axis=None)
    max_weight = max(weight_sorted[int(len(weight_sorted)*percentile)], weight_sorted[-1*int(len(weight_sorted)*percentile)],key=abs)
    if layer.convolution_param.bias_term:
        bias_sorted = np.sort(layer_param[1].data[...], axis=None)
        max_bias = max(bias_sorted[int(len(bias_sorted)*percentile)], bias_sorted[-1*int(len(bias_sorted)*percentile)],key=abs)
        #max_bias = max(layer_param[1].data[...].max(), layer_param[1].data[...].min(), key=abs)
    else:
        max_bias = 0
    #print layer.name, max_weight, max(weight_sorted[0],weight_sorted[-1],key=abs),  max(weight_sorted[int(len(weight_sorted)/100)], weight_sorted[-1*int(len(weight_sorted)/100)],key=abs)
    print max_weight,max_bias
    max_param = max_weight
    max_bias = max_bias
    return math.ceil(math.log(abs(max_param), 2)) + 1, math.ceil(math.log(abs(max_bias), 2)) + 1


def analyze_net_param(net, net_proto):
    net_param_IL = dict()
    net_bias_IL = dict()
    for layer in net_proto.layer:
        if layer.type == 'ConvolutionIVS' \
            or layer.type == 'FcIVS' \
            or layer.type == 'DeconvolutionRistretto':
            net_param_IL[layer.name], net_bias_IL[layer.name] = calc_layer_param_IL(net, layer)
    return net_param_IL, net_bias_IL


#calculate needed Integer Length of layer output
def calc_layer_inout_IL(net, layer_bottom_name):
    layer_output = net.blobs[layer_bottom_name].data
    layer_output_max = abs(max(layer_output.max(), layer_output.min(), key=abs))
    #if layer_bottom_name == 'data':
    #    print net.blobs[layer_bottom_name].data 
    #    print math.ceil(math.log(layer_output_max, 2)) + 1
    return math.ceil(math.log(layer_output_max, 2)) + 1


def analyze_net_output(net, net_proto):
    #num_images = len(imdb.image_index)
    #_t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    #if not cfg.TEST.HAS_RPN:
    #    roidb = imdb.roidb
    net_output_IL = dict()
    net_input_IL = dict()
    #for layer in net_proto.layer:
    #    if layer.type == 'Convolution' \
    #        or layer.type == 'DeconvolutionRistretto':
    #        assert layer.top[0] != layer.bottom[0],"bottom name cannot be the same as top name in the same layer, at layer:{} top:{} bottom:{}".format(layer.name,layer.top[0],layer.bottom[0])
    #        net_output_IL[layer.name] = -sys.maxint - 1
    #        net_input_IL[layer.name] = -sys.maxint - 1
    output_sum1 = np.zeros(net.blobs['layer14-conv'].data.shape)
    output_sum2 = np.zeros(net.blobs['layer13-conv'].data.shape)
    
    for i in xrange(cal_iters):
        #if cfg.TEST.HAS_RPN:
        #    box_proposals = None
        #else:
        #    box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        #im = cv2.imread(imdb.image_path_at(i))
        #scores, boxes = im_detect(net, im, _t, box_proposals)
        net.forward()
        output_sum1 += net.blobs['layer14-conv'].data
        output_sum2 += net.blobs['layer13-conv'].data
        #for layer in net_proto.layer:
        #    if layer.type == 'ConvolutionIVS' \
        #        or layer.type == 'FcIVS' \
        #        or layer.type == 'DeconvolutionRistretto':
        #        net_output_IL[layer.name] = max(calc_layer_inout_IL(net, layer.top[0]), net_output_IL[layer.name])
        #        net_input_IL[layer.name] = max(calc_layer_inout_IL(net, layer.bottom[0]), net_input_IL[layer.name])
        #        #print layer.type, layer.name, net_output_IL[layer.name],net_input_IL[layer.name]
    #for i in range(output_sum.shape[1]):
    output_ave1 = np.average(output_sum1,axis=(2,3))
    output_std1 = np.std(output_sum1,axis=(2,3))
    min_ = output_std1.min()
    max_ = output_std1.max()
    step_ = (max_-min_)/10
    std = []
    for i in range(10):
        std.append(0)
        _min_ = min_ + i * step_
        _max_ = min_ + (i+1) * step_
        for j in range(1024):
            if (_min_ <= output_std1[0][j]) and (output_std1[0][j] < _max_):
                std[i] += 1
        print _min_, _max_, std[i]
    std[9] += 1
    for i in range(10):
        print std[i]

    output_ave2 = np.average(output_sum2,axis=(2,3))
    output_std2 = np.std(output_sum2,axis=(2,3))
    min_ = output_std2.min()
    max_ = output_std2.max()
    step_ = (max_-min_)/10
    std = []
    for i in range(10):
        std.append(0)
        _min_ = min_ + i * step_
        _max_ = min_ + (i+1) * step_
        for j in range(512):
            if (_min_ <= output_std2[0][j]) and (output_std2[0][j] < _max_):
                std[i] += 1
        print _min_, _max_, std[i]
    std[9] += 1
    for i in range(10):
        print std[i]

    for p in range(10):
        for i in range(102):
            #print 'min=' +  str(output_ave.min()) + ', location=' + str(output_ave.argmin()) + ', stddev=' + str(output_std[0][output_ave.argmin()])
            #print 'min=' +  str(output_std.min()) + ', location=' + str(output_std.argmin()) + ', ave=' + str(output_ave[0][output_std.argmin()])
            net.params['layer14-conv_'][0].data[output_ave1.argmin()][...] = np.zeros(net.params['layer14-conv_'][0].data[output_ave1.argmin()].shape)
            output_ave1[0][output_ave1.argmin()] = sys.maxint
        net.save('./temp/temp11-' + str(p) + '.caffemodel')

    for p in range(10):
        break
        for i in range(51):
            #print 'min=' +  str(output_ave.min()) + ', location=' + str(output_ave.argmin()) + ', stddev=' + str(output_std[0][output_ave.argmin()])
            #print 'min=' +  str(output_std.min()) + ', location=' + str(output_std.argmin()) + ', ave=' + str(output_ave[0][output_std.argmin()])
            net.params['layer13-conv_'][0].data[output_std2.argmin()][...] = np.zeros(net.params['layer13-conv_'][0].data[output_std2.argmin()].shape)
            output_std2[0][output_std2.argmin()] = sys.maxint
        net.save('./temp/temp11-' + str(p) + '.caffemodel')
        

    net.save('./temp.caffemodel')  
    return net_output_IL, net_input_IL
    
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

def mAP_worker(i, def_path, net_path, shared_dict, GPU_ID):
    #caffe.set_mode_cpu()
    #GPU_ID = 2 # Switch between 0 and 1 depending on the GPU you want to use.
    #cfg.GPU_ID = GPU_ID
    #caffe.set_device(GPU_ID)
    #caffe.set_mode_gpu()

    #imdb = get_imdb(args.imdb_name)
    #imdb.competition_mode(args.comp_mode)
    #if not cfg.TEST.HAS_RPN:
    #    imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    ap_string = check_output('./caffe-fast-rcnn/build/tools/caffe test_detection --model=' + def_path + ' --weights=' + net_path + ' -iterations=' + str(test_iters) + ' -gpu='+str(GPU_ID),shell=True)
    #ap_string = check_output('./caffe-fast-rcnn-c3d/caffe-fast-rcnn-2/build/tools/caffe test_detection --model=' + net_path + ' --weights=' + args.caffemodel + ' -iterations=' + str(30) + ' -gpu='+str(GPU_ID),shell=True)
    ap = 0.
    if len(ap_string) != 0:
        ap = float(ap_string)
    #ap = test_qnet(net_path, args.caffemodel, imdb)

    #ap = test_qnet(net_path, args.caffemodel, imdb)
    shared_dict[i] = ap


def analyze_net_output_worker(net_output, net_input, GPU_ID):
    cfg.GPU_ID = GPU_ID

    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()
    net_proto = read_from_prototxt(args.prototxt)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TRAIN)
    #imdb = get_imdb(args.imdb_name)
    #imdb.competition_mode(args.comp_mode)
    #if not cfg.TEST.HAS_RPN:
    #    imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    net_output_, net_input_ = analyze_net_output(net, net_proto)
    for t in net_output_.keys():
        net_output[t] = net_output_[t]
    for t in net_input_.keys():
        net_input[t] = net_input_[t]

def analyze_net_param_worker(net_param_IL, net_bias_IL, GPU_ID):
    cfg.GPU_ID = GPU_ID

    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    net_proto = read_from_prototxt(args.prototxt_quantized)
    net = caffe.Net(args.prototxt_quantized, args.caffemodel, caffe.TEST)
    net_param_IL_, net_bias_IL_ = analyze_net_param(net, net_proto)
    for t in net_param_IL_.keys():
        net_param_IL[t] = net_param_IL_[t]
        net_bias_IL[t] = net_bias_IL_[t]



if __name__ == '__main__':
    args = parse_args()
    test_iters = 4952
    cal_iters = 1000
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    print 'Testing Full Precision Accuracy'
    manager = multiprocessing.Manager()

    mAP_dict = manager.dict()
    GPU1 = args.gpu_id
    GPU2 = args.gpu_id
    p = multiprocessing.Process(target=mAP_worker, args=('Full', args.prototxt, args.caffemodel, mAP_dict, GPU1))
    timer = Timer()
    timer.tic()
    p.start()
    p.join()
    timer.toc()
    print ('Took {:.3f}s').format(timer.total_time)
    full_ap = mAP_dict['Full']
    #full_ap = 0.575724
    print 'Full accuracy : {}'.format(full_ap)
    
    print 'Analyzing network'
    net_proto = read_from_prototxt(args.prototxt)
    has_conv, has_deconv, has_fc = analyze_network(net_proto)
    print 'Network Structure'
    print 'CONV:{}, DECONV:{}, FC:{}'.format(has_conv, has_deconv, has_fc)    
   
    print '-----------------------------------'
    #net_proto = read_from_prototxt(args.prototxt)
    #print 'Analyzing network parameter IL'
    #net_param = manager.dict()
    #net_bias = manager.dict()
    #p = multiprocessing.Process(target=analyze_net_param_worker,
    #                            args=(net_param, net_bias, GPU1, ))
    #p.start()
    #p.join()
    net_output = manager.dict()
    net_input = manager.dict()
    print 'Analyzing network output'
    p = multiprocessing.Process(target=analyze_net_output_worker, 
                                args=(net_output, net_input, GPU1))
    p.start()
    p.join()
    with open('act_analysis.json', 'w') as outfile:
        act_analysis = dict()
        act_analysis['net_output'] = dict()
        act_analysis['net_input'] = dict()
        for t in net_output.keys():
            act_analysis['net_output'][t] = net_output[t]
        for t in net_input.keys():
            act_analysis['net_input'][t] = net_input[t]
        json.dump(act_analysis, outfile)
    #Make Final Quantized Prototxt
    print 'Final Pruning Testing'
    for i in range(10):
        p = multiprocessing.Process(target=mAP_worker, args=('Pruned'+str(i), args.prototxt,
                                    './temp/temp11-' + str(i) + '.caffemodel',
                                    mAP_dict, GPU1))
        p.start()
        p.join()
        ap = mAP_dict['Pruned'+str(i)]
        pruned_ap = ap
        print 'Pruned {}% Accuracy {}'.format((i+1)*10,ap)
