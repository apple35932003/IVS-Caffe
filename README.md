# IVS Bit-Accurate Hardware Simulation Caffe.

IVS Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by IVSLab in National Chiao-Tung University.


IVS Caffe provide bit-accurate hardware simulation layers and other frequently used layers, including the following layers.
### Supported Bit Accurate Layers
* ConvolutionIVS - bit-accurate convolution layer  
* FcIVS - bit-accurate fully-connected layer  
* ConvolutionXNORIVS - bit-accurate XNOR convolution layer  
    * support input 1 , 0 , -1( 0 treat as -1 )
### Supported IVS defined Layers
* BinarizeIVS
  * binarize input to 0 and 1 at custom thresh and condition
### Supported Normal Layers
* [All BVLC/caffe supported layer (CAFFE)](https://github.com/BVLC/caffe)  
* [Al sanghoon/pva-faster-rcnn supported layer (PVANET)](https://github.com/sanghoon/pva-faster-rcnn/blob/master/README.md)  
* [All weiliu89/caffe supprted layer (SSD)](https://github.com/weiliu89/caffe/tree/ssd)  
* [All rbgirshick/py-faster-rcnn supported layer (Faster-RCNN)](https://github.com/rbgirshick/py-faster-rcnn)  
* [All hotzeng/Ristretto-caffe supported layer (RISTRETTO-CAFFE)](https://github.com/hotzeng/Ristretto-caffe)  
* Self-defined pyhon layer  
### Supported Detector
* YOLOv1~v3
* SSD
* Faster-RCNN



### Bit Accurate Layers Definition in prototxt  
every IVS layer can define the quantization_param in their corresponding layer, including the following parameters
1. bw_layer_in - bit-length of input 
    * number from 0 ~ [max of uint32]
    * default: 32
2. bw_layer_out - bit-length of output 
    * number from 0 ~ [max of uint32]
    * default: 32
3. bw_params - bit-length of weights
    * number from 0 ~ [max of uint32]
    * default: 32
4. bw_add - bit-length of adder
    * number from 0 ~ [max of uint32]
    * default: 32
5. bw_multiply - bit-length of multiplier
    * number from [min of int32] ~ [max of int32]
    * default: 16
6. fl_layer_in - floating length of input
    * number from [min of int32] ~ [max of int32]
    * default: 16
7. fl_layer_out - floating length of output
    * number from [min of int32] ~ [max of int32]
    * default: 16
8. fl_params - floating length of weights
    * number from [min of int32] ~ [max of int32]
    * default: 16
9. fl_add - floating length of adder
    * number from [min of int32] ~ [max of int32]
    * default: 16
10. fl_multiply - floating length of multiplier
    * number from [min of int32] ~ [max of int32]
    * default: 16
11. rounding_time - when to rounding 
    * LAYER_BY_LAYER 
    * EVERY_OPERATION - folding addition (faster)
    * EVERY_OPERATION_SERIAL - ripple adder
    * default: EVERY_OPERATION
12. overflow_behavior - behavior when overflow in adder and multiplier
    * OVERFLOW_SIM - overflow behavior
    * TRIM_AT_THRESH - trim overflowed nubmer at thresh (thresh defined in adder or multiplier bw and fl)
    * default: TRIM_AT_THRESH
13. analyze_mode - using to analyze bit-length of adder and multipler 
    * NO_ANALYZE - fastest
    * ANALYZE_ADD - record and print max and min value in adder 
    * ANALYZE_MULTIPLY - record and print max and min value in multiplier 
    * ANALYZE_BOTH - record and print max and min value in both adder and multiplier
    * default: NO_ANALYZE
#### Example of ConvolutionIVS Layer
```
layer {
  name: "conv1"
  type: "ConvolutionIVS"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
  quantization_param {
    bw_layer_in: 32
    bw_layer_out: 32
    bw_params: 32
    bw_add: 32
    bw_multiply: 32
    fl_layer_in: 16
    fl_layer_out: 16
    fl_params: 16
    fl_add: 16
    fl_multiply: 16
    rounding_time: LAYER_BY_LAYER
    overflow_behavior: TRIM_AT_THRESH
    analyze_mode: ANALYZE_BOTH
  }
}
```
#### Example of FcIVS Layer
```
layer {
  name: "ip2"
  type: "FcIVS"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
  quantization_param {
    bw_layer_in: 32
    bw_layer_out: 32
    bw_params: 32
    bw_add: 32
    bw_multiply: 32
    fl_layer_in: 16
    fl_layer_out: 16
    fl_params: 16
    fl_add: 16
    fl_multiply: 16
    rounding_time: LAYER_BY_LAYER
    overflow_behavior: TRIM_AT_THRESH
    analyze_mode: NO_ANALYZE
  }
}
```
#### Example of ConvolutionXNORIVS Layer
```
layer {
  name: "ip2"
  type: "ConvolutionXNORIVS"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
  quantization_param {
    bw_layer_in: 32
    bw_layer_out: 32
    bw_params: 32
    bw_add: 32
    bw_multiply: 32
    fl_layer_in: 16
    fl_layer_out: 16
    fl_params: 16
    fl_add: 16
    fl_multiply: 16
    rounding_time: LAYER_BY_LAYER
    overflow_behavior: TRIM_AT_THRESH
    analyze_mode: NO_ANALYZE
  }
}
```
### Example of training a Bit-accurate Quantized network
* lenet_IVS
```Shell
cd IVS-Caffe/caffe-fast-rcnn
./examples/mnist/train_lenet_IVS.sh
```



### BinarizeIVS Layer definition in prototxt  
every IVS layer can define the quantization_param in their corresponding layer, including the following parameters
1. thresh - threshold for binarization
    * number from [min of float] ~ [max of float]
    * default: 0
2. thresh_condition - threshold include in uppder bound or lower bound 
    * INCLUDE_IN_UPPER - y = x >= thresh ? 1 : 0
    * INCLUDE_IN_LOWER - y = x > thresh ? 1 : 0
    * default - INCLUDE_IN_UPPER
3. inverse_binarize - inverse binarize output 0 and 1
    * FALSE
    * TRUE
    * default - FALSE

#### Example of BinarizeIVS Layer
```
layer {
  name: "binarize1"
  type: "BinarizeIVS"
  bottom: "conv1"
  top: "conv1"
  binarize_param {
    thresh: 0
    thresh_condition: INCLUDE_IN_UPPER
    inverse_binarize: FALSE
  }
}
```



#### Note
* Please use make to compile cafe-fast-rcnn module, cmake is currently unsupported
* GPU version is currently unsupported in this version, will be release in the future
