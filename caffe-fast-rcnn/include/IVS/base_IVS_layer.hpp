#ifndef CAFFE_BASE_IVS_LAYER_HPP_
#define CAFFE_BASE_IVS_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides quantization methods used by other IVS layers.
 */
template <typename Dtype>
class BaseIVSLayer{
 public:
  explicit BaseIVSLayer();
 protected:
  void QuantizeLayerOutputs_cpu(Dtype* data, const int count);
  void QuantizeLayerInputs_cpu(Dtype* data, const int count);
  void QuantizeLayerOutputs_gpu(Dtype* data, const int count);
  void QuantizeIVS_conv_inter_add_cpu(Dtype* data, const int count);
  void QuantizeIVS_conv_inter_add_gpu(Dtype* data, const int count);
  void QuantizeIVS_conv_inter_multiply_cpu(Dtype* data, const int count);
  void QuantizeIVS_conv_inter_multiply_gpu(Dtype* data, const int count);
  void QuantizeLayerInputs_gpu(Dtype* data, const int count);
  void QuantizeWeights_cpu(vector<shared_ptr<Blob<Dtype> > > weights_quantized,
      const int rounding, const bool bias_term = true);
  void QuantizeWeights_gpu(vector<shared_ptr<Blob<Dtype> > > weights_quantized,
      const int rounding, const bool bias_term = true);
  void Analyze_add(const Dtype* data, const int count);
  void Analyze_multiply(const Dtype* data, const int count);
  /**
   * @brief Trim data to fixed point.
   * @param fl The number of bits in the fractional part.
   */
  void Trim2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  void Trim2FixedPoint_overflow_cpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  void Trim2FixedPoint_gpu(Dtype* data, const int cnt, const int bit_width,
      const int rounding, int fl);
  /**
   * @brief Trim data to minifloat.
   * @param bw_mant The number of bits used to represent the mantissa.
   * @param bw_exp The number of bits used to represent the exponent.
   */
  void Trim2MiniFloat_cpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  void Trim2MiniFloat_gpu(Dtype* data, const int cnt, const int bw_mant,
      const int bw_exp, const int rounding);
  /**
   * @brief Trim data to integer-power-of-two numbers.
   * @param min_exp The smallest quantized value is 2^min_exp.
   * @param min_exp The largest quantized value is 2^max_exp.
   */
  void Trim2IntegerPowerOf2_cpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  void Trim2IntegerPowerOf2_gpu(Dtype* data, const int cnt, const int min_exp,
      const int max_exp, const int rounding);
  /**
   * @brief Generate random number in [0,1) range.
   */
  inline double RandUniform_cpu();

  // The number of bits used for dynamic fixed point parameters and layer
  // activations.
  int bw_params_, bw_layer_in_, bw_layer_out_, bw_add_, bw_multiply_, bw_bias_;
  // The fractional length of dynamic fixed point numbers.
  int fl_params_, fl_layer_in_, fl_layer_out_, fl_add_, fl_multiply_, fl_bias_;
  // The number of bits used to represent mantissa and exponent of minifloat
  // numbers.
  int fp_mant_, fp_exp_;
  // Integer-power-of-two numbers are in range +/- [2^min_exp, 2^max_exp].
  int pow_2_min_exp_, pow_2_max_exp_;
  // The rounding mode for quantization and the quantization scheme.
  int rounding_, precision_;
  // When to rounding 
  int rounding_time_;
  // Overflow behaviro when rounding 
  int overflow_behavior_;
  // Analyze mode, used to analyze adder and multiplier
  int analyze_mode_;
  double max_multiply_, min_multiply_;
  double max_add_, min_add_;
  int IVS_q_stat_;
  int ivs_ste_param_;
  int ivs_if_deploy_;
  int train_fl_param_;
  int train_fl_io_;
  double ns_ratio_param_;
  double ns_ratio_act_;
  double max_out_;
  double max_in_;
  // For parameter layers: reduced word with parameters.
  vector<shared_ptr<Blob<Dtype> > > weights_quantized_;
  //For bit accurate element-wisr multiplication and row-wise addidtion temporary storage.
};

/**
 * @brief Convolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class ConvolutionIVSLayer : public ConvolutionLayer<Dtype>,
      public BaseIVSLayer<Dtype> {
 public:
  explicit ConvolutionIVSLayer(const LayerParameter& param);
  virtual inline const char* type() const { return "ConvolutionIVS"; }

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void forward_cpu_IVS(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
#ifndef CPU_ONLY
//  virtual void forward_gpu_IVS(const Dtype* input, const Dtype* weights,
//      Dtype* output, Dtype* IVS_top_q_stat_data, bool skip_im2col = false);
//  virtual void forward_gpu_IVS_DLA(const Dtype* input, const Dtype* weights,
//      Dtype* output, Dtype* IVS_top_q_stat_data, bool skip_im2col = false);
#endif
  virtual void forward_cpu_IVS_serial(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  virtual void forward_cpu_IVS_serial_mult(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  virtual void forward_cpu_IVS_bias(Dtype* output, const Dtype* bias);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void im2col_cpu_IVS(const Dtype* data_im, const int channels,
  //    const int height, const int width, const int kernel_h, const int kernel_w,
  //    const int pad_h, const int pad_w, const int stride_h,
  //    const int stride_w, const int dilation_h, const int dilation_w,
  //    Dtype* data_col);
  //virtual void col2im_nd_cpu_IVS(const Dtype* data_col, const int num_spatial_axes,
  //    const int* im_shape, const int* col_shape,
  //    const int* kernel_shape, const int* pad, const int* stride,
  //    const int* dilation, Dtype* data_im);


  inline void conv_im2col_cpu_IVS(const Dtype* data, Dtype* col_buff) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      im2col_cpu_IVS(data, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1], this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0], this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0], this->dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu_IVS(data, this->num_spatial_axes_, this->conv_input_shape_.cpu_data(),
          this->col_buffer_shape_.data(), this->kernel_shape_.cpu_data(),
          this->pad_.cpu_data(), this->stride_.cpu_data(), this->dilation_.cpu_data(), col_buff);
    }
  }
  vector<int> IVS_conv_inter_shape_;
#ifndef CPU_ONLY
  inline void conv_im2col_gpu_IVS(const Dtype* data, Dtype* col_buff) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      im2col_gpu(data, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1], this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0], this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0], this->dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, this->num_spatial_axes_, this->num_kernels_im2col_,
          this->conv_input_shape_.gpu_data(), this->col_buffer_.gpu_shape(),
          this->kernel_shape_.gpu_data(), this->pad_.gpu_data(),
          this->stride_.gpu_data(), this->dilation_.gpu_data(), col_buff);
    }
  }
  Blob<Dtype> IVS_analyze_add_max_inter_;
  Blob<Dtype> IVS_analyze_add_min_inter_;
  Blob<Dtype> IVS_analyze_mul_max_inter_;
  Blob<Dtype> IVS_analyze_mul_min_inter_;
  int analyze_initialized;
  int parallel_factor_;
#endif

  Blob<Dtype> IVS_conv_inter_;
  vector<Blob<Dtype>*> IVS_top_q_stat;
};

/**
 * @brief Convolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class ConvolutionXNORIVSLayer : public ConvolutionLayer<Dtype>,
      public BaseIVSLayer<Dtype> {
 public:
  explicit ConvolutionXNORIVSLayer(const LayerParameter& param);
  virtual inline const char* type() const { return "ConvolutionXNORIVS"; }

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void forward_cpu_IVS(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  virtual void forward_cpu_IVS_no_rounding(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  virtual void forward_cpu_IVS_serial(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  virtual void forward_cpu_IVS_bias(Dtype* output, const Dtype* bias);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void im2col_cpu_IVS(const Dtype* data_im, const int channels,
  //    const int height, const int width, const int kernel_h, const int kernel_w,
  //    const int pad_h, const int pad_w, const int stride_h,
  //    const int stride_w, const int dilation_h, const int dilation_w,
  //    Dtype* data_col);
  //virtual void col2im_nd_cpu_IVS(const Dtype* data_col, const int num_spatial_axes,
  //    const int* im_shape, const int* col_shape,
  //    const int* kernel_shape, const int* pad, const int* stride,
  //    const int* dilation, Dtype* data_im);


  inline void conv_im2col_cpu_IVS(const Dtype* data, Dtype* col_buff) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      im2col_cpu_IVS(data, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1], this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0], this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0], this->dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu_IVS(data, this->num_spatial_axes_, this->conv_input_shape_.cpu_data(),
          this->col_buffer_shape_.data(), this->kernel_shape_.cpu_data(),
          this->pad_.cpu_data(), this->stride_.cpu_data(), this->dilation_.cpu_data(), col_buff);
    }
  }
  Blob<Dtype> IVS_conv_inter_;
};
/**
 * @brief Deconvolutional layer with quantized layer parameters and activations.
 */
template <typename Dtype>
class DeconvolutionIVSLayer : public DeconvolutionLayer<Dtype>,
      public BaseIVSLayer<Dtype> {
 public:
  explicit DeconvolutionIVSLayer(const LayerParameter& param);

  virtual inline const char* type() const { return "DeconvolutionIVS"; }

 protected:
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void backward_cpu_IVS(const Dtype* output, const Dtype* weights,
      Dtype* input);
  inline void conv_im2col_cpu_IVS(const Dtype* data, Dtype* col_buff) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      im2col_cpu_IVS(data, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1], this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0], this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0], this->dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu_IVS(data, this->num_spatial_axes_, this->conv_input_shape_.cpu_data(),
          this->col_buffer_shape_.data(), this->kernel_shape_.cpu_data(),
          this->pad_.cpu_data(), this->stride_.cpu_data(), this->dilation_.cpu_data(), col_buff);
    }
  }
  vector<int> IVS_conv_inter_shape_;
  Blob<Dtype> IVS_conv_inter_;
  Blob<Dtype> IVS_weights_trans_;
  vector<Blob<Dtype>*> IVS_top_q_stat;
#ifndef CPU_ONLY
//  virtual void backward_gpu_IVS(const Dtype* output, const Dtype* weights,
//      Dtype* input, Dtype* IVS_top_q_stat_data);
  inline void conv_im2col_gpu_IVS(const Dtype* data, Dtype* col_buff) {
    if (!this->force_nd_im2col_ && this->num_spatial_axes_ == 2) {
      im2col_gpu(data, this->conv_in_channels_,
          this->conv_input_shape_.cpu_data()[1], this->conv_input_shape_.cpu_data()[2],
          this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1],
          this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
          this->stride_.cpu_data()[0], this->stride_.cpu_data()[1],
          this->dilation_.cpu_data()[0], this->dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, this->num_spatial_axes_, this->num_kernels_im2col_,
          this->conv_input_shape_.gpu_data(), this->col_buffer_.gpu_shape(),
          this->kernel_shape_.gpu_data(), this->pad_.gpu_data(),
          this->stride_.gpu_data(), this->dilation_.gpu_data(), col_buff);
    }
  }
  Blob<Dtype> IVS_analyze_add_max_inter_;
  Blob<Dtype> IVS_analyze_add_min_inter_;
  Blob<Dtype> IVS_analyze_mul_max_inter_;
  Blob<Dtype> IVS_analyze_mul_min_inter_;
  int analyze_initialized;
  int parallel_factor_;
#endif
};

/**
 * @brief Inner product (fully connected) layer with quantized layer parameters
 * and activations.
 */
template <typename Dtype>
class FcIVSLayer : public InnerProductLayer<Dtype>,
      public BaseIVSLayer<Dtype>{
 public:
  explicit FcIVSLayer(const LayerParameter& param);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "FcIVS"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void forward_cpu_IVS(const CBLAS_TRANSPOSE TransB,
      const int M, const int N, const int K,
      const Dtype* A, const Dtype* B, Dtype* C);
#ifndef CPU_ONLY
//  virtual void forward_gpu_IVS(const CBLAS_TRANSPOSE TransB,
//      const int M, const int N, const int K,
//      const Dtype* A, const Dtype* B, Dtype* C, Dtype* IVS_top_q_stat_data_);
#endif
  virtual void forward_cpu_IVS_serial(const CBLAS_TRANSPOSE TransB,
      const int M, const int N, const int K,
      const Dtype* A, const Dtype* B, Dtype* C);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> IVS_fc_inter_;
  vector<Blob<Dtype>*> IVS_top_q_stat;
#ifndef CPU_ONLY
  Blob<Dtype> IVS_analyze_add_max_inter_;
  Blob<Dtype> IVS_analyze_add_min_inter_;
  Blob<Dtype> IVS_analyze_mul_max_inter_;
  Blob<Dtype> IVS_analyze_mul_min_inter_;
  int analyze_initialized;
#endif
};

/**
 * @brief Batch Nrom layer with quantized layer parameters
 * and activations.
 */

template <typename Dtype>
class BatchNormIVSLayer : public Layer<Dtype>,
      public BaseIVSLayer<Dtype>{
 public:
  explicit BatchNormIVSLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BatchNormIVS"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> mean_, variance_, temp_, x_norm_;
  bool use_global_stats_;
  Dtype moving_average_fraction_;
  int channels_;
  Dtype eps_;

  // extra temporarary variables is used to carry out sums/broadcasting
  // using BLAS
  Blob<Dtype> batch_sum_multiplier_;
  Blob<Dtype> num_by_chans_;
  Blob<Dtype> spatial_sum_multiplier_;
};


/**
 * @brief scale layer with quantized layer parameters
 * and activations.
 */
template <typename Dtype>
class ScaleIVSLayer: public Layer<Dtype>,
      public BaseIVSLayer<Dtype>{
 public:
  explicit ScaleIVSLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ScaleIVS"; }
  // Scale
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * In the below shape specifications, @f$ i @f$ denotes the value of the
   * `axis` field given by `this->layer_param_.scale_param().axis()`, after
   * canonicalization (i.e., conversion from negative to positive index,
   * if applicable).
   *
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the first factor @f$ x @f$
   *   -# @f$ (d_i \times ... \times d_j) @f$
   *      the second factor @f$ y @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (d_0 \times ... \times
   *           d_i \times ... \times d_j \times ... \times d_n) @f$
   *      the product @f$ z = x y @f$ computed after "broadcasting" y.
   *      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$,
   *      then computing the elementwise product.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  shared_ptr<Layer<Dtype> > bias_layer_;
  vector<Blob<Dtype>*> bias_bottom_vec_;
  vector<bool> bias_propagate_down_;
  int bias_param_id_;

  Blob<Dtype> sum_multiplier_;
  Blob<Dtype> sum_result_;
  Blob<Dtype> temp_;
  int axis_;
  int outer_dim_, scale_dim_, inner_dim_;
};


/**
 * @brief NdConvolution layer with quantized layer parameters
 * and activations.
 */
template <typename Dtype>
class CudnnNdConvolutionIVSLayer : public Layer<Dtype>,
      public BaseIVSLayer<Dtype>{
 public:
  explicit CudnnNdConvolutionIVSLayer(const LayerParameter& param)
      : Layer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CudnnNdConvolutionIVSLayer();

  virtual inline const char* type() const { return "NdConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape();

  vector<int> kernel_shape_;
  vector<int> stride_shape_;
  int num_;
  int channels_;
  vector<int> pad_shape_;
  vector<int> input_shape_;
  int group_;
  int num_output_;
  vector<int> output_shape_;
  bool bias_term_;

  int conv_out_spatial_dim_;
  int kernel_dim_;
  int output_offset_;

  Blob<Dtype> bias_multiplier_;

  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t*  stream_;
  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, weight_offset_, bias_offset_;
  size_t workspaceSizeInBytes;
  void* workspace_data_;  // underlying storage
  void** workspace_;  // aliases into workspaceData
  cudnnConvolutionBwdFilterAlgo_t* bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t* bwd_data_algo_;
  size_t* workspace_bwd_filter_sizes_;
  size_t* workspace_bwd_data_sizes_;
};



/**
 * @brief Local response normalization (LRN) layer with minifloat layer inputs,
 * intermediate results and outputs.
 */
//template <typename Dtype>
//class LRNIVSLayer : public LRNLayer<Dtype>,
//      public BaseIVSLayer<Dtype>{
// public:
//  explicit LRNIVSLayer(const LayerParameter& param);
//  virtual inline const char* type() const { return "LRNIVS"; }
//
// protected:
//  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
//  //void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//  //  const vector<Blob<Dtype>*>& top);
//  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  //void Backward_gpu(const vector<Blob<Dtype>*>& top,
//  //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  //virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
//  //    const vector<Blob<Dtype>*>& top);
//};

}  // namespace caffe

#endif  // CAFFE_BASE_IVS_LAYER_HPP_
