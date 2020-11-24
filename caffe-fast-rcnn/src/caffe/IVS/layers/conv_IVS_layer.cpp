#include <vector>
#include <float.h>

#include "IVS/base_IVS_layer.hpp"
#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
ConvolutionIVSLayer<Dtype>::ConvolutionIVSLayer(
      const LayerParameter& param) : ConvolutionLayer<Dtype>(param),
      BaseIVSLayer<Dtype>() {
  this->precision_ = this->layer_param_.quantization_param().precision();
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
  switch (this->precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->bw_params_ = this->layer_param_.quantization_param().bw_params();
    this->bw_add_ = this->layer_param_.quantization_param().bw_add();
    this->bw_multiply_ = this->layer_param_.quantization_param().bw_multiply();
    this->bw_bias_ = this->layer_param_.quantization_param().bw_bias();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    this->fl_params_ = this->layer_param_.quantization_param().fl_params();
    this->fl_add_ = this->layer_param_.quantization_param().fl_add();
    this->fl_multiply_ = this->layer_param_.quantization_param().fl_multiply();
    this->fl_bias_ = this->layer_param_.quantization_param().fl_bias();
    this->rounding_time_ = this->layer_param_.quantization_param().rounding_time();
    this->overflow_behavior_ = this->layer_param_.quantization_param().overflow_behavior();
    this->analyze_mode_ = this->layer_param_.quantization_param().analyze_mode();
    this->IVS_q_stat_ = this->layer_param_.quantization_param().ivs_q_stat();
    this->train_fl_param_ = this->layer_param_.quantization_param().train_fl_param();
    this->train_fl_io_ = this->layer_param_.quantization_param().train_fl_io();
    this->ns_ratio_param_ = this->layer_param_.quantization_param().ns_ratio_param();
    this->ns_ratio_act_ = this->layer_param_.quantization_param().ns_ratio_act();
    this->ivs_ste_param_ = this->layer_param_.quantization_param().ivs_ste_param();
    this->ivs_if_deploy_ = this->layer_param_.quantization_param().ivs_if_deploy();
    break;
  case QuantizationParameter_Precision_MINIFLOAT:
    this->fp_mant_ = this->layer_param_.quantization_param().mant_bits();
    this->fp_exp_ = this->layer_param_.quantization_param().exp_bits();
    break;
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    this->pow_2_min_exp_ = this->layer_param_.quantization_param().exp_min();
    this->pow_2_max_exp_ = this->layer_param_.quantization_param().exp_max();
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    break;
  default:
    LOG(FATAL) << "Unknown precision mode: " << this->precision_;
    break;
  }
}

template <typename Dtype>
void ConvolutionIVSLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  this->force_nd_im2col_ = conv_param.force_nd_im2col();
  this->channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int num_axes = bottom[0]->num_axes();
  if (num_axes == 5 && this->channel_axis_ == 1 && bottom[0]->shape(2) == 1) {
    this->forced_3d_ = true;
  } else {
    this->forced_3d_ = false;
  }
  const int first_spatial_axis = this->channel_axis_ + 1 + this->forced_3d_;
  this->num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(this->num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = this->stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = this->pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  this->dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = this->dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == this->num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << this->num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    this->is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!this->is_1x1_) { break; }
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(this->channel_axis_);
  this->num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";
  if (this->reverse_dimensions()) {
    this->conv_out_channels_ = this->channels_;
    this->conv_in_channels_ = this->num_output_;
  } else {
    this->conv_out_channels_ = this->num_output_;
    this->conv_in_channels_ = this->channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = this->conv_out_channels_;
  weight_shape[1] = this->conv_in_channels_ / this->group_;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  this->bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(this->bias_term_, this->num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    // true_blob_shape is original blob_shape (n,c,h,w) in case of forced_3d_
    // where blob_shape is expanded to (n,c,1,h,w)
    vector<int> true_blob_shape = this->blobs_[0]->shape();
    if (this->forced_3d_) true_blob_shape.erase(true_blob_shape.begin()+2);
    if (weight_shape != true_blob_shape) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  this->kernel_dim_ = this->blobs_[0]->count(1);
  this->weight_offset_ =
      this->conv_out_channels_ * this->kernel_dim_ / this->group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // Prepare quantized weights
  this->weights_quantized_.resize(2);
  this->weights_quantized_[0].reset(new Blob<Dtype>(weight_shape));
  if (this->bias_term_) {
      this->weights_quantized_[1].reset(new Blob<Dtype>(bias_shape));
  }
  //analyze mode for add and multiply
  this->min_multiply_ = DBL_MAX;
  this->min_add_ = DBL_MAX;
  this->max_multiply_ = DBL_MIN;
  this->max_add_ = DBL_MIN;
  this->max_out_ = (powf(2, this->bw_layer_out_ - 1) - 1) * powf(2, -1* this->fl_layer_out_);
  this->max_in_ = (powf(2, this->bw_layer_in_ - 1) - 1) * powf(2, -1* this->fl_layer_in_);
#ifndef CPU_ONLY
  this->analyze_initialized = 0;
#endif
}
template <typename Dtype>
void ConvolutionIVSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_axes = bottom[0]->num_axes();
  // Setting forced_3d_ in LayerSetup() alone is not sufficient as that can be
  // skipped and Reshape() is directed called.
  if (num_axes == 5 && this->channel_axis_ == 1 && bottom[0]->shape(2) == 1) {
    this->forced_3d_ = true;
  } else {
    this->forced_3d_ = false;
  }
  const int first_spatial_axis = this->channel_axis_ + 1 + this->forced_3d_;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + this->num_spatial_axes_)
      << "bottom num_axes may not change.";
  this->num_ = bottom[0]->count(0, this->channel_axis_);
  CHECK_EQ(bottom[0]->shape(this->channel_axis_), this->channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  this->bottom_shape_ = &bottom[0]->shape();
  this->compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + this->channel_axis_);
  top_shape.push_back(this->num_output_);
  if (this->forced_3d_)
    top_shape.push_back(1);  // in place of length
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    top_shape.push_back(this->output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (this->reverse_dimensions()) {
    this->conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    this->conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  this->col_offset_ = this->kernel_dim_ * this->conv_out_spatial_dim_;
  this->output_offset_ = this->conv_out_channels_ * this->conv_out_spatial_dim_ / this->group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
  this->conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = this->conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < this->num_spatial_axes_ + 1; ++i) {
    if (this->reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(this->channel_axis_ + i + this->forced_3d_);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(this->channel_axis_ + i + this->forced_3d_);
    }
  }
  this->col_buffer_shape_.clear();
  this->col_buffer_shape_.push_back(this->kernel_dim_ * this->group_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    if (this->reverse_dimensions()) {
      this->col_buffer_shape_.push_back(this->input_shape(i + 1));
    } else {
      this->col_buffer_shape_.push_back(this->output_shape_[i]);
    }
  }
  this->col_buffer_.Reshape(this->col_buffer_shape_);
  this->IVS_conv_inter_shape_.clear();
#if (CONV_ADD_PARALLEL_LEVEL == 1)
  this->parallel_factor_ = 1;
#elif (CONV_ADD_PARALLEL_LEVEL == 2)
  this->parallel_factor_ = 2;
#elif (CONV_ADD_PARALLEL_LEVEL == 3)
  this->parallel_factor_ = 4;
#endif

  this->IVS_conv_inter_shape_.push_back(this->kernel_dim_*this->parallel_factor_);
  //this->IVS_conv_inter_shape_.push_back(this->conv_out_channels_/this->group_);
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    if (this->reverse_dimensions()) {
      this->IVS_conv_inter_shape_.push_back(this->input_shape(i + 1));
    } else {
        this->IVS_conv_inter_shape_.push_back(this->output_shape_[i]);
    }
  }
  this->IVS_conv_inter_.Reshape(this->IVS_conv_inter_shape_);
#ifndef CPU_ONLY
#ifdef LOW_MEM_MODE
  vector<int> inter_size(1);
  inter_size[0] = ANALYZE_MODE_PARALLEL_FACTOR;
  if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
      || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
    this->IVS_analyze_add_max_inter_.Reshape(inter_size);
    this->IVS_analyze_add_min_inter_.Reshape(inter_size);
  }
  if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY\
      || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
    this->IVS_analyze_mul_max_inter_.Reshape(inter_size);
    this->IVS_analyze_mul_min_inter_.Reshape(inter_size);
    //this->IVS_analyze_mul_min_inter_.Reshape(this->IVS_conv_inter_shape_);
  }
#else
  if(this->IVS_conv_inter_.count() > this->IVS_analyze_add_max_inter_.count()){
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
        || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
      this->IVS_analyze_add_max_inter_.Reshape(this->IVS_conv_inter_shape_);
      this->IVS_analyze_add_min_inter_.Reshape(this->IVS_conv_inter_shape_);
    }
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY\
        || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
      this->IVS_analyze_mul_max_inter_.Reshape(this->IVS_conv_inter_shape_);
      this->IVS_analyze_mul_min_inter_.Reshape(this->IVS_conv_inter_shape_);
    }
  }
#endif
  if(this->analyze_initialized == 0){
    this->analyze_initialized = 1;
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
      || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
      caffe_gpu_set(IVS_analyze_add_max_inter_.count(), (Dtype)FLT_MIN, IVS_analyze_add_max_inter_.mutable_gpu_data());
      caffe_gpu_set(IVS_analyze_add_min_inter_.count(), (Dtype)FLT_MAX, IVS_analyze_add_min_inter_.mutable_gpu_data());
    }
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY\
      || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
    caffe_gpu_set(IVS_analyze_mul_max_inter_.count(), (Dtype)FLT_MIN, IVS_analyze_mul_max_inter_.mutable_gpu_data());
    caffe_gpu_set(IVS_analyze_mul_min_inter_.count(), (Dtype)FLT_MAX, IVS_analyze_mul_min_inter_.mutable_gpu_data());
    }
  }
  if(this->phase_ == TRAIN && this->IVS_q_stat_ == QuantizationParameter_IVS_Q_Stat_Q_STAT_ON){
    vector<int> IVS_top_q_stat_shape(bottom[0]->shape().begin(),
        bottom[0]->shape().begin() + this->channel_axis_);
    IVS_top_q_stat_shape.push_back(this->num_output_);
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      IVS_top_q_stat_shape.push_back(this->output_shape_[i]);
    }
    if(IVS_top_q_stat.size()<=0){
      for(int i = 0; i < top.size();i++){
        IVS_top_q_stat.push_back(new Blob<Dtype>(IVS_top_q_stat_shape));
      }
    }
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      IVS_top_q_stat[top_id]->Reshape(IVS_top_q_stat_shape);
    }
  }
#endif
  this->bottom_dim_ = bottom[0]->count(this->channel_axis_);
  this->top_dim_ = top[0]->count(this->channel_axis_);
  this->num_kernels_im2col_ = this->conv_in_channels_ * this->conv_out_spatial_dim_;
  this->num_kernels_col2im_ = this->reverse_dimensions() ? this->top_dim_ : this->bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  this->out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1, this->out_spatial_dim_);
    this->bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(this->bias_multiplier_.count(), Dtype(1),
        this->bias_multiplier_.mutable_cpu_data());
  }
}



template <typename Dtype>
void ConvolutionIVSLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if(*(this->blobs_[0]->cpu_data()) == 2610214){
    *(top[0]->mutable_cpu_data())   = this->max_add_;
    *(top[0]->mutable_cpu_data()+1) = this->min_add_;
    *(top[0]->mutable_cpu_data()+2) = this->max_multiply_;
    *(top[0]->mutable_cpu_data()+3) = this->min_multiply_;
    return;
  }
  // Trim layer input
  if (this->phase_ == TEST) {
    for (int i = 0; i < bottom.size(); ++i) {
      this->QuantizeLayerInputs_cpu(bottom[i]->mutable_cpu_data(),
          bottom[i]->count());
    }
  }
  // Trim weights
  caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(),
      this->weights_quantized_[0]->mutable_cpu_data());
  if (this->bias_term_) {
    caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(),
        this->weights_quantized_[1]->mutable_cpu_data());
  }
  int rounding = this->phase_ == TEST ? this->rounding_ :
      QuantizationParameter_Rounding_STOCHASTIC;
  this->QuantizeWeights_cpu(this->weights_quantized_, rounding,
      this->bias_term_);
  // Do forward propagation
  const Dtype* weight = this->weights_quantized_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      switch(this->rounding_time_){
      case QuantizationParameter_Rounding_time_LAYER_BY_LAYER:
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                top_data + n * this->top_dim_);
        break;
      case QuantizationParameter_Rounding_time_EVERY_OPERATION:
        this->forward_cpu_IVS(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
        break;
      case QuantizationParameter_Rounding_time_EVERY_OPERATION_SERIAL:
        this->forward_cpu_IVS_serial(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
        break;
      default:
        LOG(FATAL) << "Unknown rounding time mode: " << this->rounding_time_;
        break;
      }
      if (this->bias_term_) {
        const Dtype* bias = this->weights_quantized_[1]->cpu_data();
        switch(this->rounding_time_){
        case QuantizationParameter_Rounding_time_LAYER_BY_LAYER:
          this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
          break;
        case QuantizationParameter_Rounding_time_EVERY_OPERATION:
          this->forward_cpu_IVS_bias(top_data + n * this->top_dim_, bias);
          break;
        case QuantizationParameter_Rounding_time_EVERY_OPERATION_SERIAL:
          this->forward_cpu_IVS_bias(top_data + n * this->top_dim_, bias);
          break;
        default:
          LOG(FATAL) << "Unknown rounding time mode: " << this->rounding_time_;
          break;
        }
      }
    }
    //Analyze bias add 
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
      || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
      this->Analyze_add(top_data,top[i]->count());
    }
    //Quantize bias add
    if (this->phase_ == TEST && this->bias_term_){
      switch(this->rounding_time_){
      case QuantizationParameter_Rounding_time_LAYER_BY_LAYER:
        break;
      case QuantizationParameter_Rounding_time_EVERY_OPERATION:
        this->QuantizeIVS_conv_inter_add_cpu(top_data, top[i]->count());
        break;
      case QuantizationParameter_Rounding_time_EVERY_OPERATION_SERIAL:
        this->QuantizeIVS_conv_inter_add_cpu(top_data, top[i]->count());
        break;
      default:
        LOG(FATAL) << "Unknown rounding time mode: " << this->rounding_time_;
        break;
      }
    }
    // Trim layer output
    if (this->phase_ == TEST) {
      this->QuantizeLayerOutputs_cpu(top_data, top[i]->count());
    }
  }
  //Log analyze result
  switch(this->analyze_mode_){
  case QuantizationParameter_Analyze_mode_ANALYZE_ADD:
    LOG(INFO) << "adder max value = "<< this->max_add_ << " adder min value = " << this->min_add_;
    break;
  case QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY:
    LOG(INFO) << "multiplier max value = "<< this->max_multiply_ << " multiplier min value = " << this->min_multiply_;
    break;
  case QuantizationParameter_Analyze_mode_ANALYZE_BOTH:
    LOG(INFO) << "adder max value = "<< this->max_add_ << " adder min value = " << this->min_add_;
    LOG(INFO) << "multiplier max value = "<< this->max_multiply_ << " multiplier min value = " << this->min_multiply_;
    break;
  case QuantizationParameter_Analyze_mode_NO_ANALYZE:
    break;
  default:
    LOG(FATAL) << "Unknown analyze mode: " << this->analyze_mode_;
    break;
  }
}

template <typename Dtype>
void ConvolutionIVSLayer<Dtype>::Backward_cpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->weights_quantized_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionIVSLayer<Dtype>::forward_cpu_IVS(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu_IVS(input, this->col_buffer_.mutable_cpu_data());
    }
    col_buff = this->col_buffer_.cpu_data();
  }
  //const int channels = conv_in_channels_;
  const int height = this->conv_input_shape_.cpu_data()[1];
  const int width = this->conv_input_shape_.cpu_data()[2];
  const int kernel_h = this->kernel_shape_.cpu_data()[0];
  const int kernel_w = this->kernel_shape_.cpu_data()[1];
  const int pad_h = this->pad_.cpu_data()[0];
  const int pad_w = this->pad_.cpu_data()[1];
  const int stride_h = this->stride_.cpu_data()[0];
  const int stride_w = this->stride_.cpu_data()[1];
  const int dilation_h = this->dilation_.cpu_data()[0];
  const int dilation_w = this->dilation_.cpu_data()[1];
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  //const int channel_size = height * width;
  Dtype* IVS_conv_inter_data;
  Dtype* output_data;
  output_data = output;
  for (int g = 0; g < this->group_; ++g){
    //printf("g= %d\n",g);
    for( int kernels_col = 0; kernels_col < this->conv_out_channels_/this->group_; kernels_col++){
      //element-wise multiplication
      //this->IVS_conv_inter_.CopyFrom(col_buff);
      memcpy((Dtype*)this->IVS_conv_inter_.cpu_data(), 
            col_buff, sizeof(Dtype) *IVS_conv_inter_.count());
      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data() + this->col_offset_*g;
      for (int kernels_row = 0; 
            kernels_row < kernel_h * kernel_w * this->conv_in_channels_/this->group_;\
            kernels_row++){
        caffe_cpu_axpby(output_h * output_w, (Dtype)0, (Dtype*)IVS_conv_inter_data, \
                        (Dtype)*weights,(Dtype*)IVS_conv_inter_data); 
        IVS_conv_inter_data +=output_h * output_w;
        weights++;
      }
      // Analyze after multiplier
      if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY\
        || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
        this->Analyze_multiply(this->IVS_conv_inter_.cpu_data()+this->col_offset_*g,\
                              output_h * output_w \
                                * kernel_h * kernel_w * this->conv_in_channels_/this->group_);
      }
      // Quantize after element-wise multiplier
      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data() + this->col_offset_*g;
      this->QuantizeIVS_conv_inter_multiply_cpu(this->IVS_conv_inter_.mutable_cpu_data()\
                                + this->col_offset_*g,\
                                output_h * output_w \
                                * kernel_h * kernel_w * this->conv_in_channels_/this->group_);
      //matrix acculumation (folding adding)
      for (int IVS_conv_inter_col_left = kernel_h * kernel_w * this->conv_in_channels_/this->group_;\
           IVS_conv_inter_col_left != 1;\
           IVS_conv_inter_col_left /= 2){
        IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data() + this->col_offset_*g;
        for (int IVS_conv_inter_col = IVS_conv_inter_col_left/2 ; IVS_conv_inter_col;\
             IVS_conv_inter_col--){
          caffe_add(output_h * output_w, IVS_conv_inter_data, \
                    IVS_conv_inter_data + output_h * output_w * (int)(IVS_conv_inter_col_left/2),
                    IVS_conv_inter_data);
          IVS_conv_inter_data += output_h * output_w;
        }
        // Analyze afer adder
        if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
          || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
          this->Analyze_add(this->IVS_conv_inter_.cpu_data()+this->col_offset_*g,\
                            (IVS_conv_inter_col_left/2) * output_h * output_w);
        }
        // If number of row is odd add the rest one row
        this->QuantizeIVS_conv_inter_add_cpu(this->IVS_conv_inter_.mutable_cpu_data()\
                                                +this->col_offset_*g,\
                                              (IVS_conv_inter_col_left/2) * output_h * output_w);
        if((IVS_conv_inter_col_left)%2 == 1){
          IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data();
          IVS_conv_inter_data += this->col_offset_*g;
          caffe_add(output_h * output_w, IVS_conv_inter_data, \
                    IVS_conv_inter_data + output_h * output_w * (IVS_conv_inter_col_left-1),
                    IVS_conv_inter_data);
          // Analyze afer adder
          if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
            || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
            this->Analyze_add(this->IVS_conv_inter_.cpu_data()+this->col_offset_*g,\
                            output_h * output_w);
          }
          //Quantize after odd handler
          this->QuantizeIVS_conv_inter_add_cpu(this->IVS_conv_inter_.mutable_cpu_data()
                                                + this->col_offset_*g,
                                                output_h * output_w);
        }

      }
      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data() + this->col_offset_*g;
      for (int output_row = 0 ; output_row < output_h * output_w ;\
              output_row++){
        *(output_data++)=*(IVS_conv_inter_data++);
      }
    }
    //Original Caffe
    //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
    //    group_, conv_out_spatial_dim_, kernel_dim_,
    //    (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
    //    (Dtype)0., output + output_offset_ * g);

  }
}

template <typename Dtype>
void ConvolutionIVSLayer<Dtype>::forward_cpu_IVS_serial(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu_IVS(input, this->col_buffer_.mutable_cpu_data());
    }
    col_buff = this->col_buffer_.cpu_data();
  }
  //const int channels = conv_in_channels_;
  const int height = this->conv_input_shape_.cpu_data()[1];
  const int width = this->conv_input_shape_.cpu_data()[2];
  const int kernel_h = this->kernel_shape_.cpu_data()[0];
  const int kernel_w = this->kernel_shape_.cpu_data()[1];
  const int pad_h = this->pad_.cpu_data()[0];
  const int pad_w = this->pad_.cpu_data()[1];
  const int stride_h = this->stride_.cpu_data()[0];
  const int stride_w = this->stride_.cpu_data()[1];
  const int dilation_h = this->dilation_.cpu_data()[0];
  const int dilation_w = this->dilation_.cpu_data()[1];
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  //const int channel_size = height * width;
  Dtype* IVS_conv_inter_data;
  Dtype* output_data;
//  IVS_conv_inter_data = col_buff.cpu_data();
  output_data = output;
  for (int g = 0; g < this->group_; ++g){
    //printf("g= %d\n",g);
    for( int kernels_col = 0; kernels_col < this->conv_out_channels_/this->group_; kernels_col++){
      //element-wise multiplication
      memcpy((Dtype*)this->IVS_conv_inter_.cpu_data(), 
            col_buff, sizeof(Dtype) *IVS_conv_inter_.count());
      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data() + this->col_offset_*g;
      for (int kernels_row = 0; 
            kernels_row < kernel_h * kernel_w * this->conv_in_channels_/this->group_;\
            kernels_row++){
        caffe_cpu_axpby(output_h * output_w, (Dtype)0, (Dtype*)IVS_conv_inter_data, (Dtype)*weights,(Dtype*)IVS_conv_inter_data); 
        IVS_conv_inter_data +=output_h * output_w;
        weights++;
      }
      //Analyze Multiply
      if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY\
          || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
        this->Analyze_multiply(this->IVS_conv_inter_.cpu_data()+this->col_offset_*g,\
                           output_h * output_w \
                            * kernel_h * kernel_w * this->conv_in_channels_/this->group_);
      }

      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data() + this->col_offset_*g;
      this->QuantizeIVS_conv_inter_multiply_cpu(this->IVS_conv_inter_.mutable_cpu_data()\
                                            +this->col_offset_*g ,\
                                            output_h * output_w \
                                        * kernel_h * kernel_w * this->conv_in_channels_/this->group_);
      //matrix acculumation
      for (int IVS_conv_inter_col_left = 1;\
           IVS_conv_inter_col_left < kernel_h * kernel_w * this->conv_in_channels_/this->group_;\
           IVS_conv_inter_col_left++){
        IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data() + this->col_offset_*g;
        for (int input_cols_row = 0 ; input_cols_row < output_h * output_w ; input_cols_row++){
          *(IVS_conv_inter_data) += *(IVS_conv_inter_data \
                                     + output_h * output_w * IVS_conv_inter_col_left);
          IVS_conv_inter_data++;
        }
        if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
            || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
          this->Analyze_add(this->IVS_conv_inter_.cpu_data() + this->col_offset_*g,\
                            output_h * output_w);
        }
        this->QuantizeIVS_conv_inter_add_cpu(this->IVS_conv_inter_.mutable_cpu_data()\
                                            + this->col_offset_*g,
                                             output_h * output_w);
      }

      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data() + this->col_offset_*g;
      for (int output_row = 0 ; output_row < output_h * output_w ;\
              output_row++){
        *(output_data++)=*(IVS_conv_inter_data++);
      }
    }
    //Original Caffe
    //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
    //    group_, conv_out_spatial_dim_, kernel_dim_,
    //    (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
    //    (Dtype)0., output + output_offset_ * g);

  }
}

template <typename Dtype>
void ConvolutionIVSLayer<Dtype>::forward_cpu_IVS_serial_mult(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu_IVS(input, this->col_buffer_.mutable_cpu_data());
    }
    col_buff = this->col_buffer_.cpu_data();
  }
  const int height = this->conv_input_shape_.cpu_data()[1];
  const int width = this->conv_input_shape_.cpu_data()[2];
  const int kernel_h = this->kernel_shape_.cpu_data()[0];
  const int kernel_w = this->kernel_shape_.cpu_data()[1];
  const int pad_h = this->pad_.cpu_data()[0];
  const int pad_w = this->pad_.cpu_data()[1];
  const int stride_h = this->stride_.cpu_data()[0];
  const int stride_w = this->stride_.cpu_data()[1];
  const int dilation_h = this->dilation_.cpu_data()[0];
  const int dilation_w = this->dilation_.cpu_data()[1];
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  //const int channel_size = height * width;
  Dtype* IVS_conv_inter_data;
  Dtype* output_data;
  //IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data();
  output_data = output;
  for (int g = 0; g < this->group_; ++g){
    for( int kernels_col = 0; kernels_col < this->conv_out_channels_/this->group_; kernels_col++){
      memcpy((Dtype*)this->IVS_conv_inter_.cpu_data(),
            col_buff, sizeof(Dtype) *IVS_conv_inter_.count());

      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data();
      IVS_conv_inter_data += this->col_offset_*g;
      for (int kernels_row = 0; 
            kernels_row < kernel_h * kernel_w * this->conv_in_channels_/this->group_;\
            kernels_row++){
        for (int input_cols_row = 0 ; input_cols_row < output_h * output_w ; input_cols_row++){
          *(IVS_conv_inter_data) = *IVS_conv_inter_data * *weights;
          IVS_conv_inter_data++;
        }
        weights++;
      }
      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data();
      IVS_conv_inter_data += this->col_offset_*g;
      this->QuantizeIVS_conv_inter_multiply_cpu(this->IVS_conv_inter_.mutable_cpu_data(),
        this->IVS_conv_inter_.count());
      //matrix acculumation
      for (int IVS_conv_inter_col_left = kernel_h * kernel_w * this->conv_in_channels_/this->group_;\
           IVS_conv_inter_col_left != 1;\
           IVS_conv_inter_col_left /= 2){
        IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data();
        IVS_conv_inter_data += this->col_offset_*g;
        for (int IVS_conv_inter_col = IVS_conv_inter_col_left/2 ; IVS_conv_inter_col;\
             IVS_conv_inter_col--){
          for (int input_cols_row = 0 ; input_cols_row < output_h * output_w ;\
               input_cols_row++){
            *(IVS_conv_inter_data) += *(IVS_conv_inter_data \
                                          + output_h * output_w \
                                            * (int)(IVS_conv_inter_col_left/2));
            IVS_conv_inter_data++;
          }
        }

        this->QuantizeIVS_conv_inter_add_cpu(this->IVS_conv_inter_.mutable_cpu_data(),
          this->IVS_conv_inter_.count());
        if((IVS_conv_inter_col_left)%2 == 1){
            //printf("odd\n");
          IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data();
          IVS_conv_inter_data += this->col_offset_*g;
          for (int input_cols_row = 0 ; input_cols_row < output_h * output_w ;\
               input_cols_row++){
            *(IVS_conv_inter_data) += *(IVS_conv_inter_data \
                                          + output_h * output_w \
                                            * (IVS_conv_inter_col_left-1));
            IVS_conv_inter_data++;
          }
          this->QuantizeIVS_conv_inter_add_cpu(this->IVS_conv_inter_.mutable_cpu_data(),
            this->IVS_conv_inter_.count());
        }

      }

      this->QuantizeIVS_conv_inter_add_cpu(this->IVS_conv_inter_.mutable_cpu_data(),
          this->IVS_conv_inter_.count());
      IVS_conv_inter_data = this->IVS_conv_inter_.mutable_cpu_data();
      IVS_conv_inter_data += this->col_offset_*g;
      for (int output_row = 0 ; output_row < output_h * output_w ;\
              output_row++){
        *(output_data++)=*(IVS_conv_inter_data++);
      }
    }
    //Original Caffe
    //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
    //    group_, conv_out_spatial_dim_, kernel_dim_,
    //    (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
    //    (Dtype)0., output + output_offset_ * g);

  }
}
template <typename Dtype>
void ConvolutionIVSLayer<Dtype>::forward_cpu_IVS_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_,
      this->out_spatial_dim_, 1, (Dtype)1., bias, this->bias_multiplier_.cpu_data(),
      (Dtype)1., output);
  
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionIVSLayer);
#endif

INSTANTIATE_CLASS(ConvolutionIVSLayer);
REGISTER_LAYER_CLASS(ConvolutionIVS);

}  // namespace caffe

