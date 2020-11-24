#include <vector>
#include <float.h>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "IVS/base_IVS_layer.hpp"

namespace caffe {

template <typename Dtype>
FcIVSLayer<Dtype>::FcIVSLayer(const LayerParameter& param)
      : InnerProductLayer<Dtype>(param), BaseIVSLayer<Dtype>() {
  this->precision_ = this->layer_param_.quantization_param().precision();
  this->rounding_ = this->layer_param_.quantization_param().rounding_scheme();
  switch (this->precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    this->bw_layer_in_ = this->layer_param_.quantization_param().bw_layer_in();
    this->bw_layer_out_ = this->layer_param_.quantization_param().bw_layer_out();
    this->bw_params_ = this->layer_param_.quantization_param().bw_params();
    this->bw_bias_ = this->layer_param_.quantization_param().bw_bias();
    this->bw_add_ = this->layer_param_.quantization_param().bw_add();
    this->bw_multiply_ = this->layer_param_.quantization_param().bw_multiply();
    this->bw_add_ = this->layer_param_.quantization_param().bw_add();
    this->bw_multiply_ = this->layer_param_.quantization_param().bw_multiply();
    this->fl_layer_in_ = this->layer_param_.quantization_param().fl_layer_in();
    this->fl_layer_out_ = this->layer_param_.quantization_param().fl_layer_out();
    this->fl_params_ = this->layer_param_.quantization_param().fl_params();
    this->fl_bias_ = this->layer_param_.quantization_param().fl_bias();
    this->fl_add_ = this->layer_param_.quantization_param().fl_add();
    this->fl_multiply_ = this->layer_param_.quantization_param().fl_multiply();
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
void FcIVSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  this->bias_term_ = this->layer_param_.inner_product_param().bias_term();
  this->transpose_ = this->layer_param_.inner_product_param().transpose();
  this->N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  this->K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (this->bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (this->transpose_) {
      weight_shape[0] = this->K_;
      weight_shape[1] = this->N_;
    } else {
      weight_shape[0] = this->N_;
      weight_shape[1] = this->K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (this->bias_term_) {
      vector<int> bias_shape(1, this->N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // Prepare quantized weights
  this->weights_quantized_.resize(2);
  vector<int> weight_shape(2);
  weight_shape[0] = this->N_;
  weight_shape[1] = this->K_;
  this->weights_quantized_[0].reset(new Blob<Dtype>(weight_shape));
  vector<int> fc_inter_shape(2);
  if(this->transpose_){
    fc_inter_shape[0] = this->K_;
    fc_inter_shape[1] = this->N_;
  }
  else{
    fc_inter_shape[0] = this->N_;
    fc_inter_shape[1] = this->K_;
  }
  this->IVS_fc_inter_.Reshape(fc_inter_shape);
  this->analyze_initialized = 0;
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
    //this->IVS_analyze_mul_min_inter_.Reshape(fc_inter_shape);
  }
#else
  if(this->IVS_fc_inter_.count() > this->IVS_analyze_add_max_inter_.count()){
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
        || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
      this->IVS_analyze_add_max_inter_.Reshape(fc_inter_shape);
      this->IVS_analyze_add_min_inter_.Reshape(fc_inter_shape);
    }
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY\
        || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
      this->IVS_analyze_mul_max_inter_.Reshape(fc_inter_shape);
      this->IVS_analyze_mul_min_inter_.Reshape(fc_inter_shape);
    }
  }
  if(this->phase_ == TRAIN && this->IVS_q_stat_ == QuantizationParameter_IVS_Q_Stat_Q_STAT_ON){
    if(IVS_top_q_stat.size() <= 0){
      IVS_top_q_stat.push_back(new Blob<Dtype>( top[0]->shape() ));
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
#endif

  vector<int> bias_shape(1, this->N_);
  if (this->bias_term_) {
      this->weights_quantized_[1].reset(new Blob<Dtype>(bias_shape));
  }
  this->min_multiply_ = DBL_MAX;
  this->min_add_ = DBL_MAX;
  this->max_multiply_ = DBL_MIN;
  this->max_add_ = DBL_MIN;
  this->max_out_ = (powf(2, this->bw_layer_out_ - 1) - 1) * powf(2, -1* this->fl_layer_out_);
  this->max_in_ = (powf(2, this->bw_layer_in_ - 1) - 1) * powf(2, -1* this->fl_layer_in_);

}

template <typename Dtype>
void FcIVSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Trim layer input
  if (this->phase_ == TEST) {
      this->QuantizeLayerInputs_cpu(bottom[0]->mutable_cpu_data(),
          bottom[0]->count());
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
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->weights_quantized_[0]->cpu_data();
  switch(this->rounding_time_){
  case QuantizationParameter_Rounding_time_LAYER_BY_LAYER:
    caffe_cpu_gemm<Dtype>(CblasNoTrans, this->transpose_ ? CblasNoTrans :
        CblasTrans, this->M_, this->N_, this->K_, (Dtype)1., bottom_data, weight,
        (Dtype)0., top_data);
    break;
  case QuantizationParameter_Rounding_time_EVERY_OPERATION:
    forward_cpu_IVS(this->transpose_ ? CblasNoTrans :CblasTrans,
        this->M_, this->N_, this->K_, bottom_data, weight,
        top_data);
    break;
  case QuantizationParameter_Rounding_time_EVERY_OPERATION_SERIAL:
    forward_cpu_IVS_serial(this->transpose_ ? CblasNoTrans :CblasTrans,
        this->M_, this->N_, this->K_, bottom_data, weight,
        top_data);
    break;
  default:
    LOG(FATAL) << "Unknown rounding time mode: " << this->rounding_time_;
    break;
  }
  if (this->bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->N_, 1,
        (Dtype)1., this->bias_multiplier_.cpu_data(),
        this->weights_quantized_[1]->cpu_data(), (Dtype)1., top_data);
    if(this->rounding_time_ != QuantizationParameter_Rounding_time_LAYER_BY_LAYER){
      this->QuantizeIVS_conv_inter_add_cpu(top_data,top[0]->count());
    }
  }
  // Trim layer output
  if (this->phase_ == TEST) {
    this->QuantizeLayerOutputs_cpu(top_data, top[0]->count());
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
void FcIVSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (this->transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->K_, this->N_, this->M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          this->N_, this->K_, this->M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, (Dtype)1., top_diff,
        this->bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (this->transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          this->M_, this->K_, this->N_,
          (Dtype)1., top_diff, this->weights_quantized_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          this->M_, this->K_, this->N_,
          (Dtype)1., top_diff, this->weights_quantized_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}
template <typename Dtype>
void FcIVSLayer<Dtype>::forward_cpu_IVS(const CBLAS_TRANSPOSE TransB,
      const int M, const int N, const int K,
      const Dtype* A, const Dtype* B, Dtype* C){
  
  const Dtype* input_data;
  Dtype* IVS_fc_inter_data;
  const Dtype* weight;
  Dtype* output_data;
  output_data = C;
  for(int batch_iter = 0; batch_iter<M; batch_iter++){
    weight = B;
    IVS_fc_inter_data = this->IVS_fc_inter_.mutable_cpu_data();
    for(int weight_column=0; weight_column<N; weight_column++){
      input_data = A + batch_iter * K;
      for(int weight_row=0; weight_row<K; weight_row++){
        if(TransB!=CblasNoTrans)
          *(IVS_fc_inter_data + weight_row * N + weight_column) = *(input_data++) * *(weight++);
        else
          *(IVS_fc_inter_data + weight_row * N + weight_column) = *(input_data++) * *(weight++ + weight_row * N + weight_column);
      }
    }
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY\
      || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
      this->Analyze_multiply(this->IVS_fc_inter_.cpu_data(),this->IVS_fc_inter_.count());
    }
    //Quantize after element-wise multiplier
    this->QuantizeIVS_conv_inter_multiply_cpu(this->IVS_fc_inter_.mutable_cpu_data(),
      this->IVS_fc_inter_.count());

    for (int IVS_fc_inter_col_left = K; IVS_fc_inter_col_left != 1; IVS_fc_inter_col_left /= 2){
      IVS_fc_inter_data = this->IVS_fc_inter_.mutable_cpu_data();
      for (int IVS_fc_inter_col = IVS_fc_inter_col_left/2 ; IVS_fc_inter_col;\
           IVS_fc_inter_col--){
        for (int fc_inter_row = 0 ; fc_inter_row < N ; fc_inter_row++){
          *(IVS_fc_inter_data) += *(IVS_fc_inter_data \
                                        + N * (int)(IVS_fc_inter_col_left/2));
          IVS_fc_inter_data++;
        }
      }
      if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
        || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
        this->Analyze_add(this->IVS_fc_inter_.cpu_data(), N * IVS_fc_inter_col_left/2);
      }

      this->QuantizeIVS_conv_inter_add_cpu(this->IVS_fc_inter_.mutable_cpu_data(),
        N * IVS_fc_inter_col_left/2);
      if((IVS_fc_inter_col_left)%2 == 1){
          //printf("odd\n");
        IVS_fc_inter_data = this->IVS_fc_inter_.mutable_cpu_data();
        for (int fc_inter_row = 0 ; fc_inter_row < N ;\
             fc_inter_row++){
          *(IVS_fc_inter_data) += *(IVS_fc_inter_data \
                                        + N * (IVS_fc_inter_col_left-1));
          IVS_fc_inter_data++;
        }
        if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
          || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
          this->Analyze_add(this->IVS_fc_inter_.cpu_data(), N);
        }
        this->QuantizeIVS_conv_inter_add_cpu(this->IVS_fc_inter_.mutable_cpu_data(), N);
      }
    }
    IVS_fc_inter_data = this->IVS_fc_inter_.mutable_cpu_data();
    for (int output_row = 0 ;output_row < N ;output_row++){
      *(output_data++) = *(IVS_fc_inter_data++);
    }
  }

}

template <typename Dtype>
void FcIVSLayer<Dtype>::forward_cpu_IVS_serial(const CBLAS_TRANSPOSE TransB,
      const int M, const int N, const int K,
      const Dtype* A, const Dtype* B, Dtype* C){
  
  const Dtype* input_data;
  Dtype* IVS_fc_inter_data;
  const Dtype* weight;
  Dtype* output_data;
  output_data = C;
  for(int batch_iter = 0; batch_iter<M; batch_iter++){
    weight = B;
    IVS_fc_inter_data = this->IVS_fc_inter_.mutable_cpu_data();
    for(int weight_column=0; weight_column<N; weight_column++){
      input_data = A + batch_iter * K;
      for(int weight_row=0; weight_row<K; weight_row++){
        if(TransB!=CblasNoTrans)
          *(IVS_fc_inter_data + weight_row * N + weight_column) = *(input_data++) * *(weight++);
        else
          *(IVS_fc_inter_data + weight_row * N + weight_column) = *(input_data++) * *(weight++ + weight_row * N + weight_column);
      }
    }
    if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_MULTIPLY\
      || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
      this->Analyze_multiply(this->IVS_fc_inter_.cpu_data(),this->IVS_fc_inter_.count());
    }
    this->QuantizeIVS_conv_inter_multiply_cpu(this->IVS_fc_inter_.mutable_cpu_data(),
      this->IVS_fc_inter_.count());
    
    //serial accumulator
    for (int IVS_fc_inter_col_left = 1; IVS_fc_inter_col_left < K ; IVS_fc_inter_col_left ++){
      IVS_fc_inter_data = this->IVS_fc_inter_.mutable_cpu_data();
      for (int fc_inter_row = 0 ; fc_inter_row < N ; fc_inter_row++){
        *(IVS_fc_inter_data) += *(IVS_fc_inter_data + N * IVS_fc_inter_col_left);
        IVS_fc_inter_data++;
      }
      if(this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_ADD\
        || this->analyze_mode_ == QuantizationParameter_Analyze_mode_ANALYZE_BOTH){
        this->Analyze_add(this->IVS_fc_inter_.cpu_data(), N);
      }
      this->QuantizeIVS_conv_inter_add_cpu(this->IVS_fc_inter_.mutable_cpu_data(), N);
    }
    IVS_fc_inter_data = this->IVS_fc_inter_.mutable_cpu_data();
    for (int output_row = 0 ;output_row < N ;output_row++){
      *(output_data++) = *(IVS_fc_inter_data++);
    }
  }
}
#ifdef CPU_ONLY
STUB_GPU(FcIVSLayer);
#endif

INSTANTIATE_CLASS(FcIVSLayer);
REGISTER_LAYER_CLASS(FcIVS);

}  // namespace caffe
