#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "IVS/base_IVS_layer.hpp"

namespace caffe {

template <typename Dtype>
BaseIVSLayer<Dtype>::BaseIVSLayer() {
  // Initialize random number generator
  srand(time(NULL));
}

template <typename Dtype>
void BaseIVSLayer<Dtype>::QuantizeWeights_cpu(
      vector<shared_ptr<Blob<Dtype> > > weights_quantized, const int rounding,
      const bool bias_term) {
  Dtype* weight = weights_quantized[0]->mutable_cpu_data();
  const int cnt_weight = weights_quantized[0]->count();
  switch (precision_) {
  case QuantizationParameter_Precision_MINIFLOAT:
    Trim2MiniFloat_cpu(weight, cnt_weight, fp_mant_, fp_exp_, rounding);
    if (bias_term) {
      Trim2MiniFloat_cpu(weights_quantized[1]->mutable_cpu_data(),
          weights_quantized[1]->count(), fp_mant_, fp_exp_, rounding);
    }
    break;
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    switch(this->overflow_behavior_){
    case QuantizationParameter_Overflow_behavior_TRIM_AT_THRESH:
      Trim2FixedPoint_cpu(weight, cnt_weight, bw_params_, rounding, fl_params_);
      break;
    case QuantizationParameter_Overflow_behavior_OVERFLOW_SIM:
      Trim2FixedPoint_overflow_cpu(weight, cnt_weight, bw_params_, rounding, fl_params_);
      break;
    }
    if (bias_term) {
      switch(this->overflow_behavior_){
      case QuantizationParameter_Overflow_behavior_TRIM_AT_THRESH:
        Trim2FixedPoint_cpu(weights_quantized[1]->mutable_cpu_data(),
          weights_quantized[1]->count(), bw_bias_, rounding, fl_bias_);
        break;
      case QuantizationParameter_Overflow_behavior_OVERFLOW_SIM:
        Trim2FixedPoint_overflow_cpu(weights_quantized[1]->mutable_cpu_data(),
          weights_quantized[1]->count(), bw_bias_, rounding, fl_bias_);
        break;
      }
      //Trim2FixedPoint_cpu(weights_quantized[1]->mutable_cpu_data(),
      //    weights_quantized[1]->count(), bw_bias_, rounding, fl_bias_);
    }
    break;
  case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
    Trim2IntegerPowerOf2_cpu(weight, cnt_weight, pow_2_min_exp_, pow_2_max_exp_,
        rounding);
    // Don't trim bias
    break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << precision_;
    break;
  }
}

template <typename Dtype>
void BaseIVSLayer<Dtype>::QuantizeLayerInputs_cpu(Dtype* data,
      const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      switch(this->overflow_behavior_){
      case QuantizationParameter_Overflow_behavior_TRIM_AT_THRESH:
        Trim2FixedPoint_cpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
        break;
      case QuantizationParameter_Overflow_behavior_OVERFLOW_SIM:
        Trim2FixedPoint_overflow_cpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
        break;
      }
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
void BaseIVSLayer<Dtype>::QuantizeIVS_conv_inter_add_cpu(Dtype* data,
      const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      switch(this->overflow_behavior_){
      case QuantizationParameter_Overflow_behavior_TRIM_AT_THRESH:
        Trim2FixedPoint_cpu(data, count, bw_add_, rounding_, fl_add_);
        break;
      case QuantizationParameter_Overflow_behavior_OVERFLOW_SIM:
        Trim2FixedPoint_overflow_cpu(data, count, bw_add_, rounding_, fl_add_);
        break;
      }
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
void BaseIVSLayer<Dtype>::QuantizeIVS_conv_inter_multiply_cpu(Dtype* data,
      const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      switch(this->overflow_behavior_){
      case QuantizationParameter_Overflow_behavior_TRIM_AT_THRESH:
        Trim2FixedPoint_cpu(data, count, bw_multiply_, rounding_, fl_multiply_);
        break;
      case QuantizationParameter_Overflow_behavior_OVERFLOW_SIM:
        Trim2FixedPoint_overflow_cpu(data, count, bw_multiply_, rounding_, fl_multiply_);
        break;
      }
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}
template <typename Dtype>
void BaseIVSLayer<Dtype>::QuantizeLayerOutputs_cpu(
      Dtype* data, const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      switch(this->overflow_behavior_){
      case QuantizationParameter_Overflow_behavior_TRIM_AT_THRESH:
        Trim2FixedPoint_cpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
        break;
      case QuantizationParameter_Overflow_behavior_OVERFLOW_SIM:
        Trim2FixedPoint_overflow_cpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
        break;
      }
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
void BaseIVSLayer<Dtype>::Trim2FixedPoint_cpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
  Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
  Dtype temp_pow = pow(2, -fl);
  for (int index = 0; index < cnt; ++index) {
    // Saturate data
    //Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    //Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    //data[index] /= pow(2, -fl);
    data[index] /= temp_pow;
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
    //data[index] *= pow(2, -fl);
    data[index] *= temp_pow;
	}
}

template <typename Dtype>
void BaseIVSLayer<Dtype>::Trim2FixedPoint_overflow_cpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
  Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
  Dtype temp_pow = pow(2, -fl);
  for (int index = 0; index < cnt; ++index) {
    // Saturate data
    //Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    //Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    while(data[index] > max_data)
    {
        data[index] -= (max_data - min_data + 1);
    }
    while(data[index] < min_data)
    {
        data[index] += (max_data - min_data + 1);
    }
    //data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    //data[index] /= pow(2, -fl);
    data[index] /= temp_pow;
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
    //data[index] *= pow(2, -fl);
    data[index] *= temp_pow;
	}
}

typedef union{
  float d;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;

template <typename Dtype>
void BaseIVSLayer<Dtype>::Trim2MiniFloat_cpu(Dtype* data, const int cnt,
      const int bw_mant, const int bw_exp, const int rounding) {
  for (int index = 0; index < cnt; ++index) {
    int bias_out = pow(2, bw_exp - 1) - 1;
    float_cast d2;
    // This casts the input to single precision
    d2.d = (float)data[index];
    int exponent=d2.parts.exponent - 127 + bias_out;
    double mantisa = d2.parts.mantisa;
    // Special case: input is zero or denormalized number
    if (d2.parts.exponent == 0) {
      data[index] = 0;
      return;
    }
    // Special case: denormalized number as output
    if (exponent < 0) {
      data[index] = 0;
      return;
    }
    // Saturation: input float is larger than maximum output float
    int max_exp = pow(2, bw_exp) - 1;
    int max_mant = pow(2, bw_mant) - 1;
    if (exponent > max_exp) {
      exponent = max_exp;
      mantisa = max_mant;
    } else {
      // Convert mantissa from long format to short one. Cut off LSBs.
      double tmp = mantisa / pow(2, 23 - bw_mant);
      switch (rounding) {
      case QuantizationParameter_Rounding_NEAREST:
        mantisa = round(tmp);
        break;
      case QuantizationParameter_Rounding_STOCHASTIC:
        mantisa = floor(tmp + RandUniform_cpu());
        break;
      default:
        break;
      }
    }
    // Assemble result
    data[index] = pow(-1, d2.parts.sign) * ((mantisa + pow(2, bw_mant)) /
        pow(2, bw_mant)) * pow(2, exponent - bias_out);
	}
}

template <typename Dtype>
void BaseIVSLayer<Dtype>::Trim2IntegerPowerOf2_cpu(Dtype* data,
      const int cnt, const int min_exp, const int max_exp, const int rounding) {
	for (int index = 0; index < cnt; ++index) {
    float exponent = log2f((float)fabs(data[index]));
    int sign = data[index] >= 0 ? 1 : -1;
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      exponent = round(exponent);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      exponent = floorf(exponent + RandUniform_cpu());
      break;
    default:
      break;
    }
    exponent = std::max(std::min(exponent, (float)max_exp), (float)min_exp);
    data[index] = sign * pow(2, exponent);
	}
}

template <typename Dtype>
double BaseIVSLayer<Dtype>::RandUniform_cpu(){
  return rand() / (RAND_MAX+1.0);
}

template <typename Dtype>
void BaseIVSLayer<Dtype>::Analyze_multiply(const Dtype* data,
      const int count) {
   for (int i = 0; i < count; i++){
     if(max_multiply_ < *data && *data!= FLT_MAX)
       max_multiply_ = *data;
     if(min_multiply_ > *data && *data!= FLT_MIN)
       min_multiply_ = *data;
     data++;
   }

}
template <typename Dtype>
void BaseIVSLayer<Dtype>::Analyze_add(const Dtype* data,
      const int count) {
   for (int i = 0; i < count; i++){
     if(max_add_ < *data && *data!= FLT_MAX)
       max_add_ = *data;
     if(min_add_ > *data && *data!= FLT_MIN)
       min_add_ = *data;
     data++;
   }

}

template BaseIVSLayer<double>::BaseIVSLayer();
template BaseIVSLayer<float>::BaseIVSLayer();
template void BaseIVSLayer<double>::QuantizeWeights_cpu(
    vector<shared_ptr<Blob<double> > > weights_quantized, const int rounding,
    const bool bias_term);
template void BaseIVSLayer<float>::QuantizeWeights_cpu(
    vector<shared_ptr<Blob<float> > > weights_quantized, const int rounding,
    const bool bias_term);
template void BaseIVSLayer<double>::QuantizeLayerInputs_cpu(double* data,
    const int count);
template void BaseIVSLayer<float>::QuantizeLayerInputs_cpu(float* data,
    const int count);
template void BaseIVSLayer<double>::QuantizeIVS_conv_inter_add_cpu(double* data,
    const int count);
template void BaseIVSLayer<float>::QuantizeIVS_conv_inter_add_cpu(float* data,
    const int count);
template void BaseIVSLayer<double>::QuantizeIVS_conv_inter_multiply_cpu(double* data,
    const int count);
template void BaseIVSLayer<float>::QuantizeIVS_conv_inter_multiply_cpu(float* data,
    const int count);
template void BaseIVSLayer<double>::QuantizeLayerOutputs_cpu(double* data,
    const int count);
template void BaseIVSLayer<float>::QuantizeLayerOutputs_cpu(float* data,
    const int count);
template void BaseIVSLayer<double>::Analyze_add(const double* data,
    const int count);
template void BaseIVSLayer<float>::Analyze_add(const float* data,
    const int count);
template void BaseIVSLayer<double>::Analyze_multiply(const double* data,
    const int count);
template void BaseIVSLayer<float>::Analyze_multiply(const float* data,
    const int count);
template void BaseIVSLayer<double>::Trim2FixedPoint_cpu(double* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseIVSLayer<float>::Trim2FixedPoint_cpu(float* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseIVSLayer<double>::Trim2FixedPoint_overflow_cpu(double* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseIVSLayer<float>::Trim2FixedPoint_overflow_cpu(float* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseIVSLayer<double>::Trim2MiniFloat_cpu(double* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseIVSLayer<float>::Trim2MiniFloat_cpu(float* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseIVSLayer<double>::Trim2IntegerPowerOf2_cpu(double* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);
template void BaseIVSLayer<float>::Trim2IntegerPowerOf2_cpu(float* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);
template double BaseIVSLayer<double>::RandUniform_cpu();
template double BaseIVSLayer<float>::RandUniform_cpu();

}  // namespace caffe
