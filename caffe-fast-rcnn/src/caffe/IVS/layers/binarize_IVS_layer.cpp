#include <algorithm>
#include <vector>

#include "IVS/binarize_IVS_layer.hpp"

namespace caffe {

template <typename Dtype>
void BinarizeIVSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  //Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  float thresh = this->layer_param_.binarize_param().thresh();
  for (int i = 0; i < count; ++i) {
    if( this->layer_param_.binarize_param().inverse_binarize() == BinarizeParameter_InverseBinarize_FALSE )
    {
      if( this->layer_param_.binarize_param().thresh_condition() == BinarizeParameter_ThreshCondition_INCLUDE_IN_UPPER )
      {
        if( bottom_data[i] >= thresh )
          top_data[i] = 1;
        else
          top_data[i] = 0;
      }
      else
      {
        if( bottom_data[i] > thresh )
          top_data[i] = 1;
        else
          top_data[i] = 0;
      }
    }
    else
    {
      if( this->layer_param_.binarize_param().thresh_condition() == BinarizeParameter_ThreshCondition_INCLUDE_IN_UPPER )
      {
        if( bottom_data[i] < thresh )
          top_data[i] = 1;
        else
          top_data[i] = 0;
      }
      else
      {
        if( bottom_data[i] <= thresh )
          top_data[i] = 1;
        else
          top_data[i] = 0;
      }
    }
  }
}

template <typename Dtype>
void BinarizeIVSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    //const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    //Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i];
      //bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
      //    + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
//STUB_GPU(BinarizeIVSLayer);
#endif

INSTANTIATE_CLASS(BinarizeIVSLayer);
REGISTER_LAYER_CLASS(BinarizeIVS);
}  // namespace caffe
