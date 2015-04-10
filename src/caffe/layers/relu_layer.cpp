#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

static int call_count = 0;

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  bool correct_layer=false;  
  std::string str1("relu7");
  //std::ofstream myfile;
  FILE * myfile;

  if(str1.compare(this->layer_param_.name())==0){
	char *filename = new char[1000];
	//printf("%s\n",this->layer_param_.name().c_str());
	correct_layer=true;
	//myfile.open("example.txt");
	sprintf(filename, "example_%d.txt", call_count);
	myfile= fopen(filename,"w");
  }

  //printf("count = %d\n", count);

  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
    if(correct_layer){
	if(i>0 && (i%4096==0)) {
		fprintf(myfile, "\n");
	}
	fprintf(myfile, "%20.19e ", (double) top_data[i]);
	if(i==0){
		printf("%20.19e\n", top_data[i]);
	}
	//myfile << std::setprecision(20) << top_data[i] << '\n';
    }
  }

  if(correct_layer){
	//myfile.close();
	fclose(myfile);
	call_count++;
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
