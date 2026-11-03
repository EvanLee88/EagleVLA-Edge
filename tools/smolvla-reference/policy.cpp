#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
static bool read(const char * path,void * data,size_t size){std::ifstream f(path,std::ios::binary);f.read(static_cast<char *>(data),size);return bool(f);}
int main(int argc,char **argv){
 if(argc!=10){std::fprintf(stderr,"usage: %s MODEL MMPROJ IMAGE0 IMAGE1 IMAGE2 STATE NOISE TEXT OUTPUT\n",argv[0]);return 2;}
 std::vector<std::vector<uint8_t>> rgb(3,std::vector<uint8_t>(256*256*3));std::vector<smolvla_rgb_image> images;
 for(int i=0;i<3;++i){if(!read(argv[3+i],rgb[i].data(),rgb[i].size()))return 3;images.push_back({rgb[i].data(),256,256});}
 std::vector<float> state(6),noise(50*32),out;
 if(!read(argv[6],state.data(),state.size()*sizeof(float))||!read(argv[7],noise.data(),noise.size()*sizeof(float)))return 3;
 std::string error;auto model=smolvla_policy::open(argv[1],argv[2],4,error);
 if(!model||!model->infer(images,argv[8],state,noise,out,error)){std::fprintf(stderr,"%s\n",error.c_str());return 4;}
 if (std::getenv("SMOLVLA_TEST_REUSE")) {
  const auto first = out;
  auto alternate_noise = noise;
  for (auto & value : alternate_noise) value = -value;
  std::vector<float> alternate;
  if (!model->infer(images,argv[8],state,alternate_noise,alternate,error) ||
      !model->infer(images,argv[8],state,noise,out,error) || out != first) {
   std::fprintf(stderr,"policy graph reuse failed: %s\n",error.c_str()); return 4;
  }
 }
 std::ofstream f(argv[9],std::ios::binary);f.write(reinterpret_cast<const char *>(out.data()),out.size()*sizeof(float));return f?0:5;
}
