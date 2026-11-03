#include "jetson_pi_smolvla.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
static bool read(const char * path,void * dst,size_t size){std::ifstream f(path,std::ios::binary);f.read(static_cast<char *>(dst),size);return bool(f);}
int main(int argc,char **argv){
 if(argc!=6){std::fprintf(stderr,"usage: %s MODEL IMAGE_EMB TOKENS STATE OUTPUT\n",argv[0]);return 2;}
 std::vector<float> images(3*64*960),state(32),projected,output;std::vector<int32_t> tokens(48);
 if(!read(argv[2],images.data(),images.size()*sizeof(float))||!read(argv[3],tokens.data(),tokens.size()*sizeof(int32_t))||!read(argv[4],state.data(),state.size()*sizeof(float)))return 3;
 std::string error;auto prefix=smolvla_prefix::open(argv[1],4,error);auto action=smolvla_action_input::open(argv[1],4,error);
 std::vector<std::vector<float>> frames(3);
 for(size_t i=0;i<3;++i) frames[i].assign(images.begin()+i*64*960,images.begin()+(i+1)*64*960);
 std::vector<uint8_t> valid(48,1),padding,attention;std::vector<int32_t> positions;
 if(!prefix||!action||!action->project_state(state.data(),projected,error)||!prefix->assemble(frames,tokens,valid,projected,output,padding,attention,positions,error)){std::fprintf(stderr,"%s\n",error.c_str());return 4;}
 std::ofstream f(argv[5],std::ios::binary);f.write(reinterpret_cast<const char *>(output.data()),output.size()*sizeof(float));return f?0:5;
}
