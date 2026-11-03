#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
static bool read(const char * p,void * dst,size_t size){std::ifstream f(p,std::ios::binary);f.read(static_cast<char *>(dst),size);return bool(f);}
int main(int argc,char **argv){
 if(argc!=8){std::fprintf(stderr,"usage: %s MODEL N PREFIX MASK POS KEYS VALUES\n",argv[0]);return 2;}
 int n=std::atoi(argv[2]);if(n<1||n>1024)return 2;
 std::vector<float> input((size_t)n*960);std::vector<uint8_t> mask((size_t)n*n);std::vector<int32_t> pos(n);
 if(!read(argv[3],input.data(),input.size()*sizeof(float))||!read(argv[4],mask.data(),mask.size())||!read(argv[5],pos.data(),pos.size()*sizeof(int32_t)))return 3;
 std::string error;auto model=smolvla_prefix::open(argv[1],4,error);std::vector<std::vector<float>> keys,values;
 if(!model||!model->forward(input,mask,pos,keys,values,error)){std::fprintf(stderr,"%s\n",error.c_str());return 4;}
 std::ofstream k(argv[6],std::ios::binary),v(argv[7],std::ios::binary);
 for(size_t i=0;i<keys.size();++i){k.write(reinterpret_cast<const char *>(keys[i].data()),keys[i].size()*sizeof(float));v.write(reinterpret_cast<const char *>(values[i].data()),values[i].size()*sizeof(float));}
 return k&&v?0:5;
}
