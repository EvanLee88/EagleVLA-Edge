#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
static bool read(const char * path, void * data, size_t size) { std::ifstream f(path,std::ios::binary); f.read(static_cast<char *>(data),size); return bool(f); }
int main(int argc,char **argv) {
    if(argc!=8) { std::fprintf(stderr,"usage: %s MODEL SUFFIX KEYS VALUES MASK NTOK OUTPUT\n",argv[0]); return 2; }
    int n=std::atoi(argv[6]); if(n<1 || n>1024) return 2;
    std::vector<float> suffix(50*720), all_k((size_t)16*n*320), all_v(all_k.size()), output;
    std::vector<uint8_t> mask(n);
    if(!read(argv[2],suffix.data(),suffix.size()*4)||!read(argv[3],all_k.data(),all_k.size()*4)||!read(argv[4],all_v.data(),all_v.size()*4)||!read(argv[5],mask.data(),mask.size())) return 3;
    std::vector<std::vector<float>> k(16),v(16);
    for(size_t i=0;i<16;++i) { k[i].assign(all_k.begin()+i*n*320,all_k.begin()+(i+1)*n*320); v[i].assign(all_v.begin()+i*n*320,all_v.begin()+(i+1)*n*320); }
    std::string error; auto model=smolvla_expert::open(argv[1],4,error);
    if(!model || !model->forward(suffix.data(),k,v,mask,output,error)) { std::fprintf(stderr,"%s\n",error.c_str()); return 4; }
    std::ofstream f(argv[7],std::ios::binary); f.write(reinterpret_cast<const char *>(output.data()),output.size()*4); return f?0:5;
}
