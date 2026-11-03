#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
static bool read(const char * path, void * data, size_t bytes) { std::ifstream f(path,std::ios::binary); f.read(static_cast<char *>(data),bytes); return bool(f); }
int main(int argc,char **argv) {
    if(argc != 9) { std::fprintf(stderr,"usage: %s MODEL LAYER HIDDEN PREFIX_K PREFIX_V MASK NTOK OUTPUT\n",argv[0]); return 2; }
    int layer=std::atoi(argv[2]), n=std::atoi(argv[7]); if(n<1 || n>1024) return 2;
    std::vector<float> h(50*720),k((size_t)n*320),v(k.size()),out; std::vector<uint8_t> mask(n);
    if(!read(argv[3],h.data(),h.size()*4)||!read(argv[4],k.data(),k.size()*4)||!read(argv[5],v.data(),v.size()*4)||!read(argv[6],mask.data(),mask.size())) return 3;
    std::string error; auto model=smolvla_expert_mlp::open(argv[1],layer,4,error);
    if(!model || !model->forward(h.data(),k.data(),v.data(),mask,out,error)) { std::fprintf(stderr,"%s\n",error.c_str()); return 4; }
    std::ofstream f(argv[8],std::ios::binary); f.write(reinterpret_cast<const char *>(out.data()),out.size()*sizeof(float)); return f?0:5;
}
