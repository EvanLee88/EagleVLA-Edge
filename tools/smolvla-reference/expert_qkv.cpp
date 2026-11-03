#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
static bool read(const char * path, std::vector<float> & data) {
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    return bool(in);
}
static bool write(const char * path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    return bool(out);
}
int main(int argc, char ** argv) {
    if (argc != 10) { std::fprintf(stderr, "usage: %s MODEL LAYER HIDDEN CACHE_K CACHE_V NTOK Q K V\n", argv[0]); return 2; }
    const int layer=std::atoi(argv[2]), count=std::atoi(argv[6]);
    if (layer < 0 || layer >= 16 || count < 1 || count > 1024) return 2;
    std::vector<float> hidden(50*720), ck((size_t)count*320), cv(ck.size()), q,k,v;
    if (!read(argv[3], hidden) || !read(argv[4], ck) || !read(argv[5], cv)) return 3;
    std::string error;
    auto model=smolvla_expert_mlp::open(argv[1],layer,4,error);
    if (!model || !model->project_qkv(hidden.data(),ck.data(),cv.data(),count,q,k,v,error)) { std::fprintf(stderr,"%s\n",error.c_str()); return 4; }
    return write(argv[7],q) && write(argv[8],k) && write(argv[9],v) ? 0 : 5;
}
