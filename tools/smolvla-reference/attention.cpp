#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 8) { std::fprintf(stderr,"usage: %s NQ NK QUERY KEY VALUE MASK OUTPUT\n", argv[0]); return 2; }
    int nq=std::atoi(argv[1]), nk=std::atoi(argv[2]); if (nq<1 || nq>1024 || nk<1 || nk>1024) return 2;
    std::vector<float> q((size_t)nq*15*64), k((size_t)nk*5*64), v(k.size()), output;
    std::vector<uint8_t> mask((size_t)nq*nk);
    std::ifstream a(argv[3],std::ios::binary), b(argv[4],std::ios::binary), c(argv[5],std::ios::binary), d(argv[6],std::ios::binary);
    a.read(reinterpret_cast<char *>(q.data()),q.size()*sizeof(float)); b.read(reinterpret_cast<char *>(k.data()),k.size()*sizeof(float));
    c.read(reinterpret_cast<char *>(v.data()),v.size()*sizeof(float)); d.read(reinterpret_cast<char *>(mask.data()),mask.size());
    if (!a || !b || !c || !d) return 3;
    std::string error;
    if (!smolvla::attention(q.data(),k.data(),v.data(),nq,nk,mask.data(),4,output,error)) { std::fprintf(stderr,"%s\n",error.c_str()); return 4; }
    std::ofstream f(argv[7],std::ios::binary); f.write(reinterpret_cast<const char *>(output.data()),output.size()*sizeof(float));
    return f ? 0 : 5;
}
