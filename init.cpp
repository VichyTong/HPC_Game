#include <iostream>
#include <fstream>
#include <random>

using namespace std;
int problem_size[5] = {0, 4, 512, 1024, 2048};
float b[2048 * 2048];

void randomize(const int size, float *mat){
    random_device dev;
    mt19937 engine(dev());
    uniform_real_distribution<float> distribution(0.f, 1.f);
    for(int i = 0; i < size; i ++){
        mat[i] = distribution(engine);
    }
}

int main(){
    for(int i = 1; i <= 4; i ++){
        int size = problem_size[i];
        for(int j = 1; j <= 5; j ++){
            string file_name = "b_" + to_string(i) + "_" + to_string(size) + "_" + to_string(j) + ".bin";
            randomize(size * size, b);
            ofstream ofs(file_name, ios::binary | ios::out);
            ofs.write((const char*)b, sizeof(float) * size * size);
            ofs.close();
        }
    }
    return 0;
}