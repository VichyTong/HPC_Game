#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <cmath>

#define PRINT_TIME(code) do { \
    auto start = std::chrono::system_clock::now(); \
    code \
    auto end   = std::chrono::system_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
    std::cout << "time spent: " << double(duration.count()) << "us" << std::endl; \
} while(0)

__global__ void compute_Ap(int n, const float *p, float *Ap){
#define Ap(i, j) Ap[(i) * n + j]
#define p(i, j) p[(i) * n + j]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || j <= 0 || i > n || j > n){
        Ap(i, j) = 0.f;
        return;
    }
    Ap(i, j) = 4.0 * p(i, j) - p(i - 1,j) - p(i + 1, j) - p(i, j - 1) - p(i, j + 1);
#undef Ap
#undef p
}

__global__ void reduce(int n, const float *p, const float *q, float *result){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float res = 0.f;
    for(int i = index; i < n; i += stride){
        res += p[i] * q[i];
    }
    *result += res;
}

__global__ void update_x(int n, float *x, const float *p, const float alpha){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i += stride){
        x[i] += alpha * p[i];
    }
}

__global__ void update_r(int n, float *r, const float *p, const float alpha){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i += stride){
        r[i] -= alpha * p[i];
    }
}

__global__ void update_p(int n, const float *r, float *p, const float beta){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i += stride){
        p[i] = r[i] + beta * p[i];
    }
}

float B[2048 * 2048];
float X[2048 * 2048];

#define BLOCK_SIZE 32
void cgSolver(int n, float eps){
    int size = n * n;

    float *r, *b, *x, *p, *Ap, *Ax;
    float alpha, beta;
    cudaMalloc(&r, size * sizeof(float));
    cudaMalloc(&b, size * sizeof(float));
    cudaMalloc(&x, size * sizeof(float));
    cudaMalloc(&p, size * sizeof(float));
    cudaMalloc(&Ap, size * sizeof(float));
    cudaMalloc(&Ax, size * sizeof(float));

    cudaMemcpy(b, B, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(r, B, size * sizeof(float), cudaMemcpyHostToDevice);

    float initial_rTr = 0.f;
    reduce<<<n * n / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, r, &initial_rTr);
    float old_rTr = initial_rTr;
    update_p<<<n * n / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, p, beta);

    for(int i = 0; i < size; i ++){
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE);
        compute_Ap<<< dimGrid, dimBlock >>>(n, p, Ap);
        float pAp = 0.f;
        reduce<<<n * n / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, r, &pAp);
        alpha = old_rTr / pAp;
        update_x<<<n * n / BLOCK_SIZE, BLOCK_SIZE>>>(n, x, p, alpha);
        update_r<<<n * n / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, p, alpha);
        float new_rTr = 0.f;
        reduce<<<n * n / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, r, &new_rTr);
        if (sqrt(new_rTr) < eps){
            break;
        }
        beta = new_rTr / old_rTr;
        update_p<<<n * n / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, p, beta);
        old_rTr = new_rTr;
    }

    cudaMemcpy(X, x, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(r);
    cudaFree(b);
    cudaFree(x);
    cudaFree(p);
    cudaFree(Ap);
    cudaFree(Ax);
}

float eps = 1e-8;
int problem_size[5] = {0,256, 512, 1024, 2048};
int repeats = 5;

int main() {
    for(int i = 1; i <= 4; i ++){
        for(int j = 1; j <= 5; j ++){
            int p_size = problem_size[i];
            std::string input_name = "b_" + std::to_string(i) + "_" + std::to_string(p_size) + "_" + std::to_string(j) + ".bin";
            std::ifstream ifs(input_name, std::ios::binary | std::ios::in);
            ifs.read((char *)B, sizeof(float) * p_size * p_size);
            ifs.close();
            PRINT_TIME(
                    cgSolver(p_size, eps);
                    );
            std::string output_name = "ans_" + std::to_string(i) + "_" + std::to_string(p_size) + "_" + std::to_string(j) + ".bin";
            std::ofstream ofs(output_name, std::ios::binary | std::ios::out);
            ofs.write((const char*)X, sizeof(float) * p_size * p_size);
            ofs.close();
        }
    }
}