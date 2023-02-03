#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <cmath>
#include <assert.h>
#include <cstdio>

#define BLOCK_SIZE 64

#define ADD_TIME(code) do { \
    auto start = std::chrono::system_clock::now(); \
    code \
    auto end   = std::chrono::system_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
    time += double(duration.count());\
} while(0)

__global__ void compute_Ap(int n, const float *p, float *Ap){
#define Ap(i, j) Ap[(i) * n + j]
#define p(i, j) p[(i) * n + j]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n){
        Ap(i, j) = 0.f;
        return;
    }
    Ap(i, j) = 4.0 * p(i, j) - p(i - 1,j) - p(i + 1, j) - p(i, j - 1) - p(i, j + 1);
#undef Ap
#undef p
}

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val += __shfl_down(val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    static __shared__ int shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void deviceReduceKernelStep1(int n, float *p, float *q, float *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0;
    for (int i = index; i < n; i += stride) {
        sum += p[i] * q[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        output[blockIdx.x] = sum;
}
__global__ void deviceReduceKernelStep2(int n, float *p, float *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0;
    for (int i = index; i < n; i += stride) {
        sum += p[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        output[blockIdx.x] = sum;
}

float reduce(int n, float *p, float *q){
//    int blocks = std::min((n * n + BLOCK_SIZE - 1)/ BLOCK_SIZE, 1024);
//    auto output = new float[1024];
//    deviceReduceKernelStep1<<<blocks, BLOCK_SIZE>>>(n * n, p, q, output);
//    deviceReduceKernelStep2<<<1, 1024>>>(n * n, output, output);
//    return output[0];
    float ans = 0.f;
    for(int i = 0; i < n * n; i ++){
        ans += p[i] * q[i];
    }
    return ans;
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

__global__ void check_solution(int n, float *Ax, const float *x, const float *b, float *residual){
#define Ax(i, j) Ax[(i) * n + (j)]
#define x(i, j) x[(i) * n + (j)]
#define b(i, j) b[(i) * n + (j)]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= n || j >= n){
        Ax(i, j) = 0.f;
    }
    Ax(i, j) = 4.0 * x(i, j) - x(i - 1, j) - x(i + 1, j) - x(i, j - 1) - x(i, j + 1);
    *residual += (b(i, j) - Ax(i, j)) * (b(i, j) - Ax(i, j));
#undef Ax
#undef x
#undef b
}

float B[2048 * 2048];
float X[2048 * 2048];

void cgSolver(int n, float eps, float *r, float *b, float *x,float *p, float *Ap, float *Ax){
    int size = n * n;
    float alpha = 0.f, beta = 0.f;
    float initial_rTr = reduce(n, r, r);
    printf(">>> Initial residual = %f\n", initial_rTr);
    float old_rTr = initial_rTr;
    update_p<<<n * n / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, p, beta);


    for(int i = 0; i < size; i ++){
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        compute_Ap<<< dimGrid, dimBlock >>>(n, p, Ap);

        cudaDeviceSynchronize();

        float pAp = reduce(n, p, Ap);
        alpha = old_rTr / pAp;
        update_x<<<(n * n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n, x, p, alpha);

        cudaDeviceSynchronize();

        update_r<<<(n * n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, p, alpha);

        cudaDeviceSynchronize();

        float new_rTr = reduce(n, r, r);

        if (sqrt(new_rTr) < eps){
            printf(">>> Conjugate Gradient method converged at time %d.\n", i + 1);
            break;
        }
        beta = new_rTr / old_rTr;
        update_p<<<(n * n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(n, r, p, beta);
        old_rTr = new_rTr;
    }

    float residual_cg = 0.f;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    check_solution<<<dimGrid, dimBlock >>>(n, Ax, x, b, &residual_cg);

    cudaDeviceSynchronize();

    printf(">>> Checking the residual norm(Ax-b)...\n");
    printf(">>> Residual CGPoissonSolver: %f\n",sqrt(residual_cg));
    assert(residual_cg < eps);

    cudaMemcpy(X, x, size * sizeof(float), cudaMemcpyDeviceToHost);
}

float eps = 1e-8;
int problem_size[5] = {0,256, 512, 1024, 2048};
int repeats = 5;

int main() {
    for(int i = 1; i <= 4; i ++){
        printf("\n>>> Current problem size: %d x %d\n", problem_size[i], problem_size[i]);
        double time = 0.0;
        for(int j = 1; j <= 5; j ++){
            printf(">>> Solving Poisson\'s equation using CG [%d/%d]\n", j, 5);
            int p_size = problem_size[i];
            std::string input_name = "b_" + std::to_string(i) + "_" + std::to_string(p_size) + "_" + std::to_string(j) + ".bin";
            std::ifstream ifs(input_name, std::ios::binary | std::ios::in);
            ifs.read((char *)B, sizeof(float) * p_size * p_size);
            ifs.close();

            int size = p_size * p_size;
            float *r, *b, *x, *p, *Ap, *Ax;
            cudaMalloc(&r, size * sizeof(float));
            cudaMalloc(&b, size * sizeof(float));
            cudaMalloc(&x, size * sizeof(float));
            cudaMalloc(&p, size * sizeof(float));
            cudaMalloc(&Ap, size * sizeof(float));
            cudaMalloc(&Ax, size * sizeof(float));
            cudaMemset(&x, 0, size * sizeof(float));
            cudaMemset(&p, 0, size * sizeof(float));
            cudaMemset(&Ap, 0, size * sizeof(float));
            cudaMemset(&Ax, 0, size * sizeof(float));

            cudaMemcpy(b, B, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(r, B, size * sizeof(float), cudaMemcpyHostToDevice);

            ADD_TIME(
                    cgSolver(p_size, eps, r, b, x, p, Ap, Ax);
                    );

            cudaFree(r);
            cudaFree(b);
            cudaFree(x);
            cudaFree(p);
            cudaFree(Ap);
            cudaFree(Ax);

            std::string output_name = "ans_" + std::to_string(i) + "_" + std::to_string(p_size) + "_" + std::to_string(j) + ".bin";
            std::ofstream ofs(output_name, std::ios::binary | std::ios::out);
            ofs.write((const char*)X, sizeof(float) * p_size * p_size);
            ofs.close();
        }
        printf("*** Average kernel time: %lf ms\n",time / 5000.0);
    }
}