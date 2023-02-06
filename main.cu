#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
#include <assert.h>
#include <cstdio>

#define BLOCK_SIZE 128
#define SMALL_BLOCK_SIZE 32
#define WARP_SIZE 32

#define ADD_TIME(code) do { \
    auto start = std::chrono::system_clock::now(); \
    code \
    auto end   = std::chrono::system_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
    time += double(duration.count());\
} while(0)

__global__ void reductionKernel(const int n, float *p, float *q, float *res){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = index / WARP_SIZE;
    int lane_id = index % WARP_SIZE;
    const unsigned int FULL_MASK = 0xffffffff;
    if(warp_id < n){
        float sum = 0.f;
        for(int i = lane_id; i < n; i += WARP_SIZE){
            sum += p[warp_id * n + i] * q[warp_id * n + i];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
        if(lane_id == 0){
            res[warp_id] = sum;
        }
    }
}

float reduce(int n, float *p, float *q){
    dim3 dim_block(BLOCK_SIZE);
    dim3 dim_grid((n + (BLOCK_SIZE / WARP_SIZE)- 1) / (BLOCK_SIZE / WARP_SIZE));
    float *res_host = new float [n];
    float *res_device;
    cudaMalloc(&res_device, n * sizeof(float));
    reductionKernel<<<dim_grid, dim_block>>>(n, p, q, res_device);
    cudaDeviceSynchronize();
    cudaMemcpy(res_host, res_device, n * sizeof(float), cudaMemcpyDeviceToHost);
    float res = 0.f;
    for(int i = 0; i < n ; i ++){
        res += res_host[i];
    }
    cudaFree(res_device);
    delete[] res_host;
    return res;
}

__global__ void update_x(int size, float *x, const float *p, const float alpha){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < size; i += stride){
        x[i] += alpha * p[i];
    }
}

__global__ void update_r(int size, float *r, const float *Ap, const float alpha){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < size; i += stride){
        r[i] -= alpha * Ap[i];
    }
}

__global__ void update_p(int size, const float *r, float *p, const float beta){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < size; i += stride){
        p[i] = r[i] + beta * p[i];
    }
}

__global__ void check_solution(int n, float *Ax, const float *x, const float *b, float *c){
#define Ax(i, j) Ax[(i) * n + (j)]
#define x(i, j) x[(i) * n + (j)]
#define b(i, j) b[(i) * n + (j)]
#define c(i, j) c[(i) * n + (j)]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= n || j >= n){
        Ax(i, j) = 0.f;
        return;
    }
    float res = 4.0 * x(i, j);
    if(i > 0){
        res -= x(i - 1, j);
    }
    if(i <= n){
        res -= x(i + 1, j);
    }
    if(j > 0){
        res -= x(i, j - 1);
    }
    if(j <= n){
        res -= x(i, j + 1);
    }
    Ax(i, j) = res;
    c(i, j) = b(i, j) - res;
#undef Ax
#undef x
#undef b
#undef c
}

__global__ void compute_Ap(int n, float *p, float *Ap){
#define Ap(i, j) Ap[(i) * n + (j)]
#define p(i, j) p[(i) * n + (j)]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float res = 0.f;
    res = 4.0 * p(i, j);
    if(i > 0){
        res -= p(i - 1, j);
    }
    if(i <= n){
        res -= p(i + 1, j);
    }
    if(j > 0){
        res -= p(i, j - 1);
    }
    if(j <= n){
        res -= p(i, j + 1);
    }
    Ap(i, j) = res;
#undef Ap
#undef p
}

float B[2048 * 2048];
float X[2048 * 2048];

void cgSolver(int n, float eps, float *r, float *b, float *x,float *p, float *Ap, float *Ax, float *c){
    int size = n * n;
    float alpha = 0.f, beta = 0.f;
    float initial_rTr = reduce(n, r, r);
    printf(">>> Initial residual = %f\n", sqrt(initial_rTr));
    float old_rTr = initial_rTr;
    update_p<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(size, r, p, beta);
    cudaDeviceSynchronize();
    float pTp = reduce(n, p, p);
    printf(">>> pTp = %f\n", pTp);
    for(int i = 0; i < size; i ++){
        dim3 dimBlock(SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE);
        dim3 dimGrid((n + SMALL_BLOCK_SIZE - 1) / SMALL_BLOCK_SIZE, (n + SMALL_BLOCK_SIZE - 1) / SMALL_BLOCK_SIZE);

        compute_Ap<<<dimGrid, dimBlock>>>(n, p, Ap);
        cudaDeviceSynchronize();


        float pAp = reduce(n, p, Ap);
        alpha = old_rTr / pAp;
        update_x<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(size, x, p, alpha);
        cudaDeviceSynchronize();

        update_r<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(size, r, Ap, alpha);

        cudaDeviceSynchronize();

        float new_rTr = reduce(n, r, r);
        if (sqrt(new_rTr) < eps){
            printf(">>> Conjugate Gradient method converged at time %d.\n", i + 1);
            break;
        }
        beta = new_rTr / old_rTr;
        update_p<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(size, r, p, beta);
        cudaDeviceSynchronize();

        old_rTr = new_rTr;
    }

    dim3 dimBlock(SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE);
    dim3 dimGrid((n + SMALL_BLOCK_SIZE - 1) / SMALL_BLOCK_SIZE, (n + SMALL_BLOCK_SIZE - 1) / SMALL_BLOCK_SIZE);
    check_solution<<<dimGrid, dimBlock >>>(n, Ax, x, b, c);
    cudaDeviceSynchronize();
    float  residual_cg = reduce(n, c, c);

    printf(">>> Checking the residual norm(Ax-b)...\n");
    printf(">>> Residual CGPoissonSolver: %f\n",sqrt(residual_cg));
    assert(residual_cg < eps);

    cudaMemcpy(X, x, size * sizeof(float), cudaMemcpyDeviceToHost);
}

float eps = 1e-4;
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
            float *r, *b, *x, *p, *Ap, *Ax, *c;
            cudaMalloc(&r, size * sizeof(float));
            cudaMalloc(&b, size * sizeof(float));
            cudaMalloc(&x, size * sizeof(float));
            cudaMalloc(&p, size * sizeof(float));
            cudaMalloc(&Ap, size * sizeof(float));
            cudaMalloc(&Ax, size * sizeof(float));
            cudaMalloc(&c, size * sizeof(float));
            cudaMemcpy(b, B, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(r, B, size * sizeof(float), cudaMemcpyHostToDevice);

            ADD_TIME(
                    cgSolver(p_size, eps, r, b, x, p, Ap, Ax, c);
                    );

            cudaFree(r);
            cudaFree(b);
            cudaFree(x);
            cudaFree(p);
            cudaFree(Ap);
            cudaFree(Ax);
            cudaFree(c);

            std::string output_name = "ans_" + std::to_string(i) + "_" + std::to_string(p_size) + "_" + std::to_string(j) + ".bin";
            std::ofstream ofs(output_name, std::ios::binary | std::ios::out);
            ofs.write((const char*)X, sizeof(float) * p_size * p_size);
            ofs.close();
        }
        printf("*** Average kernel time: %lf ms\n",time / 5000.0);
    }
}