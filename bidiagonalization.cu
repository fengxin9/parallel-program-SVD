// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
//
// 算法说明（你需要结合代码看）：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步（k = 0, 1, ..., n-1）：
//    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
//    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
//
// 例如，对一个 4x4 矩阵 A，第一步 k=0：
//   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
//   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
//
// 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
//
// 本组件输出：A = U * B * V^T
// 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵

#include "matrix.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    double sum = 0.0;
    for (double x : v)
        sum += x * x;
    return std::sqrt(sum);
}

// 手写 kernel：计算 w = v^T * B_sub（左乘 GEMV）
__global__ void gemv_left_kernel(const double *B, double *w, const double *v, int m, int n, int lda) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        sum += v[i] * B[i + j * lda];
    }
    w[j] = sum;
}

// 手写 kernel：计算 w = B_sub * v（右乘 GEMV）
__global__ void gemv_right_kernel(const double *B, double *w, const double *v, int m, int n, int lda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        sum += B[i + j * lda] * v[j];
    }
    w[i] = sum;
}

// 手写 kernel：更新 B_sub = B_sub - beta * v * w^T（左乘 GER）
__global__ void ger_left_kernel(double *B, const double *v, const double *w, double beta, int m, int n, int lda) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    B[i + j * lda] -= beta * v[i] * w[j];
}

// 手写 kernel：更新 B_sub = B_sub - beta * w * v^T（右乘 GER）
__global__ void ger_right_kernel(double *B, const double *w, const double *v, double beta, int m, int n, int lda) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    B[i + j * lda] -= beta * w[i] * v[j];
}

// 手写 kernel：累积正交矩阵 U 或 V
__global__ void update_ortho_kernel(double *Q, const double *wQ, const double *v, double beta, int m, int n, int lda) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    Q[i + j * lda] -= beta * wQ[i] * v[j];
}

// 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    // U = I_m，V = I_n
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    // 分配 GPU 内存（列主序存储）
    double *d_B, *d_U, *d_V, *d_v, *d_w, *d_wUV;
    cudaMalloc(&d_B, m * n * sizeof(double));
    cudaMalloc(&d_U, m * m * sizeof(double));
    cudaMalloc(&d_V, n * n * sizeof(double));
    cudaMalloc(&d_v, m * sizeof(double));
    cudaMalloc(&d_w, m * sizeof(double));
    cudaMalloc(&d_wUV, m * sizeof(double));

    // 将 B、U、V 转换为列主序并拷贝到 GPU
    std::vector<double> B_col(m * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            B_col[j * m + i] = B.at(i, j);
        }
    }
    cudaMemcpy(d_B, B_col.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> U_col(m * m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            U_col[j * m + i] = U.at(i, j);
        }
    }
    cudaMemcpy(d_U, U_col.data(), m * m * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> V_col(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            V_col[j * n + i] = V.at(i, j);
        }
    }
    cudaMemcpy(d_V, V_col.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    // 配置 kernel 的线程块大小
    dim3 block_size(16, 16);

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i) {
            cudaMemcpy(&x[i], d_B + (k + i) + k * m, sizeof(double), cudaMemcpyDeviceToHost);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;
            std::vector<double> v(x);
            v[0] += sigma;
            double vTv = 0.0;
            for (double vi : v) vTv += vi * vi;

            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                int sub_m = m - k;
                int sub_n = n - k;

                // 子矩阵 B_sub 在列主序 d_B 中的起始地址，lda = m
                double *d_B_sub = d_B + k + k * m;

                // 拷贝 v 到 GPU
                cudaMemcpy(d_v, v.data(), sub_m * sizeof(double), cudaMemcpyHostToDevice);

                // 配置 kernel 参数（左乘）
                dim3 grid_gemv((sub_n + block_size.x - 1) / block_size.x, 1);
                dim3 grid_ger((sub_n + block_size.x - 1) / block_size.x, (sub_m + block_size.y - 1) / block_size.y);

                // w = v^T * B_sub（左乘 GEMV）
                gemv_left_kernel<<<grid_gemv, block_size>>>(d_B_sub, d_w, d_v, sub_m, sub_n, m);
                cudaDeviceSynchronize();

                // B_sub = B_sub - beta * v * w^T（左乘 GER）
                ger_left_kernel<<<grid_ger, block_size>>>(d_B_sub, d_v, d_w, beta, sub_m, sub_n, m);
                cudaDeviceSynchronize();

                // 累积 U：U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                double *d_U_sub = d_U + k * m;
                int u_sub_n = sub_m;

                // 配置 kernel 参数（U 累积）
                dim3 grid_gemv_U((m + block_size.x - 1) / block_size.x, 1);
                dim3 grid_ger_U((u_sub_n + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);

                // wU = U_sub * v（右乘 GEMV）
                gemv_right_kernel<<<grid_gemv_U, block_size>>>(d_U_sub, d_wUV, d_v, m, u_sub_n, m);
                cudaDeviceSynchronize();

                // U_sub = U_sub - beta * wU * v^T（右乘 GER）
                update_ortho_kernel<<<grid_ger_U, block_size>>>(d_U_sub, d_wUV, d_v, beta, m, u_sub_n, m);
                cudaDeviceSynchronize();
            }
        }

        // 清除第 k 列中对角线以下的元素（GPU 上置零）
        for (int i = k + 1; i < m; ++i) {
            double zero = 0.0;
            cudaMemcpy(d_B + i + k * m, &zero, sizeof(double), cudaMemcpyHostToDevice);
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j) {
                cudaMemcpy(&y[j], d_B + k + (k + 1 + j) * m, sizeof(double), cudaMemcpyDeviceToHost);
            }

            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;
                std::vector<double> v(y);
                v[0] += sigma;
                double vTv = 0.0;
                for (double vi : v) vTv += vi * vi;

                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    int sub_m = m - k;
                    int sub_n = n - k - 1;

                    // B_sub 的起始地址：列从 k+1 开始，行从 k 开始
                    double *d_B_sub = d_B + k + (k + 1) * m;

                    // 拷贝 v 到 GPU
                    cudaMemcpy(d_v, v.data(), sub_n * sizeof(double), cudaMemcpyHostToDevice);

                    // 配置 kernel 参数（右乘）
                    dim3 grid_gemv((sub_m + block_size.x - 1) / block_size.x, 1);
                    dim3 grid_ger((sub_n + block_size.x - 1) / block_size.x, (sub_m + block_size.y - 1) / block_size.y);

                    // w = B_sub * v（右乘 GEMV）
                    gemv_right_kernel<<<grid_gemv, block_size>>>(d_B_sub, d_w, d_v, sub_m, sub_n, m);
                    cudaDeviceSynchronize();

                    // B_sub = B_sub - beta * w * v^T（右乘 GER）
                    ger_right_kernel<<<grid_ger, block_size>>>(d_B_sub, d_w, d_v, beta, sub_m, sub_n, m);
                    cudaDeviceSynchronize();

                    // 累积 V：V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    double *d_V_sub = d_V + (k + 1) * n;

                    // 配置 kernel 参数（V 累积）
                    dim3 grid_gemv_V((n + block_size.x - 1) / block_size.x, 1);
                    dim3 grid_ger_V((sub_n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

                    // wV = V_sub * v（右乘 GEMV）
                    gemv_right_kernel<<<grid_gemv_V, block_size>>>(d_V_sub, d_wUV, d_v, n, sub_n, n);
                    cudaDeviceSynchronize();

                    // V_sub = V_sub - beta * wV * v^T（右乘 GER）
                    update_ortho_kernel<<<grid_ger_V, block_size>>>(d_V_sub, d_wUV, d_v, beta, n, sub_n, n);
                    cudaDeviceSynchronize();
                }
            }

            // 强制置零（GPU 上）
            for (int j = k + 2; j < n; ++j) {
                double zero = 0.0;
                cudaMemcpy(d_B + k + j * m, &zero, sizeof(double), cudaMemcpyHostToDevice);
            }
        }
    }

    // 将结果从 GPU 拷回 CPU（列主序转行主序）
    cudaMemcpy(B_col.data(), d_B, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            B.at(i, j) = B_col[j * m + i];
        }
    }

    cudaMemcpy(U_col.data(), d_U, m * m * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            U.at(i, j) = U_col[j * m + i];
        }
    }

    cudaMemcpy(V_col.data(), d_V, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            V.at(i, j) = V_col[j * n + i];
        }
    }

    // 释放 GPU 内存
    cudaFree(d_B);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_wUV);

    return B;
}