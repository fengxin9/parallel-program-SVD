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
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    double sum = 0.0;
    for (double x : v)
        sum += x * x;
    return std::sqrt(sum);
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

    // 创建 cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配需要的临时空间
    double *d_B_temp, *d_U_temp,*d_V_temp, *d_v, *d_w, *d_wU;
    cudaMalloc(&d_B_temp, m * n * sizeof(double));      // 最大 B 子矩阵
    cudaMalloc(&d_U_temp, m * m * sizeof(double));      // 最大 U 子矩阵
    cudaMalloc(&d_V_temp, n * n * sizeof(double));      // 最大 V 子矩阵
    cudaMalloc(&d_v, m * sizeof(double));               // 最大 v 向量长度
    cudaMalloc(&d_w, n * sizeof(double));               // 最大 w 向量长度
    cudaMalloc(&d_wU, m * sizeof(double));              // 最大 wU 向量长度

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            // sign(x[0])：此处规定 x[0]==0 时取 +1
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            // 实际上这里是+或者-都可以，手册里 Householder 一节是 -αe_1
            // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
            std::vector<double> v(x);
            v[0] += sigma; // v = x + sigma * e_1

            // 计算 v^T v
            double vTv = 0.0;
            for (double vi : v)
                vTv += vi * vi;

            // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                int sub_m = m - k;
                int sub_n = n - k;

                // 将 B_sub 转置为列主序（在 CPU 上转置）
                std::vector<double> B_col(sub_m * sub_n);
                for (int i = 0; i < sub_m; ++i) {
                    for (int j = 0; j < sub_n; ++j) {
                        B_col[j * sub_m + i] = B.at(k + i, k + j);
                    }
                }

                // 拷贝转置后的数据
                cudaMemcpy(d_B_temp, B_col.data(), sub_m * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_v, v.data(), sub_m * sizeof(double), cudaMemcpyHostToDevice);

                double alpha = 1.0;
                double beta_cublas = 0.0;

                // w = v^T * B_sub
                cublasDgemv(handle, CUBLAS_OP_T, sub_m, sub_n, &alpha, d_B_temp, sub_m, d_v, 1, &beta_cublas, d_w, 1);
                cudaDeviceSynchronize();

                // 更新 B_sub = B_sub - beta * v * w^T
                alpha = -beta;
                cublasDger(handle, sub_m, sub_n, &alpha, d_v, 1, d_w, 1, d_B_temp, sub_m);
                cudaDeviceSynchronize();

                // 拷回并转置回来
                std::vector<double> B_new_col(sub_m * sub_n);
                cudaMemcpy(B_new_col.data(), d_B_temp, sub_m * sub_n * sizeof(double), cudaMemcpyDeviceToHost);

                for (int i = 0; i < sub_m; ++i) {
                    for (int j = 0; j < sub_n; ++j) {
                        B.at(k + i, k + j) = B_new_col[j * sub_m + i];
                    }
                }

                // 累积 U：U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                int u_sub_n = sub_m;
                std::vector<double> U_col(m * u_sub_n);
                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < u_sub_n; ++j) {
                        U_col[j * m + i] = U.at(i, k + j);
                    }
                }

                cudaMemcpy(d_U_temp, U_col.data(), m * u_sub_n * sizeof(double), cudaMemcpyHostToDevice);

                alpha = 1.0;
                beta_cublas = 0.0;
                cublasDgemv(handle, CUBLAS_OP_N, m, u_sub_n, &alpha, d_U_temp, m, d_v, 1, &beta_cublas, d_wU, 1);
                cudaDeviceSynchronize();

                alpha = -beta;
                cublasDger(handle, m, u_sub_n, &alpha, d_wU, 1, d_v, 1, d_U_temp, m);
                cudaDeviceSynchronize();

                std::vector<double> U_new_col(m * u_sub_n);
                cudaMemcpy(U_new_col.data(), d_U_temp, m * u_sub_n * sizeof(double), cudaMemcpyDeviceToHost);

                for (int i = 0; i < m; ++i) {
                    for (int j = 0; j < u_sub_n; ++j) {
                        U.at(i, k + j) = U_new_col[j * m + i];
                    }
                }
            }
        }

        // 清除第 k 列中对角线以下的元素
        // 理论上应为 0，但不能完全保证全是 0，这里强制置零
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        //        （只在 k < n-2 时需要）
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            // 与之前类似，计算模长
            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                // 构造 Householder 向量 v = y + sigma * e_1
                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = 0.0;
                for (double vi : v)
                    vTv += vi * vi;

                // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    int sub_m = m - k;
                    int sub_n = n - k - 1;

                    // 将 B_sub 转置为列主序
                    std::vector<double> B_col(sub_m * sub_n);
                    for (int i = 0; i < sub_m; ++i) {
                        for (int j = 0; j < sub_n; ++j) {
                            B_col[j * sub_m + i] = B.at(k + i, k + 1 + j);
                        }
                    }

                    cudaMemcpy(d_B_temp, B_col.data(), sub_m * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_v, v.data(), sub_n * sizeof(double), cudaMemcpyHostToDevice);

                    double alpha = 1.0;
                    double beta_cublas = 0.0;

                    // w = B_sub * v，B 是列主序，lda = sub_m
                    cublasDgemv(handle, CUBLAS_OP_N, sub_m, sub_n, &alpha, d_B_temp, sub_m, d_v, 1, &beta_cublas, d_w, 1);
                    cudaDeviceSynchronize();

                    // 更新 B_sub = B_sub - beta * w * v^T
                    alpha = -beta;
                    cublasDger(handle, sub_m, sub_n, &alpha, d_w, 1, d_v, 1, d_B_temp, sub_m);
                    cudaDeviceSynchronize();

                    // 拷回并转置
                    std::vector<double> B_new_col(sub_m * sub_n);
                    cudaMemcpy(B_new_col.data(), d_B_temp, sub_m * sub_n * sizeof(double), cudaMemcpyDeviceToHost);

                    for (int i = 0; i < sub_m; ++i) {
                        for (int j = 0; j < sub_n; ++j) {
                            B.at(k + i, k + 1 + j) = B_new_col[j * sub_m + i];
                        }
                    }

                    // V_sub 转置
                    std::vector<double> V_col(n * sub_n);
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < sub_n; ++j) {
                            V_col[j * n + i] = V.at(i, k + 1 + j);
                        }
                    }
                    cudaMemcpy(d_V_temp, V_col.data(), n * sub_n * sizeof(double), cudaMemcpyHostToDevice);

                    // wV = V_sub * v
                    alpha = 1.0;
                    cublasDgemv(handle, CUBLAS_OP_N, n, sub_n, &alpha, d_V_temp, n, d_v, 1, &beta_cublas, d_wU, 1);
                    cudaDeviceSynchronize();

                    // 更新 V_sub
                    alpha = -beta;
                    cublasDger(handle, n, sub_n, &alpha, d_wU, 1, d_v, 1, d_V_temp, n);
                    cudaDeviceSynchronize();

                    // 拷回并转置
                    std::vector<double> V_new_col(n * sub_n);
                    cudaMemcpy(V_new_col.data(), d_V_temp, n * sub_n * sizeof(double), cudaMemcpyDeviceToHost);

                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < sub_n; ++j) {
                            V.at(i, k + 1 + j) = V_new_col[j * n + i];
                        }
                    }
                }
            }

            // 强制置零
            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    // 销毁 cuBLAS handle，释放临时空间
    cublasDestroy(handle);
    cudaFree(d_B_temp);
    cudaFree(d_U_temp);
    cudaFree(d_V_temp);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_wU);

    return B;
}