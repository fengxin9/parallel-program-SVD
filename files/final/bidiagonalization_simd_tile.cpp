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
#include <arm_neon.h>
#include <omp.h>

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

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            std::vector<double> v(x);
            v[0] += sigma;

            double vTv = 0.0;
            for (double vi : v)
                vTv += vi * vi;

            // 左乘Householder变换（分块）
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;
                const int rows = m - k;
                const int cols = n - k;
                const int BLOCK = 64;
                float64x2_t vbeta_simd = vdupq_n_f64(beta);

                // 1. 计算 w = v^T * B (GEMV)
                std::vector<double> w(cols, 0.0);
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < cols; ++j)
                {
                    float64x2_t sum = vdupq_n_f64(0.0);
                    int i = 0;
                    for (; i + 1 < rows; i += 2)
                    {
                        float64x2_t vi = vld1q_f64(&v[i]);
                        float64x2_t bj = vld1q_f64(&B.at(k + i, k + j));
                        sum = vfmaq_f64(sum, vi, bj);
                    }
                    w[j] = vaddvq_f64(sum);
                    for (; i < rows; ++i)
                    {
                        w[j] += v[i] * B.at(k + i, k + j);
                    }
                }

                // 2. 分块更新 B = B - beta * v * w^T
                #pragma omp parallel for schedule(dynamic)
                for (int bi = 0; bi < rows; bi += BLOCK)
                {
                    int row_end = std::min(bi + BLOCK, rows);
                    for (int bj = 0; bj < cols; bj += BLOCK)
                    {
                        int col_end = std::min(bj + BLOCK, cols);
                        for (int i = bi; i < row_end; ++i)
                        {
                            float64x2_t vi = vdupq_n_f64(v[i]);
                            float64x2_t vbeta_vi = vmulq_f64(vbeta_simd, vi);
                            int j = bj;
                            for (; j + 1 < col_end; j += 2)
                            {
                                float64x2_t wj = vld1q_f64(&w[j]);
                                float64x2_t update = vmulq_f64(vbeta_vi, wj);
                                float64x2_t bval = vld1q_f64(&B.at(k + i, k + j));
                                bval = vsubq_f64(bval, update);
                                vst1q_f64(&B.at(k + i, k + j), bval);
                            }
                            for (; j < col_end; ++j)
                            {
                                B.at(k + i, k + j) -= beta * v[i] * w[j];
                            }
                        }
                    }
                }

                // 3. 计算 wU = U * v (GEMV)
                std::vector<double> wU(m, 0.0);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < m; ++i)
                {
                    float64x2_t sum = vdupq_n_f64(0.0);
                    int j = 0;
                    for (; j + 1 < rows; j += 2)
                    {
                        float64x2_t vj = vld1q_f64(&v[j]);
                        float64x2_t uj = vld1q_f64(&U.at(i, k + j));
                        sum = vfmaq_f64(sum, uj, vj);
                    }
                    wU[i] = vaddvq_f64(sum);
                    for (; j < rows; ++j)
                    {
                        wU[i] += U.at(i, k + j) * v[j];
                    }
                }

                // 4. 分块更新 U = U - beta * wU * v^T
                #pragma omp parallel for schedule(dynamic)
                for (int bi = 0; bi < m; bi += BLOCK)
                {
                    int row_end = std::min(bi + BLOCK, m);
                    for (int bj = 0; bj < rows; bj += BLOCK)
                    {
                        int col_end = std::min(bj + BLOCK, rows);
                        for (int i = bi; i < row_end; ++i)
                        {
                            float64x2_t vbeta_wU = vdupq_n_f64(beta * wU[i]);
                            int j = bj;
                            for (; j + 1 < col_end; j += 2)
                            {
                                float64x2_t vj = vld1q_f64(&v[j]);
                                float64x2_t update = vmulq_f64(vbeta_wU, vj);
                                float64x2_t uval = vld1q_f64(&U.at(i, k + j));
                                uval = vsubq_f64(uval, update);
                                vst1q_f64(&U.at(i, k + j), uval);
                            }
                            for (; j < col_end; ++j)
                            {
                                U.at(i, k + j) -= beta * wU[i] * v[j];
                            }
                        }
                    }
                }
            }
        }

        // 清除第 k 列中对角线以下的元素
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        // ================================================================

        if (k < n - 2)
        {
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = 0.0;
                for (double vi : v)
                    vTv += vi * vi;

                // 右乘Householder变换（分块）
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;
                    const int rows = m - k;
                    const int cols = n - k - 1;
                    const int BLOCK = 64;
                    float64x2_t vbeta_simd = vdupq_n_f64(beta);

                    // 1. 计算 w = B * v (GEMV)
                    std::vector<double> w(rows, 0.0);
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < rows; ++i)
                    {
                        float64x2_t sum = vdupq_n_f64(0.0);
                        int j = 0;
                        for (; j + 1 < cols; j += 2)
                        {
                            float64x2_t vj = vld1q_f64(&v[j]);
                            float64x2_t bj = vld1q_f64(&B.at(k + i, k + 1 + j));
                            sum = vfmaq_f64(sum, bj, vj);
                        }
                        w[i] = vaddvq_f64(sum);
                        for (; j < cols; ++j)
                        {
                            w[i] += B.at(k + i, k + 1 + j) * v[j];
                        }
                    }

                    // 2. 分块更新 B = B - beta * w * v^T
                    #pragma omp parallel for schedule(dynamic, 1)
                    for (int bi = 0; bi < rows; bi += BLOCK)
                    {
                        int row_end = std::min(bi + BLOCK, rows);
                        for (int bj = 0; bj < cols; bj += BLOCK)
                        {
                            int col_end = std::min(bj + BLOCK, cols);
                            for (int i = bi; i < row_end; ++i)
                            {
                                float64x2_t vbeta_w = vdupq_n_f64(beta * w[i]);
                                int j = bj;
                                for (; j + 1 < col_end; j += 2)
                                {
                                    float64x2_t vj = vld1q_f64(&v[j]);
                                    float64x2_t update = vmulq_f64(vbeta_w, vj);
                                    float64x2_t bval = vld1q_f64(&B.at(k + i, k + 1 + j));
                                    bval = vsubq_f64(bval, update);
                                    vst1q_f64(&B.at(k + i, k + 1 + j), bval);
                                }
                                for (; j < col_end; ++j)
                                {
                                    B.at(k + i, k + 1 + j) -= beta * w[i] * v[j];
                                }
                            }
                        }
                    }

                    // 3. 计算 wV = V * v (GEMV)
                    std::vector<double> wV(n, 0.0);
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < n; ++i)
                    {
                        float64x2_t sum = vdupq_n_f64(0.0);
                        int j = 0;
                        for (; j + 1 < cols; j += 2)
                        {
                            float64x2_t vj = vld1q_f64(&v[j]);
                            float64x2_t vv = vld1q_f64(&V.at(i, k + 1 + j));
                            sum = vfmaq_f64(sum, vv, vj);
                        }
                        wV[i] = vaddvq_f64(sum);
                        for (; j < cols; ++j)
                        {
                            wV[i] += V.at(i, k + 1 + j) * v[j];
                        }
                    }

                    // 4. 分块更新 V = V - beta * wV * v^T
                    #pragma omp parallel for schedule(dynamic)
                    for (int bi = 0; bi < n; bi += BLOCK)
                    {
                        int row_end = std::min(bi + BLOCK, n);
                        for (int bj = 0; bj < cols; bj += BLOCK)
                        {
                            int col_end = std::min(bj + BLOCK, cols);
                            for (int i = bi; i < row_end; ++i)
                            {
                                float64x2_t vbeta_wV = vdupq_n_f64(beta * wV[i]);
                                int j = bj;
                                for (; j + 1 < col_end; j += 2)
                                {
                                    float64x2_t vj = vld1q_f64(&v[j]);
                                    float64x2_t update = vmulq_f64(vbeta_wV, vj);
                                    float64x2_t vval = vld1q_f64(&V.at(i, k + 1 + j));
                                    vval = vsubq_f64(vval, update);
                                    vst1q_f64(&V.at(i, k + 1 + j), vval);
                                }
                                for (; j < col_end; ++j)
                                {
                                    V.at(i, k + 1 + j) -= beta * wV[i] * v[j];
                                }
                            }
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

    return B;
}