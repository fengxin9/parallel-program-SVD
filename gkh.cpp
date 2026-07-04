#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <arm_neon.h>
#include <omp.h>

namespace
{

    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，你可以认为通过这个抽象结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        int cols = M.cols();
        float64x2_t simd_c = vdupq_n_f64(c);
        float64x2_t simd_s = vdupq_n_f64(s);
        float64x2_t neg_simd_s = vnegq_f64(simd_s);
        
        int j = 0;
        for (; j + 1 < cols; j += 2)   // 一次处理2列
        {
            float64x2_t a = vld1q_f64(&M.at(r0, j));
            float64x2_t b = vld1q_f64(&M.at(r1, j));
            
            // new_a = c * a + s * b
            // 使用vmlaq_f64融合乘加指令，先计算simd_s*b，再加simd_c*a
            float64x2_t new_a = vfmaq_f64(vmulq_f64(simd_s, b), simd_c, a);
            
            // new_b = -s * a + c * b
            float64x2_t new_b = vfmaq_f64(vmulq_f64(neg_simd_s, a), simd_c, b);
            
            // 存储结果
            vst1q_f64(&M.at(r0, j), new_a);
            vst1q_f64(&M.at(r1, j), new_b);
        }
     
        for (; j < cols; ++j)    // 处理剩余列
        {
            double a = M.at(r0, j);
            double b = M.at(r1, j);
            M.at(r0, j) = c * a + s * b;
            M.at(r1, j) = -s * a + c * b;
        }
    }

    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    // 即 M <- M * R，其中 R 只作用在第 c0/c1 两列上。
    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        int rows = M.rows();
        int i = 0;
        // 四路循环展开
        for (; i + 3 < rows; i += 4)
        {
            double a0 = M.at(i, c0);
            double b0 = M.at(i, c1);
            M.at(i, c0) = a0 * c - b0 * s;
            M.at(i, c1) = a0 * s + b0 * c;
            
            double a1 = M.at(i + 1, c0);
            double b1 = M.at(i + 1, c1);
            M.at(i + 1, c0) = a1 * c - b1 * s;
            M.at(i + 1, c1) = a1 * s + b1 * c;
            
            double a2 = M.at(i + 2, c0);
            double b2 = M.at(i + 2, c1);
            M.at(i + 2, c0) = a2 * c - b2 * s;
            M.at(i + 2, c1) = a2 * s + b2 * c;
            
            double a3 = M.at(i + 3, c0);
            double b3 = M.at(i + 3, c1);
            M.at(i + 3, c0) = a3 * c - b3 * s;
            M.at(i + 3, c1) = a3 * s + b3 * c;
        }
        
        // 处理剩余行
        for (; i < rows; ++i)
        {
            double a = M.at(i, c0);
            double b = M.at(i, c1);
            M.at(i, c0) = a * c - b * s;
            M.at(i, c1) = a * s + b * c;
        }
    }

    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们B <- L * B 时，我们必须维护的等式是 A = U * B * V^T
        // 如果 A = U * B * V^T 不成立，那么我们最终的SVD结果显然不是 A 的正确分解。
        // 由于正交矩阵和其转置的乘积是I，一个自然的想法是让 U <- U * L^T。
        // 这样就变成 A = (U * L^T) * (L * B) * V^T = U * B * V^T，等式得以保持。

        // 由于 L^T = [c -s; s c]，此处复用“右乘两列”接口并传入 -s。
        apply_right_cols(U, r0, r1, c, -s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // 偏移用于加速 QR 迭代收敛，并让 bulge chasing 过程更稳定。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 将上二对角结构以外、且绝对值很小的元素强制置零。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    // 对活动块 [l, r] 执行一次“单块 GKH bulge chasing”迭代。
    // 流程：首次右乘引入 bulge -> 首次左乘消 bulge -> 交替右乘/左乘将 bulge 追赶到块末端。
    static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘：由 (d_l^2-mu, d_l*e_l) 构造。
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols(B, l, l + 1, c, s);
        apply_right_cols(V, l, l + 1, c, s);

        // 首次左乘：消去 (l+1, l)。
        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows(B, l, l + 1, c, s);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘：消去 (k-1, k+1)
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols(B, k, k + 1, c, s);
            apply_right_cols(V, k, k + 1, c, s);

            // 左乘：消去 (k+1, k)
            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows(B, k, k + 1, c, s);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    // 处理“对角元 d_k 近零但超对角 e_k 未近零”的情况。
    // 思路与单块追赶类似：先右乘把 e_i 消掉，再左乘清理新引入的次对角 bulge，
    // 把这个问题逐步向右传递，直到块末端。
    static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
        // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
        // 然后把问题传递到下一行，直到末端。
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols(B, i, i + 1, c, s);
            apply_right_cols(V, i, i + 1, c, s);

            // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows(B, i, i + 1, c, s);
                accumulate_left_into_U(U, i, i + 1, c, s);
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    // 扫描所有 d_k≈0 的位置：若对应 e_k 仍显著非零，则调用追赶过程压缩该异常结构。
    // 返回值表示本轮是否对 B/U/V 做了实际更新。
    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal(U, B, V, k, tol))
                {
                    changed = true;
                }
            }
        }

        return changed;
    }

    // 根据超对角线是否“足够小”对问题进行分块。
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置可解耦并直接置零。
    // 最终会得到一系列小矩阵。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
    // 最终得到常见的 SVD 规范形式：sigma_1 >= sigma_2 >= ... >= 0。
    // 这个函数你不用太在意，后续任务也不会明确涉及它。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }

} // namespace

// 对一个块反复 bulge chasing 直至可分割
static std::vector<Block> chase_block_until_split(Matrix &U, Matrix &B, Matrix &V,
Block blk, double tol, int max_inner_iter)
{
    std::vector<Block> result;
    if (blk.r <= blk.l) return result;   // 1x1 块无需处理

    int iter = 0;
    while (iter < max_inner_iter) {
        // 执行一步 bulge chasing
        one_block_step(U, B, V, blk.l, blk.r);

        // 清理数值噪声，处理对角零元
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        // 获取当前整个矩阵的块划分（split_active_blocks 会置零超对角线小元素）
        auto all_blocks = split_active_blocks(B, B.cols(), tol);

        // 筛选出完全位于当前块 [blk.l, blk.r] 内的非平凡子块
        std::vector<Block> inside;
        for (const auto& sb : all_blocks) {
            if (sb.l >= blk.l && sb.r <= blk.r && sb.r > sb.l) {
                inside.push_back(sb);
            }
        }

        // 判断是否发生了分割
        if (inside.empty()) {   // 原块已完全收敛为 1x1
            return result;   // 返回空
        }
        if (inside.size() == 1 && inside[0].l == blk.l && inside[0].r == blk.r) {
            // 仍未分割，继续下一轮迭代
            ++iter;
            continue;
        }

        // 分割发生，返回所有非平凡子块
        return inside;
    }

    // 达到最大迭代仍未分割，返回空
    return result;
}

// 递归处理所有任务
static void process_block_task(Block blk, Matrix &U, Matrix &B, Matrix &V,
                               double tol, int max_inner_iter)
{
    auto new_blocks = chase_block_until_split(U, B, V, blk, tol, max_inner_iter);

    // 检查是否返回了原块
    bool is_original = (new_blocks.size() == 1 &&
                        new_blocks[0].l == blk.l &&
                        new_blocks[0].r == blk.r);

    if (is_original) {  // 未分割，丢弃（外层循环会重新扫描）
        return;
    }

    for (auto &nb : new_blocks) {
        if (nb.r > nb.l) {
            #pragma omp task shared(U, B, V)
            process_block_task(nb, U, B, V, tol, max_inner_iter);
        }
    }
}

// 输入输出满足 A = U * B * V^T 不变；
// 迭代中自动分块、处理对角近零、并在每个活动块上做 bulge chasing；
// 成功收敛后，B 被整理为非负且降序的对角矩阵（其对角元即奇异值）
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n) {
        throw std::invalid_argument("gkh_svd_from_bidiagonal: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m) {
        throw std::invalid_argument("gkh_svd_from_bidiagonal: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n) {
        throw std::invalid_argument("gkh_svd_from_bidiagonal: V must be n x n");
    }

    const int inner_max_iter = 500;

    // 外层循环，持续处理直到全部收敛
    for (int outer = 0; outer < max_iter; ++outer) {
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);
        auto blocks = split_active_blocks(B, n, tol);

        std::vector<Block> task_pool;
        for (const auto &blk : blocks) {
            if (blk.r > blk.l) {
                task_pool.push_back(blk);
            }
        }

        if (task_pool.empty()) {
            break;
        }

        // 并行处理所有非平凡块
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (auto &blk : task_pool) {
                    #pragma omp task shared(U, B, V)
                    process_block_task(blk, U, B, V, tol, inner_max_iter);
                }
            }
        }
    }

    // 收尾
    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i) {
        B.at(i, i + 1) = 0.0;
    }
    make_nonnegative_and_sort(U, B, V);

    return true;
}