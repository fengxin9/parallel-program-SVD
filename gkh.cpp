#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <arm_neon.h>
#include <mpi.h>
#include <queue>
#include <cstdio>

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


// MPI消息标签
static const int MPI_TAG_TASK_REQ   = 100;  // 从进程请求任务
static const int MPI_TAG_TASK_DATA  = 101;  // 主进程发送block给从进程
static const int MPI_TAG_NEW_BLOCK  = 102;  // 从进程返回新分割出的子块
static const int MPI_TAG_NO_MORE    = 103;  // 主进程通知从进程没有更多任务了
static const int MPI_TAG_DONE       = 104;  // 从进程通知主进程自己已完成
static const int MPI_TAG_FINAL_U    = 105;  // 从进程发送最终U矩阵
static const int MPI_TAG_FINAL_V    = 106;  // 发送V矩阵
static const int MPI_TAG_FINAL_B    = 107;  // 发送B矩阵
static const int MPI_TAG_B_SYNC     = 108;  // B矩阵对角/超对角同步

// 发送子块的辅助函数
static void mpi_send_block(int dest, const Block& blk, int tag) {
    int data[2] = {blk.l, blk.r};
    MPI_Send(data, 2, MPI_INT, dest, tag, MPI_COMM_WORLD);
}

// 接收子块的辅助函数
static Block mpi_recv_block(int src, int tag, MPI_Status* status) {
    int data[2];
    MPI_Recv(data, 2, MPI_INT, src, tag, MPI_COMM_WORLD, status);
    return {data[0], data[1]};
}

// 加载B的对角线和超对角线到缓冲区 (2n-1个元素)
static std::vector<double> pack_b_state(const Matrix& B, int n) {
    std::vector<double> buf(2 * n - 1);
    for (int i = 0; i < n; ++i) buf[i] = B.at(i, i);
    for (int i = 0; i < n - 1; ++i) buf[n + i] = B.at(i, i + 1);
    return buf;
}

// 从缓冲区恢复B的对角线和超对角线
static void unpack_b_state(Matrix& B, int n, const std::vector<double>& buf) {
    for (int i = 0; i < n; ++i) B.at(i, i) = buf[i];
    for (int i = 0; i < n - 1; ++i) B.at(i, i + 1) = buf[n + i];
}

// 判断块[l, r]内是否存在可分割的位置
static bool can_split_in_range(const Matrix& B, int l, int r, double tol) {
    for (int k = l; k < r; ++k) {
        const double a = std::fabs(B.at(k, k));
        const double d = std::fabs(B.at(k + 1, k + 1));
        const double crit = tol * (a + d + 1.0);
        if (std::fabs(B.at(k, k + 1)) <= crit) {
            return true;
        }
    }
    return false;
}

// 对一个块反复执行 one_block_step，直到块内出现可分割的位置，然后分割出子块
static void bulge_chase_segment(Matrix& U, Matrix& B, Matrix& V, Block blk, 
                                std::vector<Block>& out_new_blocks, double tol) {
    while (blk.r > blk.l) {
        one_block_step(U, B, V, blk.l, blk.r);
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);
        if (can_split_in_range(B, blk.l, blk.r, tol)) {
            break;           // 块内出现可分割位置，停止追赶，准备分块
        }
    }

    auto all_blocks = split_active_blocks(B, B.cols(), tol);  // 全局扫描分块
    for (const auto& sb : all_blocks) {
        if (sb.l >= blk.l && sb.r <= blk.r && sb.r > sb.l) {
            out_new_blocks.push_back(sb);   // 收集新子块
        }
    }
}

// 从进程函数
static void mpi_slave_process(int m, int n, double tol, Matrix& U, Matrix& B, Matrix& V) {
    std::vector<double> b_buf(2 * n - 1);
    while (true) {
        MPI_Send(NULL, 0, MPI_INT, 0, MPI_TAG_TASK_REQ, MPI_COMM_WORLD);
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // 收到 NO_MORE 后退出循环
        if (status.MPI_TAG == MPI_TAG_NO_MORE) {
            MPI_Recv(NULL, 0, MPI_INT, 0, MPI_TAG_NO_MORE, MPI_COMM_WORLD, &status);
            break;
        }

        // 领取块+同步B状态
        Block blk = mpi_recv_block(0, MPI_TAG_TASK_DATA, &status);
        MPI_Recv(b_buf.data(), 2 * n - 1, MPI_DOUBLE, 0, MPI_TAG_B_SYNC, MPI_COMM_WORLD, &status);
        unpack_b_state(B, n, b_buf);

        // bulge chase直到可分割
        std::vector<Block> new_blocks;
        bulge_chase_segment(U, B, V, blk, new_blocks, tol);

        // 返回子块
        for (const auto& nb : new_blocks) {
            mpi_send_block(0, nb, MPI_TAG_NEW_BLOCK);
        }
        MPI_Send(NULL, 0, MPI_INT, 0, MPI_TAG_NEW_BLOCK, MPI_COMM_WORLD);

        // 回传更新后的B状态
        b_buf = pack_b_state(B, n);
        MPI_Send(b_buf.data(), 2 * n - 1, MPI_DOUBLE, 0, MPI_TAG_B_SYNC, MPI_COMM_WORLD);
    }

    // 发送最终 U, V, B 给主进程
    MPI_Send(U.data(), m * m, MPI_DOUBLE, 0, MPI_TAG_FINAL_U, MPI_COMM_WORLD);
    MPI_Send(V.data(), n * n, MPI_DOUBLE, 0, MPI_TAG_FINAL_V, MPI_COMM_WORLD);
    MPI_Send(B.data(), m * n, MPI_DOUBLE, 0, MPI_TAG_FINAL_B, MPI_COMM_WORLD);
    MPI_Send(NULL, 0, MPI_INT, 0, MPI_TAG_DONE, MPI_COMM_WORLD);
}

// 主进程函数
static void mpi_master_process(int nprocs, int m, int n, double tol, Matrix& U, Matrix& B, Matrix& V) {
    auto blocks = split_active_blocks(B, n, tol);
    std::queue<Block> task_queue;   // 维护待处理块的队列（进程池）
    for (const auto& blk : blocks) {
        if (blk.r > blk.l) task_queue.push(blk);  // 初始块入队
    }

    int active_slaves = nprocs - 1;  // 从进程数量
    MPI_Status status;
    std::vector<double> b_buf(2 * n - 1);

    // 循环响应从进程请求，分发任务并回收子块，直到所有从进程都收到 NO_MORE
    while (active_slaves > 0) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int src = status.MPI_SOURCE;

        if (status.MPI_TAG == MPI_TAG_TASK_REQ) {
            MPI_Recv(NULL, 0, MPI_INT, src, MPI_TAG_TASK_REQ, MPI_COMM_WORLD, &status);

            if (task_queue.empty()) {
                MPI_Send(NULL, 0, MPI_INT, src, MPI_TAG_NO_MORE, MPI_COMM_WORLD);
                --active_slaves;
            } else {
                Block blk = task_queue.front(); task_queue.pop();
                mpi_send_block(src, blk, MPI_TAG_TASK_DATA);

                // 发送当前B矩阵
                b_buf = pack_b_state(B, n);
                MPI_Send(b_buf.data(), 2 * n - 1, MPI_DOUBLE, src, MPI_TAG_B_SYNC,
                         MPI_COMM_WORLD);
            }

        } else if (status.MPI_TAG == MPI_TAG_NEW_BLOCK) { 
            // 回收子块
            while (true) {
                MPI_Probe(src, MPI_TAG_NEW_BLOCK, MPI_COMM_WORLD, &status);
                int count;
                MPI_Get_count(&status, MPI_INT, &count);
                if (count == 0) {
                    MPI_Recv(NULL, 0, MPI_INT, src, MPI_TAG_NEW_BLOCK, MPI_COMM_WORLD, &status);
                    break;
                }
                Block nb = mpi_recv_block(src, MPI_TAG_NEW_BLOCK, &status);
                if (nb.r > nb.l) task_queue.push(nb);
            }

            // 接收更新后的B状态，更新权威副本
            MPI_Recv(b_buf.data(), 2 * n - 1, MPI_DOUBLE, src, MPI_TAG_B_SYNC, MPI_COMM_WORLD, &status);
            unpack_b_state(B, n, b_buf);
        }
    }

    // 收集各从进程最终结果并合并
    for (int p = 1; p < nprocs; ++p) {
        Matrix worker_U(m, m), worker_V(n, n), worker_B(m, n);
        MPI_Recv(worker_U.data(), m * m, MPI_DOUBLE, p, MPI_TAG_FINAL_U, MPI_COMM_WORLD, &status);
        MPI_Recv(worker_V.data(), n * n, MPI_DOUBLE, p, MPI_TAG_FINAL_V, MPI_COMM_WORLD, &status);
        MPI_Recv(worker_B.data(), m * n, MPI_DOUBLE, p, MPI_TAG_FINAL_B, MPI_COMM_WORLD, &status);
        MPI_Recv(NULL, 0, MPI_INT, p, MPI_TAG_DONE, MPI_COMM_WORLD, &status);  // 确认完成

        for (int col = 0; col < n; ++col) {
            bool changed = false;
            for (int row = 0; row < m; ++row) {  // 只要U的某个元素有更新，就认为整列都要更新
                if (std::abs(U.at(row, col) - worker_U.at(row, col)) > 1e-12) {
                    changed = true;
                    break;
                }
            }
            if (changed) {   // 如果U的这一列有更新，则认为V的这一列也有更新
                for (int row = 0; row < m; ++row) U.at(row, col) = worker_U.at(row, col);
                for (int row = 0; row < n; ++row) V.at(row, col) = worker_V.at(row, col);
            }
        }

        for (int col = 0; col < n; ++col) {  // B的对角线更新
            if (std::abs(B.at(col, col) - worker_B.at(col, col)) > 1e-12) {
                B.at(col, col) = worker_B.at(col, col);
            }
        }
    }

    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i) B.at(i, i + 1) = 0.0;
    make_nonnegative_and_sort(U, B, V);
}


// 从”上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代：
// - 输入输出满足 A = U * B * V^T 不变；
// - 迭代中自动分块、处理对角近零、并在每个活动块上做 bulge chasing；
// - 成功收敛后，B 被整理为非负且降序的对角矩阵（其对角元即奇异值）。

// 实现新的子块迭代方法和主从进程池，未进行子块排序，矩阵B每轮同步更新，U、V矩阵本地更新且最后一次性合并
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_v2: V must be n x n");
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);     // 获取当前进程的rank
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);   // 获取总进程数

    std::cout<< "Process " << rank << " of " << nprocs << " started." << std::endl;

    // 单进程采用串行算法
    if (nprocs == 1) {
        bool converged = false;
        for (int iter = 0; iter < max_iter; ++iter) {
            cleanup_bidiagonal(B, tol);
            handle_diagonal_zeros(U, B, V, tol);

            std::vector<Block> blocks = split_active_blocks(B, n, tol);

            bool all_singletons = true;
            for (const auto &blk : blocks) {
                if (blk.r > blk.l) { 
                    all_singletons = false; 
                    break; 
                }
            }
            if (all_singletons) { 
                converged = true; 
                break; 
            }

            for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
                if (blocks[i].r > blocks[i].l) {
                    one_block_step(U, B, V, blocks[i].l, blocks[i].r);
                }
            }
        }

        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i) B.at(i, i + 1) = 0.0;
        make_nonnegative_and_sort(U, B, V);
        return converged;
    }

    // 多进程：MPI 并行版本
    if (rank != 0) {      
        U = Matrix(m, m);
        B = Matrix(m, n);
        V = Matrix(n, n);
    }

    // 广播初始矩阵（所有进程参与）
    MPI_Bcast(U.data(), m * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(V.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {   // 主进程负责调度
        mpi_master_process(nprocs, m, n, tol, U, B, V);
    } else {   // 从进程负责计算
        mpi_slave_process(m, n, tol, U, B, V);
    }
    return true;
}