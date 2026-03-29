/*=============================================================
  resamp_kriging_refine.c - 源SLC向参考SLC重采样（克里金+64×64区块SIFT二次精配准）
  核心：源SLC → 参考SLC网格校正，兼顾全局趋势+局部细节，提升配准精度
  处理流程（正向映射warping）：
    目标像素(参考SLC的x,y) → 融合型克里金变形场 → 源SLC位置(x+dx,y+dy) → sinc插值 → 写入目标
  完整配准流程：
    1. 读取初始配准点，迭代剔除异常点，构建全局粗配准克里金变形场
    2. 读取源SLC/参考SLC，转强度图（幅度值+高斯去噪），用于SIFT匹配
    3. 按64×64区块划分全图，对每个区块做SIFT特征匹配（源强度图↔参考强度图）
    4. RANSAC剔除区块匹配异常点，生成局部精配准点
    5. 融合原始全局配准点和区块SIFT精配准点，生成全图高精度配准点集
    6. 基于融合配准点，重构高精度克里金变形场
    7. 对参考SLC网格的每个像素，计算源SLC对应位置，sinc插值后输出
  输入：源SLC+参考SLC（同格式复数short）、初始配准点、各尺寸参数
  输出：校正到参考SLC网格的高精度SLC数据
=============================================================*/
#include <omp.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include <algorithm>
// OpenCV头文件：适配OpenCV4+，SIFT移至主命名空间
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

/*---------------- 常量定义 ------------------------*/
#define NS 8                     // Sinc插值窗口大小（高精度）
#define I2MAX 32767              // int16最大值（SLC数据类型）
#define PI 3.1415926535897932    // 圆周率
#define KRIGING_RANGE_FACTOR 1.5 // 克里金范围因子，确保覆盖全图
#define MAX_ITERATIONS 10         // 初始点迭代去异常次数
#define OUTLIER_THRESHOLD 2.0    // 异常点阈值（2倍标准差）
#define KRIGING_NUGGET 0.01      // 克里金块金效应，防止矩阵奇异
#define BLOCK_SIZE 256            // SIFT二次精配准区块大小（64×64）
#define SIFT_MATCH_RATIO 0.75    // SIFT匹配Lowe距离比准则（经典阈值）
#define SIFT_MIN_MATCHES 5      // 每个区块最小有效匹配数（低于则放弃）
#define GAUSS_BLUR_KERNEL 3      // 强度图高斯模糊核（3×3，去噪提升SIFT稳定性）
#define TARGET_GRID_STEP 1       // 目标网格步长（与参考SLC一致）

/*---------------- 全局变量 ------------------------*/
// 配准点数据：参考SLC(目标)位置 → 源SLC变形量
double *target_x_points = NULL;     // 参考SLC x坐标（目标）
double *target_y_points = NULL;     // 参考SLC y坐标（目标）
double *dx_points = NULL;           // x方向变形量：源x = 参考x + dx
double *dy_points = NULL;           // y方向变形量：源y = 参考y + dy

// 克里金权重：前n为权重，最后1个为拉格朗日乘子（普通克里金）
double *kriging_x_weights = NULL;   // dx方向克里金解（权重+乘子）
double *kriging_y_weights = NULL;   // dy方向克里金解（权重+乘子）

double kriging_range = 100.0;       // 克里金变差函数范围
int kriging_n_points = 0;           // 融合后配准点总数（全局+SIFT局部）

// 图像尺寸：源SLC / 参考SLC（目标网格）→ 全局变量，所有函数可见
int src_width = 0, src_height = 0;  // 源SLC宽度/高度
int ref_width = 0, ref_height = 0;  // 参考SLC宽度/高度（目标网格）

// SLC内存映射指针（mmap，高效IO）
short *src_slc_data = NULL;         // 源SLC数据指针（复数short：实+虚）
short *ref_slc_data = NULL;         // 参考SLC数据指针（复数short：实+虚）

// 强度图（用于SIFT匹配，灰度图）
cv::Mat src_intensity;              // 源SLC强度图（√(实²+虚²)，float32）
cv::Mat ref_intensity;              // 参考SLC强度图（√(实²+虚²)，float32）

/*-------------------------------------------------------------*/
// 修复：C++规范 - 字符串常量改为const char*
const char *USAGE = 
    "\nUsage: resamp_kriging src.SLC src_width ref.SLC ref_width reg.txt output.SLC\n"
    "  必选参数（共6个，顺序不可变）：\n"
    "    1. src.SLC:       待校正源SLC文件（复数short，无表头二进制）\n"
    "    2. src_width:     源SLC图像宽度（像素数，正整数）\n"
    "    3. ref.SLC:       参考SLC文件（配准标准，同格式复数short）\n"
    "    4. ref_width:     参考SLC图像宽度（像素数，正整数，目标网格宽度）\n"
    "    5. reg.txt:       初始配准点文件（格式：x dx y dy [corr]，#为注释）\n"
    "    6. output.SLC:    输出文件（校正到参考SLC网格的高精度SLC）\n\n"
    "  初始配准点格式说明：\n"
    "    x: 参考SLC(目标)x坐标, dx: x变形量; y: 参考SLC(目标)y坐标, dy: y变形量\n"
    "    源SLC位置 = (x+dx, y+dy), [corr]为配准相关系数（可选，0~1）\n\n"
    "  核心处理特性：\n"
    "    ✔ 双SLC强度图生成：复数→幅度值，高斯模糊去噪，适配SIFT灰度图要求\n"
    "    ✔ 64×64区块SIFT：局部精配准，解决全局配准局部精度不足问题\n"
    "    ✔ 配准点融合：初始全局点+SLC局部点，兼顾全局趋势和局部细节\n"
    "    ✔ 高效IO：mmap内存映射，避免频繁磁盘读写，支持大尺寸SLC\n"
    "    ✔ 多核并行：OpenMP并行重采样，充分利用CPU多核资源\n"
    "    ✔ 异常处理：全流程边界检查、内存校验、配准点有效性判断\n";

/*=============================================================
  辅助函数：数值裁剪（浮点→short，防止SLC数据溢出）
=============================================================*/
short clipi2(double x) {
    if (x > I2MAX) return I2MAX;
    if (x < -I2MAX) return -I2MAX;
    return (short)round(x); // 四舍五入，提升精度
}

/*=============================================================
  读取初始配准点文件（支持注释/空行，格式：x dx y dy [corr]）
=============================================================*/
int read_registration_points(const char *filename, double **t_x, double **t_y,
                            double **dx, double **dy, int *n_points) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "❌ 无法打开初始配准点文件: %s\n", filename); return -1; }

    // 第一步：统计有效行数（跳过注释/空行）
    int count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        int is_empty = 1;
        for (int i = 0; line[i] != '\0'; i++) if (!isspace(line[i])) { is_empty = 0; break; }
        if (!is_empty && line[0] != '#') count++;
    }
    if (count == 0) { fprintf(stderr, "❌ 初始配准点文件无有效数据\n"); fclose(fp); return -1; }
    printf("✅ 找到初始配准点有效行：%d\n", count);

    // 第二步：分配内存
    *t_x = (double *)malloc(count * sizeof(double));
    *t_y = (double *)malloc(count * sizeof(double));
    *dx = (double *)malloc(count * sizeof(double));
    *dy = (double *)malloc(count * sizeof(double));
    if (!*t_x || !*t_y || !*dx || !*dy) {
        fprintf(stderr, "❌ 配准点内存分配失败\n");
        fclose(fp);
        return -1;
    }

    // 第三步：读取有效数据
    rewind(fp);
    int i = 0, line_num = 0;
    while (fgets(line, sizeof(line), fp) && i < count) {
        line_num++;
        int is_empty = 1;
        for (int j = 0; j < strlen(line); j++) if (!isspace(line[j])) { is_empty = 0; break; }
        if (is_empty || line[0] == '#') continue;

        double x, dx_val, y, dy_val, corr = 1.0;
        int n = sscanf(line, "%lf %lf %lf %lf %lf", &x, &dx_val, &y, &dy_val, &corr);
        if (n >= 4) {
            (*t_x)[i] = x; (*t_y)[i] = y;
            (*dx)[i] = dx_val; (*dy)[i] = dy_val;
            i++;
        } else if (n > 0) {
            fprintf(stderr, "⚠️  第%d行格式错误，跳过：%s", line_num, line);
        }
    }
    *n_points = i;
    fclose(fp);

    if (i < count) fprintf(stderr, "⚠️  仅读取%d/%d个有效初始配准点\n", i, count);
    if (i < 4) { fprintf(stderr, "❌ 有效配准点不足4个，无法构建克里金场\n"); return -1; }
    printf("✅ 成功读取有效初始配准点：%d\n", *n_points);
    return 0;
}

/*=============================================================
  二次线性拟合（y = a0 + a1*x + a2*x²）→ 用于初始点去异常
=============================================================*/
void quadratic_fit(double *x, double *y, int n, double *coeffs) {
    double sum_x=0, sum_x2=0, sum_x3=0, sum_x4=0, sum_y=0, sum_xy=0, sum_x2y=0;
    for (int i = 0; i < n; i++) {
        double xi = x[i], xi2 = xi*xi, yi = y[i];
        sum_x += xi; sum_x2 += xi2; sum_x3 += xi2*xi; sum_x4 += xi2*xi2;
        sum_y += yi; sum_xy += xi*yi; sum_x2y += xi2*yi;
    }

    // 构建正规方程矩阵
    double A[3][3] = {{(double)n, sum_x, sum_x2}, {sum_x, sum_x2, sum_x3}, {sum_x2, sum_x3, sum_x4}};
    double b[3] = {sum_y, sum_xy, sum_x2y};

    // 高斯消元法求解
    for (int i = 0; i < 3; i++) {
        // 主元选择（避免除零）
        int pivot = i;
        double max_val = fabs(A[i][i]);
        for (int j = i+1; j < 3; j++) if (fabs(A[j][i]) > max_val) { max_val = fabs(A[j][i]); pivot = j; }
        if (max_val < 1e-12) { // 矩阵奇异，退化为线性拟合
            coeffs[0] = 0.0;
            coeffs[1] = (sum_xy - sum_x*sum_y/n) / (sum_x2 - sum_x*sum_x/n);
            coeffs[2] = 0.0;
            return;
        }
        // 交换行
        if (pivot != i) {
            for (int k = i; k < 3; k++) { double temp = A[i][k]; A[i][k] = A[pivot][k]; A[pivot][k] = temp; }
            double temp = b[i]; b[i] = b[pivot]; b[pivot] = temp;
        }
        // 消元
        double diag = A[i][i];
        for (int j = i+1; j < 3; j++) {
            double factor = A[j][i]/diag;
            for (int k = i; k < 3; k++) A[j][k] -= factor*A[i][k];
            b[j] -= factor*b[i];
        }
    }
    // 回代求解系数
    coeffs[2] = b[2]/A[2][2];
    coeffs[1] = (b[1] - A[1][2]*coeffs[2])/A[1][1];
    coeffs[0] = (b[0] - A[0][1]*coeffs[1] - A[0][2]*coeffs[2])/A[0][0];
}

/*=============================================================
  计算拟合值 + RMSE（均方根误差）→ 评估拟合效果
=============================================================*/
double evaluate_fit(double x, double *coeffs) { return coeffs[0] + coeffs[1]*x + coeffs[2]*x*x; }
double calculate_rmse(double *x, double *y, int n, double *coeffs) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) { double e = y[i] - evaluate_fit(x[i], coeffs); sum_sq += e*e; }
    return sqrt(sum_sq / n);
}

/*=============================================================
  迭代剔除初始配准点异常点（x-dx/y-dy分别拟合，2倍标准差）
=============================================================*/
void filter_outliers(double *x, double *dx, double *y, double *dy, int *n_points) {
    int n = *n_points;
    int *valid = (int *)calloc(n, sizeof(int));
    if (!valid) { fprintf(stderr, "❌ 去异常点内存分配失败\n"); return; }
    for (int i = 0; i < n; i++) valid[i] = 1; // 初始所有点有效

    printf("📊 初始配准点迭代去异常（最大%d次）：\n", MAX_ITERATIONS);
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // 统计有效点
        int valid_cnt = 0;
        for (int i = 0; i < n; i++) if (valid[i]) valid_cnt++;
        if (valid_cnt < 4) { printf("⚠️  迭代%d：有效点仅剩%d，停止去异常\n", iter+1, valid_cnt); break; }

        // 提取有效点
        double *vx = (double *)malloc(valid_cnt*sizeof(double)), *vdx = (double *)malloc(valid_cnt*sizeof(double));
        double *vy = (double *)malloc(valid_cnt*sizeof(double)), *vdy = (double *)malloc(valid_cnt*sizeof(double));
        if (!vx || !vdx || !vy || !vdy) { fprintf(stderr, "❌ 有效点内存分配失败\n"); break; }
        int idx = 0;
        for (int i = 0; i < n; i++) if (valid[i]) { vx[idx]=x[i]; vdx[idx]=dx[i]; vy[idx]=y[i]; vdy[idx]=dy[i]; idx++; }

        // 拟合+计算RMSE
        double cx[3], cy[3];
        quadratic_fit(vx, vdx, valid_cnt, cx); quadratic_fit(vy, vdy, valid_cnt, cy);
        double rmse_x = calculate_rmse(vx, vdx, valid_cnt, cx);
        double rmse_y = calculate_rmse(vy, vdy, valid_cnt, cy);

        // 标记异常点（残差>2倍标准差）
        int outlier_cnt = 0; idx = 0;
        for (int i = 0; i < n; i++) {
            if (valid[i]) {
                double e_x = fabs(vdx[idx] - evaluate_fit(vx[idx], cx)) / rmse_x;
                double e_y = fabs(vdy[idx] - evaluate_fit(vy[idx], cy)) / rmse_y;
                if (e_x > OUTLIER_THRESHOLD || e_y > OUTLIER_THRESHOLD) { valid[i] = 0; outlier_cnt++; }
                idx++;
            }
        }

        // 打印迭代信息
        printf("  迭代%d：有效点%d | RMSE(x)=%.4f | RMSE(y)=%.4f | 剔除异常点%d\n",
               iter+1, valid_cnt, rmse_x, rmse_y, outlier_cnt);
        free(vx); free(vdx); free(vy); free(vdy);
        if (outlier_cnt == 0) { printf("  ✅ 无新异常点，停止迭代\n"); break; }
    }

    // 收集去异常后的有效点（覆盖原数组，节省内存）
    int new_n = 0;
    for (int i = 0; i < n; i++) {
        if (valid[i]) { x[new_n]=x[i]; dx[new_n]=dx[i]; y[new_n]=y[i]; dy[new_n]=dy[i]; new_n++; }
    }
    *n_points = new_n;
    free(valid);
    printf("✅ 初始配准点去异常完成，剩余有效点：%d\n", new_n);
}

/*=============================================================
  球形变差函数（克里金核心，描述空间点相关性）
  h：两点距离，range：相关范围 → 输出0~1（距离越远，相关性越弱）
=============================================================*/
double variogram_spherical(double h, double range) {
    if (h <= 0) return 0.0;
    if (h >= range) return 1.0;
    double h_r = h / range;
    return 1.5 * h_r - 0.5 * h_r * h_r * h_r;
}

/*=============================================================
  创建克里金矩阵（普通克里金，n+1阶，含拉格朗日乘子约束）
=============================================================*/
double **create_kriging_matrix(double *x, double *y, int n, double range) {
    double **mat = (double **)malloc((n+1) * sizeof(double *));
    for (int i = 0; i <= n; i++) mat[i] = (double *)calloc(n+2, sizeof(double)); // 初始化0

    // 填充矩阵：i/j为配准点，对角线加块金效应，非对角线为-变差函数
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) mat[i][j] = KRIGING_NUGGET;
            else {
                double dx = x[i]-x[j], dy = y[i]-y[j];
                double dist = sqrt(dx*dx + dy*dy);
                mat[i][j] = -variogram_spherical(dist, range);
            }
        }
        mat[i][n] = 1.0; // 拉格朗日乘子列
    }
    // 最后一行：无偏约束（权重和为1）
    for (int j = 0; j < n; j++) mat[n][j] = 1.0;
    mat[n][n] = 0.0;

    return mat;
}

/*=============================================================
  求解克里金系统（高斯消元法，返回解：前n为权重，最后1个为拉格朗日乘子）
=============================================================*/
double *solve_kriging_system(double *x, double *y, double *val, int n, double range) {
    double **mat = create_kriging_matrix(x, y, n, range);
    int m = n + 1; // 矩阵阶数

    // 设置右侧向量（变形量）
    for (int i = 0; i < n; i++) mat[i][m] = val[i];
    mat[n][m] = 0.0;

    // 高斯消元：前向消元为上三角
    for (int i = 0; i < m; i++) {
        // 主元选择
        int pivot = i;
        double max_val = fabs(mat[i][i]);
        for (int j = i+1; j < m; j++) if (fabs(mat[j][i]) > max_val) { max_val = fabs(mat[j][i]); pivot = j; }
        if (max_val < 1e-12) { fprintf(stderr, "⚠️  克里金矩阵接近奇异\n"); }
        // 交换行
        if (pivot != i) { double *temp = mat[i]; mat[i] = mat[pivot]; mat[pivot] = temp; }
        // 归一化主元行
        double diag = mat[i][i];
        if (fabs(diag) < 1e-12) diag = 1.0;
        for (int k = i; k <= m; k++) mat[i][k] /= diag;
        // 消元其他行
        for (int j = 0; j < m; j++) {
            if (j != i && fabs(mat[j][i]) > 1e-12) {
                double factor = mat[j][i];
                for (int k = i; k <= m; k++) mat[j][k] -= factor * mat[i][k];
            }
        }
    }

    // 提取解
    double *sol = (double *)malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) sol[i] = mat[i][m];

    // 释放矩阵内存
    for (int i = 0; i < m; i++) free(mat[i]);
    free(mat);

    // 权重和校验
    double w_sum = 0.0;
    for (int i = 0; i < n; i++) w_sum += sol[i];
    printf("  克里金权重和：%.6f（理想1.0）\n", w_sum);
    return sol;
}

/*=============================================================
  克里金插值计算变形量（普通克里金，输入参考点坐标，输出dx/dy）
=============================================================*/
double kriging_interp(double x, double y, double *sx, double *sy, double *sol, int n, double range) {
    double res = 0.0;
    double lambda = sol[n]; // 拉格朗日乘子
    for (int i = 0; i < n; i++) {
        double dx = x - sx[i], dy = y - sy[i];
        double dist = sqrt(dx*dx + dy*dy);
        double gamma = variogram_spherical(dist, range);
        res += sol[i] * gamma;
    }
    res += lambda; // 无偏约束修正
    return res;
}

/*=============================================================
  SLC转强度图（核心修复：输出CV_8U单通道灰度图，适配SIFT强制要求）
  slc_data：SLC内存指针，w/h：尺寸 → 输出8位单通道灰度图（0~255）
  修复点：1. 归一化后强制转换为CV_8U 2. 增加SLC数据有效性校验 3. 释放临时浮点图
=============================================================*/
cv::Mat slc2intensity(short *slc_data, int w, int h) {
    // 健壮性校验：避免空指针/无效尺寸
    if (slc_data == NULL || w <= 0 || h <= 0) {
        fprintf(stderr, "❌ SLC数据无效：空指针或尺寸为负（w=%d, h=%d）\n", w, h);
        return cv::Mat();
    }

    // 第一步：生成32位浮点幅度图（√(实²+虚²)）
    cv::Mat img_f32(h, w, CV_32F, cv::Scalar(0));
    #pragma omp parallel for schedule(static) // 多核并行生成
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int pos = 2 * (y * w + x); // SLC：每个像素2个short（实+虚）
            // 校验SLC数据位置有效性，避免越界
            if (pos + 1 >= 2 * w * h) {
                img_f32.at<float>(y, x) = 0.0;
                continue;
            }
            double real = (double)slc_data[pos];
            double imag = (double)slc_data[pos+1];
            double amp = sqrt(real*real + imag*imag); // 幅度值（强度）
            img_f32.at<float>(y, x) = (float)amp;
        }
    }

    // 第二步：高斯模糊去噪（提升SIFT匹配稳定性）
    cv::GaussianBlur(img_f32, img_f32, cv::Size(GAUSS_BLUR_KERNEL, GAUSS_BLUR_KERNEL), 0);

    // 第三步：归一化到0~255，并强制转换为SIFT要求的CV_8U（8位单通道）
    cv::Mat img_8u;
    double min_val, max_val;
    cv::minMaxLoc(img_f32, &min_val, &max_val);
    // 避免除零（若全为0，直接返回空图）
    if (fabs(max_val - min_val) < 1e-8) {
        fprintf(stderr, "❌ SLC强度图全为0，无法生成有效灰度图\n");
        return cv::Mat();
    }
    // 归一化+类型转换：CV_32F → CV_8U（0~255）
    img_f32 = (img_f32 - min_val) / (max_val - min_val) * 255.0;
    img_f32.convertTo(img_8u, CV_8U); // 核心修复：转换为8位无符号整型

    // 释放临时浮点图，避免内存泄漏
    img_f32.release();
    return img_8u;
}

/*=============================================================
  64×64区块SIFT二次精配准（源强度图↔参考强度图）
  输入：全局粗配准克里金解 → 输出：新增SIFT精配准点数量
  修复：1. SIFT改为cv::SIFT 2. 增加图像深度/通道数校验 3. 全局变量名修正
=============================================================*/
int block_sift_registration(double *sol_x_rough, double *sol_y_rough, double range_rough,
                            double **new_t_x, double **new_t_y, double **new_dx, double **new_dy) {
    printf("\n🎯 开始64×64区块SIFT二次精配准：\n");
    // 增强健壮性校验：空图/非8位单通道直接返回
    if (src_intensity.empty() || ref_intensity.empty()) {
        fprintf(stderr, "❌ 强度图未生成/为空，无法进行SIFT配准\n");
        return 0;
    }
    if (src_intensity.depth() != CV_8U || ref_intensity.depth() != CV_8U) {
        fprintf(stderr, "❌ 强度图深度错误：要求CV_8U，源=%d，参考=%d\n", src_intensity.depth(), ref_intensity.depth());
        return 0;
    }
    if (src_intensity.channels() != 1 || ref_intensity.channels() != 1) {
        fprintf(stderr, "❌ 强度图通道数错误：要求单通道，源=%d，参考=%d\n", src_intensity.channels(), ref_intensity.channels());
        return 0;
    }
    if (src_intensity.size() != ref_intensity.size()) {
        fprintf(stderr, "⚠️  源/参考强度图尺寸不一致，SIFT匹配可能精度下降\n");
    }

    // 修复：OpenCV4+ SIFT移至主命名空间cv::SIFT
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::FlannBasedMatcher matcher; // 快速近邻匹配器
    std::vector<cv::KeyPoint> kp_src, kp_ref; // 特征点
    cv::Mat des_src, des_ref; // 特征描述子
    // 检测+计算描述子（源+参考）
    sift->detectAndCompute(src_intensity, cv::noArray(), kp_src, des_src);
    sift->detectAndCompute(ref_intensity, cv::noArray(), kp_ref, des_ref);
    printf("✅ SIFT特征检测完成：源%d个特征点 | 参考%d个特征点\n", (int)kp_src.size(), (int)kp_ref.size());
    if (kp_src.size() < SIFT_MIN_MATCHES || kp_ref.size() < SIFT_MIN_MATCHES) {
        //fprintf(stderr, "❌ 特征点数量不足，无法进行SIFT配准\n");
        return 0;
    }

    // 区块划分：按64×64遍历参考SLC（目标）全图
    int ref_h = ref_intensity.rows, ref_w = ref_intensity.cols;
    int block_cols = (ref_w + BLOCK_SIZE - 1) / BLOCK_SIZE; // 向上取整
    int block_rows = (ref_h + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("✅ 参考SLC区块划分：%d行×%d列 = %d个64×64区块\n", block_rows, block_cols, block_rows*block_cols);

    // 分配SIFT配准点内存（预分配足够空间）
    int max_sift_points = block_rows * block_cols * 100; // 每个区块最多100个点
    *new_t_x = (double *)malloc(max_sift_points * sizeof(double));
    *new_t_y = (double *)malloc(max_sift_points * sizeof(double));
    *new_dx = (double *)malloc(max_sift_points * sizeof(double));
    *new_dy = (double *)malloc(max_sift_points * sizeof(double));
    if (!*new_t_x || !*new_t_y || !*new_dx || !*new_dy) {
        fprintf(stderr, "❌ SIFT配准点内存分配失败\n");
        return 0;
    }
    int sift_point_cnt = 0;

    // 遍历每个区块
    for (int br = 0; br < block_rows; br++) {
        for (int bc = 0; bc < block_cols; bc++) {
            // 计算区块在参考SLC中的坐标范围（目标）
            int ref_x0 = bc * BLOCK_SIZE;
            int ref_y0 = br * BLOCK_SIZE;
            int ref_x1 = std::min(ref_x0 + BLOCK_SIZE - 1, ref_w - 1);
            int ref_y1 = std::min(ref_y0 + BLOCK_SIZE - 1, ref_h - 1);
            if (ref_x0 >= ref_w || ref_y0 >= ref_h) continue;
            cv::Rect ref_rect(ref_x0, ref_y0, ref_x1-ref_x0+1, ref_y1-ref_y0+1); // 参考区块

            // 全局粗配准：计算参考区块中心对应的源SLC位置，限定源区块范围
            double ref_cx = (ref_x0 + ref_x1) / 2.0, ref_cy = (ref_y0 + ref_y1) / 2.0;
            double src_cx_rough = ref_cx + kriging_interp(ref_cx, ref_cy, target_x_points, target_y_points, sol_x_rough, kriging_n_points, range_rough);
            double src_cy_rough = ref_cy + kriging_interp(ref_cx, ref_cy, target_x_points, target_y_points, sol_y_rough, kriging_n_points, range_rough);
            // 修复：全局变量为src_width/src_height，非src_w/src_h
            int src_x0 = std::max(0, (int)(src_cx_rough - BLOCK_SIZE*1.5));
            int src_y0 = std::max(0, (int)(src_cy_rough - BLOCK_SIZE*1.5));
            int src_x1 = std::min(src_width - 1, (int)(src_cx_rough + BLOCK_SIZE*1.5));
            int src_y1 = std::min(src_height - 1, (int)(src_cy_rough + BLOCK_SIZE*1.5));
            if (src_x0 >= src_width || src_y0 >= src_height) continue;
            cv::Rect src_rect(src_x0, src_y0, src_x1-src_x0+1, src_y1-src_y0+1); // 源区块

            // 提取区块内特征点和描述子
            std::vector<cv::KeyPoint> kp_src_block, kp_ref_block;
            cv::Mat des_src_block, des_ref_block;
            // 筛选参考区块特征点
            for (size_t i = 0; i < kp_ref.size(); i++) {
                if (ref_rect.contains(kp_ref[i].pt)) { kp_ref_block.push_back(kp_ref[i]); }
            }
            // 筛选源区块特征点
            for (size_t i = 0; i < kp_src.size(); i++) {
                if (src_rect.contains(kp_src[i].pt)) { kp_src_block.push_back(kp_src[i]); }
            }
            if (kp_src_block.size() < SIFT_MIN_MATCHES || kp_ref_block.size() < SIFT_MIN_MATCHES) {
                //printf("  区块(%d,%d)：特征点不足，跳过\n", br, bc);
                continue;
            }
            // 重新计算区块描述子
            sift->compute(src_intensity(src_rect), kp_src_block, des_src_block);
            sift->compute(ref_intensity(ref_rect), kp_ref_block, des_ref_block);

            // SIFT特征匹配（Lowe距离比准则，剔除虚假匹配）
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(des_src_block, des_ref_block, knn_matches, 2); // k=2
            std::vector<cv::DMatch> good_matches;
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i][0].distance < SIFT_MATCH_RATIO * knn_matches[i][1].distance) {
                    good_matches.push_back(knn_matches[i][0]);
                }
            }
            if ((int)good_matches.size() < SIFT_MIN_MATCHES) {
                printf("  区块(%d,%d)：有效匹配%d个（不足%d），跳过\n", br, bc, (int)good_matches.size(), SIFT_MIN_MATCHES);
                continue;
            }

            // RANSAC剔除匹配异常点（进一步提升精度）
            std::vector<cv::Point2f> src_pts, ref_pts;
            for (size_t i = 0; i < good_matches.size(); i++) {
                src_pts.push_back(kp_src_block[good_matches[i].queryIdx].pt + cv::Point2f(src_x0, src_y0));
                ref_pts.push_back(kp_ref_block[good_matches[i].trainIdx].pt + cv::Point2f(ref_x0, ref_y0));
            }
            std::vector<uchar> inlier_mask;
            cv::findHomography(src_pts, ref_pts, cv::RANSAC, 5.0, inlier_mask); // 单应性矩阵+RANSAC
            // 收集RANSAC内点（有效匹配）
            std::vector<cv::Point2f> src_inliers, ref_inliers;
            for (size_t i = 0; i < inlier_mask.size(); i++) {
                if (inlier_mask[i]) { src_inliers.push_back(src_pts[i]); ref_inliers.push_back(ref_pts[i]); }
            }
            if ((int)src_inliers.size() < SIFT_MIN_MATCHES) {
                printf("  区块(%d,%d)：RANSAC后有效匹配%d个，跳过\n", br, bc, (int)src_inliers.size());
                continue;
            }

            // 计算SLC配准点（参考x,y → 源dx,dy：dx=src_x-ref_x，dy=src_y-ref_y）
            for (size_t i = 0; i < src_inliers.size(); i++) {
                double rx = ref_inliers[i].x, ry = ref_inliers[i].y;
                double sx = src_inliers[i].x, sy = src_inliers[i].y;
                (*new_t_x)[sift_point_cnt] = rx;
                (*new_t_y)[sift_point_cnt] = ry;
                (*new_dx)[sift_point_cnt] = sx - rx;
                (*new_dy)[sift_point_cnt] = sy - ry;
                sift_point_cnt++;
                // 防止内存溢出
                if (sift_point_cnt >= max_sift_points) { goto block_sift_end; }
            }
            printf("  区块(%d,%d)：有效匹配%d个 → 生成%d个精配准点\n",
                   br, bc, (int)src_inliers.size(), (int)src_inliers.size());
        }
    }
block_sift_end:
    printf("\n✅ SIFT区块精配准完成，新增局部精配准点：%d\n", sift_point_cnt);
    return sift_point_cnt;
}

/*=============================================================
  融合配准点（初始全局点 + SIFT局部点）→ 重构配准点集
=============================================================*/
int fuse_registration_points(double *t_x_g, double *t_y_g, double *dx_g, double *dy_g, int n_g,
                             double *t_x_s, double *t_y_s, double *dx_s, double *dy_s, int n_s,
                             double **t_x_f, double **t_y_f, double **dx_f, double **dy_f) {
    printf("\n🔗 融合配准点：全局%d个 + 局部SIFT%d个 = 总计%d个\n", n_g, n_s, n_g+n_s);
    int n_f = n_g + n_s;
    // 分配融合点内存
    *t_x_f = (double *)malloc(n_f * sizeof(double));
    *t_y_f = (double *)malloc(n_f * sizeof(double));
    *dx_f = (double *)malloc(n_f * sizeof(double));
    *dy_f = (double *)malloc(n_f * sizeof(double));
    if (!*t_x_f || !*t_y_f || !*dx_f || !*dy_f) {
        fprintf(stderr, "❌ 融合配准点内存分配失败\n");
        return -1;
    }
    // 复制全局点
    memcpy(*t_x_f, t_x_g, n_g * sizeof(double));
    memcpy(*t_y_f, t_y_g, n_g * sizeof(double));
    memcpy(*dx_f, dx_g, n_g * sizeof(double));
    memcpy(*dy_f, dy_g, n_g * sizeof(double));
    // 复制SIFT局部点
    memcpy(*t_x_f + n_g, t_x_s, n_s * sizeof(double));
    memcpy(*t_y_f + n_g, t_y_s, n_s * sizeof(double));
    memcpy(*dx_f + n_g, dx_s, n_s * sizeof(double));
    memcpy(*dy_f + n_g, dy_s, n_s * sizeof(double));
    // 释放SIFT临时点内存
    free(t_x_s); free(t_y_s); free(dx_s); free(dy_s);
    printf("✅ 配准点融合完成，最终配准点集：%d个\n", n_f);
    return n_f;
}

/*=============================================================
  Sinc插值核函数（sin(πx)/(πx)，x→0时返回1，避免除零）
=============================================================*/
double sinc_kernel(double x) {
    double arg = fabs(PI * x);
    return (arg > 1e-8) ? (sin(arg) / arg) : 1.0;
}

/*=============================================================
  单通道Sinc插值（8×8窗口，高精度）
=============================================================*/
void sinc_interp_single(double *win_data, double x, double y, double *out) {
    const int ns2 = NS / 2; // 窗口半宽（4，8×8窗口）
    double wx[NS], wy[NS];
    double wsum = 0.0, val = 0.0;

    // 计算x/y方向Sinc权重
    for (int i = 0; i < NS; i++) { wx[i] = sinc_kernel(x - (i - ns2)); wy[i] = sinc_kernel(y - (i - ns2)); }
    // 8×8窗口加权求和
    int idx = 0;
    for (int j = 0; j < NS; j++) {
        for (int i = 0; i < NS; i++) {
            double w = wx[i] * wy[j];
            val += win_data[idx + i] * w;
            wsum += w;
            idx++;
        }
    }
    // 权重归一化（避免权重和为0）
    *out = (wsum > 1e-8) ? (val / wsum) : 0.0;
}

/*=============================================================
  复数Sinc插值（针对SLC数据，实部+虚部分别插值，8×8窗口）
  src_x/src_y：源SLC浮点坐标 → 输出：插值后的复数short（实+虚）
=============================================================*/
void bisinc_interp(double src_x, double src_y, short *src_data, int src_w, int src_h, short *out) {
    out[0] = 0; out[1] = 0; // 初始化0
    const int ns2 = NS / 2;
    // 计算整数坐标和小数部分（窗口中心）
    int x0 = (int)floor(src_x), y0 = (int)floor(src_y);
    double fx = src_x - x0, fy = src_y - y0;
    // 边界检查：确保8×8窗口完全在源SLC内
    if (x0 - ns2 < 0 || x0 + ns2 >= src_w || y0 - ns2 < 0 || y0 + ns2 >= src_h) { return; }

    // 提取8×8窗口的实部和虚部数据
    double real_win[NS*NS], imag_win[NS*NS];
    int win_idx = 0;
    for (int y = y0 - ns2; y < y0 + ns2; y++) {
        for (int x = x0 - ns2; x < x0 + ns2; x++) {
            int src_pos = 2 * (y * src_w + x);
            real_win[win_idx] = (double)src_data[src_pos];   // 实部
            imag_win[win_idx] = (double)src_data[src_pos+1]; // 虚部
            win_idx++;
        }
    }

    // 实部+虚部分别Sinc插值
    double real_val, imag_val;
    sinc_interp_single(real_win, fx, fy, &real_val);
    sinc_interp_single(imag_win, fx, fy, &imag_val);
    // 裁剪到short范围，写入输出
    out[0] = clipi2(real_val);
    out[1] = clipi2(imag_val);
}

/*=============================================================
  测试克里金变形场（关键位置+抽样，验证映射有效性）
  修复：多字符常量改为普通字符，全局变量ref_w/ref_h→ref_width/ref_height
=============================================================*/
void test_kriging_field(double *sol_x, double *sol_y, double range, int n_pts) {
    printf("\n📈 测试高精度克里金变形场：\n");
    // 9个关键位置（参考SLC四角、中心、四边中点）
    int test_pts[9][2] = {{0,0}, {ref_width/2,0}, {ref_width-1,0}, {0,ref_height/2}, {ref_width/2,ref_height/2},
                          {ref_width-1,ref_height/2}, {0,ref_height-1}, {ref_width/2,ref_height-1}, {ref_width-1,ref_height-1}};
    for (int i = 0; i < 9; i++) {
        double rx = (double)test_pts[i][0], ry = (double)test_pts[i][1];
        double sx = rx + kriging_interp(rx, ry, target_x_points, target_y_points, sol_x, n_pts, range);
        double sy = ry + kriging_interp(rx, ry, target_x_points, target_y_points, sol_y, n_pts, range);
        // 修复：多字符常量警告，改为'Y'/'N'
        char valid = (sx>=0 && sx<src_width && sy>=0 && sy<src_height) ? 'Y' : 'N';
        printf("  参考(%4d,%4d) → 源(%8.2f,%8.2f) [%c]\n", test_pts[i][0], test_pts[i][1], sx, sy, valid);
    }
    // 抽样20×20验证有效比例
    int valid_cnt = 0, sample_cnt = 0;
    for (int y = 0; y < ref_height; y += ref_height/20) {
        for (int x = 0; x < ref_width; x += ref_width/20) {
            double rx = (double)x, ry = (double)y;
            double sx = rx + kriging_interp(rx, ry, target_x_points, target_y_points, sol_x, n_pts, range);
            double sy = ry + kriging_interp(rx, ry, target_x_points, target_y_points, sol_y, n_pts, range);
            if (sx>=0 && sx<src_width && sy>=0 && sy<src_height) valid_cnt++;
            sample_cnt++;
        }
    }
    double valid_ratio = 100.0 * valid_cnt / sample_cnt;
    printf("✅ 变形场抽样有效比例：%.1f%%（%d/%d）\n", valid_ratio, valid_cnt, sample_cnt);
    if (valid_ratio < 50.0) fprintf(stderr, "⚠️  变形场有效比例低于50%，重采样可能大量零值\n");
}

/*=============================================================
  主函数：全流程调度（读取→去异常→SIFT精配准→融合→重采样）
=============================================================*/
int main(int argc, char **argv) {
    // 命令行参数校验（必须6个）
    if (argc != 7) { fprintf(stderr, "%s", USAGE); return EXIT_FAILURE; }
    char *src_slc_file = argv[1];  src_width = atoi(argv[2]);
    char *ref_slc_file = argv[3];  ref_width = atoi(argv[4]);
    char *reg_file = argv[5];      char *out_slc_file = argv[6];
    // 尺寸合法性校验
    if (src_width <=0 || ref_width <=0) { fprintf(stderr, "❌ 源/参考SLC宽度必须为正整数\n"); return EXIT_FAILURE; }
    printf("=========================================\n");
    printf("SLC重采样：源→参考网格（克里金+SIFT精配准）\n");
    printf("=========================================\n");
    printf("源SLC：%s（宽度：%d）\n参考SLC：%s（宽度：%d）\n初始配准点：%s\n输出SLC：%s\n",
           src_slc_file, src_width, ref_slc_file, ref_width, reg_file, out_slc_file);

    // 第一步：读取初始配准点
    printf("\n【第一步：读取初始配准点】\n");
    if (read_registration_points(reg_file, &target_x_points, &target_y_points, &dx_points, &dy_points, &kriging_n_points) != 0) {
        return EXIT_FAILURE;
    }

    // 第二步：初始配准点迭代去异常
    printf("\n【第二步：初始配准点去异常】\n");
    int init_n_pts = kriging_n_points;
    filter_outliers(target_x_points, dx_points, target_y_points, dy_points, &kriging_n_points);
    if (kriging_n_points < 4) { fprintf(stderr, "❌ 去异常后配准点不足4个\n"); return EXIT_FAILURE; }

    // 第三步：读取源SLC和参考SLC（mmap内存映射，高效IO）
    printf("\n【第三步：读取源/参考SLC（mmap）】\n");
    // 读取源SLC
    int fd_src = open(src_slc_file, O_RDONLY);
    if (fd_src < 0) { perror("❌ 打开源SLC失败"); return EXIT_FAILURE; }
    off_t src_file_size = lseek(fd_src, 0, SEEK_END);
    src_height = src_file_size / (src_width * 2 * sizeof(short));
    src_slc_data = (short *)mmap(NULL, src_file_size, PROT_READ, MAP_SHARED, fd_src, 0);
    if (src_slc_data == MAP_FAILED) { perror("❌ 源SLC mmap失败"); close(fd_src); return EXIT_FAILURE; }
    // 读取参考SLC
    int fd_ref = open(ref_slc_file, O_RDONLY);
    if (fd_ref < 0) { perror("❌ 打开参考SLC失败"); munmap(src_slc_data, src_file_size); close(fd_src); return EXIT_FAILURE; }
    off_t ref_file_size = lseek(fd_ref, 0, SEEK_END);
    ref_height = ref_file_size / (ref_width * 2 * sizeof(short));
    ref_slc_data = (short *)mmap(NULL, ref_file_size, PROT_READ, MAP_SHARED, fd_ref, 0);
    if (ref_slc_data == MAP_FAILED) { perror("❌ 参考SLC mmap失败"); munmap(src_slc_data, src_file_size); close(fd_src); close(fd_ref); return EXIT_FAILURE; }
    // 打印SLC信息
    printf("✅ 源SLC：%d×%d 像素，文件大小：%.1f MB\n", src_width, src_height, src_file_size/(1024.0*1024.0));
    printf("✅ 参考SLC：%d×%d 像素，文件大小：%.1f MB\n", ref_width, ref_height, ref_file_size/(1024.0*1024.0));
    if (src_height <=0 || ref_height <=0) { fprintf(stderr, "❌ 无法推导源/参考SLC高度\n"); return EXIT_FAILURE; }

    // 第四步：生成源/参考SLC强度图（用于SIFT匹配）
    printf("\n【第四步：SLC转强度图（高斯去噪）】\n");
    src_intensity = slc2intensity(src_slc_data, src_width, src_height);
    ref_intensity = slc2intensity(ref_slc_data, ref_width, ref_height);
    if (src_intensity.empty() || ref_intensity.empty()) { fprintf(stderr, "❌ 强度图生成失败\n"); return EXIT_FAILURE; }
    printf("✅ 源强度图：%d×%d | 参考强度图：%d×%d\n", src_intensity.cols, src_intensity.rows, ref_intensity.cols, ref_intensity.rows);

    // 第五步：构建全局粗配准克里金变形场（用于SIFT区块粗配准）
    printf("\n【第五步：构建全局粗配准克里金变形场】\n");
    // 计算克里金范围
    // 原错误自动计算逻辑（注释掉）
    // double x_range = *std::max_element(target_x_points, target_x_points+kriging_n_points) - *std::min_element(target_x_points, target_x_points+kriging_n_points);
    // double y_range = *std::max_element(target_y_points, target_y_points+kriging_n_points) - *std::min_element(target_y_points, target_y_points+kriging_n_points);
    // double max_range = std::max(x_range, y_range);
    // double kriging_range_rough = std::max(max_range * KRIGING_RANGE_FACTOR, ref_diag * 0.5);

    // 修复后：固定克里金范围为参考SLC对角线的1/2（合理值，适配绝大多数场景）
    double ref_diag = sqrt((double)ref_width*ref_width + (double)ref_height*ref_height);
    double kriging_range_rough = ref_diag * 0.5; // 核心：固定为参考SLC对角线的1/2
    printf("  克里金范围：%.1f 像素（参考SLC对角线的1/2，覆盖全图）\n", kriging_range_rough);
    // 求解克里金系统（dx/dy方向）
    double *kriging_x_rough = solve_kriging_system(target_x_points, target_y_points, dx_points, kriging_n_points, kriging_range_rough);
    double *kriging_y_rough = solve_kriging_system(target_x_points, target_y_points, dy_points, kriging_n_points, kriging_range_rough);
    if (!kriging_x_rough || !kriging_y_rough) { fprintf(stderr, "❌ 全局克里金求解失败\n"); return EXIT_FAILURE; }
    printf("✅ 全局粗配准克里金变形场构建完成\n");

    // 第六步：64×64区块SIFT二次精配准，生成局部精配准点
    printf("\n【第六步：64×64区块SIFT二次精配准】\n");
    double *sift_t_x = NULL, *sift_t_y = NULL, *sift_dx = NULL, *sift_dy = NULL;
    int n_sift_pts = block_sift_registration(kriging_x_rough, kriging_y_rough, kriging_range_rough, &sift_t_x, &sift_t_y, &sift_dx, &sift_dy);

    // 第七步：融合全局点+SIFT局部点，生成最终配准点集
    printf("\n【第七步：融合配准点集】\n");
    double *fuse_t_x = NULL, *fuse_t_y = NULL, *fuse_dx = NULL, *fuse_dy = NULL;
    int n_fuse_pts = kriging_n_points;
    if (n_sift_pts > 0) {
        n_fuse_pts = fuse_registration_points(target_x_points, target_y_points, dx_points, dy_points, kriging_n_points,
                                              sift_t_x, sift_t_y, sift_dx, sift_dy, n_sift_pts,
                                              &fuse_t_x, &fuse_t_y, &fuse_dx, &fuse_dy);
        if (n_fuse_pts < 4) { fprintf(stderr, "❌ 融合后配准点不足4个\n"); return EXIT_FAILURE; }
        // 替换为融合后的配准点
        free(target_x_points); free(target_y_points); free(dx_points); free(dy_points);
        target_x_points = fuse_t_x; target_y_points = fuse_t_y;
        dx_points = fuse_dx; dy_points = fuse_dy;
        kriging_n_points = n_fuse_pts;
    } else {
        printf("⚠️  无SIFT精配准点，使用全局配准点进行重采样\n");
    }

    // 第八步：构建高精度克里金变形场（基于融合配准点）
    printf("\n【第八步：构建高精度克里金变形场（融合点集）】\n");
    // 重新计算克里金范围（基于融合点）
    // 原错误自动计算逻辑（注释掉）
    // x_range = *std::max_element(target_x_points, target_x_points+kriging_n_points) - *std::min_element(target_x_points, target_x_points+kriging_n_points);
    // y_range = *std::max_element(target_y_points, target_y_points+kriging_n_points) - *std::min_element(target_y_points, target_y_points+kriging_n_points);
    // max_range = std::max(x_range, y_range);
    // kriging_range = std::max(max_range * KRIGING_RANGE_FACTOR, ref_diag * 0.5);

    // 修复后：固定为参考SLC对角线的1/2，与粗配准一致，保证变形场一致性
    kriging_range = ref_diag * 0.5;
    printf("  融合点克里金范围：%.1f 像素（参考SLC对角线的1/2）\n", kriging_range);
    // 求解高精度克里金系统
    kriging_x_weights = solve_kriging_system(target_x_points, target_y_points, dx_points, kriging_n_points, kriging_range);
    kriging_y_weights = solve_kriging_system(target_x_points, target_y_points, dy_points, kriging_n_points, kriging_range);
    if (!kriging_x_weights || !kriging_y_weights) { fprintf(stderr, "❌ 高精度克里金求解失败\n"); return EXIT_FAILURE; }
    // 测试变形场有效性
    test_kriging_field(kriging_x_weights, kriging_y_weights, kriging_range, kriging_n_points);

    // 第九步：创建输出SLC文件（mmap映射，预分配空间，多核并行写入）
    printf("\n【第九步：初始化输出SLC（mmap）】\n");
    long long out_size = (long long)ref_width * ref_height * 2 * sizeof(short);
    int fd_out = open(out_slc_file, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_out < 0) { perror("❌ 创建输出SLC失败"); return EXIT_FAILURE; }
    if (ftruncate(fd_out, out_size) < 0) { perror("❌ 预分配输出SLC空间失败"); close(fd_out); return EXIT_FAILURE; }
    short *out_slc_data = (short *)mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);
    if (out_slc_data == MAP_FAILED) { perror("❌ 输出SLC mmap失败"); close(fd_out); return EXIT_FAILURE; }
    printf("✅ 输出SLC：%d×%d 像素，预分配空间：%.1f MB\n", ref_width, ref_height, out_size/(1024.0*1024.0));

    // 第十步：多核并行重采样（核心步骤，正向映射）
    printf("\n【第十步：多核并行重采样（Sinc插值）】\n");
    int zero_cnt = 0, out_bound_cnt = 0;
    double start_time = omp_get_wtime();
    // OpenMP并行：按行划分，无数据竞争，reduction统计计数
    #pragma omp parallel for schedule(static) reduction(+:zero_cnt, out_bound_cnt)
    for (int ref_y = 0; ref_y < ref_height; ref_y++) {
        // 每10%行打印一次进度，避免刷屏
        if (ref_y % (ref_height / 10) == 0) {
            #pragma omp critical
            {
                double progress = 100.0 * ref_y / ref_height;
                double elapsed = omp_get_wtime() - start_time;
                printf("  进度：%.1f%% | 已用时间：%.1f 秒\n", progress, elapsed);
            }
        }
        // 计算当前行在输出中的起始位置
        size_t line_start = 2 * (size_t)ref_width * ref_y;
        for (int ref_x = 0; ref_x < ref_width; ref_x++) {
            size_t pix_pos = line_start + 2 * ref_x;
            // 高精度克里金插值计算变形量，得到源SLC浮点坐标
            double dx = kriging_interp((double)ref_x, (double)ref_y, target_x_points, target_y_points, kriging_x_weights, kriging_n_points, kriging_range);
            double dy = kriging_interp((double)ref_x, (double)ref_y, target_x_points, target_y_points, kriging_y_weights, kriging_n_points, kriging_range);
            double src_x = (double)ref_x + dx;
            double src_y = (double)ref_y + dy;
            // 边界检查
            if (src_x >=0 && src_x < src_width && src_y >=0 && src_y < src_height) {
                // 8×8 Sinc复数插值
                short pix[2];
                bisinc_interp(src_x, src_y, src_slc_data, src_width, src_height, pix);
                out_slc_data[pix_pos] = pix[0];
                out_slc_data[pix_pos+1] = pix[1];
                if (pix[0] == 0 && pix[1] == 0) zero_cnt++;
            } else {
                // 超出源SLC范围，填充0
                out_slc_data[pix_pos] = 0;
                out_slc_data[pix_pos+1] = 0;
                zero_cnt++;
                out_bound_cnt++;
            }
        }
    }
    // 重采样耗时统计
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    long long total_pix = (long long)ref_width * ref_height;
    double speed = (double)total_pix / (total_time * 10000); // 万像素/秒
    printf("✅ 重采样完成！总时间：%.2f 秒 | 处理速度：%.1f 万像素/秒\n", total_time, speed);

    // 第十一步：统计重采样结果，同步输出到磁盘
    printf("\n【第十一步：结果统计与磁盘同步】\n");
    msync(out_slc_data, out_size, MS_SYNC); // 强制将内存映射写入磁盘
    double zero_ratio = 100.0 * zero_cnt / total_pix;
    double out_bound_ratio = 100.0 * out_bound_cnt / total_pix;
    printf("📊 重采样统计：\n");
    printf("  总像素数：%lld\n", total_pix);
    printf("  零值像素：%d (%.2f%%)\n", zero_cnt, zero_ratio);
    printf("  超出源范围像素：%d (%.2f%%)\n", out_bound_cnt, out_bound_ratio);
    if (zero_ratio > 90.0) {
        fprintf(stderr, "⚠️  零值像素超过90%，请检查配准点或克里金参数\n");
    }

    // 第十二步：释放所有资源（内存映射、文件句柄、动态内存）
    printf("\n【第十二步：释放所有资源】\n");
    // 释放mmap映射
    munmap(src_slc_data, src_file_size);
    munmap(ref_slc_data, ref_file_size);
    munmap(out_slc_data, out_size);
    // 关闭文件句柄
    close(fd_src);
    close(fd_ref);
    close(fd_out);
    // 释放动态分配的内存
    free(target_x_points); free(target_y_points); free(dx_points); free(dy_points);
    free(kriging_x_rough); free(kriging_y_rough); free(kriging_x_weights); free(kriging_y_weights);
    printf("✅ 所有资源释放完成\n");

    printf("\n=========================================\n");
    printf("🎉 重采样完成！输出文件：%s\n", out_slc_file);
    printf("=========================================\n");
    return EXIT_SUCCESS;
}