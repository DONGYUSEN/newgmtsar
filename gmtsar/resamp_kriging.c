/*=============================================================
  resamp_kriging.c - 基于克里金变形场的SLC数据重采样
  
  处理流程（正向映射warping）：
    目标像素(x,y) → 克里金变形场 → 源位置(x+dx,y+dy) → sinc插值 → 写入目标(x,y)
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

/*---------------- 常量定义 ------------------------*/
#define NS 8                     // Sinc插值窗口大小
#define I2MAX 32767              // int16最大值
#define PI 3.1415926535897932
#define KRIGING_RANGE_FACTOR 1.5 // 增加克里金范围因子，确保覆盖整个图像
#define MAX_ITERATIONS 5         // 迭代拟合次数
#define OUTLIER_THRESHOLD 2.0    // 异常点阈值（标准差倍数）
#define KRIGING_NUGGET 0.01      // 克里金块金效应，防止奇异矩阵

/*---------------- 全局变量 ------------------------*/
// 配准点数据：目标位置和变形量
double *target_x_points = NULL;     // 目标图像x坐标
double *target_y_points = NULL;     // 目标图像y坐标
double *dx_points = NULL;           // x方向变形量
double *dy_points = NULL;           // y方向变形量

// 克里金权重（包含拉格朗日乘子，权重[n], 乘子[n+1]）
double *kriging_x_weights = NULL;   // dx方向的克里金权重+乘子
double *kriging_y_weights = NULL;   // dy方向的克里金权重+乘子

double kriging_range = 100.0;       // 变差函数范围
int kriging_n_points = 0;           // 配准点数量

// 图像尺寸
int target_width = 0;               // 目标图像宽度
int target_height = 0;              // 目标图像高度
int source_width = 0;               // 源图像宽度
int source_height = 0;              // 源图像高度

/*-------------------------------------------------------------*/
char *USAGE = 
    "\nUsage: resamp_kriging source.SLC source_width target_width target_height registration.txt output.SLC\n"
    "  source.SLC:       源SLC数据文件（变形前的数据）\n"
    "  source_width:     源图像宽度（像素数）\n"
    "  target_width:     目标图像宽度（理想网格宽度）\n"
    "  target_height:    目标图像高度（理想网格高度）\n"
    "  registration.txt: 配准点文件 (格式: x dx y dy correlation)\n"
    "  output.SLC:       输出重采样后的SLC数据（校正到理想网格）\n\n"
    "处理流程（图像warping）:\n"
    "  1. 读取配准点，建立从目标到源的变形场\n"
    "  2. 对每个目标像素，计算在源图像中的对应位置\n"
    "  3. 从源图像对应位置使用sinc插值获取像素值\n"
    "  4. 将结果写回目标像素位置\n\n"
    "这实现的是正向映射的图像变形（warping）\n";

/*=============================================================
  辅助函数：数值裁剪
=============================================================*/
short clipi2(double x) {
    if (x > I2MAX) return I2MAX;
    if (x < -I2MAX) return -I2MAX;
    return (short)x;
}

/*=============================================================
  读取配准点文件
  格式：x dx y dy correlation
  (x,y): 目标位置
  (x+dx, y+dy): 源位置
=============================================================*/
int read_registration_points(const char *filename, double **target_x, double **target_y,
                            double **dx, double **dy, double **corr, int *n_points) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "无法打开配准点文件: %s\n", filename);
        return -1;
    }
    
    // 第一遍：计算行数
    int count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        // 跳过空行、注释行和只有空白字符的行
        int is_empty = 1;
        for (int i = 0; line[i] != '\0'; i++) {
            if (!isspace(line[i])) {
                is_empty = 0;
                break;
            }
        }
        
        if (!is_empty && line[0] != '#') {
            count++;
        }
    }
    
    if (count == 0) {
        fprintf(stderr, "错误: 配准点文件为空\n");
        fclose(fp);
        return -1;
    }
    
    printf("   找到 %d 行有效数据\n", count);
    
    // 分配内存
    *target_x = (double *)malloc(count * sizeof(double));
    *target_y = (double *)malloc(count * sizeof(double));
    *dx = (double *)malloc(count * sizeof(double));
    *dy = (double *)malloc(count * sizeof(double));
    if (corr) *corr = (double *)malloc(count * sizeof(double));
    
    if (!*target_x || !*target_y || !*dx || !*dy) {
        fprintf(stderr, "错误: 内存分配失败\n");
        fclose(fp);
        return -1;
    }
    
    // 第二遍：读取数据
    rewind(fp);
    int i = 0;
    int line_num = 0;
    
    while (fgets(line, sizeof(line), fp) && i < count) {
        line_num++;
        
        // 跳过空行和注释行
        int is_empty = 1;
        for (int j = 0; line[j] != '\0'; j++) {
            if (!isspace(line[j])) {
                is_empty = 0;
                break;
            }
        }
        
        if (is_empty || line[0] == '#') continue;
        
        double x, dx_val, y, dy_val, correlation = 1.0;
        // 格式：x dx y dy correlation
        int n = sscanf(line, "%lf %lf %lf %lf %lf", &x, &dx_val, &y, &dy_val, &correlation);
        
        if (n >= 4) {
            (*target_x)[i] = x;
            (*target_y)[i] = y;
            (*dx)[i] = dx_val;
            (*dy)[i] = dy_val;
            if (corr && n >= 5) (*corr)[i] = correlation;
            
            i++;
        } else if (n > 0) {
            fprintf(stderr, "警告: 第%d行格式错误: %s", line_num, line);
        }
    }
    
    *n_points = i;
    fclose(fp);
    
    if (i < count) {
        fprintf(stderr, "警告: 只读取到 %d/%d 个有效点\n", i, count);
    }
    
    if (i < 4) {
        fprintf(stderr, "错误: 有效点数太少 (%d)，需要至少4个\n", i);
        return -1;
    }
    
    return 0;
}

/*=============================================================
  二次线性拟合（y = a0 + a1*x + a2*x²）
=============================================================*/
void quadratic_fit(double *x, double *y, int n, double *coeffs) {
    // 初始化统计量
    double sum_x = 0.0, sum_x2 = 0.0, sum_x3 = 0.0, sum_x4 = 0.0;
    double sum_y = 0.0, sum_xy = 0.0, sum_x2y = 0.0;
    
    // 计算统计量
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        double xi2 = xi * xi;
        double yi = y[i];
        
        sum_x += xi;
        sum_x2 += xi2;
        sum_x3 += xi2 * xi;
        sum_x4 += xi2 * xi2;
        sum_y += yi;
        sum_xy += xi * yi;
        sum_x2y += xi2 * yi;
    }
    
    // 构建正规方程矩阵
    double A[3][3] = {
        {(double)n, sum_x, sum_x2},
        {sum_x, sum_x2, sum_x3},
        {sum_x2, sum_x3, sum_x4}
    };
    
    double b[3] = {sum_y, sum_xy, sum_x2y};
    
    // 解线性方程组（使用高斯消元法）
    for (int i = 0; i < 3; i++) {
        // 寻找主元
        int pivot = i;
        double max_val = fabs(A[i][i]);
        for (int j = i + 1; j < 3; j++) {
            if (fabs(A[j][i]) > max_val) {
                max_val = fabs(A[j][i]);
                pivot = j;
            }
        }
        
        if (max_val < 1e-12) {
            // 如果矩阵奇异，使用线性拟合
            coeffs[0] = 0.0;
            coeffs[1] = (sum_xy - sum_x * sum_y / n) / (sum_x2 - sum_x * sum_x / n);
            coeffs[2] = 0.0;
            return;
        }
        
        // 交换行
        if (pivot != i) {
            for (int k = i; k < 3; k++) {
                double temp = A[i][k];
                A[i][k] = A[pivot][k];
                A[pivot][k] = temp;
            }
            double temp = b[i];
            b[i] = b[pivot];
            b[pivot] = temp;
        }
        
        // 消元
        double diag = A[i][i];
        for (int j = i + 1; j < 3; j++) {
            double factor = A[j][i] / diag;
            for (int k = i; k < 3; k++) {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
    
    // 回代
    coeffs[2] = b[2] / A[2][2];
    coeffs[1] = (b[1] - A[1][2] * coeffs[2]) / A[1][1];
    coeffs[0] = (b[0] - A[0][1] * coeffs[1] - A[0][2] * coeffs[2]) / A[0][0];
}

/*=============================================================
  计算拟合值
=============================================================*/
double evaluate_fit(double x, double *coeffs) {
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x * x;
}

/*=============================================================
  计算RMSE
=============================================================*/
double calculate_rmse(double *x, double *y, int n, double *coeffs) {
    double sum_sq_error = 0.0;
    for (int i = 0; i < n; i++) {
        double predicted = evaluate_fit(x[i], coeffs);
        double error = y[i] - predicted;
        sum_sq_error += error * error;
    }
    return sqrt(sum_sq_error / n);
}

/*=============================================================
  迭代剔除异常点
  对x-dx和y-dy分别进行5次迭代二次线性拟合
  每次剔除超过OUTLIER_THRESHOLD倍标准差的点
=============================================================*/
void filter_outliers(double *x, double *dx, double *y, double *dy, int *n_points, int max_iterations) {
    int n = *n_points;
    int *valid = (int *)malloc(n * sizeof(int));
    if (!valid) {
        fprintf(stderr, "错误: 内存分配失败\n");
        return;
    }
    
    // 初始所有点都有效
    for (int i = 0; i < n; i++) {
        valid[i] = 1;
    }
    
    printf("   迭代拟合过程:\n");
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // 统计有效点数量
        int valid_count = 0;
        for (int i = 0; i < n; i++) {
            if (valid[i]) valid_count++;
        }
        
        if (valid_count < 4) {
            printf("     迭代%d: 有效点太少(%d)，停止迭代\n", iter+1, valid_count);
            break;
        }
        
        // 为有效点分配临时数组
        double *valid_x = (double *)malloc(valid_count * sizeof(double));
        double *valid_dx = (double *)malloc(valid_count * sizeof(double));
        double *valid_y = (double *)malloc(valid_count * sizeof(double));
        double *valid_dy = (double *)malloc(valid_count * sizeof(double));
        
        if (!valid_x || !valid_dx || !valid_y || !valid_dy) {
            fprintf(stderr, "错误: 内存分配失败\n");
            free(valid_x); free(valid_dx); free(valid_y); free(valid_dy);
            break;
        }
        
        // 填充有效点数据
        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (valid[i]) {
                valid_x[idx] = x[i];
                valid_dx[idx] = dx[i];
                valid_y[idx] = y[i];
                valid_dy[idx] = dy[i];
                idx++;
            }
        }
        
        // 对x-dx进行二次拟合
        double coeffs_x[3];
        quadratic_fit(valid_x, valid_dx, valid_count, coeffs_x);
        double rmse_x = calculate_rmse(valid_x, valid_dx, valid_count, coeffs_x);
        
        // 对y-dy进行二次拟合
        double coeffs_y[3];
        quadratic_fit(valid_y, valid_dy, valid_count, coeffs_y);
        double rmse_y = calculate_rmse(valid_y, valid_dy, valid_count, coeffs_y);
        
        // 计算残差并标记异常点
        int outliers_removed = 0;
        idx = 0;
        for (int i = 0; i < n; i++) {
            if (valid[i]) {
                double predicted_dx = evaluate_fit(valid_x[idx], coeffs_x);
                double predicted_dy = evaluate_fit(valid_y[idx], coeffs_y);
                double residual_x = fabs(valid_dx[idx] - predicted_dx) / rmse_x;
                double residual_y = fabs(valid_dy[idx] - predicted_dy) / rmse_y;
                
                // 如果任一方向的残差超过阈值，标记为异常点
                if (residual_x > OUTLIER_THRESHOLD || residual_y > OUTLIER_THRESHOLD) {
                    valid[i] = 0;
                    outliers_removed++;
                }
                idx++;
            }
        }
        
        printf("     迭代%d: %d个有效点, RMSE_x=%.4f, RMSE_y=%.4f, 剔除%d个异常点\n",
               iter+1, valid_count, rmse_x, rmse_y, outliers_removed);
        
        free(valid_x);
        free(valid_dx);
        free(valid_y);
        free(valid_dy);
        
        if (outliers_removed == 0) {
            printf("     没有发现新的异常点，停止迭代\n");
            break;
        }
    }
    
    // 收集有效点
    int new_n = 0;
    for (int i = 0; i < n; i++) {
        if (valid[i]) {
            x[new_n] = x[i];
            dx[new_n] = dx[i];
            y[new_n] = y[i];
            dy[new_n] = dy[i];
            new_n++;
        }
    }
    
    *n_points = new_n;
    free(valid);
}

/*=============================================================
  球形变差函数（修复版本）
=============================================================*/
double variogram_spherical(double h, double range) {
    if (h <= 0) return 0.0;
    if (h >= range) return 1.0;
    
    double h_r = h / range;
    return 1.5 * h_r - 0.5 * h_r * h_r * h_r;
}

/*=============================================================
  创建克里金矩阵（修复版本）
=============================================================*/
double **create_kriging_matrix(double *x, double *y, int n_points, double range) {
    double **matrix = (double **)malloc((n_points + 1) * sizeof(double *));
    for (int i = 0; i <= n_points; i++) {
        matrix[i] = (double *)calloc(n_points + 2, sizeof(double));
    }
    
    // 填充矩阵（克里金系统）
    for (int i = 0; i < n_points; i++) {
        for (int j = 0; j < n_points; j++) {
            if (i == j) {
                // 对角线：块金效应 + 变差函数
                matrix[i][j] = KRIGING_NUGGET;
            } else {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                double dist = sqrt(dx * dx + dy * dy);
                matrix[i][j] = -variogram_spherical(dist, range);
            }
        }
        matrix[i][n_points] = 1.0;  // 拉格朗日乘子列
    }
    
    // 最后一行：无偏估计约束
    for (int j = 0; j < n_points; j++) {
        matrix[n_points][j] = 1.0;
    }
    matrix[n_points][n_points] = 0.0;
    
    return matrix;
}

/*=============================================================
  求解克里金系统（修复版本：正确回代拉格朗日乘子）
=============================================================*/
double *solve_kriging_system(double *x, double *y, double *values, int n_points, double range) {
    double **matrix = create_kriging_matrix(x, y, n_points, range);
    int n = n_points + 1;  // 矩阵大小：n x n+1（含右侧向量）
    
    // 设置右侧向量
    for (int i = 0; i < n_points; i++) {
        matrix[i][n_points + 1] = values[i];
    }
    matrix[n_points][n_points + 1] = 0.0;
    
    // 高斯消元：前向消元为上三角矩阵
    for (int i = 0; i < n; i++) {
        // 寻找主元
        int pivot = i;
        double max_val = fabs(matrix[i][i]);
        for (int j = i + 1; j < n; j++) {
            if (fabs(matrix[j][i]) > max_val) {
                max_val = fabs(matrix[j][i]);
                pivot = j;
            }
        }
        
        if (max_val < 1e-12) {
            fprintf(stderr, "警告: 矩阵接近奇异 (行%d, 值=%e)\n", i, max_val);
        }
        
        // 交换行
        if (pivot != i) {
            double *temp = matrix[i];
            matrix[i] = matrix[pivot];
            matrix[pivot] = temp;
        }
        
        // 归一化主元行
        double diag = matrix[i][i];
        if (fabs(diag) < 1e-12) diag = 1.0;
        for (int k = i; k < n + 1; k++) {
            matrix[i][k] /= diag;
        }
        
        // 消元其他行
        for (int j = 0; j < n; j++) {
            if (j != i && fabs(matrix[j][i]) > 1e-12) {
                double factor = matrix[j][i];
                for (int k = i; k < n + 1; k++) {
                    matrix[j][k] -= factor * matrix[i][k];
                }
            }
        }
    }
    
    // 提取解：前n_points个是权重，最后1个是拉格朗日乘子
    double *solution = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        solution[i] = matrix[i][n];
    }
    
    // 检查权重总和（应为1或接近1）
    double weight_sum = 0.0;
    for (int i = 0; i < n_points; i++) {
        weight_sum += solution[i];
    }
    printf("     权重总和检查: %.6f (应为1.0)\n", weight_sum);
    
    // 释放矩阵内存
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
    return solution;
}

/*=============================================================
  克里金插值（核心修复：普通克里金正确计算逻辑）
=============================================================*/
double kriging_interpolate(double x, double y, double *sample_x, double *sample_y, 
                          double *solution, int n_points, double range) {
    double result = 0.0;
    double lambda = solution[n_points];  // 拉格朗日乘子
    double sum_gamma = 0.0;
    
    // 步骤1：计算待插值点与所有配准点的变差函数值，并累加权重*变差函数
    for (int i = 0; i < n_points; i++) {
        double dx = x - sample_x[i];
        double dy = y - sample_y[i];
        double dist = sqrt(dx * dx + dy * dy);
        double gamma = variogram_spherical(dist, range);  // 移除负号，正确变差函数
        result += solution[i] * gamma;
        sum_gamma += gamma;
    }
    
    // 步骤2：加上拉格朗日乘子（普通克里金无偏约束）
    result += lambda;
    
    return result;
}

/*=============================================================
  计算源图像X坐标
=============================================================*/
double compute_source_x(double target_x, double target_y) {
    // 使用克里金插值计算x方向变形量（传入完整solution：权重+乘子）
    double dx = kriging_interpolate(target_x, target_y, target_x_points, target_y_points,
                                   kriging_x_weights, kriging_n_points, kriging_range);
    
    // 源坐标 = 目标坐标 + 变形量
    return target_x + dx;
}

/*=============================================================
  计算源图像Y坐标
=============================================================*/
double compute_source_y(double target_x, double target_y) {
    // 使用克里金插值计算y方向变形量（传入完整solution：权重+乘子）
    double dy = kriging_interpolate(target_x, target_y, target_x_points, target_y_points,
                                   kriging_y_weights, kriging_n_points, kriging_range);
    
    // 源坐标 = 目标坐标 + 变形量
    return target_y + dy;
}

/*=============================================================
  测试克里金变形场覆盖情况
=============================================================*/
void test_kriging_field(int target_width, int target_height) {
    printf("   测试点覆盖检查:\n");
    
    // 测试9个关键位置
    int test_positions[9][2] = {
        {0, 0},  // 左上角
        {target_width/2, 0},  // 上中
        {target_width-1, 0},  // 右上角
        {0, target_height/2},  // 左中
        {target_width/2, target_height/2},  // 中心
        {target_width-1, target_height/2},  // 右中
        {0, target_height-1},  // 左下角
        {target_width/2, target_height-1},  // 下中
        {target_width-1, target_height-1}   // 右下角
    };
    
    for (int i = 0; i < 9; i++) {
        int x = test_positions[i][0];
        int y = test_positions[i][1];
        
        double source_x = compute_source_x((double)x, (double)y);
        double source_y = compute_source_y((double)x, (double)y);
        
        printf("     目标(%4d,%4d) -> 源(%8.2f,%8.2f)", x, y, source_x, source_y);
        
        if (source_x >= 0 && source_x < source_width && 
            source_y >= 0 && source_y < source_height) {
            printf(" [有效]\n");
        } else {
            printf(" [超出边界]\n");
        }
    }
    
    // 检查变形量是否合理
    printf("\n   变形量合理性检查:\n");
    int inside_count = 0;
    int total_samples = 0;
    
    // 抽样检查
    for (int y = 0; y < target_height; y += target_height/20) {
        for (int x = 0; x < target_width; x += target_width/20) {
            double source_x = compute_source_x((double)x, (double)y);
            double source_y = compute_source_y((double)x, (double)y);
            
            if (source_x >= 0 && source_x < source_width && 
                source_y >= 0 && source_y < source_height) {
                inside_count++;
            }
            total_samples++;
        }
    }
    
    double inside_ratio = 100.0 * inside_count / total_samples;
    printf("     抽样点中 %.1f%% 在源图像范围内\n", inside_ratio);
    
    if (inside_ratio < 50.0) {
        printf("     警告: 很多像素映射到源图像外，可能需要调整变形场\n");
    }
}

/*=============================================================
  sinc插值核函数
=============================================================*/
double sinc_kernel(double x) {
    double arg = fabs(PI * x);
    if (arg > 1e-8) {
        return sin(arg) / arg;
    } else {
        return 1.0;
    }
}

/*=============================================================
  sinc插值函数（修复窗口半宽参数）
=============================================================*/
void sinc_one(double *rdata, double *idata, double x, double y, double *cz) {
    int i, j, ij;
    const int ns2 = NS / 2;  // 核心修复：窗口半宽为4（NS=8），而非3
    double wx[NS], wy[NS];
    double w, wsum = 0.0, rsum = 0.0, isum = 0.0;
    
    // 计算权重：x/y为小数部分，窗口中心为0
    for (i = 0; i < NS; i++) {
        wx[i] = sinc_kernel(x - (i - ns2));
        wy[i] = sinc_kernel(y - (i - ns2));
    }
    
    ij = 0;
    for (j = 0; j < NS; j++) {
        for (i = 0; i < NS; i++) {
            w = wx[i] * wy[j];
            rsum += rdata[ij + i] * w;
            isum += idata[ij + i] * w;
            wsum += w;
        }
        ij += NS;
    }
    
    if (wsum > 1e-8) {
        cz[0] = rsum / wsum;
        cz[1] = isum / wsum;
    } else {
        cz[0] = 0.0;
        cz[1] = 0.0;
    }
}

/*=============================================================
  双sinc插值（针对SLC复数数据，修复边界判断和窗口参数）
=============================================================*/
void bisinc(double source_x, double source_y, short *source_data, 
           int source_height, int source_width, short *output) {
    // 初始化输出为0
    output[0] = 0;
    output[1] = 0;
    
    const int ns2 = NS / 2;  // 核心修复：窗口半宽
    // 计算整数部分和小数部分（无四舍五入，直接取整，避免偏移）
    int j0 = (int)floor(source_x);  // x方向整数坐标（列）
    int i0 = (int)floor(source_y);  // y方向整数坐标（行）
    double dr = source_x - j0;      // x方向小数部分
    double da = source_y - i0;      // y方向小数部分
    
    // 唯一边界检查：确保插值窗口完全在源图像内
    if (i0 - ns2 < 0 || i0 + ns2 >= source_height ||
        j0 - ns2 < 0 || j0 + ns2 >= source_width) {
        return;  // 窗口越界，返回0
    }
    
    // 分配插值窗口缓存
    double rdata[NS * NS], idata[NS * NS], cz[2];
    short *line_ptr = source_data + 2 * (size_t)source_width * (i0 - ns2);
    
    // 提取NS×NS的复数插值窗口（实部+虚部）
    for (int i = 0; i < NS; i++) {
        short *row_ptr = line_ptr + 2 * (size_t)source_width * i;
        for (int j = 0; j < NS; j++) {
            int k = i * NS + j;
            int kk = 2 * (j0 + j - ns2);
            rdata[k] = (double)row_ptr[kk];      // 实部
            idata[k] = (double)row_ptr[kk + 1];  // 虚部
        }
    }
    
    // 执行sinc插值
    sinc_one(rdata, idata, dr, da, cz);
    
    // 裁剪到int16范围并赋值
    output[0] = clipi2(cz[0]);
    output[1] = clipi2(cz[1]);
}

/*=============================================================
  主函数（修复并行写入、权重统计）
=============================================================*/
int main(int argc, char **argv) {
    
    if (argc != 7) {
        fprintf(stderr, "%s", USAGE);
        return EXIT_FAILURE;
    }
    
    char *source_file = argv[1];
    source_width = atoi(argv[2]);
    target_width = atoi(argv[3]);
    target_height = atoi(argv[4]);
    char *registration_file = argv[5];
    char *output_file = argv[6];
    
    if (source_width <= 0 || target_width <= 0 || target_height <= 0) {
        fprintf(stderr, "错误: 宽度和高度必须为正数\n");
        return EXIT_FAILURE;
    }
    
    printf("=========================================\n");
    printf("SLC数据重采样 - 图像Warping（正向映射）\n");
    printf("=========================================\n\n");
    
    // 1. 读取配准点文件
    printf("1. 读取配准点文件: %s\n", registration_file);
    printf("   格式: x dx y dy correlation\n");
    
    if (read_registration_points(registration_file, &target_x_points, &target_y_points,
                               &dx_points, &dy_points, NULL, &kriging_n_points) != 0) {
        return EXIT_FAILURE;
    }
    
    if (kriging_n_points < 4) {
        fprintf(stderr, "错误: 需要至少4个配准点，当前只有 %d 个\n", kriging_n_points);
        return EXIT_FAILURE;
    }
    
    printf("   读取到 %d 个配准点\n", kriging_n_points);
    
    // 显示配准点统计
    printf("\n   配准点变形量统计:\n");
    double dx_min = dx_points[0], dx_max = dx_points[0], dx_sum = 0;
    double dy_min = dy_points[0], dy_max = dy_points[0], dy_sum = 0;
    
    for (int i = 0; i < kriging_n_points; i++) {
        if (dx_points[i] < dx_min) dx_min = dx_points[i];
        if (dx_points[i] > dx_max) dx_max = dx_points[i];
        if (dy_points[i] < dy_min) dy_min = dy_points[i];
        if (dy_points[i] > dy_max) dy_max = dy_points[i];
        dx_sum += dx_points[i];
        dy_sum += dy_points[i];
    }
    
    printf("     dx范围: [%.4f, %.4f], 平均: %.4f\n", dx_min, dx_max, dx_sum/kriging_n_points);
    printf("     dy范围: [%.4f, %.4f], 平均: %.4f\n", dy_min, dy_max, dy_sum/kriging_n_points);
    
    // 2. 迭代二次线性拟合，剔除异常点
    printf("\n2. 迭代二次线性拟合，剔除异常点\n");
    
    int original_n_points = kriging_n_points;
    
    // 进行迭代拟合
    filter_outliers(target_x_points, dx_points, target_y_points, dy_points, 
                   &kriging_n_points, MAX_ITERATIONS);
    
    printf("   剔除异常点后剩余 %d 个点 (剔除 %d 个点)\n", 
           kriging_n_points, original_n_points - kriging_n_points);
    
    if (kriging_n_points < 4) {
        fprintf(stderr, "错误: 剔除异常点后点数太少 (%d)，需要至少4个\n", kriging_n_points);
        return EXIT_FAILURE;
    }
    
    // 3. 确定源图像高度
    printf("\n3. 确定源图像尺寸\n");
    FILE *fp = fopen(source_file, "rb");
    if (!fp) {
        fprintf(stderr, "错误: 无法打开源文件 %s\n", source_file);
        return EXIT_FAILURE;
    }
    
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fclose(fp);
    
    // SLC数据：每个像素2个short（实部和虚部）
    source_height = file_size / (source_width * 2 * sizeof(short));
    
    if (source_height <= 0) {
        fprintf(stderr, "错误: 无法确定源图像高度\n");
        fprintf(stderr, "      文件大小: %ld 字节\n", file_size);
        fprintf(stderr, "      源宽度: %d 像素\n", source_width);
        return EXIT_FAILURE;
    }
    
    printf("   源图像尺寸: %d x %d\n", source_width, source_height);
    printf("   目标图像尺寸: %d x %d\n", target_width, target_height);
    printf("   源文件大小: %ld 字节\n", file_size);
    
    // 4. 计算克里金范围（关键修改：确保覆盖整个图像）
    printf("\n4. 计算克里金参数\n");
    
    // 计算目标图像对角线长度作为最大距离
    double target_diagonal = sqrt((double)target_width * target_width + (double)target_height * target_height);
    
    // 计算配准点覆盖范围
    double min_x = target_x_points[0], max_x = target_x_points[0];
    double min_y = target_y_points[0], max_y = target_y_points[0];
    
    for (int i = 1; i < kriging_n_points; i++) {
        if (target_x_points[i] < min_x) min_x = target_x_points[i];
        if (target_x_points[i] > max_x) max_x = target_x_points[i];
        if (target_y_points[i] < min_y) min_y = target_y_points[i];
        if (target_y_points[i] > max_y) max_y = target_y_points[i];
    }
    
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    double max_range = (range_x > range_y) ? range_x : range_y;
    
    // 关键：使用较大的范围确保整个图像都有变形场
    kriging_range = fmax(max_range * KRIGING_RANGE_FACTOR, target_diagonal * 0.5);
    
    printf("   目标点范围: x=[%.1f, %.1f], y=[%.1f, %.1f]\n", min_x, max_x, min_y, max_y);
    printf("   目标图像对角线: %.1f 像素\n", target_diagonal);
    printf("   克里金范围: %.1f 像素 (确保覆盖整个图像)\n", kriging_range);
    
    // 5. 计算克里金权重（包含拉格朗日乘子）
    printf("\n5. 计算克里金权重（建立变形场）\n");
    
    // 计算dx方向的权重+乘子
    printf("   计算dx方向变形场...\n");
    kriging_x_weights = solve_kriging_system(target_x_points, target_y_points, 
                                            dx_points, kriging_n_points, kriging_range);
    
    // 计算dy方向的权重+乘子
    printf("   计算dy方向变形场...\n");
    kriging_y_weights = solve_kriging_system(target_x_points, target_y_points,
                                            dy_points, kriging_n_points, kriging_range);
    
    if (!kriging_x_weights || !kriging_y_weights) {
        fprintf(stderr, "错误: 克里金系统求解失败\n");
        return EXIT_FAILURE;
    }
    
    printf("   克里金变形场建立完成\n");
    printf("   权重统计（不含拉格朗日乘子）:\n");
    
    // 检查权重是否合理（仅统计前n_points个权重）
    double weight_sum_x = 0, weight_sum_y = 0;
    double weight_min_x = kriging_x_weights[0], weight_max_x = kriging_x_weights[0];
    double weight_min_y = kriging_y_weights[0], weight_max_y = kriging_y_weights[0];
    
    for (int i = 0; i < kriging_n_points; i++) {
        weight_sum_x += kriging_x_weights[i];
        weight_sum_y += kriging_y_weights[i];
        if (kriging_x_weights[i] < weight_min_x) weight_min_x = kriging_x_weights[i];
        if (kriging_x_weights[i] > weight_max_x) weight_max_x = kriging_x_weights[i];
        if (kriging_y_weights[i] < weight_min_y) weight_min_y = kriging_y_weights[i];
        if (kriging_y_weights[i] > weight_max_y) weight_max_y = kriging_y_weights[i];
    }
    
    printf("     dx权重: 范围[%.6f, %.6f], 总和=%.6f\n", weight_min_x, weight_max_x, weight_sum_x);
    printf("     dy权重: 范围[%.6f, %.6f], 总和=%.6f\n", weight_min_y, weight_max_y, weight_sum_y);
    printf("     拉格朗日乘子: dx=%.6f, dy=%.6f\n", kriging_x_weights[kriging_n_points], kriging_y_weights[kriging_n_points]);
    
    // 6. 测试变形场
    printf("\n6. 测试变形场覆盖情况\n");
    test_kriging_field(target_width, target_height);
    
    // 7. 读取源图像数据（SLC复数数据，mmap映射）
    printf("\n7. 读取源图像数据（SLC格式）\n");
    int fdin;
    size_t st_size = (size_t)source_width * (size_t)source_height * 2 * sizeof(short);
    
    printf("   打开源文件: %s\n", source_file);
    if ((fdin = open(source_file, O_RDONLY)) < 0) {
        perror("无法打开源SLC");
        return EXIT_FAILURE;
    }
    
    short *source_data = mmap(NULL, st_size, PROT_READ, MAP_SHARED, fdin, 0);
    if (source_data == MAP_FAILED) {
        perror("mmap失败");
        close(fdin);
        return EXIT_FAILURE;
    }
    
    printf("   源数据大小: %ld 字节\n", st_size);
    printf("   数据类型: short (复数: 实部 + 虚部)\n");
    
    // 8. 创建输出文件并预分配空间（mmap映射输出，避免磁盘IO竞争）
    printf("\n8. 创建输出文件\n");
    int fdout = open(output_file, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fdout < 0) {
        perror("无法创建输出SLC");
        munmap(source_data, st_size);
        close(fdin);
        return EXIT_FAILURE;
    }
    long long output_size = (long long)target_width * target_height * 2 * sizeof(short);
    // 预分配文件空间
    if (ftruncate(fdout, output_size) < 0) {
        perror("无法预分配输出文件空间");
        close(fdout);
        munmap(source_data, st_size);
        close(fdin);
        return EXIT_FAILURE;
    }
    // mmap映射输出文件到内存，并行直接写入内存，无磁盘竞争
    short *output_data = mmap(NULL, output_size, PROT_READ | PROT_WRITE, MAP_SHARED, fdout, 0);
    if (output_data == MAP_FAILED) {
        perror("输出文件mmap失败");
        close(fdout);
        munmap(source_data, st_size);
        close(fdin);
        return EXIT_FAILURE;
    }
    printf("   输出文件: %s (%.1f MB)\n", output_file, output_size / (1024.0 * 1024.0));
    
    // 9. 重采样处理（正向映射warping，OpenMP并行优化）
    printf("\n9. 开始重采样处理（图像Warping）\n");
    printf("   处理方式: 对每个目标像素计算源坐标 -> sinc插值 -> 写入目标位置\n");
    printf("   并行策略: 内存映射输出，无磁盘IO竞争，满效率并行\n\n");
    
    int zero_count = 0;
    int out_of_bound_count = 0;
    double start_time = omp_get_wtime();
    
    // OpenMP并行处理：按行划分，无竞争，无需critical
    #pragma omp parallel for schedule(static) reduction(+:zero_count, out_of_bound_count)
    for (int target_y = 0; target_y < target_height; target_y++) {
        // 计算当前行在输出内存中的起始位置
        size_t line_start = 2 * (size_t)target_width * target_y;
        // 每10%行打印一次进度，避免刷屏
        if (target_y % (target_height / 10) == 0) {
            #pragma omp critical
            {
                double progress = 100.0 * target_y / target_height;
                double elapsed = omp_get_wtime() - start_time;
                printf("     进度: %.1f%% (已用: %.1fs)\n", progress, elapsed);
            }
        }
        
        for (int target_x = 0; target_x < target_width; target_x++) {
            // 步骤1: 计算对应的源位置（正向映射）
            double source_x = compute_source_x((double)target_x, (double)target_y);
            double source_y = compute_source_y((double)target_x, (double)target_y);
            // 计算当前像素在输出内存中的位置
            size_t pix_pos = line_start + 2 * target_x;
            
            // 步骤2: 检查是否在源图像范围内
            if (source_x >= 0 && source_x < source_width && 
                source_y >= 0 && source_y < source_height) {
                // 步骤3: 使用sinc插值获取复数像素值
                short pix[2];
                bisinc(source_x, source_y, source_data, source_height, source_width, pix);
                // 写入输出内存
                output_data[pix_pos] = pix[0];
                output_data[pix_pos + 1] = pix[1];
                // 统计零值像素
                if (pix[0] == 0 && pix[1] == 0) zero_count++;
            } else {
                // 超出边界，填充0
                output_data[pix_pos] = 0;
                output_data[pix_pos + 1] = 0;
                zero_count++;
                out_of_bound_count++;
            }
        }
    }
    
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    
    // 10. 处理完成，同步输出内存到磁盘
    msync(output_data, output_size, MS_SYNC);
    munmap(output_data, output_size);
    close(fdout);
    
    printf("\n10. 处理完成\n");
    printf("   总时间: %.2f 秒\n", total_time);
    printf("   处理速度: %.1f 万像素/秒\n", 
           (target_width * target_height) / (total_time * 10000));
    
    printf("\n11. 统计信息\n");
    long long total_pixels = (long long)target_width * target_height;
    printf("   总像素数: %lld\n", total_pixels);
    printf("   零值像素: %d (%.2f%%)\n", 
           zero_count, 100.0 * zero_count / total_pixels);
    printf("   超出边界: %d (%.2f%%)\n", 
           out_of_bound_count, 100.0 * out_of_bound_count / total_pixels);
    
    if (zero_count > total_pixels * 0.9) {
        printf("\n   警告: 超过90%%的像素是零值！\n");
        printf("   可能原因:\n");
        printf("     1. 配准点的变形量过大，导致源坐标超出图像范围\n");
        printf("     2. 克里金范围设置不合理，变形场外推失效\n");
        printf("     3. 配准点分布不均，部分区域无有效变形场\n");
    }
    
    // 12. 清理资源
    munmap(source_data, st_size);
    close(fdin);
    
    // 释放内存
    free(target_x_points);
    free(target_y_points);
    free(dx_points);
    free(dy_points);
    free(kriging_x_weights);
    free(kriging_y_weights);
    
    printf("\n输出文件: %s\n", output_file);
    printf("文件大小: %.1f MB\n", output_size / (1024.0 * 1024.0));
    printf("=========================================\n");
    
    return EXIT_SUCCESS;
}