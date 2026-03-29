#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define MAX_POINTS 5000
#define MAX_COEFFS 7  // 最大7个系数: a0, a1, a2, a3, a4, a5, a6

// 数据点结构
typedef struct {
    double x;      // 原始X坐标
    double dx;     // X方向位移
    double y;      // 原始Y坐标
    double dy;     // Y方向位移
    double r;      // 相关系数
    int valid;     // 是否有效点
} DataPoint;

// 矩阵结构
typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

// 创建矩阵
Matrix* create_matrix(int rows, int cols) {
    Matrix *mat = (Matrix*)malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat->data[i] = (double*)malloc(cols * sizeof(double));
        memset(mat->data[i], 0, cols * sizeof(double));
    }
    return mat;
}

// 释放矩阵
void free_matrix(Matrix *mat) {
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
    free(mat);
}

// 高斯消元法
int gauss_elimination(Matrix *A, double *b, double *x, int n) {
    double **aug = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        aug[i] = (double*)malloc((n + 1) * sizeof(double));
        for (int j = 0; j < n; j++) {
            aug[i][j] = A->data[i][j];
        }
        aug[i][n] = b[i];
    }
    
    for (int i = 0; i < n; i++) {
        // 寻找主元
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(aug[k][i]) > fabs(aug[max_row][i])) {
                max_row = k;
            }
        }
        
        if (max_row != i) {
            double *temp = aug[i];
            aug[i] = aug[max_row];
            aug[max_row] = temp;
        }
        
        if (fabs(aug[i][i]) < 1e-15) {
            for (int j = 0; j < n; j++) free(aug[j]);
            free(aug);
            return 0;
        }
        
        double pivot = aug[i][i];
        for (int j = i; j <= n; j++) {
            aug[i][j] /= pivot;
        }
        
        for (int k = i + 1; k < n; k++) {
            double factor = aug[k][i];
            for (int j = i; j <= n; j++) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }
    
    for (int i = n - 1; i >= 0; i--) {
        x[i] = aug[i][n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= aug[i][j] * x[j];
        }
    }
    
    for (int i = 0; i < n; i++) free(aug[i]);
    free(aug);
    return 1;
}

// 计算基函数值：根据系数个数选择模型
void compute_basis_functions(double x, double y, double *basis, int num_coeffs) {
    // 根据系数个数选择不同的基函数组合
    switch (num_coeffs) {
        case 1:  // 只有常数项
            basis[0] = 1.0;
            break;
        case 2:  // 常数项 + x
            basis[0] = 1.0;
            basis[1] = x;
            break;
        case 3:  // 常数项 + x + y
            basis[0] = 1.0;
            basis[1] = x;
            basis[2] = y;
            break;
        case 4:  // 常数项 + x + y + x*y
            basis[0] = 1.0;
            basis[1] = x;
            basis[2] = y;
            basis[3] = x * y;
            break;
        case 5:  // 常数项 + x + y + x*y + x^2
            basis[0] = 1.0;
            basis[1] = x;
            basis[2] = y;
            basis[3] = x * y;
            basis[4] = x * x;
            break;
        case 6:  // 常数项 + x + y + x*y + x^2 + y^2
            basis[0] = 1.0;
            basis[1] = x;
            basis[2] = y;
            basis[3] = x * y;
            basis[4] = x * x;
            basis[5] = y * y;
            break;
        case 7:  // 常数项 + x + y + x*y + x^2 + y^2 + x^2*y
            basis[0] = 1.0;
            basis[1] = x;
            basis[2] = y;
            basis[3] = x * y;
            basis[4] = x * x;
            basis[5] = y * y;
            basis[6] = x * x * y;
            break;
        default:  // 默认使用6个系数
            basis[0] = 1.0;
            basis[1] = x;
            basis[2] = y;
            basis[3] = x * y;
            basis[4] = x * x;
            basis[5] = y * y;
            break;
    }
}

// 获取系数个数的描述
const char* get_model_description(int num_coeffs) {
    switch (num_coeffs) {
        case 1: return "常数项 (dx = a0)";
        case 2: return "线性 (dx = a0 + a1*x)";
        case 3: return "双线性 (dx = a0 + a1*x + a2*y)";
        case 4: return "带交叉项 (dx = a0 + a1*x + a2*y + a3*x*y)";
        case 5: return "带二次项X (dx = a0 + a1*x + a2*y + a3*x*y + a4*x^2)";
        case 6: return "全二次项 (dx = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2)";
        case 7: return "扩展二次项 (dx = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2 + a6*x^2*y)";
        default: return "未知模型";
    }
}

// 多元多项式拟合（非加权）
int multivariate_fit(DataPoint *points, int n, double *coeffs, int is_x, int num_coeffs) {
    if (n < num_coeffs) {
        printf("错误: 数据点数 %d 少于系数个数 %d\n", n, num_coeffs);
        return 0;
    }
    
    int m = num_coeffs;
    Matrix *A = create_matrix(m, m);
    double *b = (double*)calloc(m, sizeof(double));
    double *basis = (double*)malloc(MAX_COEFFS * sizeof(double));
    
    // 计算 A = X^T * X 和 b = X^T * y
    // 其中X是设计矩阵，y是目标变量
    for (int i = 0; i < n; i++) {
        if (!points[i].valid) continue;
        
        // 计算基函数值
        compute_basis_functions(points[i].x, points[i].y, basis, num_coeffs);
        
        double target = is_x ? points[i].dx : points[i].dy;
        
        // 更新A矩阵和b向量
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
                A->data[j][k] += basis[j] * basis[k];
            }
            b[j] += basis[j] * target;
        }
    }
    
    int success = gauss_elimination(A, b, coeffs, m);
    
    free_matrix(A);
    free(b);
    free(basis);
    
    return success;
}

// 读取数据
DataPoint* read_data(const char *filename, int *n_points) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        return NULL;
    }
    
    DataPoint *points = (DataPoint*)malloc(MAX_POINTS * sizeof(DataPoint));
    *n_points = 0;
    
    while (*n_points < MAX_POINTS) {
        DataPoint p;
        if (fscanf(fp, "%lf %lf %lf %lf %lf", 
                   &p.x, &p.dx, &p.y, &p.dy, &p.r) == 5) {
            p.valid = 1;
            points[*n_points] = p;
            (*n_points)++;
        } else {
            break;
        }
    }
    
    fclose(fp);
    return points;
}

// 综合评估数据点质量并剔除异常值
void assess_and_remove_outliers(DataPoint *points, int n_points, 
                               double *coeffs_x, double *coeffs_y,
                               int num_coeffs) {
    
    double *residuals_x = (double*)malloc(n_points * sizeof(double));
    double *scores_x = (double*)malloc(n_points * sizeof(double));
    
    double *residuals_y = (double*)malloc(n_points * sizeof(double));
    double *scores_y = (double*)malloc(n_points * sizeof(double));
    
    double *combined_scores = (double*)malloc(n_points * sizeof(double));
    double basis[MAX_COEFFS];
    
    // 计算X方向残差
    for (int i = 0; i < n_points; i++) {
        if (!points[i].valid) continue;
        
        compute_basis_functions(points[i].x, points[i].y, basis, num_coeffs);
        
        double dx_fit = 0.0;
        for (int j = 0; j < num_coeffs; j++) {
            dx_fit += coeffs_x[j] * basis[j];
        }
        residuals_x[i] = points[i].dx - dx_fit;
    }
    
    // 计算Y方向残差
    for (int i = 0; i < n_points; i++) {
        if (!points[i].valid) continue;
        
        compute_basis_functions(points[i].x, points[i].y, basis, num_coeffs);
        
        double dy_fit = 0.0;
        for (int j = 0; j < num_coeffs; j++) {
            dy_fit += coeffs_y[j] * basis[j];
        }
        residuals_y[i] = points[i].dy - dy_fit;
    }
    
    // 计算统计量
    double mean_x = 0.0, std_x = 0.0, count_x = 0.0;
    double mean_y = 0.0, std_y = 0.0, count_y = 0.0;
    
    for (int i = 0; i < n_points; i++) {
        if (!points[i].valid) continue;
        mean_x += residuals_x[i];
        mean_y += residuals_y[i];
        count_x += 1.0;
        count_y += 1.0;
    }
    mean_x /= count_x;
    mean_y /= count_y;
    
    for (int i = 0; i < n_points; i++) {
        if (!points[i].valid) continue;
        std_x += (residuals_x[i] - mean_x) * (residuals_x[i] - mean_x);
        std_y += (residuals_y[i] - mean_y) * (residuals_y[i] - mean_y);
    }
    std_x = sqrt(std_x / (count_x - 1));
    std_y = sqrt(std_y / (count_y - 1));
    
    // 计算综合得分
    for (int i = 0; i < n_points; i++) {
        if (!points[i].valid) {
            combined_scores[i] = 0.0;
            continue;
        }
        
        double z_x = (residuals_x[i] - mean_x) / std_x;
        double z_y = (residuals_y[i] - mean_y) / std_y;
        double r_score = 1.0 / (fabs(points[i].r) + 0.1);
        
        scores_x[i] = fabs(z_x) * r_score;
        scores_y[i] = fabs(z_y) * r_score;
        combined_scores[i] = (scores_x[i] > scores_y[i]) ? scores_x[i] : scores_y[i];
    }
    
    // 找出得分的分布
    double *sorted_scores = (double*)malloc(count_x * sizeof(double));
    int idx = 0;
    for (int i = 0; i < n_points; i++) {
        if (points[i].valid) {
            sorted_scores[idx++] = combined_scores[i];
        }
    }
    
    // 排序（简单冒泡）
    for (int i = 0; i < idx - 1; i++) {
        for (int j = 0; j < idx - i - 1; j++) {
            if (sorted_scores[j] > sorted_scores[j + 1]) {
                double temp = sorted_scores[j];
                sorted_scores[j] = sorted_scores[j + 1];
                sorted_scores[j + 1] = temp;
            }
        }
    }
    
    // 计算阈值
    double q1 = sorted_scores[(int)(idx * 0.25)];
    double q3 = sorted_scores[(int)(idx * 0.75)];
    double iqr = q3 - q1;
    double threshold = q3 + 1.5 * iqr;
    
    // 标记异常值
    int removed = 0;
    for (int i = 0; i < n_points; i++) {
        if (!points[i].valid) continue;
        if (combined_scores[i] > threshold) {
            points[i].valid = 0;
            removed++;
        }
    }
    
    printf("剔除 %d 个异常值 (阈值: %.3f)\n", removed, threshold);
    
    free(residuals_x);
    free(scores_x);
    free(residuals_y);
    free(scores_y);
    free(combined_scores);
    free(sorted_scores);
}

// 获取有效数据
DataPoint* get_valid_data(DataPoint *points, int n_points, int *n_valid) {
    *n_valid = 0;
    for (int i = 0; i < n_points; i++) {
        if (points[i].valid) {
            (*n_valid)++;
        }
    }
    
    DataPoint *valid_points = (DataPoint*)malloc((*n_valid) * sizeof(DataPoint));
    int idx = 0;
    for (int i = 0; i < n_points; i++) {
        if (points[i].valid) {
            valid_points[idx++] = points[i];
        }
    }
    
    return valid_points;
}

// 计算误差统计
void calculate_errors(DataPoint *points, int n,
                     double *coeffs_x, double *coeffs_y,
                     double *rms_error_x, double *rms_error_y,
                     int num_coeffs) {
    
    double total_error_x = 0.0;
    double total_error_y = 0.0;
    
    double basis[MAX_COEFFS];
    
    for (int i = 0; i < n; i++) {
        compute_basis_functions(points[i].x, points[i].y, basis, num_coeffs);
        
        // X方向拟合值
        double dx_fit = 0.0;
        for (int j = 0; j < num_coeffs; j++) {
            dx_fit += coeffs_x[j] * basis[j];
        }
        
        // Y方向拟合值
        double dy_fit = 0.0;
        for (int j = 0; j < num_coeffs; j++) {
            dy_fit += coeffs_y[j] * basis[j];
        }
        
        // X方向误差
        double error_x = points[i].dx - dx_fit;
        total_error_x += error_x * error_x;
        
        // Y方向误差
        double error_y = points[i].dy - dy_fit;
        total_error_y += error_y * error_y;
    }
    
    *rms_error_x = sqrt(total_error_x / n);
    *rms_error_y = sqrt(total_error_y / n);
}

// 按照指定格式输出系数和误差到文件
void output_coefficients_and_errors_to_file(FILE *output_file,
                                          double *coeffs_x, double *coeffs_y,
                                          double rms_error_x, double rms_error_y,
                                          int num_coeffs) {
    // 输出X方向系数 a0-an
    for (int i = 0; i < num_coeffs; i++) {
        fprintf(output_file, "%.12e ", coeffs_x[i]);
    }
    
    // 如果系数个数不足6，用0补齐
    for (int i = num_coeffs; i < 6; i++) {
        fprintf(output_file, "0.000000e+00 ");
    }
    
    // 输出Y方向系数 b0-bn
    for (int i = 0; i < num_coeffs; i++) {
        fprintf(output_file, "%.12e ", coeffs_y[i]);
    }
    
    // 如果系数个数不足6，用0补齐
    for (int i = num_coeffs; i < 6; i++) {
        fprintf(output_file, "0.000000e+00 ");
    }
    
    // 输出误差
    fprintf(output_file, "%.12e %.12e\n", rms_error_x, rms_error_y);
}

// 输出详细系数信息
void print_detailed_coefficients(double *coeffs_x, double *coeffs_y, int num_coeffs) {
    printf("\n=== 详细拟合系数 ===\n");
    
    // 输出X方向系数
    printf("dx = ");
    switch (num_coeffs) {
        case 1:
            printf("a0");
            break;
        case 2:
            printf("a0 + a1*x");
            break;
        case 3:
            printf("a0 + a1*x + a2*y");
            break;
        case 4:
            printf("a0 + a1*x + a2*y + a3*x*y");
            break;
        case 5:
            printf("a0 + a1*x + a2*y + a3*x*y + a4*x^2");
            break;
        case 6:
            printf("a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2");
            break;
        case 7:
            printf("a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2 + a6*x^2*y");
            break;
    }
    printf("\n");
    
    for (int i = 0; i < num_coeffs; i++) {
        printf("a%d = %.12e\n", i, coeffs_x[i]);
    }
    
    // 输出Y方向系数
    printf("\ndy = ");
    switch (num_coeffs) {
        case 1:
            printf("b0");
            break;
        case 2:
            printf("b0 + b1*x");
            break;
        case 3:
            printf("b0 + b1*x + b2*y");
            break;
        case 4:
            printf("b0 + b1*x + b2*y + b3*x*y");
            break;
        case 5:
            printf("b0 + b1*x + b2*y + b3*x*y + b4*x^2");
            break;
        case 6:
            printf("b0 + b1*x + b2*y + b3*x*y + b4*x^2 + b5*y^2");
            break;
        case 7:
            printf("b0 + b1*x + b2*y + b3*x*y + b4*x^2 + b5*y^2 + b6*x^2*y");
            break;
    }
    printf("\n");
    
    for (int i = 0; i < num_coeffs; i++) {
        printf("b%d = %.12e\n", i, coeffs_y[i]);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "用法: %s <数据文件> <输出文件> <系数个数>\n", argv[0]);
        fprintf(stderr, "系数个数选项:\n");
        fprintf(stderr, "  1: 常数项模型 (dx = a0)\n");
        fprintf(stderr, "  2: 线性模型 (dx = a0 + a1*x)\n");
        fprintf(stderr, "  3: 双线性模型 (dx = a0 + a1*x + a2*y)\n");
        fprintf(stderr, "  4: 带交叉项 (dx = a0 + a1*x + a2*y + a3*x*y)\n");
        fprintf(stderr, "  5: 带二次项X (dx = a0 + a1*x + a2*y + a3*x*y + a4*x^2)\n");
        fprintf(stderr, "  6: 全二次项 (dx = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2)\n");
        fprintf(stderr, "  7: 扩展二次项 (dx = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2 + a6*x^2*y)\n");
        return 1;
    }
    
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];
    int num_coeffs = atoi(argv[3]);
    
    // 验证系数个数参数
    if (num_coeffs < 1 || num_coeffs > 7) {
        fprintf(stderr, "错误: 系数个数必须在1-7之间\n");
        return 1;
    }
    
    printf("使用拟合模型: %s\n", get_model_description(num_coeffs));
    
    // 读取原始数据
    int n_total;
    DataPoint *points = read_data(input_filename, &n_total);
    if (!points || n_total == 0) {
        fprintf(stderr, "错误: 无法读取文件 %s 或文件为空\n", input_filename);
        return 1;
    }
    
    printf("读取到 %d 个数据点\n", n_total);
    
    double coeffs_x[MAX_COEFFS] = {0};
    double coeffs_y[MAX_COEFFS] = {0};
    
    // 迭代拟合（5次）
    for (int iteration = 1; iteration <= 5; iteration++) {
        printf("\n=== 第 %d 次迭代拟合 ===\n", iteration);
        
        // 统计有效数据
        int n_valid = 0;
        for (int i = 0; i < n_total; i++) {
            if (points[i].valid) n_valid++;
        }
        
        printf("有效数据点: %d\n", n_valid);
        
        // 检查是否有足够的数据
        if (n_valid < num_coeffs) {
            printf("错误: 有效数据点不足（需要至少 %d 个，当前 %d 个）\n", num_coeffs, n_valid);
            free(points);
            return 1;
        }
        
        // X方向拟合
        if (!multivariate_fit(points, n_total, coeffs_x, 1, num_coeffs)) {
            printf("X方向拟合失败\n");
            free(points);
            return 1;
        }
        
        // Y方向拟合
        if (!multivariate_fit(points, n_total, coeffs_y, 0, num_coeffs)) {
            printf("Y方向拟合失败\n");
            free(points);
            return 1;
        }
        
        // 计算当前迭代的误差
        double rms_error_x, rms_error_y;
        calculate_errors(points, n_total, coeffs_x, coeffs_y,
                        &rms_error_x, &rms_error_y, num_coeffs);
        
        printf("X方向RMS误差: %.6f\n", rms_error_x);
        printf("Y方向RMS误差: %.6f\n", rms_error_y);
        
        // 如果不是最后一次迭代，评估并剔除异常值
        if (iteration < 5) {
            printf("数据质量评估与异常值剔除:\n");
            assess_and_remove_outliers(points, n_total, coeffs_x, coeffs_y, num_coeffs);
        }
    }
    
    // 最终统计有效数据点
    int final_valid = 0;
    for (int i = 0; i < n_total; i++) {
        if (points[i].valid) final_valid++;
    }
    
    // 使用最终的有效数据进行最终误差计算
    DataPoint *final_points = get_valid_data(points, n_total, &final_valid);
    
    double rms_error_x, rms_error_y;
    calculate_errors(final_points, final_valid, coeffs_x, coeffs_y,
                    &rms_error_x, &rms_error_y, num_coeffs);
    
    printf("\n=== 拟合结果汇总 ===\n");
    printf("初始数据点: %d\n", n_total);
    printf("最终数据点: %d\n", final_valid);
    printf("剔除数据点: %d (%.1f%%)\n", n_total - final_valid, 
           (n_total - final_valid) * 100.0 / n_total);
    printf("拟合模型: %s\n", get_model_description(num_coeffs));
    
    // 输出详细系数信息
    print_detailed_coefficients(coeffs_x, coeffs_y, num_coeffs);
    
    // 打开输出文件
    FILE *output_file = fopen(output_filename, "w");
    if (!output_file) {
        fprintf(stderr, "错误: 无法创建输出文件 %s\n", output_filename);
        free(final_points);
        free(points);
        return 1;
    }
    
    // 输出最终系数和误差到文件（一行格式，固定14个值）
    printf("\n=== 最终输出（程序使用格式） ===\n");
    printf("输出到文件: %s\n", output_filename);
    output_coefficients_and_errors_to_file(output_file, coeffs_x, coeffs_y, 
                                          rms_error_x, rms_error_y, num_coeffs);
    
    // 同时也在屏幕上显示输出内容
    printf("输出内容: ");
    for (int i = 0; i < num_coeffs; i++) {
        printf("%.12e ", coeffs_x[i]);
    }
    // 补齐到6个系数
    for (int i = num_coeffs; i < 6; i++) {
        printf("0.000000e+00 ");
    }
    for (int i = 0; i < num_coeffs; i++) {
        printf("%.12e ", coeffs_y[i]);
    }
    // 补齐到6个系数
    for (int i = num_coeffs; i < 6; i++) {
        printf("0.000000e+00 ");
    }
    printf("%.12e %.12e\n", rms_error_x, rms_error_y);
    
    // 关闭输出文件
    fclose(output_file);
    
    // 清理内存
    free(final_points);
    free(points);
    
    printf("\n拟合完成！结果已保存到 %s\n", output_filename);
    
    return 0;
}