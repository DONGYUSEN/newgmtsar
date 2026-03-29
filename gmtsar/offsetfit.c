#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_POINTS 100
#define MAX_DEGREE 10

// 矩阵结构体
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
        for (int j = 0; j < cols; j++) {
            mat->data[i][j] = 0.0;
        }
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

// 高斯消元法解线性方程组
int gauss_elimination(Matrix *A, double *b, double *x, int n) {
    // 创建增广矩阵
    double **aug = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        aug[i] = (double*)malloc((n + 1) * sizeof(double));
        for (int j = 0; j < n; j++) {
            aug[i][j] = A->data[i][j];
        }
        aug[i][n] = b[i];
    }
    
    // 前向消元
    for (int i = 0; i < n; i++) {
        // 寻找主元
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(aug[k][i]) > fabs(aug[max_row][i])) {
                max_row = k;
            }
        }
        
        // 交换行
        if (max_row != i) {
            double *temp = aug[i];
            aug[i] = aug[max_row];
            aug[max_row] = temp;
        }
        
        // 如果主元为0，矩阵奇异
        if (fabs(aug[i][i]) < 1e-15) {
            for (int j = 0; j < n; j++) {
                free(aug[j]);
            }
            free(aug);
            return 0; // 失败
        }
        
        // 归一化
        double pivot = aug[i][i];
        for (int j = i; j <= n; j++) {
            aug[i][j] /= pivot;
        }
        
        // 消元
        for (int k = i + 1; k < n; k++) {
            double factor = aug[k][i];
            for (int j = i; j <= n; j++) {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }
    
    // 回代
    for (int i = n - 1; i >= 0; i--) {
        x[i] = aug[i][n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= aug[i][j] * x[j];
        }
    }
    
    // 清理
    for (int i = 0; i < n; i++) {
        free(aug[i]);
    }
    free(aug);
    
    return 1; // 成功
}

// 多项式拟合函数
int polyfit(double *x, double *y, int n, int degree, double *coeffs) {
    // 检查参数
    if (n <= degree) {
        printf("错误: 数据点数必须大于拟合阶数\n");
        return 0;
    }
    
    // 创建法方程矩阵
    int m = degree + 1;
    Matrix *A = create_matrix(m, m);
    double *b = (double*)malloc(m * sizeof(double));
    
    // 计算法方程矩阵
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += pow(x[k], i + j);
            }
            A->data[i][j] = sum;
        }
    }
    
    // 计算右侧向量
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += y[k] * pow(x[k], i);
        }
        b[i] = sum;
    }
    
    // 解线性方程组
    int success = gauss_elimination(A, b, coeffs, m);
    
    // 清理
    free_matrix(A);
    free(b);
    
    return success;
}

// 读取数据文件
int read_data(const char *filename, double *x, double *dx, double *y, double *dy) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("错误: 无法打开文件 %s\n", filename);
        return 0;
    }
    
    int count = 0;
    double temp[5];
    
    while (count < MAX_POINTS && fscanf(fp, "%lf %lf %lf %lf %lf", 
           &temp[0], &temp[1], &temp[2], &temp[3], &temp[4]) == 5) {
        x[count] = temp[0];
        dx[count] = temp[1];
        y[count] = temp[2];
        dy[count] = temp[3];
        count++;
    }
    
    fclose(fp);
    return count;
}

// 输出拟合方程
void print_equation(double *coeffs, int degree, const char *var_name, const char *dep_var) {
    printf("%s = ", dep_var);
    
    // 从最高次项开始输出
    int first_term = 1;
    for (int i = degree; i >= 0; i--) {
        double coeff = coeffs[i];
        
        // 忽略接近0的系数
        if (fabs(coeff) < 1e-10) {
            continue;
        }
        
        // 输出符号
        if (!first_term) {
            if (coeff > 0) {
                printf(" + ");
            } else {
                printf(" - ");
                coeff = -coeff;
            }
        }
        
        // 输出系数
        if (i == 0 || fabs(coeff - 1.0) > 1e-10) {
            printf("%.6f", coeff);
        }
        
        // 输出变量部分
        if (i > 0) {
            printf("*%s", var_name);
            if (i > 1) {
                printf("^%d", i);
            }
        }
        
        first_term = 0;
    }
    
    printf("\n");
}

// 计算拟合误差
void calculate_error(double *x, double *y, double *coeffs, int n, int degree) {
    double total_error = 0.0;
    double max_error = 0.0;
    
    printf("\n拟合误差分析:\n");
    printf("序号\t实际值\t\t拟合值\t\t残差\n");
    printf("------------------------------------------------\n");
    
    for (int i = 0; i < n; i++) {
        double y_fit = 0.0;
        for (int j = 0; j <= degree; j++) {
            y_fit += coeffs[j] * pow(x[i], j);
        }
        
        double error = y[i] - y_fit;
        total_error += error * error;
        
        if (fabs(error) > fabs(max_error)) {
            max_error = error;
        }
        
        printf("%d\t%.3f\t\t%.3f\t\t%.3f\n", i+1, y[i], y_fit, error);
    }
    
    double rms_error = sqrt(total_error / n);
    printf("\n均方根误差(RMS): %.6f\n", rms_error);
    printf("最大绝对误差: %.6f\n", fabs(max_error));
}

int main(int argc, char *argv[]) {
    // 检查命令行参数
    if (argc < 2) {
        printf("用法: %s <拟合阶数> [数据文件]\n", argv[0]);
        printf("默认数据文件: freq.dat\n");
        return 1;
    }
    
    // 获取拟合阶数
    int degree = atoi(argv[1]);
    if (degree < 1 || degree > MAX_DEGREE) {
        printf("错误: 拟合阶数必须在1到%d之间\n", MAX_DEGREE);
        return 1;
    }
    
    // 获取文件名
    const char *filename = (argc > 2) ? argv[2] : "freq.dat";
    
    // 分配内存
    double x[MAX_POINTS], dx[MAX_POINTS];
    double y[MAX_POINTS], dy[MAX_POINTS];
    
    // 读取数据
    int n = read_data(filename, x, dx, y, dy);
    if (n == 0) {
        printf("错误: 未读取到数据\n");
        return 1;
    }
    
    printf("成功读取 %d 个数据点\n", n);
    
    // 分配系数数组
    double *x_coeffs = (double*)malloc((degree + 1) * sizeof(double));
    double *y_coeffs = (double*)malloc((degree + 1) * sizeof(double));
    
    // X方向拟合（原始X位置 -> X位移）
    printf("\n=== X方向拟合（原始X -> X位移）===\n");
    printf("拟合阶数: %d\n", degree);
    
    if (polyfit(x, dx, n, degree, x_coeffs)) {
        print_equation(x_coeffs, degree, "X", "dX");
        calculate_error(x, dx, x_coeffs, n, degree);
    }
    
    // Y方向拟合（原始Y位置 -> Y位移）
    printf("\n=== Y方向拟合（原始Y -> Y位移）===\n");
    printf("拟合阶数: %d\n", degree);
    
    if (polyfit(y, dy, n, degree, y_coeffs)) {
        print_equation(y_coeffs, degree, "Y", "dY");
        calculate_error(y, dy, y_coeffs, n, degree);
    }
    
    // 输出原始数据供参考
    printf("\n=== 原始数据 ===\n");
    printf("序号\t原始X\t\tX位移\t\t原始Y\t\tY位移\n");
    printf("----------------------------------------------------------------\n");
    for (int i = 0; i < n; i++) {
        printf("%d\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\n", 
               i+1, x[i], dx[i], y[i], dy[i]);
    }
    
    // 示例：使用拟合公式计算新值
    printf("\n=== 使用拟合公式计算示例 ===\n");
    double test_x = 10000.0;
    double test_y = 26000.0;
    
    double dx_predicted = 0.0;
    double dy_predicted = 0.0;
    
    for (int i = 0; i <= degree; i++) {
        dx_predicted += x_coeffs[i] * pow(test_x, i);
        dy_predicted += y_coeffs[i] * pow(test_y, i);
    }
    
    printf("在 X = %.1f 处，预测的X位移 dX = %.3f\n", test_x, dx_predicted);
    printf("在 Y = %.1f 处，预测的Y位移 dY = %.3f\n", test_y, dy_predicted);
    
    // 清理
    free(x_coeffs);
    free(y_coeffs);
    
    return 0;
}
