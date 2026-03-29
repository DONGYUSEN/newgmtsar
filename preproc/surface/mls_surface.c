#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <netcdf.h>

// 检查NetCDF错误
void check_nc_error(int status, const char* msg) {
    if (status != NC_NOERR) {
        fprintf(stderr, "NetCDF错误 (%s): %s\n", msg, nc_strerror(status));
        exit(1);
    }
}

// 数据结构定义
typedef struct {
    double x;   // 经度
    double y;   // 纬度
    double z;   // 高程/值
} DataPoint;

typedef struct {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
    double dx;   // x方向网格间隔
    double dy;   // y方向网格间隔
    int nx;      // x方向网格点数
    int ny;      // y方向网格点数
} GridInfo;

// 用于空间索引的网格单元
typedef struct {
    int count;
    int capacity;
    int* point_indices;
} GridCell;

// MLS参数
typedef struct {
    double h;          // 带宽参数
    int polynomial;    // 多项式阶数 (1: 线性, 2: 二次)
    int min_points;    // 最小点数
    double tension;    // 张力因子
    double max_z_change; // 最大高程变化限制
    int robust_iterations; // 稳健估计迭代次数
    double outlier_threshold; // 异常值阈值（标准差倍数）
} MLSConfig;

// 读取blockmedian处理后的数据
DataPoint* read_data(const char* filename, int* n_points) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("无法打开数据文件");
        return NULL;
    }
    
    // 获取文件大小
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    *n_points = file_size / (3 * sizeof(double));
    DataPoint* data = (DataPoint*)malloc(*n_points * sizeof(DataPoint));
    
    for (int i = 0; i < *n_points; i++) {
        fread(&data[i].x, sizeof(double), 1, fp);
        fread(&data[i].y, sizeof(double), 1, fp);
        fread(&data[i].z, sizeof(double), 1, fp);
    }
    
    fclose(fp);
    
    printf("读取了 %d 个数据点\n", *n_points);
    
    return data;
}

// 创建空间索引网格
GridCell** create_spatial_index(DataPoint* data, int n_points, 
                               GridInfo* grid, int cells_x, int cells_y) {
    
    GridCell** index = (GridCell**)malloc(cells_y * sizeof(GridCell*));
    for (int i = 0; i < cells_y; i++) {
        index[i] = (GridCell*)malloc(cells_x * sizeof(GridCell));
        for (int j = 0; j < cells_x; j++) {
            index[i][j].count = 0;
            index[i][j].capacity = 10;
            index[i][j].point_indices = (int*)malloc(10 * sizeof(int));
        }
    }
    
    double cell_dx = (grid->max_x - grid->min_x) / cells_x;
    double cell_dy = (grid->max_y - grid->min_y) / cells_y;
    
    for (int i = 0; i < n_points; i++) {
        int cell_x = (int)((data[i].x - grid->min_x) / cell_dx);
        int cell_y = (int)((data[i].y - grid->min_y) / cell_dy);
        
        if (cell_x < 0) cell_x = 0;
        if (cell_x >= cells_x) cell_x = cells_x - 1;
        if (cell_y < 0) cell_y = 0;
        if (cell_y >= cells_y) cell_y = cells_y - 1;
        
        GridCell* cell = &index[cell_y][cell_x];
        
        if (cell->count >= cell->capacity) {
            cell->capacity *= 2;
            cell->point_indices = (int*)realloc(cell->point_indices, 
                                               cell->capacity * sizeof(int));
        }
        
        cell->point_indices[cell->count++] = i;
    }
    
    return index;
}

// 张力权重函数
double tension_weight(double dist, double h, double tension) {
    if (dist >= h) return 0.0;
    double r = dist / h;
    double w = 1.0 - r;
    return pow(w, tension);
}

// 求解线性方程组
int solve_linear_system(double* A, double* b, double* x, int n) {
    if (n == 3) {
        double det = A[0]*(A[4]*A[8]-A[5]*A[7]) -
                    A[1]*(A[3]*A[8]-A[5]*A[6]) +
                    A[2]*(A[3]*A[7]-A[4]*A[6]);
        
        if (fabs(det) < 1e-12) return 0;
        
        double inv[9];
        inv[0] = (A[4]*A[8] - A[5]*A[7]) / det;
        inv[1] = (A[2]*A[7] - A[1]*A[8]) / det;
        inv[2] = (A[1]*A[5] - A[2]*A[4]) / det;
        inv[3] = (A[5]*A[6] - A[3]*A[8]) / det;
        inv[4] = (A[0]*A[8] - A[2]*A[6]) / det;
        inv[5] = (A[2]*A[3] - A[0]*A[5]) / det;
        inv[6] = (A[3]*A[7] - A[4]*A[6]) / det;
        inv[7] = (A[1]*A[6] - A[0]*A[7]) / det;
        inv[8] = (A[0]*A[4] - A[1]*A[3]) / det;
        
        x[0] = inv[0]*b[0] + inv[1]*b[1] + inv[2]*b[2];
        x[1] = inv[3]*b[0] + inv[4]*b[1] + inv[5]*b[2];
        x[2] = inv[6]*b[0] + inv[7]*b[1] + inv[8]*b[2];
        
        return 1;
    }
    
    // 高斯消元法
    double* AB = (double*)malloc(n * (n+1) * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AB[i*(n+1) + j] = A[i*n + j];
        }
        AB[i*(n+1) + n] = b[i];
    }
    
    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i+1; k < n; k++) {
            if (fabs(AB[k*(n+1) + i]) > fabs(AB[max_row*(n+1) + i])) {
                max_row = k;
            }
        }
        
        if (fabs(AB[max_row*(n+1) + i]) < 1e-12) {
            free(AB);
            return 0;
        }
        
        if (max_row != i) {
            for (int k = 0; k <= n; k++) {
                double temp = AB[i*(n+1) + k];
                AB[i*(n+1) + k] = AB[max_row*(n+1) + k];
                AB[max_row*(n+1) + k] = temp;
            }
        }
        
        for (int k = i+1; k < n; k++) {
            double factor = AB[k*(n+1) + i] / AB[i*(n+1) + i];
            for (int j = i; j <= n; j++) {
                AB[k*(n+1) + j] -= factor * AB[i*(n+1) + j];
            }
        }
    }
    
    for (int i = n-1; i >= 0; i--) {
        x[i] = AB[i*(n+1) + n];
        for (int j = i+1; j < n; j++) {
            x[i] -= AB[i*(n+1) + j] * x[j];
        }
        x[i] /= AB[i*(n+1) + i];
    }
    
    free(AB);
    return 1;
}

// 移动最小二乘法插值
double mls_interpolate(double x, double y, DataPoint* data, int* indices, 
                       int n_indices, MLSConfig* config) {
    
    if (n_indices < config->min_points) {
        // 返回加权平均值
        double sum_w = 0, sum_wz = 0;
        for (int i = 0; i < n_indices; i++) {
            DataPoint* p = &data[indices[i]];
            double dx = x - p->x;
            double dy = y - p->y;
            double dist = sqrt(dx*dx + dy*dy);
            double w = tension_weight(dist, config->h, config->tension);
            
            sum_w += w;
            sum_wz += w * p->z;
        }
        
        return sum_w > 0 ? sum_wz / sum_w : NAN;
    }
    
    // 使用线性基函数 (1, x, y)
    int basis_size = 3;
    double* ATA = (double*)calloc(basis_size * basis_size, sizeof(double));
    double* ATb = (double*)calloc(basis_size, sizeof(double));
    
    for (int i = 0; i < n_indices; i++) {
        DataPoint* p = &data[indices[i]];
        double dx = x - p->x;
        double dy = y - p->y;
        double dist = sqrt(dx*dx + dy*dy);
        
        double w = tension_weight(dist, config->h, config->tension);
        if (w < 1e-12) continue;
        
        // 基函数：1, dx, dy
        double basis[3] = {1.0, -dx, -dy};
        
        for (int j = 0; j < basis_size; j++) {
            ATb[j] += w * basis[j] * p->z;
            for (int k = 0; k < basis_size; k++) {
                ATA[j*basis_size + k] += w * basis[j] * basis[k];
            }
        }
    }
    
    // 添加正则化
    for (int i = 0; i < basis_size; i++) {
        ATA[i*basis_size + i] += 1e-6;
    }
    
    double coeff[3];
    int success = solve_linear_system(ATA, ATb, coeff, basis_size);
    
    double result = NAN;
    if (success) {
        result = coeff[0];  // 常数项
    } else {
        // 失败时返回加权平均
        double sum_w = 0, sum_wz = 0;
        for (int i = 0; i < n_indices; i++) {
            DataPoint* p = &data[indices[i]];
            double dx = x - p->x;
            double dy = y - p->y;
            double dist = sqrt(dx*dx + dy*dy);
            double w = tension_weight(dist, config->h, config->tension);
            
            sum_w += w;
            sum_wz += w * p->z;
        }
        result = sum_w > 0 ? sum_wz / sum_w : NAN;
    }
    
    free(ATA);
    free(ATb);
    
    return result;
}

// 主插值函数（并行化）
float** interpolate_grid(DataPoint* data, int n_points, GridInfo* grid, 
                        MLSConfig* config, GridCell** spatial_index,
                        int cells_x, int cells_y) {
    
    // 分配网格内存
    float** grid_data = (float**)malloc(grid->ny * sizeof(float*));
    for (int i = 0; i < grid->ny; i++) {
        grid_data[i] = (float*)malloc(grid->nx * sizeof(float));
        for (int j = 0; j < grid->nx; j++) {
            grid_data[i][j] = NAN;
        }
    }
    
    double cell_dx = (grid->max_x - grid->min_x) / cells_x;
    double cell_dy = (grid->max_y - grid->min_y) / cells_y;
    
    // 搜索半径
    int search_cells = (int)ceil(config->h / fmin(cell_dx, cell_dy)) + 1;
    
    printf("开始并行插值...\n");
    
    #pragma omp parallel
    {
        int* local_indices = (int*)malloc(n_points * sizeof(int));
        
        #pragma omp for schedule(dynamic)
        for (int iy = 0; iy < grid->ny; iy++) {
            double y = grid->min_y + iy * grid->dy;
            
            for (int ix = 0; ix < grid->nx; ix++) {
                double x = grid->min_x + ix * grid->dx;
                
                // 确定搜索单元
                int cell_x = (int)((x - grid->min_x) / cell_dx);
                int cell_y = (int)((y - grid->min_y) / cell_dy);
                
                int min_cx = cell_x - search_cells;
                int max_cx = cell_x + search_cells;
                int min_cy = cell_y - search_cells;
                int max_cy = cell_y + search_cells;
                
                if (min_cx < 0) min_cx = 0;
                if (max_cx >= cells_x) max_cx = cells_x - 1;
                if (min_cy < 0) min_cy = 0;
                if (max_cy >= cells_y) max_cy = cells_y - 1;
                
                // 收集邻近点
                int n_indices = 0;
                for (int cy = min_cy; cy <= max_cy; cy++) {
                    for (int cx = min_cx; cx <= max_cx; cx++) {
                        GridCell* cell = &spatial_index[cy][cx];
                        
                        for (int k = 0; k < cell->count; k++) {
                            DataPoint* p = &data[cell->point_indices[k]];
                            double dx = x - p->x;
                            double dy = y - p->y;
                            double dist = sqrt(dx*dx + dy*dy);
                            
                            if (dist <= config->h * 2.0) {
                                local_indices[n_indices++] = cell->point_indices[k];
                            }
                        }
                    }
                }
                
                // 插值
                if (n_indices >= config->min_points) {
                    grid_data[iy][ix] = (float)mls_interpolate(
                        x, y, data, local_indices, n_indices, config);
                }
            }
        }
        
        free(local_indices);
    }
    
    // 后处理：填充NaN值
    printf("后处理：填充NaN值...\n");
    int filled_count = 0;
    
    #pragma omp parallel for reduction(+:filled_count) schedule(static)
    for (int iy = 0; iy < grid->ny; iy++) {
        for (int ix = 0; ix < grid->nx; ix++) {
            if (isnan(grid_data[iy][ix])) {
                // 使用邻近点平均值
                double sum = 0;
                int count = 0;
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int new_ix = ix + dx;
                        int new_iy = iy + dy;
                        
                        if (new_ix >= 0 && new_ix < grid->nx && 
                            new_iy >= 0 && new_iy < grid->ny) {
                            if (!isnan(grid_data[new_iy][new_ix])) {
                                sum += grid_data[new_iy][new_ix];
                                count++;
                            }
                        }
                    }
                }
                
                if (count > 0) {
                    grid_data[iy][ix] = sum / count;
                    filled_count++;
                }
            }
        }
    }
    
    printf("填充了 %d 个NaN值\n", filled_count);
    
    return grid_data;
}

// 写入NetCDF文件（CF-1.7兼容）
void write_netcdf_grid(const char* filename, float** grid_data, GridInfo* grid) {
    int ncid, x_dimid, y_dimid, x_varid, y_varid, z_varid;
    int dimids[2];
    
    // 创建NetCDF文件
    int status = nc_create(filename, NC_CLOBBER, &ncid);
    check_nc_error(status, "nc_create");
    
    printf("创建NetCDF文件: %s\n", filename);
    
    // 定义维度
    status = nc_def_dim(ncid, "x", grid->nx, &x_dimid);
    check_nc_error(status, "nc_def_dim x");
    status = nc_def_dim(ncid, "y", grid->ny, &y_dimid);
    check_nc_error(status, "nc_def_dim y");
    
    dimids[0] = y_dimid;  // 注意：NetCDF是行优先，GMT是列优先
    dimids[1] = x_dimid;
    
    // 定义x坐标变量
    status = nc_def_var(ncid, "x", NC_DOUBLE, 1, &x_dimid, &x_varid);
    check_nc_error(status, "nc_def_var x");
    status = nc_put_att_text(ncid, x_varid, "long_name", 9, "x coordinate");
    status = nc_put_att_text(ncid, x_varid, "units", 6, "meters");
    
    // 定义y坐标变量
    status = nc_def_var(ncid, "y", NC_DOUBLE, 1, &y_dimid, &y_varid);
    check_nc_error(status, "nc_def_var y");
    status = nc_put_att_text(ncid, y_varid, "long_name", 9, "y coordinate");
    status = nc_put_att_text(ncid, y_varid, "units", 6, "meters");
    
    // 定义z数据变量
    status = nc_def_var(ncid, "z", NC_FLOAT, 2, dimids, &z_varid);
    check_nc_error(status, "nc_def_var z");
    status = nc_put_att_text(ncid, z_varid, "long_name", 9, "elevation");
    status = nc_put_att_text(ncid, z_varid, "units", 6, "meters");
    status = nc_put_att_text(ncid, z_varid, "coordinates", 3, "x y");
    
    // 添加全局属性（CF-1.7标准）
    status = nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 6, "CF-1.7");
    status = nc_put_att_text(ncid, NC_GLOBAL, "title", 19, "MLS Interpolated Grid");
    status = nc_put_att_text(ncid, NC_GLOBAL, "source", 21, "MLS Surface Interpolator");
    
    char history[256];
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    strftime(history, sizeof(history), "Created on %Y-%m-%d %H:%M:%S", t);
    status = nc_put_att_text(ncid, NC_GLOBAL, "history", strlen(history), history);
    
    // 结束定义模式
    status = nc_enddef(ncid);
    check_nc_error(status, "nc_enddef");
    
    // 写入x坐标
    double* x_coords = (double*)malloc(grid->nx * sizeof(double));
    for (int i = 0; i < grid->nx; i++) {
        x_coords[i] = grid->min_x + i * grid->dx;
    }
    status = nc_put_var_double(ncid, x_varid, x_coords);
    check_nc_error(status, "nc_put_var x");
    free(x_coords);
    
    // 写入y坐标
    double* y_coords = (double*)malloc(grid->ny * sizeof(double));
    for (int i = 0; i < grid->ny; i++) {
        y_coords[i] = grid->min_y + i * grid->dy;
    }
    status = nc_put_var_double(ncid, y_varid, y_coords);
    check_nc_error(status, "nc_put_var y");
    free(y_coords);
    
    // 写入z数据（注意行列顺序转换）
    float* z_data = (float*)malloc(grid->nx * grid->ny * sizeof(float));
    for (int i = 0; i < grid->ny; i++) {
        for (int j = 0; j < grid->nx; j++) {
            z_data[i * grid->nx + j] = grid_data[i][j];
        }
    }
    status = nc_put_var_float(ncid, z_varid, z_data);
    check_nc_error(status, "nc_put_var z");
    free(z_data);
    
    // 关闭文件
    status = nc_close(ncid);
    check_nc_error(status, "nc_close");
    
    printf("NetCDF文件写入完成\n");
}

// 主函数
int main(int argc, char* argv[]) {
    if (argc < 7) {
        fprintf(stderr, "用法: %s <输入文件> <输出文件.nc> <min_x> <max_x> <min_y> <max_y> <dx> <dy> [张力因子] [线程数]\n", argv[0]);
        fprintf(stderr, "示例: %s temp.rat pixel.nc 0 14256 0 28428 1 2 0.5 8\n", argv[0]);
        return 1;
    }
    
    clock_t start_time = clock();
    
    // 解析参数
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    
    GridInfo grid;
    grid.min_x = atof(argv[3]);
    grid.max_x = atof(argv[4]);
    grid.min_y = atof(argv[5]);
    grid.max_y = atof(argv[6]);
    grid.dx = atof(argv[7]);
    grid.dy = atof(argv[8]);
    
    // 计算网格大小
    grid.nx = (int)((grid.max_x - grid.min_x) / grid.dx) + 1;
    grid.ny = (int)((grid.max_y - grid.min_y) / grid.dy) + 1;
    
    // MLS配置
    MLSConfig config;
    config.tension = (argc > 9) ? atof(argv[9]) : 0.5;
    config.h = fmax(grid.dx, grid.dy) * 3.0;
    config.polynomial = 1;  // 线性基函数
    config.min_points = 4;
    config.max_z_change = 0;
    config.robust_iterations = 0;
    config.outlier_threshold = 0;
    
    // 设置线程数
    int num_threads = (argc > 10) ? atoi(argv[10]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    printf("===== MLS曲面插值（NetCDF输出） =====\n");
    printf("输入文件: %s\n", input_file);
    printf("输出文件: %s\n", output_file);
    printf("区域: x=[%.2f, %.2f], y=[%.2f, %.2f]\n", 
           grid.min_x, grid.max_x, grid.min_y, grid.max_y);
    printf("网格: %d x %d = %.1f 百万点\n", 
           grid.nx, grid.ny, (double)grid.nx * grid.ny / 1e6);
    printf("间隔: dx=%.2f, dy=%.2f\n", grid.dx, grid.dy);
    printf("张力因子: %.2f, 带宽: %.2f\n", config.tension, config.h);
    printf("线程数: %d\n", num_threads);
    
    // 读取数据
    int n_points;
    DataPoint* data = read_data(input_file, &n_points);
    if (!data) {
        fprintf(stderr, "读取数据失败\n");
        return 1;
    }
    
    // 创建空间索引
    printf("创建空间索引...\n");
    int cells_x = fmin(200, grid.nx / 20);
    int cells_y = fmin(200, grid.ny / 20);
    if (cells_x < 10) cells_x = 10;
    if (cells_y < 10) cells_y = 10;
    
    GridCell** spatial_index = create_spatial_index(data, n_points, &grid, cells_x, cells_y);
    
    // 进行插值
    printf("开始插值计算...\n");
    float** grid_data = interpolate_grid(data, n_points, &grid, &config, 
                                        spatial_index, cells_x, cells_y);
    
    // 写入NetCDF文件
    printf("写入NetCDF文件...\n");
    write_netcdf_grid(output_file, grid_data, &grid);
    
    // 清理内存
    for (int i = 0; i < grid.ny; i++) {
        free(grid_data[i]);
    }
    free(grid_data);
    
    for (int i = 0; i < cells_y; i++) {
        for (int j = 0; j < cells_x; j++) {
            free(spatial_index[i][j].point_indices);
        }
        free(spatial_index[i]);
    }
    free(spatial_index);
    
    free(data);
    
    // 计算并显示运行时间
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("处理完成！耗时: %.2f 秒\n", elapsed_time);
    
    // 显示转换命令
    printf("\n===== 转换为GMT GRD格式 =====\n");
    printf("方法1: 使用gmt grdconvert\n");
    printf("  gmt grdconvert %s pixel.grd -V\n", output_file);
    printf("\n方法2: 使用GDAL\n");
    printf("  gdal_translate -of GMT %s pixel.grd\n", output_file);
    
    return 0;
}