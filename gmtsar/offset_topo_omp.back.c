#include "gmtsar.h"
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// 辅助宏函数
#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a) < (b) ? (a) : (b))
#endif

// 兼容的内存对齐分配函数
void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
    #ifdef _ISOC11_SOURCE
    ptr = aligned_alloc(alignment, size);
    #else
    // POSIX 方式
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL;
    }
    #endif
    if (ptr == NULL) {
        // 回退到普通malloc
        ptr = malloc(size);
    }
    return ptr;
}

// 计时函数
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    double total_start = get_time();
    
    int i, j, is, js;
    int ni, nj, ntot;
    int xshft, yshft, ib = 200;
    int ns;
    int imax = 0, jmax = 0;
    double avea;
    double suma, sumt, maxcorr = -1e30;

    // 自动设置线程数（使用75%的CPU核心）
    int ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int nthreads = (ncpu * 3) / 4;
    if (nthreads < 1) nthreads = 1;
    omp_set_num_threads(nthreads);
    printf("使用 %d 个线程进行计算\n", nthreads);

    void *API = NULL;
    struct GMT_GRID *A = NULL, *T = NULL, *TS = NULL;

    if (argc < 6) {
        fprintf(stderr,
                "用法: offset_topo2 amp_master.grd topo_ra.grd rshift ashift ns [topo_shift.grd]\n");
        exit(EXIT_FAILURE);
    }

    API = GMT_Create_Session(argv[0], 0U, 0U, NULL);

    xshft = atoi(argv[3]);
    yshft = atoi(argv[4]);
    ns = atoi(argv[5]);

    // 读取网格数据
    double read_start = get_time();
    A = GMT_Read_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE,
                      GMT_GRID_HEADER_ONLY, NULL, argv[1], NULL);
    T = GMT_Read_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE,
                      GMT_GRID_HEADER_ONLY, NULL, argv[2], NULL);

    if (A->header->n_columns != T->header->n_columns) {
        fprintf(stderr, "错误: 网格宽度不匹配\n");
        exit(EXIT_FAILURE);
    }

    // 读取实际数据
    GMT_Read_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE,
                  GMT_GRID_DATA_ONLY, NULL, argv[1], A);
    GMT_Read_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE,
                  GMT_GRID_DATA_ONLY, NULL, argv[2], T);
    printf("数据读取完成: %.3f秒\n", get_time() - read_start);

    ni = min(A->header->n_rows, T->header->n_rows);
    nj = T->header->n_columns;
    printf("网格尺寸: %d x %d, 总像素数: %d\n", ni, nj, ni * nj);

    // 为输出网格分配内存（如果需要）
    if (argc >= 7) {
        TS = GMT_Create_Data(API, GMT_IS_GRID, GMT_IS_SURFACE, GMT_GRID_ALL,
                             NULL, A->header->wesn, A->header->inc,
                             A->header->registration, GMT_NOTSET, NULL);
    }

    /* ---------- 1. 预计算阶段 ---------- */
    double precompute_start = get_time();
    
    // 计算A的均值
    suma = 0.0;
    ntot = ni * nj;
    
    #pragma omp parallel for reduction(+:suma)
    for (i = 0; i < ni; i++) {
        double local_sum = 0.0;
        int row_start = i * nj;
        for (j = 0; j < nj; j++) {
            local_sum += A->data[row_start + j];
        }
        suma += local_sum;
    }
    avea = suma / ntot;
    printf("均值计算完成: avea = %g\n", avea);

    // 分配预计算数组（使用兼容的内存分配）
    size_t grid_size = ni * nj * sizeof(float);
    float *A_centered = (float*)aligned_malloc(grid_size, 32);
    float *T_grad_x = (float*)aligned_malloc(grid_size, 32);
    
    if (!A_centered || !T_grad_x) {
        fprintf(stderr, "内存分配失败，尝试使用普通malloc...\n");
        // 回退到普通malloc
        if (A_centered) free(A_centered);
        if (T_grad_x) free(T_grad_x);
        A_centered = (float*)malloc(grid_size);
        T_grad_x = (float*)malloc(grid_size);
        
        if (!A_centered || !T_grad_x) {
            fprintf(stderr, "内存分配彻底失败\n");
            exit(EXIT_FAILURE);
        }
    }

    // 预计算A的中心化值
    #pragma omp parallel for collapse(2)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            int idx = i * nj + j;
            A_centered[idx] = A->data[idx] - avea;
        }
    }

    // 预计算T的x方向梯度（中心差分）
    #pragma omp parallel for collapse(2)
    for (i = 0; i < ni; i++) {
        for (j = 1; j < nj - 1; j++) {
            int idx = i * nj + j;
            T_grad_x[idx] = T->data[idx + 1] - T->data[idx - 1];
        }
    }
    
    // 边界处理：梯度设为0
    #pragma omp parallel for
    for (i = 0; i < ni; i++) {
        T_grad_x[i * nj] = 0.0f;               // 左边界
        T_grad_x[i * nj + (nj - 1)] = 0.0f;    // 右边界
    }
    
    printf("预计算完成: %.3f秒\n", get_time() - precompute_start);

    /* ---------- 2. 互相关搜索（优化版） ---------- */
    double corr_start = get_time();
    
    // 计算有效搜索区域（避免边界检查）
    int valid_start_i = max(ib, ns);
    int valid_end_i = ni - max(ib, ns);
    int valid_start_j = max(ib, ns);
    int valid_end_j = nj - max(ib, ns);
    
    int is_start = -ns + yshft;
    int is_end = ns + yshft;
    int js_start = -ns + xshft;
    int js_end = ns + xshft;
    
    printf("搜索范围: is=[%d, %d], js=[%d, %d]\n", 
           is_start, is_end, js_start, js_end);
    printf("有效计算区域: i=[%d, %d], j=[%d, %d]\n",
           valid_start_i, valid_end_i, valid_start_j, valid_end_j);

    // 使用更保守的块大小
    int block_size_i = 4;  // 垂直方向块大小
    int block_size_j = 4;  // 水平方向块大小
    
    #pragma omp parallel
    {
        double maxcorr_t = -1e30;
        int imax_t = 0, jmax_t = 0;
        
        // 为每个线程分配局部累加器
        double local_sumc, local_sumaa, local_sumtt;
        
        #pragma omp for collapse(2) schedule(dynamic, 2) nowait
        for (int is_block = is_start; is_block <= is_end; is_block += block_size_i) {
            for (int js_block = js_start; js_block <= js_end; js_block += block_size_j) {
                
                // 处理当前块内的所有位移
                int is_block_end = min(is_block + block_size_i - 1, is_end);
                int js_block_end = min(js_block + block_size_j - 1, js_end);
                
                for (int is_local = is_block; is_local <= is_block_end; is_local++) {
                    for (int js_local = js_block; js_local <= js_block_end; js_local++) {
                        
                        // 重置局部累加器
                        local_sumc = local_sumaa = local_sumtt = 0.0;
                        
                        // 提前计算偏移后的边界
                        int start_i = max(valid_start_i, is_local);
                        int end_i = min(valid_end_i, ni + is_local);
                        int start_j = max(valid_start_j, js_local);
                        int end_j = min(valid_end_j, nj + js_local);
                        
                        // 计算相关系数
                        for (i = start_i; i < end_i; i++) {
                            int i1 = i - is_local;
                            long base_idx = i * nj;
                            long base_idx1 = i1 * nj;
                            
                            // 手动展开循环（提高性能）
                            int j;
                            for (j = start_j; j + 3 < end_j; j += 4) {
                                int j1 = j - js_local;
                                
                                // 批量加载数据
                                float ra0 = A_centered[base_idx + j];
                                float ra1 = A_centered[base_idx + j + 1];
                                float ra2 = A_centered[base_idx + j + 2];
                                float ra3 = A_centered[base_idx + j + 3];
                                
                                float rt0 = T_grad_x[base_idx1 + j1];
                                float rt1 = T_grad_x[base_idx1 + j1 + 1];
                                float rt2 = T_grad_x[base_idx1 + j1 + 2];
                                float rt3 = T_grad_x[base_idx1 + j1 + 3];
                                
                                // 批量计算乘积
                                local_sumc += ra0 * rt0 + ra1 * rt1 + ra2 * rt2 + ra3 * rt3;
                                local_sumaa += ra0 * ra0 + ra1 * ra1 + ra2 * ra2 + ra3 * ra3;
                                local_sumtt += rt0 * rt0 + rt1 * rt1 + rt2 * rt2 + rt3 * rt3;
                            }
                            
                            // 处理剩余的点
                            for (; j < end_j; j++) {
                                int j1 = j - js_local;
                                
                                float ra = A_centered[base_idx + j];
                                float rt = T_grad_x[base_idx1 + j1];
                                
                                local_sumc += ra * rt;
                                local_sumaa += ra * ra;
                                local_sumtt += rt * rt;
                            }
                        }
                        
                        // 计算相关系数
                        double denom = local_sumaa * local_sumtt;
                        if (denom > 0.0) {
                            double corr = local_sumc / sqrt(denom);
                            if (corr > maxcorr_t) {
                                maxcorr_t = corr;
                                imax_t = is_local;
                                jmax_t = js_local;
                            }
                        }
                    }
                }
            }
        }
        
        // 更新全局最大值
        #pragma omp critical
        {
            if (maxcorr_t > maxcorr) {
                maxcorr = maxcorr_t;
                imax = imax_t;
                jmax = jmax_t;
            }
        }
    }
    
    printf("互相关搜索完成: %.3f秒\n", get_time() - corr_start);
    printf("最优结果: rshift=%d ashift=%d maxcorr=%g\n", jmax, imax, maxcorr);

    /* ---------- 3. 输出移位后的DEM（如果需要） ---------- */
    if (argc >= 7) {
        double output_start = get_time();
        
        #pragma omp parallel for collapse(2)
        for (i = 0; i < ni; i++) {
            int i1 = i - imax;
            int i1_valid = (i1 >= 0 && i1 < ni);
            
            for (j = 0; j < nj; j++) {
                int j1 = j - jmax;
                int idx = i * nj + j;
                
                if (i1_valid && j1 >= 0 && j1 < nj) {
                    int idx1 = i1 * nj + j1;
                    TS->data[idx] = T->data[idx1];
                } else {
                    TS->data[idx] = 0.0f;
                }
            }
        }
        
        GMT_Write_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE,
                       GMT_GRID_ALL, NULL, argv[6], TS);
        
        printf("输出文件已保存: %s (%.3f秒)\n", argv[6], get_time() - output_start);
    }

    /* ---------- 4. 清理内存 ---------- */
    double cleanup_start = get_time();
    
    free(A_centered);
    free(T_grad_x);
    
    GMT_Destroy_Session(API);
    
    printf("总运行时间: %.3f秒\n", get_time() - total_start);
    
    return EXIT_SUCCESS;
}