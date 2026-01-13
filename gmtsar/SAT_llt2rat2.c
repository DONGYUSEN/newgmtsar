/* SAT_llt2rat2_omp.c */
/* OpenMP parallel version - optimized for performance and memory safety */

#include "gmtsar.h"
#include "llt2xyz.h"
#include "orbit.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>  // 用于标准整数类型

#define R 0.61803399
#define C (1.0 - R)  // 更准确的黄金分割常数
#define TOL 2
#define SOL 299792458.0
#define INIT_CAPACITY 1000000  // 初始容量，减少realloc调用

/* 优化轨道缓存结构 - 内存对齐 */
typedef struct {
    double time;
    double x, y, z;
} __attribute__((aligned(32))) OrbPos;  // 32字节对齐，适合AVX

/* 预计算的结构，避免重复计算 */
typedef struct {
    double inv_dr;      // 1/dr
    double prf;         // 脉冲重复频率
    double t1;          // 起始时间
    double near_range;  // 近距
    double dr;          // 距离分辨率
    double ra;          // 地球半径
    double fll;         // 扁率因子
    double RE;
} CalcParams;

/* 外部函数声明 */
void read_orb(FILE *, struct SAT_ORB *);
void set_prm_defaults(struct PRM *);
void hermite_c(double *, double *, double *, int, int, double, double *, int *);
void null_sio_struct(struct PRM *);
void get_sio_struct(FILE *, struct PRM *);
void plh2xyz(double *, double *, double, double);

/* 快速距离计算 - 使用hypot减少溢出风险 */
static inline double dist3_fast(double x, double y, double z, const OrbPos *o) {
    double dx = x - o->x;
    double dy = y - o->y;
    double dz = z - o->z;
    // 使用hypot函数，更稳定
    return hypot(hypot(dx, dy), dz);
}

/* 优化的黄金搜索算法 */
static int golden_search(const OrbPos *orb, int a, int b, 
                         double x, double y, double z, 
                         double *rng, double *tm) {
    int x0 = a, x3 = b;
    int x1 = a + (int)((b - a) * C);
    int x2 = b - (int)((b - a) * C);
    
    // 预计算距离
    double f1 = dist3_fast(x, y, z, &orb[x1]);
    double f2 = dist3_fast(x, y, z, &orb[x2]);
    
    int iterations = 0;
    const int max_iter = 100;  // 安全限制
    
    while ((x3 - x0) > TOL && iterations++ < max_iter) {
        if (f2 < f1) {
            // 最小值在[x1, x3]区间
            x0 = x1; 
            x1 = x2;
            x2 = (int)(R * x3 + C * x1);
            f1 = f2;
            f2 = dist3_fast(x, y, z, &orb[x2]);
        } else {
            // 最小值在[x0, x2]区间
            x3 = x2; 
            x2 = x1;
            x1 = (int)(R * x0 + C * x2);
            f2 = f1;
            f1 = dist3_fast(x, y, z, &orb[x1]);
        }
    }
    
    // 最终确定最小值点
    int xmin;
    if (f1 < f2) {
        xmin = x1; 
        *rng = f1;
    } else {
        xmin = x2; 
        *rng = f2;
    }
    
    *tm = orb[xmin].time;
    return xmin;
}

/* 优化的轨道计算函数 */
static int calculate_orbit_positions(struct SAT_ORB *orb, OrbPos *op, 
                                     double ts, double t1, int nrec, int npad) {
    int i, k;
    int nval = 6;
    int N = nrec + 2 * npad;
    
    // 一次性分配所有临时数组
    double *pt = malloc(sizeof(double) * orb->nd);
    double *px = malloc(sizeof(double) * orb->nd);
    double *py = malloc(sizeof(double) * orb->nd);
    double *pz = malloc(sizeof(double) * orb->nd);
    double *pvx = malloc(sizeof(double) * orb->nd);
    double *pvy = malloc(sizeof(double) * orb->nd);
    double *pvz = malloc(sizeof(double) * orb->nd);
    
    if (!pt || !px || !py || !pz || !pvx || !pvy || !pvz) {
        free(pt); free(px); free(py); free(pz);
        free(pvx); free(pvy); free(pvz);
        return -1;
    }
    
    // 预计算时间基准
    double pt0 = 86400.0 * orb->id + orb->sec;
    
    // 并行化数据准备（如果数据量足够大）
    #pragma omp parallel for schedule(static)
    for (k = 0; k < orb->nd; k++) {
        pt[k] = pt0 + k * orb->dsec;
        px[k] = orb->points[k].px;
        py[k] = orb->points[k].py;
        pz[k] = orb->points[k].pz;
        pvx[k] = orb->points[k].vx;
        pvy[k] = orb->points[k].vy;
        pvz[k] = orb->points[k].vz;
    }
    
    // 并行计算轨道位置
    #pragma omp parallel for schedule(static)
    for (i = 0; i < N; i++) {
        double time = t1 - npad * ts + i * ts;
        double xs, ys, zs;
        int ir;
        
        // 插值计算位置
        hermite_c(pt, px, pvx, orb->nd, nval, time, &xs, &ir);
        hermite_c(pt, py, pvy, orb->nd, nval, time, &ys, &ir);
        hermite_c(pt, pz, pvz, orb->nd, nval, time, &zs, &ir);
        
        // 存储结果
        op[i].time = time;
        op[i].x = xs;
        op[i].y = ys;
        op[i].z = zs;
    }
    
    // 清理临时内存
    free(pt); free(px); free(py); free(pz);
    free(pvx); free(pvy); free(pvz);
    
    return N;
}

/* 优化的主处理函数 */
static void process_points( double *in, double *out, int npt,
                          const OrbPos *orb_pos, int orb_count,
                          const CalcParams *params) {
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < npt; i++) {
        double rp[3], xp[3];
        
        // 输入数据顺序：经度、纬度、高度
        rp[1] = in[3 * i + 0];     // 经度
        rp[0] = in[3 * i + 1];     // 纬度
        rp[2] = in[3 * i + 2];     // 高度
        
        // 将地理坐标转换为地心直角坐标
        plh2xyz(rp, xp, params->ra, params->fll);

		/* compute the topography due to the difference between the local radius and
		 * center radius  修正高度？ysdong*/ 
		rp[2] = sqrt(xp[0] * xp[0] + xp[1] * xp[1] + xp[2] * xp[2]) - params->RE;
        in[3 * i + 2] = rp[2]; //add end ysdong 


        // 使用黄金搜索找到最近的点
        double rng0, tm;
        golden_search(orb_pos, 0, orb_count - 1, 
                     xp[0], xp[1], xp[2], &rng0, &tm);

        
        // 计算距离门和方位时间
        out[2 * i + 0] = (rng0 - params->near_range) * params->inv_dr;
        out[2 * i + 1] = params->prf * (tm - params->t1);
    }
}

/* 安全的内存重新分配 */
static void* safe_realloc(void *ptr, size_t new_size) {
    void *new_ptr = realloc(ptr, new_size);
    if (!new_ptr && new_size > 0) {
        fprintf(stderr, "Error: Memory reallocation failed (requested %zu bytes)\n", new_size);
        free(ptr);
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

/* 主函数 */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s prm_file precise [-bos|-bod]\n", argv[0]);
        fprintf(stderr, "  -bos: binary output (short integers)\n");
        fprintf(stderr, "  -bod: binary output (double precision)\n");
        return EXIT_FAILURE;
    }
    
    int binary_short = 0;
    int binary_double = 0;
    
    if (argc >= 4) {
        if (strcmp(argv[3], "-bos") == 0) binary_short = 1;
        else if (strcmp(argv[3], "-bod") == 0) binary_double = 1;
    }
    
    // 读取PRM参数
    struct PRM prm;
    null_sio_struct(&prm);
    set_prm_defaults(&prm);
    
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("Error opening PRM file");
        return EXIT_FAILURE;
    }
    get_sio_struct(f, &prm);
    fclose(f);
    
    int precise = atoi(argv[2]);
    if (precise <= 0) {
        fprintf(stderr, "Error: precise value must be positive\n");
        return EXIT_FAILURE;
    }
    
    // 读取轨道数据
    FILE *fo = fopen(prm.led_file, "r");
    if (!fo) {
        perror("Error opening LED file");
        return EXIT_FAILURE;
    }
    
    struct SAT_ORB *orb = malloc(sizeof(struct SAT_ORB));
    if (!orb) {
        perror("Memory allocation failed for orbit");
        fclose(fo);
        return EXIT_FAILURE;
    }
    
    read_orb(fo, orb);
    fclose(fo);
    
    // 计算时间参数
    double t1 = 86400.0 * prm.clock_start;
    double t2 = t1 + prm.num_patches * prm.num_valid_az / prm.prf;
    double ts = (prm.prf < 600.0) ? (1.0 / prm.prf) : (2.0 / prm.prf);
    int npad = (prm.prf < 600.0) ? 20000 : 8000;
    
    int nrec = (int)((t2 - t1) / ts + 0.5);
    
    // 分配轨道位置缓存
    OrbPos *orb_pos = malloc(sizeof(OrbPos) * (nrec + 2 * npad));
    if (!orb_pos) {
        perror("Memory allocation failed for orbit positions");
        free(orb->points);
        free(orb);
        return EXIT_FAILURE;
    }
    
    // 计算轨道位置
    int orb_count = calculate_orbit_positions(orb, orb_pos, ts, t1, nrec, npad);
    if (orb_count < 0) {
        fprintf(stderr, "Error calculating orbit positions\n");
        free(orb_pos);
        free(orb->points);
        free(orb);
        return EXIT_FAILURE;
    }
    
    // 读取输入数据（来自stdin）
    size_t capacity = INIT_CAPACITY;
    int npt = 0;
    double *in = malloc(sizeof(double) * 3 * capacity);
    if (!in) {
        perror("Memory allocation failed for input buffer");
        free(orb_pos);
        free(orb->points);
        free(orb);
        return EXIT_FAILURE;
    }
    
    double lon, lat, h;
    while (scanf("%lf %lf %lf", &lon, &lat, &h) == 3) {
        if (npt >= capacity) {
            capacity *= 2;
            in = safe_realloc(in, sizeof(double) * 3 * capacity);
        }
        in[3 * npt + 0] = lon;
        in[3 * npt + 1] = lat;
        in[3 * npt + 2] = h;
        npt++;
    }
    
    // 检查是否有有效数据
    if (npt == 0) {
        fprintf(stderr, "Error: No valid input data read from stdin\n");
        free(in);
        free(orb_pos);
        free(orb->points);
        free(orb);
        return EXIT_FAILURE;
    }
    
    // 输出缓冲区
    double *out = malloc(sizeof(double) * 2 * npt);
    if (!out) {
        perror("Memory allocation failed for output buffer");
        free(in);
        free(orb_pos);
        free(orb->points);
        free(orb);
        return EXIT_FAILURE;
    }
    
    // 预计算参数，避免重复计算
    CalcParams params;
    params.dr = 0.5 * SOL / prm.fs;
    params.inv_dr = 1.0 / params.dr;  // 预计算倒数，减少除法
    params.prf = prm.prf;
    params.t1 = t1;
    params.near_range = prm.near_range;
    params.ra = prm.ra;
    params.fll = (prm.ra - prm.rc) / prm.ra;
    params.RE = prm.RE; //ysdong
    
    
    // 处理所有点
    process_points(in, out, npt, orb_pos, orb_count, &params);
    
    // 输出结果
    for (int i = 0; i < npt; i++) {
        double rg = out[2 * i + 0];
        double az = out[2 * i + 1];
        double h  = in[3 * i + 2];
        double lat = in[3 * i + 1];
        double lon = in[3 * i + 0];
        
        if (binary_short) {
            // 二进制短整型输出
            int16_t buf[5];
            buf[0] = (int16_t)round(rg);
            buf[1] = (int16_t)round(az);
            buf[2] = (int16_t)round(h);
            buf[3] = (int16_t)round(lon * 1000);  // 保留3位小数
            buf[4] = (int16_t)round(lat * 1000);
            fwrite(buf, sizeof(int16_t), 5, stdout);
        } else if (binary_double) {
            // 二进制双精度输出
            double buf[5] = {rg, az, h, lon, lat};
            fwrite(buf, sizeof(double), 5, stdout);
        } else {
            // 文本输出
            printf("%.6f %.6f %.6f %.6f %.6f\n", rg, az, h, lon, lat);
        }
    }
    
    // 清理内存
    free(in);
    free(out);
    free(orb_pos);
    free(orb->points);
    free(orb);
    
    return EXIT_SUCCESS;
}