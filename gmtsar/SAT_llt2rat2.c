/*	$Id$	*/
/****************************************************************************
 *  Program to project a longitude, latitude, and topography
 *  into a file of range, azimuth, and topography.
 *  优化说明：
 *  1. 使用OpenMP并行化输入点处理循环
 *  2. 优化orb_pos的内存布局，提升缓存命中率
 *  3. 减少循环内重复计算，预计算常数
 *  4. 线程私有数据分配，避免竞争
 *  5. 优化I/O操作，批量写入
 ****************************************************************************/

#include "gmtsar.h"
#include "llt2xyz.h"
#include "orbit.h"
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define R 0.61803399
#define C 0.382
#define SHFT2(a, b, c) (a) = (b); (b) = (c);
#define SHFT3(a, b, c, d) (a) = (b); (b) = (c); (c) = (d);
#define TOL 2
#define BATCH_SIZE 1024  // 批量写入大小
#define SOL 299792458.0  // 光速常量（补充定义）

char *USAGE = " \n Usage: "
              "SAT_llt2rat2 master.PRM prec [-bo[s|d]] outputfile  \n\n"
              "             master.PRM   -  parameter file for master image and points to LED orbit file \n"
              "             precise      -  (0) standard back geocoding, (1) polynomial refinenent (slower) \n"
              "             -bos or -bod -  binary single or double precision output (only output results within \n"
              "                             data coverage, PRM num_lines, num_rng_bins ) \n"
              "             outputfile   -  range, azimuth, elevation(ref to radius in PRM), lon, lat [ASCII default] \n"
              " example: cat test.txt | SAT_llt2rat2 master.PRM 1 -bos output    \n";

int npad = 8000;

// 优化：将orb_pos改为结构体数组，提升缓存局部性
typedef struct {
    double time;
    double x;
    double y;
    double z;
} OrbPos;

// 声明外部函数（补充必要的函数声明）
void read_orb(FILE *, struct SAT_ORB *);
void set_prm_defaults(struct PRM *);
void hermite_c(double *, double *, double *, int, int, double, double *, int *);
void interpolate_SAT_orbit_slow(struct SAT_ORB *orb, double time, double *x, double *y, double *z, int *ir);
void polyfit(double *, double *, double *, int *, int *);
//void die(const char *msg, const char *file);  // 补充die函数声明
void null_sio_struct(struct PRM *prm);        // 补充null_sio_struct声明
void get_sio_struct(FILE *fp, struct PRM *prm); // 补充get_sio_struct声明
void plh2xyz(double *plh, double *xyz, double ra, double f); // 补充plh2xyz声明

// 补充die函数的实现（与声明严格一致）
/*void die(const char *msg, const char *file) {
    fprintf(stderr, "%s %s\n", msg, file);
    exit(EXIT_FAILURE);
}*/
// 优化：将dist函数内联，减少函数调用开销
static inline double dist(double x, double y, double z, int n, const OrbPos *orb_pos) {
    double dx = x - orb_pos[n].x;
    double dy = y - orb_pos[n].y;
    double dz = z - orb_pos[n].z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// 优化：黄金分割搜索函数（修改为使用OrbPos，修复索引计算和类型错误）
int goldop(double ts, double t1, const OrbPos *orb_pos, int ax, int bx, int cx, double xpx, double xpy, double xpz, double *rng, double *tm) {
    double f1, f2;
    int x0, x1, x2, x3;
    int xmin;
    int n = bx - ax;

    x0 = ax;
    x3 = bx;
    if (abs(bx - cx) > abs(cx - ax)) {
        x1 = cx;
        x2 = cx + (int)(C * (bx - cx) + 0.5);  // 四舍五入避免截断错误
    } else {
        x2 = cx;
        x1 = cx - (int)(C * (cx - ax) + 0.5);
    }

    // 边界检查：确保x1/x2在有效范围内
    x1 = (x1 < ax) ? ax : (x1 > bx) ? bx : x1;
    x2 = (x2 < ax) ? ax : (x2 > bx) ? bx : x2;

    f1 = dist(xpx, xpy, xpz, x1, orb_pos);
    f2 = dist(xpx, xpy, xpz, x2, orb_pos);

    while ((x3 - x0) > TOL && abs(x2 - x1) > 0) {
        if (f2 < f1) {
            x0 = x1;
            x1 = x2;
            x2 = (int)(R * x3 + C * x1 + 0.5);
            x2 = (x2 > bx) ? bx : x2;
            SHFT2(f1, f2, dist(xpx, xpy, xpz, x2, orb_pos));
        } else {
            x3 = x2;
            x2 = x1;
            x1 = (int)(R * x0 + C * x2 + 0.5);
            x1 = (x1 < ax) ? ax : x1;
            SHFT2(f2, f1, dist(xpx, xpy, xpz, x1, orb_pos));
        }
    }

    if (f1 < f2) {
        xmin = x1;
        *tm = orb_pos[x1].time;
        *rng = f1;
    } else {
        xmin = x2;
        *tm = orb_pos[x2].time;
        *rng = f2;
    }

    // 确保xmin在有效范围内
    xmin = (xmin >= ax && xmin <= bx) ? xmin : ax;

    return xmin;
}

// 优化：轨道计算函数（输出OrbPos结构体数组）
int calorb_alos(struct SAT_ORB *orb, OrbPos *orb_pos, double ts, double t1, int nrec) {
    int i, k, ir;
    int nval = 6;
    double xs, ys, zs;
    double *pt, *px, *py, *pz, *pvx, *pvy, *pvz;
    double pt0;
    double time;

    // 内存分配检查
    px = (double *)malloc(orb->nd * sizeof(double));
    py = (double *)malloc(orb->nd * sizeof(double));
    pz = (double *)malloc(orb->nd * sizeof(double));
    pvx = (double *)malloc(orb->nd * sizeof(double));
    pvy = (double *)malloc(orb->nd * sizeof(double));
    pvz = (double *)malloc(orb->nd * sizeof(double));
    pt = (double *)malloc(orb->nd * sizeof(double));

    if (!px || !py || !pz || !pvx || !pvy || !pvz || !pt) {
        perror("malloc failed in calorb_alos");
        exit(EXIT_FAILURE);
    }

    pt0 = 86400. * orb->id + orb->sec;
    for (k = 0; k < orb->nd; k++) {
        pt[k] = pt0 + k * orb->dsec;
        px[k] = orb->points[k].px;
        py[k] = orb->points[k].py;
        pz[k] = orb->points[k].pz;
        pvx[k] = orb->points[k].vx;
        pvy[k] = orb->points[k].vy;
        pvz[k] = orb->points[k].vz;
    }

    // 优化：使用OpenMP并行化轨道插值（如果有多个CPU核心）
    #pragma omp parallel for private(time, ir, xs, ys, zs) schedule(static)
    for (i = 0; i < nrec + npad * 2; i++) {
        time = t1 - npad * ts + i * ts;
        orb_pos[i].time = time;

        hermite_c(pt, px, pvx, orb->nd, nval, time, &xs, &ir);
        hermite_c(pt, py, pvy, orb->nd, nval, time, &ys, &ir);
        hermite_c(pt, pz, pvz, orb->nd, nval, time, &zs, &ir);

        orb_pos[i].x = xs;
        orb_pos[i].y = ys;
        orb_pos[i].z = zs;
    }

    // 释放内存
    free(px); free(py); free(pz); free(pt);
    free(pvx); free(pvy); free(pvz);

    return orb->nd;
}

int main(int argc, char **argv) {
    FILE *fprm1 = NULL;
    int otype = 1;          // 1:ASCII, 2:float, 3:double
    int precise = 0;
    int lookdir = 1;
    int nrec;
    double ts, dr;
    double r0, rf, a0, af;
    double fll;
    double t1, t2;
    double prf, RE, near_range, rshift, sub_int_r, chirp_ext, ashift, sub_int_a;
    double fd1, fdd1, vel, lambda;
    double sol_half = 0.5 * SOL;  // 预计算常数

    // 批量输出缓冲区（主进程用，线程使用私有缓冲区）
    FILE *outfp = stdout;    // 默认输出到标准输出
    char *outfilename = NULL;

    struct PRM prm;
    struct SAT_ORB *orb = NULL;
    FILE *ldrfile = NULL;

    // 命令行参数解析（修复参数解析逻辑）
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "%s\n", USAGE);
        exit(EXIT_FAILURE);
    }

    precise = atoi(argv[2]);
    if (argc == 4) {
        // 处理：SAT_llt2rat2 prm prec outputfile
        outfilename = argv[3];
        outfp = fopen(outfilename, "w");
        if (!outfp) {
            perror("Failed to open output file");
            exit(EXIT_FAILURE);
        }
    } else if (argc == 5) {
        // 处理：SAT_llt2rat2 prm prec -bos/-bod outputfile
        if (!strcmp(argv[3], "-bos")) otype = 2;
        else if (!strcmp(argv[3], "-bod")) otype = 3;
        else {
            fprintf(stderr, "Invalid option: %s\n", argv[3]);
            exit(EXIT_FAILURE);
        }
        outfilename = argv[4];
        outfp = (otype == 2 || otype == 3) ? fopen(outfilename, "wb") : fopen(outfilename, "w");
        if (!outfp) {
            perror("Failed to open output file");
            exit(EXIT_FAILURE);
        }
    }

    // 读取PRM文件
    if ((fprm1 = fopen(argv[1], "r")) == NULL) {
        fprintf(stderr, "couldn't open master.PRM \n");
        exit(EXIT_FAILURE);
    }
    null_sio_struct(&prm);
    set_prm_defaults(&prm);
    get_sio_struct(fprm1, &prm);
    fclose(fprm1);

    // 预计算常数（避免循环内重复计算）
    lookdir = (strcmp(prm.lookdir, "L") == 0) ? -1 : 1;
    fll = (prm.ra - prm.rc) / prm.ra;
    dr = sol_half / prm.fs;
    t1 = 86400. * prm.clock_start + (prm.nrows - prm.num_valid_az) / (2. * prm.prf);
    t2 = t1 + prm.num_patches * prm.num_valid_az / prm.prf;
    ts = (prm.prf < 600.) ? (1. / prm.prf) : (2. / prm.prf);
    npad = (prm.prf < 600.) ? 20000 : 8000;
    nrec = (int)((t2 - t1) / ts + 0.5);  // 四舍五入

    // 范围过滤参数
    r0 = -10.;
    rf = prm.num_rng_bins + 10.;
    a0 = -20.;
    af = prm.num_patches * prm.num_valid_az + 20.;

    // 预计算PRM中的常用参数
    prf = prm.prf;
    RE = prm.RE;
    near_range = prm.near_range;
    rshift = prm.rshift;
    sub_int_r = prm.sub_int_r;
    chirp_ext = prm.chirp_ext;
    ashift = prm.ashift;
    sub_int_a = prm.sub_int_a;
    fd1 = prm.fd1;
    fdd1 = prm.fdd1;
    vel = prm.vel;
    lambda = prm.lambda;

    // 读取轨道数据
    ldrfile = fopen(prm.led_file, "r");
    if (!ldrfile) die("can't open ", prm.led_file);
    orb = (struct SAT_ORB *)malloc(sizeof(struct SAT_ORB));
    read_orb(ldrfile, orb);
    fclose(ldrfile);

    // 分配轨道位置数组（优化为结构体数组）
    OrbPos *orb_pos = (OrbPos *)malloc((nrec + 2 * npad) * sizeof(OrbPos));
    if (!orb_pos) {
        perror("malloc orb_pos failed");
        exit(EXIT_FAILURE);
    }

    // 计算轨道位置
    calorb_alos(orb, orb_pos, ts, t1, nrec);

    // 第一步：读取所有输入点到内存（解决scanf线程不安全问题）
    double *input_data = NULL;
    int input_size = 0;
    int input_capacity = 1024 * 1024;  // 初始容量1M点

    input_data = (double *)malloc(input_capacity * 3 * sizeof(double));
    if (!input_data) {
        perror("malloc input_data failed");
        exit(EXIT_FAILURE);
    }

    double rln, rlt, rht;
    while (scanf(" %lf %lf %lf ", &rln, &rlt, &rht) == 3) {
        if (input_size >= input_capacity) {
            input_capacity *= 2;
            double *tmp = (double *)realloc(input_data, input_capacity * 3 * sizeof(double));
            if (!tmp) {
                perror("realloc input_data failed");
                exit(EXIT_FAILURE);
            }
            input_data = tmp;
        }
        input_data[input_size * 3 + 0] = rln;  // 经度
        input_data[input_size * 3 + 1] = rlt;  // 纬度
        input_data[input_size * 3 + 2] = rht;  // 高度
        input_size++;
    }
    fprintf(stderr, "Read %d input points\n", input_size);

    // 第二步：并行处理所有输入点
    // 优化：设置OpenMP线程数（可根据CPU核心数调整）
    omp_set_num_threads(omp_get_max_threads());

    #pragma omp parallel private(rln, rlt, rht) \
        private(xp, rp, xt, rng0, tm) \
        private(stai, endi, midi, dt, k, ir, xs, ys, zs) \
        private(time, rng, d, vec0, vec1, vec2, det, dtt) \
        private(dopc, rdd, daa, drr) \
        private(f1, f2, x0, x1, x2, x3, xmin) \
        shared(input_data, input_size, orb_pos, prm, otype, precise, r0, rf, a0, af)
    {
        // 线程私有变量定义
        double xp[3], rp[3], xt[3];
        double rng0, tm;
        int stai, endi, midi;
        double dt;
        int k, ir;
        double xs, ys, zs;
        double time[20], rng[20], d[3];
        double vec0[3], vec1[3], vec2[3];
        double det, dtt;
        double dopc, rdd, daa, drr;
        double f1, f2;
        int x0, x1, x2, x3, xmin;

        // 线程私有输出缓冲区
        float ds_buf[BATCH_SIZE * 5];
        double dd_buf[BATCH_SIZE * 5];
        char ascii_buf[BATCH_SIZE * 128];  // 每个条目最多128字符
        int buf_idx = 0;
        int ascii_len = 0;

        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < input_size; i++) {
            rln = input_data[i * 3 + 0];
            rlt = input_data[i * 3 + 1];
            rht = input_data[i * 3 + 2];

            // 初始化rp数组（经纬度高度）
            rp[0] = rlt;    // 纬度
            rp[1] = rln;    // 经度
            rp[2] = rht;    // 高度

            // 转换为XYZ坐标（修复参数传递顺序）
            plh2xyz(rp, xp, prm.ra, fll);
            if (rp[1] > 180.) rp[1] -= 360.;  // 经度归一化到[-180, 180]

            // 计算地形高度（相对于地球半径）
            rp[2] = sqrt(xp[0]*xp[0] + xp[1]*xp[1] + xp[2]*xp[2]) - RE;

            // 黄金分割搜索最小距离
            stai = 0;
            endi = nrec + npad * 2 - 1;
            midi = stai + (int)((endi - stai) * C + 0.5);
            goldop(ts, t1, orb_pos, stai, endi, midi, xp[0], xp[1], xp[2], &rng0, &tm);

            // 多项式精修（如果需要）
            if (precise == 1) {
                memset(d, 0, sizeof(double)*3);
                memset(time, 0, sizeof(double)*20);
                memset(rng, 0, sizeof(double)*20);

                int ntt = 10;
                dt = 1.0 / ntt;
                int interpolate_ok = 1;

                for (k = 0; k < ntt; k++) {
                    time[k] = dt * (k - ntt / 2 + 0.5);
                    double t11 = tm + time[k];
                    interpolate_SAT_orbit_slow(orb, t11, &xs, &ys, &zs, &ir);
                    if (ir != 0) {
                        interpolate_ok = 0;
                        break;
                    }
                    rng[k] = sqrt((xp[0]-xs)*(xp[0]-xs) + (xp[1]-ys)*(xp[1]-ys) + (xp[2]-zs)*(xp[2]-zs)) - rng0;
                    if (k == 0) { vec0[0] = xs; vec0[1] = ys; vec0[2] = zs; }
                    if (k == ntt-1) { vec1[0] = xs; vec1[1] = ys; vec1[2] = zs; }
                }

                if (interpolate_ok) {
                    // 计算轨道速度向量
                    vec1[0] -= vec0[0];
                    vec1[1] -= vec0[1];
                    vec1[2] -= vec0[2];
                    double vec1_norm = sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1] + vec1[2]*vec1[2]);

                    if (vec1_norm > 1e-10) {
                        int nc = 3;
                        polyfit(time, rng, d, &ntt, &nc);

                        if (fabs(d[2]) > 1e-10) {
                            dtt = -d[1]/(2*d[2]);
                            if (tm + dtt >= t1 && tm + dtt <= t2) tm += dtt;
                        }

                        // 重新插值轨道位置
                        interpolate_SAT_orbit_slow(orb, tm, &xs, &ys, &zs, &ir);
                        rng0 = sqrt((xp[0]-xs)*(xp[0]-xs) + (xp[1]-ys)*(xp[1]-ys) + (xp[2]-zs)*(xp[2]-zs));

                        // 计算视线向量
                        vec2[0] = xp[0]-xs;
                        vec2[1] = xp[1]-ys;
                        vec2[2] = xp[2]-zs;

                        // 计算行列式判断方向
                        det = (vec2[1]*vec1[2]-vec2[2]*vec1[1])*xs + 
                              (vec2[2]*vec1[0]-vec2[0]*vec1[2])*ys + 
                              (vec2[0]*vec1[1]-vec2[1]*vec1[0])*zs;
                        det = (det * lookdir > 0) ? 1.0 : -1.0;
                    }
                }
            }

            // 计算像素坐标
            xt[0] = rng0 * det;
            xt[1] = tm;

            // 距离像素坐标转换
            xt[0] = (xt[0] - near_range) / dr - (rshift + sub_int_r) + chirp_ext;
            // 方位像素坐标转换
            xt[1] = prf * (xt[1] - t1) - (ashift + sub_int_a);

            // Envisat偏置校正
            if (prm.SC_identity == 4) {
                xt[0] += 8.4;
                xt[1] += 4;
            }

            // 多普勒校正
            if (fd1 != 0.) {
                dopc = fd1 + fdd1 * (near_range + dr * prm.num_rng_bins / 2.);
                rdd = (vel * vel) / rng0;
                daa = -0.5 * (lambda * dopc) / rdd;
                drr = 0.5 * rdd * daa * daa / dr;
                daa = prf * daa;
                xt[0] += drr;
                xt[1] += daa;
            }

            // 过滤超出范围的点（仅二进制输出）
            if ((otype > 1) && (xt[0] < r0 || xt[0] > rf || xt[1] < a0 || xt[1] > af)) {
                continue;
            }

            // 写入输出缓冲区
            if (otype == 1) {
                // ASCII输出：避免缓冲区溢出
                int remaining = sizeof(ascii_buf) - ascii_len;
                int written = snprintf(ascii_buf + ascii_len, remaining,
                    "%.9f %.9f %.9f %.9f %.9f\n", 
                    xt[0], xt[1], rp[2], rp[1], rp[0]);

                if (written >= remaining) {
                    // 缓冲区不足，先刷新
                    #pragma omp critical(io_lock)
                    fwrite(ascii_buf, 1, ascii_len, outfp);
                    ascii_len = 0;
                    // 重新写入当前条目
                    snprintf(ascii_buf, sizeof(ascii_buf),
                        "%.9f %.9f %.9f %.9f %.9f\n", 
                        xt[0], xt[1], rp[2], rp[1], rp[0]);
                    ascii_len = strlen(ascii_buf);
                } else {
                    ascii_len += written;
                }
            } else if (otype == 2) {
                // 单精度二进制
                ds_buf[buf_idx * 5 + 0] = (float)xt[0];
                ds_buf[buf_idx * 5 + 1] = (float)xt[1];
                ds_buf[buf_idx * 5 + 2] = (float)rp[2];
                ds_buf[buf_idx * 5 + 3] = (float)rp[1];
                ds_buf[buf_idx * 5 + 4] = (float)rp[0];
                buf_idx++;

                if (buf_idx >= BATCH_SIZE) {
                    #pragma omp critical(io_lock)
                    fwrite(ds_buf, sizeof(float), buf_idx * 5, outfp);
                    buf_idx = 0;
                }
            } else if (otype == 3) {
                // 双精度二进制
                dd_buf[buf_idx * 5 + 0] = xt[0];
                dd_buf[buf_idx * 5 + 1] = xt[1];
                dd_buf[buf_idx * 5 + 2] = rp[2];
                dd_buf[buf_idx * 5 + 3] = rp[1];
                dd_buf[buf_idx * 5 + 4] = rp[0];
                buf_idx++;

                if (buf_idx >= BATCH_SIZE) {
                    #pragma omp critical(io_lock)
                    fwrite(dd_buf, sizeof(double), buf_idx * 5, outfp);
                    buf_idx = 0;
                }
            }
        }

        // 刷新线程私有缓冲区剩余数据
        #pragma omp critical(io_lock)
        {
            if (otype == 1 && ascii_len > 0) {
                fwrite(ascii_buf, 1, ascii_len, outfp);
            } else if (otype == 2 && buf_idx > 0) {
                fwrite(ds_buf, sizeof(float), buf_idx * 5, outfp);
            } else if (otype == 3 && buf_idx > 0) {
                fwrite(dd_buf, sizeof(double), buf_idx * 5, outfp);
            }
        }
    }

    // 释放内存
    free(input_data);
    free(orb_pos);
    if (orb) {
        if (orb->points) free(orb->points);  // 释放轨道点数据
        free(orb);
    }

    // 关闭输出文件
    if (outfp != stdout) fclose(outfp);

    return 0;
}

