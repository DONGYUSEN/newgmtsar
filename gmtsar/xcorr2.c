
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <complex.h>
#include <pthread.h>
#include <fftw3.h>
#include <glib.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/types.h>
#include <stdint.h>
#include <errno.h>

#include "xcorr2.h"
#include "xcorr2_args.h"

struct st_corr_thread_data {
    const struct st_xcorr *xc;
    complex double *c1;
    complex double *c2;
    double xoff, yoff;
    int loc_x, loc_y;
    double corr;
    volatile gint done;
};

static unsigned long long detect_available_memory_bytes(void) {
    FILE *f;
    char line[256];
    unsigned long long kb;
    long pages, page_size;

    f = fopen("/proc/meminfo", "r");
    if (f != NULL) {
        while (fgets(line, sizeof(line), f) != NULL) {
            if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                fclose(f);
                return kb * 1024ULL;
            }
        }
        fclose(f);
    }

#ifdef _SC_AVPHYS_PAGES
    pages = sysconf(_SC_AVPHYS_PAGES);
    page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return (unsigned long long)pages * (unsigned long long)page_size;
#endif

    pages = sysconf(_SC_PHYS_PAGES);
    page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return ((unsigned long long)pages * (unsigned long long)page_size) / 2ULL;

    return 0;
}

static unsigned long long estimate_worker_peak_memory_bytes(const struct st_xcorr *xc) {
    unsigned long long nx_corr, ny_corr, nx_win, ny_win;
    unsigned long long p, fft_cells, corr_cells;
    unsigned long long bytes_window, bytes_interp_peak, bytes_freq_peak;
    unsigned long long ri;

    nx_corr = (unsigned long long)xc->xsearch * 2ULL;
    ny_corr = (unsigned long long)xc->ysearch * 2ULL;
    nx_win = nx_corr * 2ULL;
    ny_win = ny_corr * 2ULL;
    p = nx_win * ny_win;
    fft_cells = ny_win * (nx_win/2ULL + 1ULL);
    corr_cells = nx_corr * ny_corr;
    ri = (xc->ri > 1) ? (unsigned long long)xc->ri : 1ULL;

    bytes_window = 2ULL * p * sizeof(complex double);  // c1 + c2

    if (xc->ri > 1) {
        // peak while computing interp2 while interp1 is still alive:
        // c1 + c2 + interp1 + (in_fft + out_fft + out_for_interp2)
        bytes_interp_peak =
            bytes_window +
            (p * ri) * sizeof(complex double) +
            (p + 2ULL * p * ri) * sizeof(complex double);
    } else {
        bytes_interp_peak = bytes_window;
    }

    // peak around frequency correlation:
    // c1r + c2r + c1r_fft + c2r_fft + c3r + corr_slice
    bytes_freq_peak =
        2ULL * p * sizeof(double) +
        2ULL * fft_cells * sizeof(complex double) +
        p * sizeof(double) +
        corr_cells * sizeof(double);

    return (bytes_interp_peak > bytes_freq_peak) ? bytes_interp_peak : bytes_freq_peak;
}

static long auto_tune_threads(const struct st_xcorr *xc, long cpu_target_threads) {
    unsigned long long avail_bytes, worker_peak, usable_bytes;
    unsigned long long queued_window_bytes, per_thread_effective;
    double window_scale, safety_factor;
    long by_memory;
    long tuned;

    tuned = cpu_target_threads;
    if (tuned < 1) tuned = 1;

    avail_bytes = detect_available_memory_bytes();
    worker_peak = estimate_worker_peak_memory_bytes(xc);

    // queue_limit is 1, so one extra queued task may hold c1+c2 windows.
    queued_window_bytes =
        2ULL * (unsigned long long)(xc->xsearch * 4ULL) *
        (unsigned long long)(xc->ysearch * 4ULL) * sizeof(complex double);

    // Dynamic safety factor:
    // keep a margin for FFTW/allocator overhead while avoiding excessive throttling.
    window_scale = ((double)xc->xsearch * (double)xc->ysearch) / (256.0 * 256.0);
    if (window_scale < 1.0) window_scale = 1.0;
    safety_factor = 1.4 + 0.6 * sqrt(window_scale);

    per_thread_effective =
        (unsigned long long)((long double)worker_peak * safety_factor) + queued_window_bytes;
    if (avail_bytes == 0 || per_thread_effective == 0)
        return tuned;

    // keep some free headroom for OS/page cache and neighboring processes.
    usable_bytes = (unsigned long long)((long double)avail_bytes * 0.85L);
    by_memory = (long)(usable_bytes / per_thread_effective);
    if (by_memory < 1) {
        fprintf(stderr,
                "Estimated memory is insufficient for one worker (need about %.2f GB per worker). "
                "Reduce xsearch/ysearch or use -norange/-nointerp.\n",
                per_thread_effective / (1024.0 * 1024.0 * 1024.0));
        exit(EXIT_FAILURE);
    }

    fprintf(stderr,
            "Auto thread caps: cpu=%ld, memory=%ld (avail %.2f GB, est %.2f GB/thread, safety %.2f)\n",
            tuned, by_memory,
            avail_bytes / (1024.0 * 1024.0 * 1024.0),
            per_thread_effective / (1024.0 * 1024.0 * 1024.0),
            safety_factor);

    if (by_memory < tuned) {
        fprintf(stderr,
                "Auto-tuning threads: CPU cap=%ld, memory cap=%ld, using %ld.\n",
                tuned, by_memory, by_memory);
        tuned = by_memory;
    }

    return tuned;
}

static void compute_axis_positions(
        int *out, int n,
        int half_win,
        int master_len, int slave_len,
        double scale, double offset,
        const char *axis_name) {
    double master_lower, master_upper;
    double lower, upper;
    double s_lower, s_upper;
    static bool warned_x_overlap = false;
    static bool warned_y_overlap = false;

    if (n < 1) {
        fprintf(stderr, "Invalid %s sample count: %d\n", axis_name, n);
        exit(EXIT_FAILURE);
    }

    if (scale <= 0.0) {
        fprintf(stderr, "Invalid %s scale factor: %.6f\n", axis_name, scale);
        exit(EXIT_FAILURE);
    }

    master_lower = half_win;
    master_upper = master_len - half_win;
    if (master_lower > master_upper) {
        fprintf(stderr,
                "No feasible %s centers: master dimension too small for window size.\n",
                axis_name);
        exit(EXIT_FAILURE);
    }

    lower = master_lower;
    upper = master_upper;

    s_lower = (half_win - offset) / scale;
    s_upper = (slave_len - half_win - offset) / scale;
    if (s_lower > s_upper) {
        double t = s_lower;
        s_lower = s_upper;
        s_upper = t;
    }

    if (s_lower > lower) lower = s_lower;
    if (s_upper < upper) upper = s_upper;

    if (lower > upper) {
        bool *warned = NULL;

        if (strcmp(axis_name, "x") == 0)
            warned = &warned_x_overlap;
        else if (strcmp(axis_name, "y") == 0)
            warned = &warned_y_overlap;

        lower = master_lower;
        upper = master_upper;

        if (warned == NULL || !(*warned)) {
            fprintf(stderr,
                    "No strict feasible %s centers under shift/stretch constraints. "
                    "Using overlap mode to preserve requested sample count.\n",
                    axis_name);
            if (warned != NULL)
                *warned = true;
        }
    }

    if (n == 1) {
        out[0] = (int)llround((lower + upper) / 2.0);
        return;
    }

    double step = (upper - lower) / (n - 1);
    for (int i=0; i<n; i++)
        out[i] = (int)llround(lower + i * step);
}

static int clamp_center(int center, int half_win, int axis_len, int *clamp_count) {
    int min_center = half_win;
    int max_center = axis_len - half_win;

    if (center < min_center) {
        if (clamp_count != NULL) (*clamp_count)++;
        return min_center;
    }

    if (center > max_center) {
        if (clamp_count != NULL) (*clamp_count)++;
        return max_center;
    }

    return center;
}

complex double *load_slc_rows(FILE *fin, int start, int n_rows, int nx) {
    long offset;
    short *tmp;
    complex double *arr;

    offset = nx * start * sizeof(short) * 2;
    fseek(fin, offset, SEEK_SET);

    tmp = malloc(nx * sizeof(short) * 2);
    arr = fftw_alloc_complex((size_t)n_rows * nx);
    if (tmp == NULL || arr == NULL) {
        perror("Failed to allocate memory for SLC rows");
        exit(-1);
    }

    for (int i=0; i<n_rows; i++) {
        if (fread(tmp, 2*sizeof(short), nx, fin) != (unsigned long)nx) {
            perror("Failed to read data from SLC file!");
            exit(-1);
        }

        for (int j=0; j<nx; j++)
            arr[i*nx + j] = tmp[2*j] + tmp[2*j+1] * I;
    }

    free(tmp);
    return arr;
}

complex double *load_slc_window(
        FILE *fin,
        int start_row, int n_rows, int total_nx,
        int start_col, int n_cols) {
    short *tmp;
    complex double *arr;

    tmp = malloc((size_t)n_cols * sizeof(short) * 2);
    arr = fftw_alloc_complex((size_t)n_rows * n_cols);
    if (tmp == NULL || arr == NULL) {
        perror("Failed to allocate memory for SLC window");
        exit(-1);
    }

    for (int i=0; i<n_rows; i++) {
        long long sample_offset = (long long)(start_row + i) * total_nx + start_col;
        off_t byte_offset = (off_t)(sample_offset * (long long)(sizeof(short) * 2));

        if (fseeko(fin, byte_offset, SEEK_SET) != 0) {
            perror("Failed to seek SLC file");
            exit(-1);
        }

        if (fread(tmp, 2*sizeof(short), n_cols, fin) != (size_t)n_cols) {
            perror("Failed to read data from SLC file");
            exit(-1);
        }

        for (int j=0; j<n_cols; j++)
            arr[i*n_cols + j] = tmp[2*j] + tmp[2*j+1] * I;
    }

    free(tmp);
    return arr;
}

long double time_corr(
        const double *c1r,
        const double *c2r,
        int xsearch, int ysearch,
        int xoff, int yoff) {

    int nx_corr, ny_corr;
    int nx_win;

    nx_corr = xsearch * 2;
    nx_win = nx_corr * 2;
    ny_corr = ysearch * 2;

    long double num, denom, denom1, denom2, result;

    num = denom1 = denom2 = 0.0;
    for (int i=0; i<ny_corr; i++)
        for (int j=0; j<nx_corr; j++) {
            long double a = c1r[(ysearch + i + yoff) * nx_win + (xsearch + j + xoff)];
            long double b = c2r[(ysearch + i) * nx_win + (xsearch + j)];

            num += a * b;
            denom1 += a * a;
            denom2 += b * b;
        }

    denom = sqrtl(denom1 * denom2);

    if (denom == 0.0) {
        fprintf(stderr, "calc_corr: denominator = zero: setting corr to 0 \n");
        result = 0.0;
    } else
        result = 100.0 * fabsl(num / denom);

    return result;
}

double *freq_corr(
        double *c1r,
        double *c2r,
        int nx_win, int ny_win,
        pthread_mutex_t *fftw_lock) {
    complex double *c1r_fft, *c2r_fft;
    double *c3r;
    fftw_plan plan1, plan2, plan3;

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    c1r_fft = fftw_alloc_complex((size_t)ny_win * (nx_win/2+1));
    c2r_fft = fftw_alloc_complex((size_t)ny_win * (nx_win/2+1));
    c3r = fftw_alloc_real((size_t)nx_win * ny_win);
    if (c1r_fft == NULL || c2r_fft == NULL || c3r == NULL) {
        perror("Failed to allocate memory for FFT correlation");
        exit(-1);
    }

    plan1 = fftw_plan_dft_r2c_2d(ny_win, nx_win, c1r, c1r_fft, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_r2c_2d(ny_win, nx_win, c2r, c2r_fft, FFTW_ESTIMATE);
    plan3 = fftw_plan_dft_c2r_2d(ny_win, nx_win, c1r_fft, c3r, FFTW_ESTIMATE);
    if (plan1 == NULL || plan2 == NULL || plan3 == NULL) {
        if (fftw_lock) pthread_mutex_unlock(fftw_lock);
        fprintf(stderr, "Failed to create FFTW plans for freq_corr\n");
        exit(EXIT_FAILURE);
    }
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    fftw_execute(plan1);
    fftw_execute(plan2);

    int isign = 1;
    for (int k=0; k<ny_win*(nx_win/2+1); k++, isign=-isign)
        c1r_fft[k] *= isign * conj(c2r_fft[k]);

    fftw_execute(plan3);

    if (fftw_lock) pthread_mutex_lock(fftw_lock);
    fftw_free(c1r_fft);
    fftw_free(c2r_fft);
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan3);
    if (fftw_lock) pthread_mutex_unlock(fftw_lock);

    // FIXME: remove scaling later
    // scale to match GMTSAR for debugging
    for (int i=0; i<nx_win*ny_win; i++)
        c3r[i] = fabs(c3r[i] / (nx_win * ny_win));

    return c3r;
}

void corr_thread(gpointer arg, gpointer user_data) {
    struct st_corr_thread_data *data = arg;
    pthread_mutex_t *lock = user_data;

    int xsearch, ysearch;
    int nx_corr, ny_corr;
    int nx_win, ny_win;
    complex double *c1, *c2;
    double *c1r, *c2r;

    xsearch = data->xc->xsearch;
    nx_corr = xsearch * 2;
    nx_win = nx_corr * 2;
    ysearch = data->xc->ysearch;
    ny_corr = ysearch * 2;
    ny_win = ny_corr * 2;

    c1 = data->c1;
    c2 = data->c2;

    // last part of assign_values
    if (data->xc->ri > 1) {
        complex double *interp1, *interp2;
        int interp_width;

        interp1 = dft_interpolate_2d(c1, ny_win, nx_win, 1, data->xc->ri, lock);
        interp2 = dft_interpolate_2d(c2, ny_win, nx_win, 1, data->xc->ri, lock);
        interp_width = data->xc->ri * nx_win;

        if (lock) pthread_mutex_lock(lock);
        fftw_free(c1);
        fftw_free(c2);
        if (lock) pthread_mutex_unlock(lock);

        c1 = c64_array_slice(interp1, interp_width,
                0, ny_win, interp_width/2 - nx_win/2, nx_win);
        c2 = c64_array_slice(interp2, interp_width,
                0, ny_win, interp_width/2 - nx_win/2, nx_win);

        if (lock) pthread_mutex_lock(lock);
        fftw_free(interp1);
        fftw_free(interp2);
        if (lock) pthread_mutex_unlock(lock);
    }

    c1r = fftw_alloc_real((size_t)nx_win * ny_win);
    c2r = fftw_alloc_real((size_t)nx_win * ny_win);
    if (c1r == NULL || c2r == NULL) {
        perror("Failed to allocate memory for amplitude buffers");
        exit(-1);
    }

    double mean1 = 0.0, mean2 = 0.0;
    for (int k=0; k<nx_win*ny_win; k++) {
        mean1 += (c1r[k] = cabs(c1[k]));
        mean2 += (c2r[k] = cabs(c2[k]));
    }

    if (lock) pthread_mutex_lock(lock);
    fftw_free(c1);
    fftw_free(c2);
    if (lock) pthread_mutex_unlock(lock);

    mean1 /= nx_win * ny_win;
    mean2 /= nx_win * ny_win;
    for (int k=0; k<nx_win*ny_win; k++) {
        c1r[k] -= mean1;
        c2r[k] -= mean2;
    }

    // make_mask and mask
    for (int i=0; i<ny_win; i++)
        for (int j=0; j<nx_win; j++) {
            if (i < ysearch
                    || i >= ny_win - ysearch
                    || j < xsearch
                    || j >= nx_win - xsearch)
                c2r[i*nx_win + j] = 0;
        }

    // calc correlation with 2D FFT
    double *c3r, *corr;
    c3r = freq_corr(c1r, c2r, nx_win, ny_win, lock);
    corr = f64_array_slice(c3r, nx_win, ysearch, ny_corr, xsearch, nx_corr);

    //puts("ARRAY corr:");
    //print_complex_double("%+04.2f%+04.2fj\t", corr, ny_corr, nx_corr);

    int xpeak, ypeak;
    double cmax, cave, max_corr;

    f64_array_stats(corr, ny_corr, nx_corr, &cave, &cmax, &ypeak, &xpeak);
    xpeak -= xsearch;
    ypeak -= ysearch;

    max_corr = time_corr(c1r, c2r, xsearch, ysearch, xpeak, ypeak);

    if (lock) pthread_mutex_lock(lock);
    fftw_free(c1r);
    fftw_free(c2r);
    if (lock) pthread_mutex_unlock(lock);

    //fprintf(stderr, "xypeak: (%d, %d)\n", xpeak, ypeak);
    //fprintf(stderr, "max_corr: %g\n", cmax);

    double xfrac = 0.0, yfrac = 0.0;

    // high-res correlation
    if (data->xc->interp_factor > 1) {
        int factor = data->xc->interp_factor;
        int nx_corr2 = data->xc->n2x;
        int ny_corr2 = data->xc->n2y;
        double *corr2;
        double *hi_corr;

        assert(nx_corr2 >= 2 && TEST_2PWR(nx_corr2));
        assert(ny_corr2 >= 2 && TEST_2PWR(ny_corr2));

        // FIXME: remove this later
        // scale to match GMTSAR for debugging
        for (int k=0; k<nx_corr*ny_corr; k++)
            corr[k] *= max_corr / cmax;

        // FIXME: original GMTSAR are vulnerable to memory violation
        // offset ypeak and xpeak to fix
        if (ypeak + ysearch < ny_corr2/2)
            ypeak = ny_corr2 / 2 - ysearch;
        else if (ypeak + ysearch >= ny_corr - ny_corr2/2)
            ypeak = ny_corr - ny_corr2/2 - ysearch - 1;

        if (xpeak + xsearch < nx_corr2/2)
            xpeak = nx_corr2 / 2 - xsearch;
        else if (xpeak + xsearch >= nx_corr - nx_corr2/2)
            xpeak = nx_corr - nx_corr2/2 - xsearch - 1;

        corr2 = f64_array_slice(
                corr, nx_corr,
                ypeak + ysearch - ny_corr2/2, ny_corr2,
                xpeak + xsearch - nx_corr2/2, nx_corr2);

        for (int i=0; i<nx_corr2*ny_corr2; i++)
            corr2[i] = pow(corr2[i], 0.25);

        hi_corr = rdft_interpolate_2d(corr2, ny_corr2, nx_corr2, factor, factor, lock);

        int ny_hi = ny_corr2 * factor;
        int nx_hi = nx_corr2 * factor;
        int xpeak2, ypeak2;

        f64_array_stats(hi_corr, ny_hi, nx_hi, NULL, NULL, &ypeak2, &xpeak2);
        ypeak2 -= ny_hi / 2;
        xpeak2 -= nx_hi / 2;

        assert(xpeak2 >= -nx_hi/2 && xpeak2 < nx_hi/2);
        assert(ypeak2 >= -ny_hi/2 && ypeak2 < ny_hi/2);

        xfrac = xpeak2 / (double)factor;
        yfrac = ypeak2 / (double)factor;

        if (lock) pthread_mutex_lock(lock);
        fftw_free(corr2);
        fftw_free(hi_corr);
        if (lock) pthread_mutex_unlock(lock);
    }

    data->xoff = data->xc->x_offset - ((xpeak + xfrac) / data->xc->ri);
    data->yoff = data->xc->y_offset - (ypeak + yfrac) + data->loc_y * data->xc->astretcha;
    data->corr = max_corr;

    // printf(" %d %6.3f %d %6.3f %6.2f \n", data->loc_x, xoff, data->loc_y, yoff, cmax);

    if (lock) pthread_mutex_lock(lock);
    fftw_free(c3r);
    fftw_free(corr);
    if (lock) pthread_mutex_unlock(lock);

    g_atomic_int_set(&data->done, 1);
}

void do_correlation(struct st_xcorr *xc, long thread_n) {
    int loc_x, loc_y;
    int slave_loc_x, slave_loc_y;
    int nx_win, ny_win;
    int nx_corr, ny_corr;
    complex double *c1, *c2;
    FILE *fmaster, *fslave;
    FILE *fout;
    struct st_corr_thread_data *row_tasks;
    int *loc_x_list, *loc_y_list;
    double scale = 1.0 + xc->astretcha;
    int clamp_x_count = 0, clamp_y_count = 0;

    if ((fmaster = fopen(xc->m_path, "rb")) == NULL) {
        perror("failed to open master SLC image");
        exit(-1);
    }

    if ((fslave = fopen(xc->s_path, "rb")) == NULL) {
        perror("failed to open slave SLC image");
        exit(-1);
    }

    if ((fout = fopen("freq_xcorr.dat", "w")) == NULL) {
        perror("failed to open output file");
        exit(-1);
    }

    nx_corr = xc->xsearch * 2;
    nx_win = nx_corr * 2;
    ny_corr = xc->ysearch * 2;
    ny_win = ny_corr * 2;

    loc_x_list = malloc((size_t)xc->nxl * sizeof(*loc_x_list));
    loc_y_list = malloc((size_t)xc->nyl * sizeof(*loc_y_list));
    if (loc_x_list == NULL || loc_y_list == NULL) {
        perror("failed to allocate location arrays");
        exit(-1);
    }

    compute_axis_positions(
            loc_x_list, xc->nxl,
            nx_win/2,
            xc->m_nx, xc->s_nx,
            scale, xc->x_offset, "x");
    compute_axis_positions(
            loc_y_list, xc->nyl,
            ny_win/2,
            xc->m_ny, xc->s_ny,
            scale, xc->y_offset, "y");

    row_tasks = calloc((size_t)xc->nxl, sizeof(*row_tasks));
    if (row_tasks == NULL) {
        perror("failed to allocate task buffers");
        exit(-1);
    }

#ifndef NO_PTHREAD
    GThreadPool *thread_pool;
    pthread_mutex_t fftw_lock;
    guint queue_limit;

    thread_pool = g_thread_pool_new(corr_thread, &fftw_lock, thread_n, TRUE, NULL);
    pthread_mutex_init(&fftw_lock, NULL);
    queue_limit = 1U;
#endif

    for (int jy=0; jy<xc->nyl; jy++) {
        int row_n = 0;

        loc_y = loc_y_list[jy];
        slave_loc_y = (1+xc->astretcha)*loc_y + xc->y_offset;
        slave_loc_y = clamp_center(slave_loc_y, ny_win/2, xc->s_ny, &clamp_y_count);

        for (int ix=0; ix<xc->nxl; ix++) {
            loc_x = loc_x_list[ix];
            slave_loc_x = (1+xc->astretcha)*loc_x + xc->x_offset;
            slave_loc_x = clamp_center(slave_loc_x, nx_win/2, xc->s_nx, &clamp_x_count);

            //fprintf(stderr, "LOC#%d (%d, %d) <=> (%d, %d)\n", loc_n, loc_x, loc_y, slave_loc_x, slave_loc_y);

#ifndef NO_PTHREAD
            while (g_thread_pool_unprocessed(thread_pool) >= queue_limit)
                g_usleep(1000);
#endif

            c1 = load_slc_window(
                    fmaster,
                    loc_y - ny_win/2, ny_win, xc->m_nx,
                    loc_x - nx_win/2, nx_win);
            c2 = load_slc_window(
                    fslave,
                    slave_loc_y - ny_win/2, ny_win, xc->s_nx,
                    slave_loc_x - nx_win/2, nx_win);
 
            struct st_corr_thread_data *p = &row_tasks[row_n++];
            *p = (struct st_corr_thread_data) {
                .xc = xc,
                .c1 = c1,
                .c2 = c2,
                .loc_x = loc_x,
                .loc_y = loc_y,
                .done = 0
            };

#ifndef NO_PTHREAD
            g_thread_pool_push(thread_pool, p, NULL);
#else
            corr_thread(p, NULL);
#endif
        }

#ifndef NO_PTHREAD
        for (int i=0; i<row_n; i++)
            while (!g_atomic_int_get(&row_tasks[i].done))
                g_usleep(1000);
#endif

        for (int i=0; i<row_n; i++) {
            struct st_corr_thread_data *p = row_tasks + i;
            fprintf(fout, " %d %6.3f %d %6.3f %6.2f \n",
                    p->loc_x, p->xoff, p->loc_y, p->yoff, p->corr);
        }
    }

#ifndef NO_PTHREAD
    g_thread_pool_free(thread_pool, FALSE, TRUE);
    pthread_mutex_destroy(&fftw_lock);
#endif
    if (clamp_x_count > 0 || clamp_y_count > 0) {
        fprintf(stderr,
                "Overlap mode: clamped slave window centers to preserve requested --nx/--ny "
                "(x clamps=%d, y clamps=%d).\n",
                clamp_x_count, clamp_y_count);
    }
    fftw_cleanup();

    free(row_tasks);
    free(loc_x_list);
    free(loc_y_list);
    fclose(fmaster);
    fclose(fslave);
    fclose(fout);
}

int main(int argc, char **argv) {
    struct st_xcorr_args args;
    struct st_xcorr xcorr;
    long thread_n = 0;

#ifndef _OPENMP
    printf("=== 警告: OpenMP 未启用，程序将串行运行 ===\n");
    printf("编译时请添加 -fopenmp 选项启用并行计算\n");
#endif

    parse_opts(&args, argc, argv);
    apply_args(&args, &xcorr);

#ifndef NO_PTHREAD
    long cpu_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_threads < 1) cpu_threads = 1;
    thread_n = cpu_threads * 3 / 4;
    if(thread_n < 1) thread_n = 1;
    thread_n = auto_tune_threads(&xcorr, thread_n);

    fprintf(stderr, "use %ld thread(s)\n", thread_n);
#endif
    do_correlation(&xcorr, thread_n);

    free(xcorr.m_path);
    free(xcorr.s_path);

    return 0;
}
