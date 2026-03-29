/*
 OpenMP + FFTW-safe modifications for xcorr:

 - Call fftw_init_threads() in main once.
 - In the OpenMP parallel region, each thread calls fftw_plan_with_nthreads(1)
   before any plan creation to avoid internal multi-threading inside FFTW
   per OpenMP thread.
 - Surround calls that likely create FFTW plans or perform complex FFT work
   (do_freq_corr, do_highres_corr) with an omp critical section to avoid
   concurrent plan creation / memory races in FFTW.
 - Allocate per-thread data buffers and only replace the buffers that must
   be private (c1,c2,c3,i1,i2,ritmp). Keep other pointers (corr, md, cd_exp, file, d1, d2, etc.)
   pointing to the shared xc to avoid accidental frees of shared memory.
 - Free only the per-thread buffers created here.

 Notes:
 - This approach trades some concurrency (the FFT-plan-related calls are
   serialized) for stability. After verifying correctness, further tuning
   (e.g., creating per-thread cached FFTW plans at program startup) can
   improve performance.
*/

#include "gmtsar.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* USAGE omitted here for brevity; keep your original USAGE string */
/* original USAGE string omitted for brevity in this header-only snippet */
char *USAGE = "xcorr [GMTSAR] - Compute 2-D cross-correlation of two images\n\n"
              "\nUsage: xcorr master.PRM aligned.PRM [-time] [-real] [-freq] [-nx n] [-ny "
              "n] [-xsearch xs] [-ysearch ys]\n"
              "master.PRM     	PRM file for reference image\n"
              "aligned.PRM     	 	PRM file of secondary image\n"
              "-time      		use time cross-correlation\n"
              "-freq      		use frequency cross-correlation (default)\n"
              "-real      		read float numbers instead of complex numbers\n"
              "-noshift  		ignore ashift and rshift in prm file (set to 0)\n"
              "-nx  nx    		number of locations in x (range) direction "
              "(int)\n"
              "-ny  ny    		number of locations in y (azimuth) direction "
              "(int)\n"
              "-nointerp     		do not interpolate correlation function\n"
              "-range_interp ri  	interpolate range by ri (power of two) [default: 2]\n"
              "-norange     		do not range interpolate \n"
              "-xsearch xs		search window size in x (range) direction (int "
              "power of 2 [32 64 128 256])\n"
              "-ysearch ys		search window size in y (azimuth) direction "
              "(int power of 2 [32 64 128 256])\n"
              "-interp  factor    	interpolate correlation function by factor "
              "(int) [default, 16]\n"
              "-v			verbose\n"
              "output: \n freq_xcorr.dat (default) \n time_xcorr.dat (if -time option))\n"
              "\nuse fitoffset.csh to convert output to PRM format\n"
              "\nExample:\n"
              "xcorr IMG-HH-ALPSRP075880660-H1.0__A.PRM "
              "IMG-HH-ALPSRP129560660-H1.0__A.PRM -nx 20 -ny 50 \n"
              "xcorr file1.grd file2.grd -nx 20 -ny 50 (takes grids with real numbers)\n";



/* ... keep do_range_interpolate and assign_values as in your code ... */
/* I'll include them unchanged (copy from your original) */

int do_range_interpolate(void *API, struct FCOMPLEX *c, int nx, int ri, struct FCOMPLEX *work) {
	int i;

	/* interpolate c and put into work */
	fft_interpolate_1d(API, c, nx, work, ri);

	/* replace original with interpolated (only half) */
	for (i = 0; i < nx; i++) {
		c[i].r = work[i + nx / 2].r;
		c[i].i = work[i + nx / 2].i;
	}

	return (EXIT_SUCCESS);
}

void assign_values(void *API, struct xcorr *xc, int iloc) {
	int i, j, k, sx, mx;
	double mean1, mean2;

	/* master and aligned x offsets */
	mx = xc->loc[iloc].x - xc->npx / 2;
	sx = xc->loc[iloc].x + xc->x_offset - xc->npx / 2;

	for (i = 0; i < xc->npy; i++) {
		for (j = 0; j < xc->npx; j++) {
			k = i * xc->npx + j;

			xc->c3[k].i = xc->c3[k].r = 0.0f;

			xc->c1[k].r = xc->d1[i * xc->m_nx + mx + j].r;
			xc->c1[k].i = xc->d1[i * xc->m_nx + mx + j].i;

			xc->c2[k].r = xc->d2[i * xc->s_nx + sx + j].r;
			xc->c2[k].i = xc->d2[i * xc->s_nx + sx + j].i;
		}
	}

	/* range interpolate */
	if (xc->ri > 1) {
		for (i = 0; i < xc->npy; i++) {
			do_range_interpolate(API, &xc->c1[i * xc->npx], xc->npx, xc->ri, xc->ritmp);
			do_range_interpolate(API, &xc->c2[i * xc->npx], xc->npx, xc->ri, xc->ritmp);
		}
	}

	/* convert to amplitude and demean */
	mean1 = mean2 = 0.0;
	for (i = 0; i < xc->npy * xc->npx; i++) {
		xc->c1[i].r = Cabs(xc->c1[i]);
		xc->c1[i].i = 0.0f;

		xc->c2[i].r = Cabs(xc->c2[i]);
		xc->c2[i].i = 0.0f;

		mean1 += xc->c1[i].r;
		mean2 += xc->c2[i].r;
	}

	mean1 /= (double)(xc->npy * xc->npx);
	mean2 /= (double)(xc->npy * xc->npx);

	for (i = 0; i < xc->npy * xc->npx; i++) {
		xc->c1[i].r = xc->c1[i].r - (float)mean1;
		xc->c2[i].r = xc->c2[i].r - (float)mean2;
	}

	/* apply mask */
	for (i = 0; i < xc->npy * xc->npx; i++) {
		xc->c1[i].i = xc->c2[i].i = 0.0f;
		xc->c2[i].r = xc->c2[i].r * (float)xc->mask[i];

		xc->i1[i] = (int)(xc->c1[i].r);
		xc->i2[i] = (int)(xc->c2[i].r);
	}

	if (debug)
		fprintf(stderr, " mean %lf\n", mean1);
	if (debug)
		fprintf(stderr, " mean %lf\n", mean2);
}

/* allocate_arrays, make_mask kept as in original */
void make_mask(struct xcorr *xc) {
	int i, j, imask;
	imask = 0;

	for (i = 0; i < xc->npy; i++) {
		for (j = 0; j < xc->npx; j++) {
			xc->mask[i * xc->npx + j] = 1;
			if ((i < xc->ysearch) || (i >= (xc->npy - xc->ysearch))) {
				xc->mask[i * xc->npx + j] = imask;
			}
			if ((j < xc->xsearch) || (j >= (xc->npx - xc->xsearch))) {
				xc->mask[i * xc->npx + j] = imask;
			}
		}
	}
}

void allocate_arrays(struct xcorr *xc) {
	int nx, ny, nx_exp, ny_exp;

	xc->d1 = (struct FCOMPLEX *)malloc(xc->m_nx * xc->npy * sizeof(struct FCOMPLEX));
	xc->d2 = (struct FCOMPLEX *)malloc(xc->s_nx * xc->npy * sizeof(struct FCOMPLEX));

	xc->i1 = (int *)malloc(xc->npx * xc->npy * sizeof(int));
	xc->i2 = (int *)malloc(xc->npx * xc->npy * sizeof(int));

	xc->c1 = (struct FCOMPLEX *)malloc(xc->npx * xc->npy * sizeof(struct FCOMPLEX));
	xc->c2 = (struct FCOMPLEX *)malloc(xc->npx * xc->npy * sizeof(struct FCOMPLEX));
	xc->c3 = (struct FCOMPLEX *)malloc(xc->npx * xc->npy * sizeof(struct FCOMPLEX));

	xc->ritmp = (struct FCOMPLEX *)malloc(xc->ri * xc->npx * sizeof(struct FCOMPLEX));
	xc->mask = (short *)malloc(xc->npx * xc->npy * sizeof(short));

	/* this is size of correlation patch */
	xc->corr = (double *)malloc(2 * xc->ri * (xc->nxc) * (xc->nyc) * sizeof(double));

	if (xc->interp_flag == 1) {
		nx = 2 * xc->n2x;
		ny = 2 * xc->n2y;
		nx_exp = nx * (xc->interp_factor);
		ny_exp = ny * (xc->interp_factor);
		xc->md = (struct FCOMPLEX *)malloc(nx * ny * sizeof(struct FCOMPLEX));
		xc->cd_exp = (struct FCOMPLEX *)malloc(nx_exp * ny_exp * sizeof(struct FCOMPLEX));
	}
}

/* the modified do_correlation with FFTW-thread-safety measures */
void do_correlation(void *API, struct xcorr *xc) {
	int i, j, iloc, istep;

	/* opportunity for multiple processors */
	istep = 1;

	/* allocate arrays   			*/
	allocate_arrays(xc);

	/* make mask 				*/
	make_mask(xc);

	/* prepare per-thread buffers */
	int npx = xc->npx;
	int npy = xc->npy;
	int nloc_patch = npx * npy;
	int ri_sz = xc->ri * npx;

	int nthreads = 1;
#ifdef _OPENMP
	nthreads = omp_get_max_threads();
#endif

	struct FCOMPLEX *c1_thr = NULL;
	struct FCOMPLEX *c2_thr = NULL;
	struct FCOMPLEX *c3_thr = NULL;
	struct FCOMPLEX *ritmp_thr = NULL;
	int *i1_thr = NULL;
	int *i2_thr = NULL;

	if (nthreads > 1) {
		c1_thr = (struct FCOMPLEX *)malloc((size_t)nthreads * nloc_patch * sizeof(struct FCOMPLEX));
		c2_thr = (struct FCOMPLEX *)malloc((size_t)nthreads * nloc_patch * sizeof(struct FCOMPLEX));
		c3_thr = (struct FCOMPLEX *)malloc((size_t)nthreads * nloc_patch * sizeof(struct FCOMPLEX));

		i1_thr = (int *)malloc((size_t)nthreads * nloc_patch * sizeof(int));
		i2_thr = (int *)malloc((size_t)nthreads * nloc_patch * sizeof(int));

		ritmp_thr = (struct FCOMPLEX *)malloc((size_t)nthreads * ri_sz * sizeof(struct FCOMPLEX));

		if (!(c1_thr && c2_thr && c3_thr && i1_thr && i2_thr && ritmp_thr)) {
			fprintf(stderr, "Warning: failed to allocate per-thread buffers, continuing single-threaded.\n");
			if (c1_thr) free(c1_thr);
			if (c2_thr) free(c2_thr);
			if (c3_thr) free(c3_thr);
			if (i1_thr) free(i1_thr);
			if (i2_thr) free(i2_thr);
			if (ritmp_thr) free(ritmp_thr);
			nthreads = 1;
		}
	}

	iloc = 0;
	for (i = 0; i < xc->nyl; i += istep) {

		/* read in data for each row (serialized) */
		read_xcorr_data(xc, iloc);

#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(static) shared(xc, c1_thr, c2_thr, c3_thr, i1_thr, i2_thr, ritmp_thr) if(nthreads>1)
#endif
		for (j = 0; j < xc->nxl; j++) {
			int tid = 0;
#ifdef _OPENMP
			tid = omp_get_thread_num();
			/* ensure FFTW in this OpenMP thread uses only 1 internal thread */
			fftwf_plan_with_nthreads(1);
#endif
			int local_iloc = iloc + j;

			/* local shallow copy of xc */
			struct xcorr local_xc = *xc;

			if (nthreads > 1) {
				local_xc.c1 = &c1_thr[(size_t)tid * nloc_patch];
				local_xc.c2 = &c2_thr[(size_t)tid * nloc_patch];
				local_xc.c3 = &c3_thr[(size_t)tid * nloc_patch];

				local_xc.i1 = &i1_thr[(size_t)tid * nloc_patch];
				local_xc.i2 = &i2_thr[(size_t)tid * nloc_patch];

				local_xc.ritmp = &ritmp_thr[(size_t)tid * ri_sz];
			} else {
				/* single-threaded: use the existing buffers in xc */
				local_xc.c1 = xc->c1;
				local_xc.c2 = xc->c2;
				local_xc.c3 = xc->c3;

				local_xc.i1 = xc->i1;
				local_xc.i2 = xc->i2;

				local_xc.ritmp = xc->ritmp;
			}

			/* d1 and d2 contain the row data read earlier and are read-only here */
			local_xc.d1 = xc->d1;
			local_xc.d2 = xc->d2;

			/* mask is shared and read-only */
			local_xc.mask = xc->mask;

			if (debug)
				fprintf(stderr, " initial: iloc %d (%d,%d) (thread %d)\n", local_iloc, xc->loc[local_iloc].x, xc->loc[local_iloc].y, tid);

			/* copy / prepare per-thread buffers */
			assign_values(API, &local_xc, local_iloc);

			if (debug)
				print_complex(local_xc.c1, local_xc.npy, local_xc.npx, 1);
			if (debug)
				print_complex(local_xc.c2, local_xc.npy, local_xc.npx, 1);

			/* time domain correlation can be called without serializing FFT plan creation */
			if (local_xc.corr_flag < 2)
				do_time_corr(&local_xc, local_iloc);

			/* Frequency-domain correlation and high-res interpolation may create FFTW plans.
			   Serialize these calls to avoid concurrent plan creation issues in FFTW. */
#ifdef _OPENMP
#pragma omp critical(fftwf_plan_create)
#endif
			{
				if (local_xc.corr_flag == 2)
					do_freq_corr(API, &local_xc, local_iloc);

				if (local_xc.interp_flag == 1)
					do_highres_corr(API, &local_xc, local_iloc);
			}

			/* write out results - protect file writes */
#ifdef _OPENMP
#pragma omp critical(write_results)
#endif
			{
				print_results(&local_xc, local_iloc);
			}
		} /* end of x iloc loop */

		iloc += xc->nxl; /* advance to start of next row's locations */
	}     /* end of y iloc loop */

	/* free per-thread buffers */
	if (c1_thr) free(c1_thr);
	if (c2_thr) free(c2_thr);
	if (c3_thr) free(c3_thr);
	if (i1_thr) free(i1_thr);
	if (i2_thr) free(i2_thr);
	if (ritmp_thr) free(ritmp_thr);
}

/* main: add fftw_init_threads() after creating GMT session */
int main(int argc, char **argv) {
	int input_flag, nfiles;
	struct xcorr *xc;
	clock_t start, end;
	double cpu_time;
	void *API = NULL; /* GMT API control structure */

	xc = (struct xcorr *)malloc(sizeof(struct xcorr));

	verbose = 0;
	debug = 0;
	input_flag = 0;
	nfiles = 2;
	xc->interp_flag = 0;
	xc->corr_flag = 2;

	/* Begin: Initializing new GMT session */
	if ((API = GMT_Create_Session(argv[0], 0U, 0U, NULL)) == NULL)
		return EXIT_FAILURE;

	/* initialize FFTW threads support (safe to call even if not using FFTW threads) */
#ifdef _OPENMP
	fftwf_init_threads();
#endif

	if (argc < 3)
		die(USAGE, "");

	set_defaults(xc);

	parse_command_line(argc, argv, xc, &nfiles, &input_flag, USAGE);

	/* read prm files */
	if (input_flag == 0)
		handle_prm(API, argv, xc, nfiles);

	if (debug)
		print_params(xc);

	/* output file */
	if (xc->corr_flag == 0)
		strcpy(xc->filename, "time_xcorr.dat");
	if (xc->corr_flag == 1)
		strcpy(xc->filename, "time_xcorr_Gatelli.dat");
	if (xc->corr_flag == 2)
		strcpy(xc->filename, "freq_xcorr.dat");

	xc->file = fopen(xc->filename, "w");
	if (xc->file == NULL)
		die("Can't open output file", xc->filename);

	/* x locations, y locations */
	get_locations(xc);

	/* calculate correlation at all points */
	start = clock();

	do_correlation(API, xc);

	end = clock();
	cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
	fprintf(stdout, " elapsed time: %lf \n", cpu_time);

        if (xc->format == 0 || xc->format == 1) {
          fclose(xc->data1);
          fclose(xc->data2);
        }

	if (GMT_Destroy_Session(API))
		return EXIT_FAILURE; /* Remove the GMT machinery */

	return (EXIT_SUCCESS);
}