/*	$Id: read_xcorr_data.c 39 2013-04-07 00:49:34Z pwessel $	*/
#include "gmtsar.h"
#include "xcorr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*-------------------------------------------------------*/
void read_complex_short2(FILE *f, struct FCOMPLEX *d, int iy, int npy, int nx, short *tmp) {
	long num_to_seek;
	int i, j;

	num_to_seek = 2 * iy * nx * sizeof(short);
	fseek(f, num_to_seek, SEEK_SET); /* from beginning */

	/* need to read two parts of complex numbers */
	for (i = 0; i < npy; i++) {
		fread(&tmp[0], 2 * sizeof(short), nx, f); /* read whole line */

		/* read into complex float */
		for (j = 0; j < nx; j++) {
			d[i * nx + j].r = (float)tmp[2 * j];
			d[i * nx + j].i = (float)tmp[2 * j + 1];
			// d[i*nx+j].r =
			// sqrt(((float)tmp[2*j])*((float)tmp[2*j])+((float)tmp[2*j+1])*((float)tmp[2*j+1]));
			// d[i*nx+j].i = 0.0;
		}
	}
}
/*-------------------------------------------------------*/
void read_real_float2(FILE *f, struct FCOMPLEX *d, int iy, int npy, int nx, float *tmp) {
	long num_to_seek;
	int i, j;

	num_to_seek = iy * nx * sizeof(float);
	fseek(f, num_to_seek, SEEK_SET); /* from beginning */

	/* need to read two parts of complex numbers */
	for (i = 0; i < npy; i++) {
		fread(&tmp[0], sizeof(float), nx, f); /* read whole line */

		/* read into complex float */
		for (j = 0; j < nx; j++) {
			d[i * nx + j].r = (float)tmp[j];
			d[i * nx + j].i = 0.0;
		}
	}
}

/*-------------------------------------------------------*/
void read_real_float_grid(struct GMT_GRID *f, struct FCOMPLEX *d, int iy, int npy, int nx, int ny) {
	// long    num;
	int i, j;

	// num_to_seek = iy*nx;
	// fseek(f, num_to_seek, SEEK_SET);        /* from beginning */

	/* need to read two parts of complex numbers */
	for (i = 0; i < npy; i++) {

		// fread(&tmp[0],sizeof(float), nx, f);  /* read whole line */

		/* read into complex float */
		for (j = 0; j < nx; j++) {
			// num = i+npy+j*ny;
			d[i * nx + j].r = f->data[(i + iy) * nx + j];
			d[i * nx + j].i = 0.0;
		}
	}
}

/*-------------------------------------------------------*/
static int ensure_xcorr_read_buffers(struct xcorr *xc) {
	size_t need_m_short, need_s_short, need_m_float, need_s_float;
	short *new_short;
	float *new_float;

	if (!xc)
		return 0;

	need_m_short = (size_t)2 * (size_t)xc->m_nx;
	need_s_short = (size_t)2 * (size_t)xc->s_nx;
	need_m_float = (size_t)2 * (size_t)xc->m_nx;
	need_s_float = (size_t)2 * (size_t)xc->s_nx;

	if ((xc->tmp_m == NULL) || (xc->tmp_m_cap < need_m_short)) {
		new_short = (short *)realloc(xc->tmp_m, need_m_short * sizeof(short));
		if (!new_short)
			return 0;
		xc->tmp_m = new_short;
		xc->tmp_m_cap = need_m_short;
	}
	if ((xc->tmp_s == NULL) || (xc->tmp_s_cap < need_s_short)) {
		new_short = (short *)realloc(xc->tmp_s, need_s_short * sizeof(short));
		if (!new_short)
			return 0;
		xc->tmp_s = new_short;
		xc->tmp_s_cap = need_s_short;
	}
	if ((xc->tmp2_m == NULL) || (xc->tmp2_m_cap < need_m_float)) {
		new_float = (float *)realloc(xc->tmp2_m, need_m_float * sizeof(float));
		if (!new_float)
			return 0;
		xc->tmp2_m = new_float;
		xc->tmp2_m_cap = need_m_float;
	}
	if ((xc->tmp2_s == NULL) || (xc->tmp2_s_cap < need_s_float)) {
		new_float = (float *)realloc(xc->tmp2_s, need_s_float * sizeof(float));
		if (!new_float)
			return 0;
		xc->tmp2_s = new_float;
		xc->tmp2_s_cap = need_s_float;
	}

	return 1;
}

/*-------------------------------------------------------*/
void free_xcorr_read_buffers(struct xcorr *xc) {
	if (!xc)
		return;
	free(xc->tmp_m);
	free(xc->tmp_s);
	free(xc->tmp2_m);
	free(xc->tmp2_s);
	xc->tmp_m = NULL;
	xc->tmp_s = NULL;
	xc->tmp2_m = NULL;
	xc->tmp2_s = NULL;
	xc->tmp_m_cap = 0;
	xc->tmp_s_cap = 0;
	xc->tmp2_m_cap = 0;
	xc->tmp2_s_cap = 0;
}

/*-------------------------------------------------------*/
void read_xcorr_data(struct xcorr *xc, int iloc) {
	int iy, ishft;

	if (!ensure_xcorr_read_buffers(xc))
		die("read_xcorr_data: failed to allocate reusable buffers", "");

	/* set locations and read data for master       */
	/* read whole line at correct y offset          */
	iy = xc->loc[iloc].y - xc->npy / 2;

	if (debug)
		fprintf(stderr, " reading data from master at y = %d and %d items\n", iy, xc->m_nx);

	if (xc->format == 0)
		read_complex_short2(xc->data1, xc->d1, iy, xc->npy, xc->m_nx, xc->tmp_m);
	if (xc->format == 1)
		read_real_float2(xc->data1, xc->d1, iy, xc->npy, xc->m_nx, xc->tmp2_m);
	if (xc->format == 2)
		read_real_float_grid(xc->D1, xc->d1, iy, xc->npy, xc->m_nx, xc->m_ny);

	/* set locations and read data for aligned */
	ishft = (int)xc->loc[iloc].y * xc->astretcha;
	iy = xc->loc[iloc].y + xc->y_offset + ishft - xc->npy / 2;

	if (debug)
		fprintf(stderr, " reading data from aligned at y = %d and %d items\n", iy, xc->s_nx);
	if (xc->format == 0)
		read_complex_short2(xc->data2, xc->d2, iy, xc->npy, xc->s_nx, xc->tmp_s);
	if (xc->format == 1)
		read_real_float2(xc->data2, xc->d2, iy, xc->npy, xc->s_nx, xc->tmp2_s);
	if (xc->format == 2)
		read_real_float_grid(xc->D2, xc->d2, iy, xc->npy, xc->s_nx, xc->s_ny);
}
