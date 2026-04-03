/************************************************************************
 * soi.h is the include file for the esarp SAR processor.		*
 ************************************************************************/
/************************************************************************
 * Creator: Evelyn J. Price	(Scripps Institution of Oceanography)	*
 * Date   : 11/18/96							*
 ************************************************************************/
/************************************************************************
 * Modification History							*
 *									*
 * Date									*
 *									*
 *  4/23/97- 	added parameters for orbit calculations: x_target,      *
 *		y_target,z_target,baseline,alpha,sc_identity,		*
 *		ref_identity,SC_clock_start,SC_clock_stop,              *
 *		clock_start,clock_stop   				*
 *		-DTS							*
 *									*
 * 4/23/97-	added parameters: rec_start, rec_stop			*
 *		-EJP							*
 *									*
 * 8/28/97-	added parameters baseline_start baseline_end		*
 *		alpha_start alpha_end					*
 *									*
 * 9/12/97	added clipi2 function to clip to short int		*
 *									*
 * 4/26/06	added nrows, num_lines					*
 ************************************************************************/
#ifndef SOI_H
#define SOI_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define SOL 299792456.0
#define PI 3.1415926535897932
#define PI2 6.2831853071795864
#define I2MAX1 32767.
#define I2SCALE 4.e6
#define TRUE 1
#define FALSE 0
#define RW 0666
#define MULT_FACT 1000.0
#define sgn(A) ((A) >= 0.0 ? 1.0 : -1.0)
#define clipi22(A) (((A) > I2MAX1) ? I2MAX1 : (((A) < -I2MAX1) ? -I2MAX1 : A))
#include "sfd_complex.h"

#ifdef SOI_DEFINE_GLOBALS
#define SOI_EXTERN
#else
#define SOI_EXTERN extern
#endif
SOI_EXTERN char *input_file;
SOI_EXTERN char *led_file;
SOI_EXTERN char *out_amp_file;
SOI_EXTERN char *out_data_file;
SOI_EXTERN char *deskew;
SOI_EXTERN char *iqflip;
SOI_EXTERN char *off_vid;
SOI_EXTERN char *srm;
SOI_EXTERN char *ref_file;
SOI_EXTERN char *orbdir;
SOI_EXTERN char *lookdir;

SOI_EXTERN int debug_flag;
SOI_EXTERN int bytes_per_line;
SOI_EXTERN int good_bytes;
SOI_EXTERN int first_line;
SOI_EXTERN int num_patches;
SOI_EXTERN int first_sample;
SOI_EXTERN int num_valid_az;
SOI_EXTERN int st_rng_bin;
SOI_EXTERN int num_rng_bins;
SOI_EXTERN int nextend;
SOI_EXTERN int nlooks;
SOI_EXTERN int xshift;
SOI_EXTERN int yshift;
SOI_EXTERN int fdc_ystrt;
SOI_EXTERN int fdc_strt;

/*New parameters 4/23/97 -EJP */
SOI_EXTERN int rec_start;
SOI_EXTERN int rec_stop;
/* End new parameters 4/23/97 -EJP */

/* New parameters 4/23/97 -DTS */
SOI_EXTERN int SC_identity;       /* (1)-ERS1 (2)-ERS2 (3)-Radarsat (4)-Envisat (5)-ALOS
                                     (6)-Envisat_SLC  (7)-TSX (8)-CSK (9)-RS2 (10)-S1A*/
SOI_EXTERN int ref_identity;      /* (1)-ERS1 (2)-ERS2 (3)-Radarsat (4)-Envisat (5)-ALOS
                                     (6)-Envisat_SLC  (7)-TSX (8)-CSK (9)-RS2 (10)-S1A*/
SOI_EXTERN double SC_clock_start; /* YYDDD.DDDD */
SOI_EXTERN double SC_clock_stop;  /* YYDDD.DDDD */
SOI_EXTERN double icu_start;      /* onboard clock counter */
SOI_EXTERN double clock_start;    /* DDD.DDDDDDDD  clock without year has more precision */
SOI_EXTERN double clock_stop;     /* DDD.DDDDDDDD  clock without year has more precision */
/* End new parameters 4/23/97 -DTS */

SOI_EXTERN double caltone;
SOI_EXTERN double RE;   /* Local Earth radius */
SOI_EXTERN double raa;  /* ellipsoid semi-major axis - added by RJM */
SOI_EXTERN double rcc;  /* ellipsoid semi-minor axis - added by RJM */
SOI_EXTERN double vel1; /* Equivalent SC velocity */
SOI_EXTERN double ht1;  /* (SC_radius - RE) center of frame*/
SOI_EXTERN double ht0;  /* (SC_radius - RE) start of frame */
SOI_EXTERN double htf;  /* (SC_radius - RE) end of frame */
SOI_EXTERN double near_range;
SOI_EXTERN double far_range;
SOI_EXTERN double prf1;
SOI_EXTERN double xmi1;
SOI_EXTERN double xmq1;
SOI_EXTERN double az_res;
SOI_EXTERN double fs;
SOI_EXTERN double slope;
SOI_EXTERN double pulsedur;
SOI_EXTERN double lambda;
SOI_EXTERN double rhww;
SOI_EXTERN double pctbw;
SOI_EXTERN double pctbwaz;
SOI_EXTERN double fd1;
SOI_EXTERN double fdd1;
SOI_EXTERN double fddd1;
SOI_EXTERN double sub_int_r;
SOI_EXTERN double sub_int_a;
SOI_EXTERN double stretch_r;
SOI_EXTERN double stretch_a;
SOI_EXTERN double a_stretch_r;
SOI_EXTERN double a_stretch_a;

/* New parameters 8/28/97 -DTS */
SOI_EXTERN double baseline_start;
SOI_EXTERN double baseline_center;
SOI_EXTERN double baseline_end;
SOI_EXTERN double alpha_start;
SOI_EXTERN double alpha_center;
SOI_EXTERN double alpha_end;
/* New parameters 9/25/18 -EXU */
SOI_EXTERN double B_offset_start;
SOI_EXTERN double B_offset_center;
SOI_EXTERN double B_offset_end;
/* End new parameters 8/28/97 -DTS */
SOI_EXTERN double bparaa; /* parallel baseline - added by RJM */
SOI_EXTERN double bperpp; /* perpendicular baseline - added by RJM */

/* New parameters 4/26/06 */
SOI_EXTERN int nrows;
SOI_EXTERN int num_lines;

/* New parameters 09/18/08 */
SOI_EXTERN double TEC_start;
SOI_EXTERN double TEC_end;
#undef SOI_EXTERN
#endif /* SOI_H	*/
