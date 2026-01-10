/*	$Id: offset_topo_omp.c 79 2013-06-10 23:43:27Z pwessel $	*/
/***************************************************************************/
/* offset_topo reads  an amplitude image and a topo_ra grid as well as     */
/* an initial guess on how to shift the topo_ra to match the master        */
/* The program uses cross correlation to estimate the refined shift to     */
/* make the topo_ra match the amplitude image more exactly.                */
/* There is an option to output a new shifted topo_ra.                     */
/***************************************************************************/

/***************************************************************************
 * Creator:  David T. Sandwell and Xiaopeng Tong                           *
 *           (Scripps Institution of Oceanography)                         *
 * Date   :  7/22/08                                                       *
 ***************************************************************************/

/***************************************************************************
 * Modification history:                                                   *
 *                                                                         *
 * DATE                                                                    *
 ***************************************************************************/

#include "gmtsar.h"

int main(int argc, char **argv) {
	int i, j, k, i1, j1, k1;
	int is, js, ns;
	int ni, nj, ntot;
	int xshft, yshft, ib = 200; /* ib is the width of the edge of the images not
	                               used for corr. must be > 2 */
	int imax = 0, jmax = 0;
	double ra, rt, avea, suma, sumt, sumc, corr, denom, maxcorr = -9999.;
	void *API = NULL;                                 /* GMT control structure */
	struct GMT_GRID *A = NULL, *T = NULL, *TS = NULL; /* Grid structure containing ->header and ->data */

	/* get the information from the command line */
	if (argc < 6) {
		printf("\offset_topo_omp [GMTSAR] - Determine topography offset\n \n");
		printf("\nUsage: offset_topo_omp amp_master.grd topo_ra.grd rshift ashift ns "
		       "[topo_shift.grd] \n \n");
		printf("   amp_master.grd - amplitude image of master \n");
		printf("   topo_ra.grd    - topo in range/azimuth coordinates of master \n");
		printf("   rshift         - guess at integer range shift \n");
		printf("   ashift         - guess at integer azimuth shift \n");
		printf("   ns             - integer search radius \n");
		printf("   topo_shift.grd - shifted topo_ra - optional, will be shifted by "
		       "rshift, ashift \n \n");
		exit(-1);
	}

	/* Begin: Initializing new GMT session */
	if ((API = GMT_Create_Session(argv[0], 0U, 0U, NULL)) == NULL)
		return EXIT_FAILURE;

	xshft = atoi(argv[3]);
	yshft = atoi(argv[4]);
	ns = atoi(argv[5]);

	/* Get header from amplitude and topo grids */
	if ((A = GMT_Read_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE, GMT_GRID_HEADER_ONLY, NULL, argv[1], NULL)) == NULL)
		return EXIT_FAILURE;
	if ((T = GMT_Read_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE, GMT_GRID_HEADER_ONLY, NULL, argv[2], NULL)) == NULL)
		return EXIT_FAILURE;

	/* make sure the dimensions match */
	if (A->header->n_columns != T->header->n_columns) {
		fprintf(stderr, "file dimensions do not match (must have same width)\n");
		exit(EXIT_FAILURE);
	}

	if (argc >= 7) {
		if ((TS = GMT_Create_Data(API, GMT_IS_GRID, GMT_IS_SURFACE, GMT_GRID_ALL, NULL, A->header->wesn, A->header->inc,
		                          A->header->registration, GMT_NOTSET, NULL)) == NULL)
			return EXIT_FAILURE;
	}

	/* Read the two grids into A->data and T->data which automatically are
	 * allocated */
	if (GMT_Read_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE, GMT_GRID_DATA_ONLY, NULL, argv[1], A) == NULL)
		return EXIT_FAILURE;
	if (GMT_Read_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE, GMT_GRID_DATA_ONLY, NULL, argv[2], T) == NULL)
		return EXIT_FAILURE;
	if (A->header->n_rows < T->header->n_rows)
		ni = A->header->n_rows;
	else
		ni = T->header->n_rows;
	fprintf(stderr, " %d %d %d \n", ni, A->header->n_rows, T->header->n_rows);
	nj = T->header->n_columns;

	/* compute average */
	ntot = 0;
	suma = sumt = 0.0;
	for (i = 0; i < ni; i++) {
		for (j = 0; j < nj; j++) {
			k = i * nj + j;
			ntot++;
			suma = suma + A->data[k];
			sumt = sumt + T->data[k];
		}
	}
	avea = suma / ntot;


    float *RA = malloc(sizeof(float) * ni * nj);
    #pragma omp parallel for
    for (i = 0; i < ni * nj; i++)
        RA[i] = A->data[i] - avea;


    float *RT = calloc(ni * nj, sizeof(float));
    #pragma omp parallel for private(i,j,k)
    for (i = 0; i < ni; i++) {
        for (j = 1; j < nj - 1; j++) {
            k = i * nj + j;
            RT[k] = T->data[k + 1] - T->data[k - 1];
        }
    }

    double global_maxcorr = -1e30;
    int global_imax = 0, global_jmax = 0;

    #pragma omp parallel
    {
        double local_maxcorr = -1e30;
        int local_imax = 0, local_jmax = 0;

    #pragma omp for collapse(2) schedule(static)
        for (is = -ns + yshft; is <= ns + yshft; is++) {
            for (js = -ns + xshft; js <= ns + xshft; js++) {

                double sumc = 0.0, suma2 = 0.0, sumt2 = 0.0;

                for (i = ib; i < ni - ib; i++) {
                    i1 = i - is;
                    if (i1 < 0 || i1 >= ni) continue;

                    for (j = ib; j < nj - ib; j++) {
                        j1 = j - js;
                        if (j1 < 1 || j1 >= nj - 1) continue;

                        k  = i  * nj + j;
                        k1 = i1 * nj + j1;

                        double ra = RA[k];
                        double rt = RT[k1];

                        sumc  += ra * rt;
                        suma2 += ra * ra;
                        sumt2 += rt * rt;
                    }
                }

                double corr = 0.0;
                if (suma2 > 0.0 && sumt2 > 0.0)
                    corr = sumc / sqrt(suma2 * sumt2);

                if (corr > local_maxcorr) {
                    local_maxcorr = corr;
                    local_imax = is;
                    local_jmax = js;
                }
            }
        }

    #pragma omp critical
        {
            if (local_maxcorr > global_maxcorr) {
                global_maxcorr = local_maxcorr;
                global_imax = local_imax;
                global_jmax = local_jmax;
            }
        }
    }

	/*  compute the normalized cross correlation function
	for (is = -ns + yshft; is < ns + 1 + yshft; is++) {
		for (js = -ns + xshft; js < ns + 1 + xshft; js++) {
			ntot = 0;
			sumc = suma = sumt = 0.0;
			for (i = 0 + ib; i < ni - ib; i++) {
				i1 = i - is;
				for (j = 0 + ib; j < nj - ib; j++) {
					j1 = j - js;
					k = i * nj + j;
					k1 = i1 * nj + j1;
					if (i1 >= 0 && i1 < ni && j1 >= 0 && j1 < nj) {
						ntot++;
						ra = A->data[k] - avea;
						rt = T->data[k1 + 1] - T->data[k1 - 1];
						sumc = sumc + ra * rt;
						suma = suma + ra * ra;
						sumt = sumt + rt * rt;
					}
				}
			}
			corr = 0;
			denom = suma * sumt;
			if (denom > 0.)
				corr = sumc / sqrt(denom);
			//printf(" rshift = %d  ashift = %d  correlation = %f\n",js,is,corr);
			if (corr > maxcorr) {
				maxcorr = corr;
				imax = is;
				jmax = js;
			}
		}
	} */

    printf(" optimal: rshift = %d  ashift = %d  max_correlation = %f\n",
       global_jmax, global_imax, global_maxcorr);

    jmax = global_jmax;
    imax = global_imax;
    maxcorr = global_maxcorr;

	// printf(" optimal: rshift = %d  ashift = %d  max_correlation = %f\n", jmax, imax, maxcorr);

	if (argc >= 7) { /* write the shifted topo phase file */
		for (i = 0; i < ni; i++) {
			i1 = i - imax;
			for (j = 0; j < nj; j++) {
				j1 = j - jmax;
				k = i * nj + j;
				k1 = i1 * nj + j1;
				TS->data[k] = 0.0f;
				if (i1 >= 0 && i1 < ni && j1 >= 0 && j1 < nj)
					TS->data[k] = T->data[k1];
			}
		}

		/*   write the shifted grd-file */
		if (GMT_Write_Data(API, GMT_IS_GRID, GMT_IS_FILE, GMT_IS_SURFACE, GMT_GRID_ALL, NULL, argv[6], TS))
			return EXIT_FAILURE;
	}

	if (GMT_Destroy_Session(API))
		return EXIT_FAILURE; /* Remove the GMT machinery */

	return (EXIT_SUCCESS);
}
