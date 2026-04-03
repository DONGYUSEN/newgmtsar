#	$Id: Makefile 109 2015-01-19 23:01:24Z sandwell $
#
#	makefile for gmtsar directory (sbas.c not included yet)
#gcc -z muldefs -s -fopenmp -Wl,-rpath,/usr/lib/x86_64-linux-gnu xcorr2.o -L/home/ysdong/GMTSAR/gmtsar -lgmtsar -L/usr/lib/x86_64-linux-gnu -lgmt -llapack -lblas -lm -L/usr/local/lib -ltiff -lm -lfftw3f -o xcorr2

include ../config.mk

INCLUDES	= $(GMT_INC) -I./ -I$(TIFF_INC)
ifeq (,$(findstring -fopenmp,$(CFLAGS)))
OPENMP_CFLAGS	?= -fopenmp
else
OPENMP_CFLAGS	?=
endif
ifeq (,$(findstring -fopenmp,$(LDFLAGS)))
OPENMP_LDFLAGS	?= -fopenmp
else
OPENMP_LDFLAGS	?=
endif
XCORR3_CFLAGS	= $(shell pkg-config --cflags glib-2.0 fftw3 2>/dev/null)
XCORR3_LIBS	= $(shell pkg-config --libs glib-2.0 fftw3 2>/dev/null)
XCORR3_HEADERS_OK = $(shell printf '%s\n' '#include <glib.h>' '#include <fftw3.h>' | $(CC) $(XCORR3_CFLAGS) -x c -E - >/dev/null 2>&1 && echo yes || echo no)
STD_LINK_LIBS	= $(GMTSAR) $(GMT_LIB) $(LALIBS) $(LIBS) -L$(TIFF_LIB) -ltiff

LIB_C		= aastretch.c acpatch.c calc_dop.c conv2d.c do_freq_xcorr.c \
		  do_time_int_xcorr.c fft_bins.c fft_interpolate_routines.c \
		  file_stuff.c geoxyz.c get_locations.c get_params.c hermite_c.c \
		  highres_corr.c interpolate_orbit.c intp_coef.c ldr_orbit.c \
		  parse_xcorr_input.c plxyz.c polyfit.c print_results.c radopp.c \
		  read_orb.c read_xcorr_data.c SAT_llt2rat_sub.c \
		  rmpatch.c rng_cmp.c rng_ref.c set_prm_defaults.c shift.c \
		  sio_struct.c siocomplex.c spline.c trans_col.c utils.c utils_complex.c \
		  write_orb.c sbas_utils.c stringutils.c update_PRM_sub.c rng_filter.c \
		  lib_strfuncs.c gmtsar_globals.c soi_globals.c
		  # xcorr2.c xcorr3.c 

LIB_O		= $(LIB_C:.c=.o)
LIB		= libgmtsar.$(LIBEXT)

#-------------------------------------------------------------------------------
#	library
#-------------------------------------------------------------------------------

$(LIB):		$(LIB_O)
		$(AR) cvur $@ $?
		$(RANLIB) $@

lib:		$(LIB)

PROGS_C		= bperp.c calc_dop_orb.c conv.c esarp.c offset_topo.c phase2topo.c \
		  phasediff.c phasefilt.c resamp.c xcorr.c extend_orbit.c update_PRM.c get_PRM.c \
		  SAT_llt2rat.c SAT_look.c SAT_baseline.c make_gaussian_filter.c sbas.c \
          nearest_grid.c fitoffset.c solid_tide.c p_scatter.c split_spectrum.c cut_slc.c \
          split_aperture.c phasediff_get_topo_phase.c geocode_slc.c SAT_llt2rat2.c \
	  resamp_omp.c offset_topo2.c offset_topo_omp.c offsetfit2.c resamp_kriging.c
 
PROGS_O         = $(PROGS_C:.c=.o)
PROGS_STD       = $(PROGS_C:.c=)
OMP_EXTRA_PROGS = resamp_lt1 sbas_parallel
OMP_OBJECTS	= conv.o SAT_llt2rat2.o resamp_omp.o offset_topo2.o \
		  offset_topo_omp.o resamp_kriging.o resamp_lt1.o \
		  sbas_parallel.o xcorr2.o
XCORR2_OMP_LINK	= $(if $(findstring -fopenmp,$(CFLAGS)),,$(OPENMP_LDFLAGS))
ifeq ($(strip $(XCORR3_LIBS)),)
PROGS_EXTRA     =
else ifneq ($(XCORR3_HEADERS_OK),yes)
PROGS_EXTRA     =
else
PROGS_EXTRA     = xcorr2
endif
PROGS           = $(PROGS_STD) $(PROGS_EXTRA)

#-------------------------------------------------------------------------------
#	software targets
#-------------------------------------------------------------------------------

all:		$(PROGS)

install:	all
		$(INSTALL) -d $(bindir)
		$(INSTALL) $(PROGS) $(bindir)
		ls csh/*.csh csh/*.sh csh/*.py > /tmp/t.lis
		$(INSTALL) `cat /tmp/t.lis` $(bindir)
		rm -f /tmp/t.lis

uninstall:
		cd $(bindir); rm -f $(PROGS)
		cd csh; ls *.csh *.sh *.py > /tmp/t.lis
		cd $(bindir); rm -f `cat /tmp/t.lis`
		rm -f /tmp/t.lis

spotless::	clean

clean:	
		rm -f *.a *.o *% core tags $(PROGS)

#-------------------------------------------------------------------------------
#	program rules
#-------------------------------------------------------------------------------

$(PROGS_STD):	$(PROGS_O) $(LIB)
		$(CC) $(LDFLAGS) $(OPENMP_LDFLAGS) $@.o $(STD_LINK_LIBS) -o $@

$(OMP_EXTRA_PROGS):	%: %.o $(LIB)
		$(CC) $(LDFLAGS) $(OPENMP_LDFLAGS) $< $(STD_LINK_LIBS) -o $@

XCORR2_X3_OBJS	= xcorr2.o array_helper.o fft_helper.o prm_helper.o xcorr2_args.o

xcorr2:		$(XCORR2_X3_OBJS)
		$(CC) -o $@ $(XCORR2_X3_OBJS) $(CFLAGS) $(XCORR2_OMP_LINK) $(XCORR3_CFLAGS) -lm $(XCORR3_LIBS)

$(OMP_OBJECTS): CFLAGS += $(OPENMP_CFLAGS)

xcorr2.o:	xcorr2.c xcorr2.h xcorr2_args.h
		$(CC) -c -o $@ $< $(CFLAGS) $(XCORR3_CFLAGS) -I./

array_helper.o:	array_helper.c xcorr2.h
		$(CC) -c -o $@ $< $(CFLAGS) $(XCORR3_CFLAGS) -I./

fft_helper.o:	fft_helper.c xcorr2.h
		$(CC) -c -o $@ $< $(CFLAGS) $(XCORR3_CFLAGS) -I./

prm_helper.o:	prm_helper.c xcorr2.h
		$(CC) -c -o $@ $< $(CFLAGS) $(XCORR3_CFLAGS) -I./

xcorr2_args.o:	xcorr2_args.c xcorr2.h xcorr2_args.h
		$(CC) -c -o $@ $< $(CFLAGS) $(XCORR3_CFLAGS) -I./
