include ../../config.mk

LIB	= lib/ibENVI.a
DIRS	= ASA_CAT Dop_orbit ENVI_decode ENVI_SLC_decode
SCRIPT	= scripts
ENVI_PRE_PROCESS = $(SCRIPT)/ENVI_pre_process
ENVI_PRE_PROCESS_IN = $(SCRIPT)/ENVI_pre_process.in
ENVI_SLC_PRE_PROCESS = $(SCRIPT)/ENVI_SLC_pre_process
ENVI_SLC_PRE_PROCESS_IN = $(SCRIPT)/ENVI_SLC_pre_process.in

all:	$(LIB)
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all) || exit $$?; \
	done

install:	all $(ENVI_PRE_PROCESS) $(ENVI_SLC_PRE_PROCESS)
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) install) || exit $$?; \
	done
	$(INSTALL) $(SCRIPT)/dump_orbit_envi.pl $(bindir)
	$(INSTALL) $(SCRIPT)/dump_time_envi.pl $(bindir)
	$(INSTALL) $(SCRIPT)/find_auxi.pl $(bindir)
	$(INSTALL) $(ENVI_PRE_PROCESS) $(bindir)
	$(INSTALL) $(ENVI_SLC_PRE_PROCESS) $(bindir)

uninstall:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) uninstall) || exit $$?; \
	done
	rm -f $(bindir)/dump_orbit_envi.pl
	rm -f $(bindir)/dump_time_envi.pl
	rm -f $(bindir)/find_auxi.pl
	rm -f $(bindir)/ENVI_pre_process
	rm -f $(bindir)/ENVI_SLC_pre_process

clean:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) clean) || exit $$?; \
	done
	(cd lib_src; $(MAKE) clean)

$(LIB):
	mkdir -p lib
	(cd lib_src; $(MAKE) all)

spotless:	clean
	rm -rf lib
	rm -f $(ENVI_PRE_PROCESS) $(ENVI_SLC_PRE_PROCESS)

$(ENVI_PRE_PROCESS): $(ENVI_PRE_PROCESS_IN)
	sed -e 's|@ORBITS_DIR@|$(ORBITS_DIR)|g' $< > $@
	chmod +x $@

$(ENVI_SLC_PRE_PROCESS): $(ENVI_SLC_PRE_PROCESS_IN)
	sed -e 's|@ORBITS_DIR@|$(ORBITS_DIR)|g' $< > $@
	chmod +x $@
