include ../../config.mk

DIRS	= ers_line_fixer read_data_file_dpaf read_data_file_ccrs read_sarleader_dpaf
SCRIPT	= scripts
ERS_PRE_PROCESS_IN = $(SCRIPT)/ERS_pre_process.in
ERS_PRE_PROCESS = $(SCRIPT)/ERS_pre_process

all:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all) || exit $$?; \
	done

install:	all $(ERS_PRE_PROCESS)
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) install) || exit $$?; \
	done
	$(INSTALL) $(SCRIPT)/dump_orbit_ers.pl $(bindir)
	$(INSTALL) $(ERS_PRE_PROCESS) $(bindir)

uninstall:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) uninstall) || exit $$?; \
	done
	rm -f $(bindir)/dump_orbit_ers.pl
	rm -f $(bindir)/ERS_pre_process

clean:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) clean) || exit $$?; \
	done

spotless:	clean
	rm -f $(ERS_PRE_PROCESS)

$(ERS_PRE_PROCESS): $(ERS_PRE_PROCESS_IN)
	sed -e 's|@ORBITS_DIR@|$(ORBITS_DIR)|g' $< > $@
	chmod +x $@
