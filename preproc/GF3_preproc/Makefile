include ../../config.mk
#
#   make all the files
#
DIRS	= lib src

all:
	$(LN_S) -f ../S1A_preproc/include include
	$(LN_S) -f ../S1A_preproc/lib lib
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all) || exit $$?; \
	done

install:	all
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) install) || exit $$?; \
	done

uninstall:
	(cd src; $(MAKE) uninstall)

clean:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) clean) || exit $$?; \
	done

spotless:	clean
	$(RM) lib include
