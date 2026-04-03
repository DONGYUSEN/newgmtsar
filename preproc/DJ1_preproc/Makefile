include ../../config.mk
#
#   make all the files
#
DIRS	= lib src_swath

all:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) all) || exit $$?; \
	done

install:	all
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) install) || exit $$?; \
	done

uninstall:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) uninstall) || exit $$?; \
	done

clean:
	for d in $(DIRS); do \
		(cd $$d; $(MAKE) clean) || exit $$?; \
	done

spotless:	clean
	$(MAKE) -C lib spotless
