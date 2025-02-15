.PHONY: build
build:
	CC=gcc CXX=g++ uv run setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -r build cython_fn*.{c,so}

.PHONY: annotate
annotate:
	cython -a -3 cython_fn.pyx

