all: serial-amr parallel-amr

serial-amr: Kernels/SERIAL/AMR/
	make -C Kernels/SERIAL/AMR/ amr
	ln -sf Kernels/SERIAL/AMR/amr serial-amr

parallel-amr: src/
	make -C src amr
	ln -sf src/amr parallel-amr

clean:
	make -C Kernels/SERIAL/AMR/ clean
	rm -f serial-amr parallel-amr
