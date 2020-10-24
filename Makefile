serial-amr:
	make -C Kernels/SERIAL/AMR/ amr
	ln -s Kernels/SERIAL/AMR/amr serial-amr

parallel-amr:
	make -C src amr
	ln -s src/amr parallel-amr

clean:
	make -C Kernels/SERIAL/AMR/ clean
	rm -f serial-amr parallel-amr
