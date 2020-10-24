serial-amr:
	make -C Kernels/SERIAL/AMR/ amr
	ln -s Kernels/SERIAL/AMR/amr serial-amr

clean:
	make -C Kernels/SERIAL/AMR/ clean
	rm -f serial-amr
