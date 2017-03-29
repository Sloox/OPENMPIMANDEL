
all:
	mpicc main.c mandle.c -o Mandle -lm -O -O3 -fopenmp -mstackrealign -msse4.1 -march=core2
