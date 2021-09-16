from libcpp.vector cimport vector

cimport cython
cimport numpy as np
import numpy as np


np.import_array()


ctypedef fused  float_ft:
	cython.float
	cython.double


cdef extern from "topn_parallel.h":

	cdef int topn_parallel[T](
		int n,
		int r[],
		int c[],
		T d[],
		int ntop,
		int n_jobs
	) except +;

cpdef topn_threaded(
	np.ndarray[int, ndim=1] r,
	np.ndarray[int, ndim=1] c,
	np.ndarray[float_ft, ndim=1] d,
	int ntop,
	int n_jobs
):
	"""
	Cython glue function:
	r, c, and d are 1D numpy arrays all of the same length N. 
	This function will return arrays rn, cn, and dn of length n <= N such
	that the set of triples {(rn[i], cn[i], dn[i]) : 0 < i < n} is a subset of 
	{(r[j], c[j], d[j]) : 0 < j < N} and that for every distinct value 
	x = rn[i], dn[i] is among the first ntop existing largest d[j]'s whose 
	r[j] = x.

	Input:
		r and c: two 1D integer arrays of the same length
		d: 1D array of single or double precision floating point type of the
		same length as r or c
		ntop maximum number of maximum d's returned
		use_threads: use multi-thread or not
		n_jobs: number of threads, must be >= 1

	Output:
		(rn, cn, dn) where rn, cn, dn are all arrays as described above.
	"""

	cdef int* var_r = &r[0]
	cdef int* var_c = &c[0]
	cdef float_ft* var_d = &d[0]
	
	n = len(r)
	
	new_len = topn_parallel(
		n, var_r, var_c, var_d, ntop, n_jobs
	)
	
	
	return new_len
