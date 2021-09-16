#ifndef UTILS_CPPCLASS_H
#define UTILS_CPPCLASS_H

template<typename T>
struct rcd {
	int r;
	int c;
	T d;

	bool operator<(const rcd& a) const
    {
        return (a.d < d && a.r == r) || a.r > r;
    }

};


template<typename T>
extern int topn_parallel(
		int n,
		int r[],
		int c[],
		T d[],
		int ntop,
		int n_jobs
);

#endif //UTILS_CPPCLASS_H
