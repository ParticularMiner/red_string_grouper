#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>

#include "./topn_parallel.h"

using namespace std;

struct job_range_type {int begin; int end;};

void distribute_load(
		int load_sz,
		int n_jobs,
		vector<job_range_type> &ranges
)
{
	// share the load among jobs:
	int equal_job_load_sz = load_sz/n_jobs;
	int rem = load_sz % n_jobs;
	ranges.resize(n_jobs);

	int start = 0;
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		ranges[job_nr].begin = start;
		ranges[job_nr].end = start + equal_job_load_sz + ((job_nr < rem)? 1 : 0);
		start = ranges[job_nr].end;
	}
}

template<typename T>
void inner_gather(
		int job_sz,
		rcd<T> v_rcd[],
		int r[],
		int c[],
		T d[]
)
{
	for (int i = 0; i < job_sz; i++){
		r[i] = v_rcd[i].r;
		c[i] = v_rcd[i].c;
		d[i] = v_rcd[i].d;
	}
}
template void inner_gather<float>(int job_sz, rcd<float> v_rcd[], int r[], int c[], float d[]);
template void inner_gather<double>(int job_sz, rcd<double> v_rcd[], int r[], int c[], double d[]);


template<typename T>
void inner_copy_to_v_rcd(
		job_range_type job_range,
		rcd<T> dest[],
		int r[],
		int c[],
		T d[]
)
{
	rcd<T>* dest_cursor = &dest[0];
	int* r_cursor = &r[0];
	int* c_cursor = &c[0];
	T* d_cursor = &d[0];

	for (int i = job_range.begin; i < job_range.end; i++){
		dest_cursor->r = *(r++);
		dest_cursor->c = *(c++);
		(dest_cursor++)->d = *(d++);
	}
}
template void inner_copy_to_v_rcd<float>(job_range_type job_range, rcd<float> dest[], int r[], int c[], float d[]);
template void inner_copy_to_v_rcd<double>(job_range_type job_range, rcd<double> dest[], int r[], int c[], double d[]);

template<typename T>
void inner_pick_nlargest(
		job_range_type job_range,
		rcd<T> v_rcd[],
		int ntop,
		int* new_end
)
{
	rcd<T>* write_cusor = &v_rcd[0];
	rcd<T>* read_cusor = &v_rcd[0];
	int c = 0;
	int r, prev_r = v_rcd[0].r;

	for (int i = job_range.begin; i < job_range.end; i++){

		r = read_cusor->r;
		if (prev_r != r) c = 0;
		if (c < ntop) *(write_cusor++) = *read_cusor;
		read_cusor++;
		prev_r = r;
		c++;
	}
	*new_end = write_cusor - &v_rcd[0];
}

template void inner_pick_nlargest<float>(job_range_type job_range, rcd<float> v_rcd[], int ntop, int* new_end);
template void inner_pick_nlargest<double>(job_range_type job_range, rcd<double> v_rcd[], int ntop, int* new_end);


/*
 * The idea of the function "find_next_boundary" is how to get from ^ to *
 * (i.e., from inside a 1D-zone to just outside it in the next zone)
 *
 * previous zone     current zone         next zone
 * _______________|_________________|__________________
 *                       ^           *
 */
template<typename T>
int find_next_boundary(rcd<T> v_rcd[], int x, int end)
{
	if (v_rcd[x - 1].r == v_rcd[end - 1].r) return end;
	while (v_rcd[x - 1].r == v_rcd[x].r){ // non-boundary condition
		int mid = (x + end)/2;
		if (mid == x) return end;
		else if (v_rcd[x].r == v_rcd[mid - 1].r) x = mid;
		else end = mid;
	}
	return x;
}

template int find_next_boundary<float>(rcd<float> v_rcd[], int x, int end);
template int find_next_boundary<double>(rcd<double> v_rcd[], int x, int end);

/*
 * The idea of the function "find_prev_boundary" is how to get from ^ to *
 * (i.e., from inside a 1D-zone to its start just outside the previous zone)
 *
 * previous zone     current zone         next zone
 * _______________|_________________|__________________
 *                 *        ^
 */
template<typename T>
int find_prev_boundary(rcd<T> v_rcd[], int begin, int x)
{
	if (v_rcd[begin].r == v_rcd[x].r) return begin;
	while (v_rcd[x - 1].r == v_rcd[x].r){ // non-boundary condition
		int mid = (begin + x)/2;
		if (v_rcd[x].r == v_rcd[mid].r) x = mid;
		else begin = mid;
	}
	return x;
}

template int find_prev_boundary<float>(rcd<float> v_rcd[], int x, int end);
template int find_prev_boundary<double>(rcd<double> v_rcd[], int x, int end);


/*
	r, c, and d are 1D arrays all of the same length n.
	This function will output arrays rn, cn, and dn of length N <= n such
    that the set of triples {(rn[i], cn[i], dn[i]) : 0 < i < N} is a subset of
    {(r[j], c[j], d[j]) : 0 < j < n} and that for every distinct value
    x = rn[i], dn[i] is among the first ntop existing largest d[j]'s whose
    r[j] = x.

    Input:
        r and c: two 1D integer arrays of the same length
        d: 1D array of single or double precision floating point type of the
        	same length as r or c
        ntop maximum number of
        use_threads: use multi-thread or not
        n_jobs: number of threads, must be >= 1

    Output:
        rn, cn, dn where rn, cn, dn are all arrays as described above and will
        	replace r, c, d.
        returns the length of the output arrays
*/
template<typename T>
int topn_parallel(
		int n,
		int r[],
		int c[],
		T d[],
		int ntop,
		int n_jobs
)
{
	// distribute the load among jobs/threads
	vector<job_range_type> job_ranges(n_jobs);
	distribute_load(n, n_jobs, job_ranges);

	// initialize temporary storage and copy in parallel
	vector<rcd<T>> v_rcd(n);
	vector<thread> thread_list(n_jobs);
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = thread(
				inner_copy_to_v_rcd<T>,
				job_ranges[job_nr],
				&v_rcd[job_ranges[job_nr].begin],
				&r[job_ranges[job_nr].begin],
				&c[job_ranges[job_nr].begin],
				&d[job_ranges[job_nr].begin]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	// sort the data in temporary location
	sort(&v_rcd[0], &v_rcd[n]);

	// adjust the job-range boundaries to coincide with row boundaries:
	for (int job_nr = 1; job_nr < n_jobs; job_nr++) {
		// This condition is always enforced:
		job_ranges[job_nr].begin = job_ranges[job_nr - 1].end;

		int x = job_ranges[job_nr].begin;	// x marks the spot
		int f = job_ranges[job_nr - 1].begin;	// f: floor
		int c = n;	// c: ceiling
		int x0 = find_prev_boundary<T>(v_rcd.data(), f, x);
		int x1 = find_next_boundary<T>(v_rcd.data(), x, c);
		if (f < x0 && (x0 - x) >= (x - x1)){
			job_ranges[job_nr - 1].end = x0;
			job_ranges[job_nr].begin = x0;
		}
		else {
			job_ranges[job_nr - 1].end = x1;
			job_ranges[job_nr].begin = x1;
		}
		if (job_ranges[job_nr].begin > job_ranges[job_nr].end)
			job_ranges[job_nr].end = job_ranges[job_nr].begin;
	}

	// pick n largest elements
	// initialize aggregate:
	vector<int> sub_counts(n_jobs, 0);

	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = thread(
				inner_pick_nlargest<T>,
				job_ranges[job_nr],
				&v_rcd[job_ranges[job_nr].begin],
				ntop,
				&sub_counts[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	vector<int> dest_job_starts(n_jobs + 1);
	dest_job_starts[0] = 0;
	partial_sum(sub_counts.begin(), sub_counts.end(), dest_job_starts.begin() + 1);

	// gather the results:
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = thread(
				inner_gather<T>,
				sub_counts[job_nr],
				&v_rcd[job_ranges[job_nr].begin],
				&r[dest_job_starts[job_nr]],
				&c[dest_job_starts[job_nr]],
				&d[dest_job_starts[job_nr]]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	return dest_job_starts[n_jobs];
}
template int topn_parallel<float>(int n, int r[], int c[], float d[], int ntop, int n_jobs);
template int topn_parallel<double>(int n, int r[], int c[], double d[], int ntop, int n_jobs);

