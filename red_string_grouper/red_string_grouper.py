import pandas as pd
import numpy as np
import functools
from string_grouper import StringGrouper
from scipy.sparse.csr import csr_matrix
from red_string_grouper.topn import awesome_topn
from red_string_grouper.sparse_dot_topn import awesome_cossim_topn


def record_linkage(data_frame,
                   fields_2b_matched_fuzzily,
                   fields_2b_matched_exactly=None,
                   hierarchical=True,
                   max_n_matches=None,
                   similarity_dtype=np.float32,
                   force_symmetries=True,
                   n_blocks=None):
    '''
    Function that combines similarity-matching results of several fields of a
    DataFrame and returns them in another DataFrame
    :param data_frame: pandas.DataFrame of strings.
    :param fields_2b_matched_fuzzily: List of tuples.  Each tuple is a triple 
        (<field name>, <threshold>, <ngram_size>, <weight>).
        <field name> is the name of a field in data_frame which is to be
        matched using a threshold similarity score of <threshold> and an ngram
        size of <ngram_size>. <weight> is a number that defines the
        **relative** importance of the field to other fields -- the field's
        contribution to the total similarity will be weighted by this number.
    :param fields_2b_matched_exactly: List of tuples.  Each tuple is a pair
        (<field name>, <weight>).  <field name> is the name of a field in
        data_frame which is to be matched exactly.  <weight> has the same
        meaning as in parameter fields_2b_matched_fuzzily. Defaults to None.
    :param hierarchical: bool.  Determines if the output DataFrame will have a
        hierarchical column-structure (True) or not (False). Defaults to True.
    :param max_n_matches: int. Maximum number of matches allowed per string.
    :param similarity_dtype: numpy type.  Either np.float32 (the default) or
        np.float64.  A value of np.float32 allows for less memory overhead
        during computation but less numerical precision, while np.float64
        allows for greater numerical precision but a larger memory overhead.
    :param force_symmetries: bool. Specifies whether corrections should be
        made to the results to account for symmetry thus removing certain
        errors due to lack of numerical precision.
    :param n_blocks: Tuple[(int, int)]. This parameter is provided to boost
        performance, if possible, by splitting the dataset into n_blocks[0]
        blocks for the left operand (of the "comparison operator") and into
        n_blocks[1] blocks for the right operand before performing the
        string-comparisons blockwise.
    :return: pandas.DataFrame containing matching results.
    '''
    def get_field_names(fields_tuples):
        return list(list(zip(*fields_tuples))[0])
    
    def get_exact_weights(exact_field_tuples):
        return list(list(zip(*exact_field_tuples))[1])
    
    def get_fuzzy_weights(fuzzy_field_tuples):
        return list(list(zip(*fuzzy_field_tuples))[3])
    
    def get_field_value_pairs(field_names, values):
        return list(zip(field_names, values))
    
    def get_field_stringGrouper_pairs(fuzzy_field_tuples, string_groupers):
        return [
            (tupl[0], ) + (x, ) for tupl, x in list(
                zip(fuzzy_field_tuples, string_groupers)
            )
        ]
    
    def get_index_names(df):
        empty_df = df.iloc[0:0]
        return [field for field in empty_df.reset_index().columns \
                if field not in empty_df.columns]
    
    def prepend(strings, prefix):
        return [f'{prefix}{i}' for i in strings]
    
    def horizontal_linkage(df,
                           match_indexes,
                           fuzzy_field_grouper_pairs,
                           fuzzy_field_names,
                           fuzzy_field_weights,
                           exact_field_value_pairs=None,
                           exact_field_weights=None,
                           hierarchical=True,
                           force_symmetries=False,
                           n_blocks=None):
        horizontal_merger_list = []
        for field, sg in fuzzy_field_grouper_pairs:
            matches = match_strings(
                df[field],
                sg,
                force_symmetries=force_symmetries,
                n_blocks=n_blocks
            )
            matches.set_index(match_indexes, inplace=True)
            matches = weed_out_trivial_matches(matches)
            if hierarchical:
                merger = matches[
                    [f'left_{field}', 'similarity', f'right_{field}']
                ]
                merger.rename(
                    columns={
                        f'left_{field}': 'left',
                        f'right_{field}': 'right'
                    },
                    inplace=True
                )
            else:
                merger = matches[['similarity']]
                merger.rename(columns={'similarity': field}, inplace=True)
            horizontal_merger_list += [merger]
            
        key_list = None if not hierarchical else fuzzy_field_names
        merged_df = pd.concat(
            horizontal_merger_list,
            axis=1,
            keys=key_list,
            join='inner'
        )
        
        title_exact = 'Exactly Matched Fields'
        title_fuzzy = 'Fuzzily Matched Fields'
        if exact_field_value_pairs:
            exact_df = build_column_precursor_to(
                merged_df,
                exact_field_value_pairs
            )
            merged_df = pd.concat(
                [exact_df, merged_df],
                axis=1,
                keys=[title_exact, title_fuzzy],
                join='inner'
            )
        totals = compute_totals(
            merged_df,
            fuzzy_field_names,
            fuzzy_field_weights,
            exact_field_weights,
            hierarchical,
            exact_field_value_pairs,
            title_fuzzy
        )
        return pd.concat([totals, merged_df], axis=1)
    
    def weed_out_trivial_matches(matches):
        num_indexes = matches.index.nlevels//2
        return matches[
            functools.reduce(
                lambda a, b: a | b, 
                [
                    (matches.index.get_level_values(i) \
                        != matches.index.get_level_values(i + num_indexes)) \
                        for i in range(num_indexes)
                ]
            )
        ]
    
    def build_column_precursor_to(df, exact_field_value_pairs):
        exact_df = df.iloc[:, 0:0]
        for field_name, field_value in exact_field_value_pairs:
            exact_df[field_name] = field_value
        return exact_df

    def compute_totals(merged_df,
                       fuzzy_field_names,
                       fuzzy_field_weights,
                       exact_field_weights,
                       hierarchical,
                       exact_field_value_pairs,
                       title_fuzzy):
        title_total = 'Weighted Mean Similarity Score'
        fuzzy_weight_array = np.array(fuzzy_field_weights, dtype=float)
        if exact_field_value_pairs:
            exact_weight_array = np.array(exact_field_weights, dtype=float)
            total = fuzzy_weight_array.sum() + exact_weight_array.sum()
            fuzzy_weight_array /= total
            exact_field_contribution = (exact_weight_array/total).sum()
            if hierarchical:
                totals = merged_df[
                    [
                        (title_fuzzy, field, 'similarity') \
                            for field in fuzzy_field_names
                    ]
                ].dot(fuzzy_weight_array) + exact_field_contribution
                totals = pd.concat(
                    [totals], axis=1, keys=[('', '', title_total)]
                )
            else:
                totals = merged_df[
                    [(title_fuzzy, field) for field in fuzzy_field_names]
                ].dot(fuzzy_weight_array) + exact_field_contribution
                totals = pd.concat([totals], axis=1, keys=[('', title_total)])
        else:
            fuzzy_weight_array /= fuzzy_weight_array.sum()
            if hierarchical:
                totals = merged_df[
                    [(field, 'similarity') for field in fuzzy_field_names]
                ].dot(fuzzy_weight_array) 
                totals = pd.concat([totals], axis=1, keys=[('', title_total)])
            else:
                totals = merged_df[fuzzy_field_names].dot(fuzzy_weight_array)
                totals.rename(title_total, inplace=True)
        return totals
    
    index_name_list = get_index_names(data_frame)
    match_indexes = prepend(index_name_list, prefix='left_') \
        + prepend(index_name_list, prefix='right_')
    
    # set the corpus for each fuzzy field
    stringGroupers = []
    for field, threshold, ngram_sz, _ in fields_2b_matched_fuzzily:
        stringGroupers += [
            PersistentCorpusStringGrouper(
                data_frame[field],
                min_similarity=threshold,
                ngram_size=ngram_sz,
                max_n_matches=max_n_matches,
                tfidf_matrix_dtype=similarity_dtype
            )
        ]
 
    fuzzy_field_grouper_pairs = get_field_stringGrouper_pairs(
        fields_2b_matched_fuzzily,
        stringGroupers
    )
    fuzzy_field_names = get_field_names(fields_2b_matched_fuzzily)
    fuzzy_field_weights = get_fuzzy_weights(fields_2b_matched_fuzzily)
    
    if not fields_2b_matched_exactly:
        return horizontal_linkage(
            data_frame,
            match_indexes,
            fuzzy_field_grouper_pairs,
            fuzzy_field_names,
            fuzzy_field_weights,
            hierarchical=hierarchical,
            force_symmetries=force_symmetries,
            n_blocks=n_blocks
        )
    else:
        exact_field_names = get_field_names(fields_2b_matched_exactly)
        exact_field_weights = get_exact_weights(fields_2b_matched_exactly)
        groups = data_frame.groupby(exact_field_names)
        vertical_merger_list = []
        for group_value, group_df in groups:
            values_matched_exactly = [group_value] if not isinstance(
                group_value,
                tuple
            ) else list(group_value)
            exact_field_value_pairs = get_field_value_pairs(
                exact_field_names,
                values_matched_exactly
            )
            vertical_merger_list += [
                horizontal_linkage(
                    group_df,
                    match_indexes,
                    fuzzy_field_grouper_pairs,
                    fuzzy_field_names,
                    fuzzy_field_weights,
                    exact_field_value_pairs,
                    exact_field_weights,
                    hierarchical=hierarchical,
                    force_symmetries=force_symmetries,
                    n_blocks=n_blocks
                )
            ]
        return pd.concat(vertical_merger_list)
    
def match_strings(master, sg, force_symmetries=True, n_blocks=None):
    sg._master = master
    sg = sg.fit(force_symmetries=force_symmetries, n_blocks=n_blocks)
    out = sg.get_matches()
    sg._matches_list = None
    sg._master = None
    return out
        

class PersistentCorpusStringGrouper(StringGrouper):
    # This class enables StringGrouper to apply matching on different
    # successive master datasets without resetting the underlying corpus, as
    # long as each master dataset is contained in the corpus. 
    # This class inherits from StringGrouper, overwriting a few of its
    # methods: __init__(), _get_tf_idf_matrices(), _build_matches() and fit(). 
    def __init__(self, corpus, **kwargs):
        # initializer is the same as before except that it now also sets the
        # corpus
        super().__init__(corpus, **kwargs)
        self._vectorizer = self._fit_vectorizer()
        self._master = None

    def _get_tf_idf_matrices(self, left_partition, right_partition):
        # _get_tf_idf_matrices() now no longer sets the corpus but rather
        # builds the matrices from the existing corpus
        # Build the two matrices
        left_matrix = self._vectorizer.transform(
            self._master.iloc[slice(*left_partition)]
        )
        right_matrix = self._vectorizer.transform(
            self._master.iloc[slice(*right_partition)]
        )
        return left_matrix, right_matrix

    def _build_matches(self, master_matrix: csr_matrix, duplicate_matrix: csr_matrix) -> csr_matrix:
        """Builds the cossine similarity matrix of two csr matrices"""
        tf_idf_matrix_1 = master_matrix
        tf_idf_matrix_2 = duplicate_matrix.transpose()

        optional_kwargs = {
            'return_best_ntop': True,
            'use_threads': self._config.number_of_processes > 1,
            'n_jobs': self._config.number_of_processes
        }

        return awesome_cossim_topn(
            tf_idf_matrix_1, tf_idf_matrix_2,
            self._max_n_matches,
            self._config.min_similarity,
            **optional_kwargs
        )

    def fit_blockwise_manual(self, n_blocks=(1, 1)):
        def divide_by(n):
            # mark blocks
            equal_block_sz = len(self._master)//n
            block_rem = len(self._master)%n
            block_ranges = []
            start = 0
            for block_id in range(n):
                block_ranges += [
                    (
                        start,
                        start + equal_block_sz + \
                            (1 if block_id < block_rem else 0)
                    )
                ]
                start = block_ranges[-1][1]
            return block_ranges
            
        block_ranges_left = divide_by(n_blocks[0])
        block_ranges_right = divide_by(n_blocks[1])
        max_n_matches = self._max_n_matches
        for left_block in block_ranges_left:
            for right_block in block_ranges_right:
                self._max_n_matches = min(
                    right_block[1] - right_block[0],
                    max_n_matches
                )
                master_matrix, duplicate_matrix = self._get_tf_idf_matrices(
                    left_block,
                    right_block
                )

                # Calculate the matches using the cosine similarity
                matches, self._true_max_n_matches = self._build_matches(
                    master_matrix,
                    duplicate_matrix
                )
                
                # build match-lists from matrix
                r, c = matches.nonzero()
                d = matches.data
                (self._r, self._c, self._d) = (
                    np.append(self._r, r + left_block[0]),
                    np.append(self._c, c + right_block[0]),
                    np.append(self._d, d)
                )
                
        self._max_n_matches = max_n_matches
        return True

    def fit_blockwise_auto(self,
                           left_partition=(None, None),
                           right_partition=(None, None)):
        """
        Builds the _matches list which contains string matches indices and
        similarity
        """
        # fit() has been extended here to enable StringGrouper to handle large
        # datasets which otherwise would lead to an OverflowError
        def begin(partition):
            return partition[0] if partition[0] else 0

        def end(partition):
            return partition[1] if partition[1] else len(self._master)
        
        def explicit(partition):
            return begin(partition), end(partition)

        master_matrix, duplicate_matrix = self._get_tf_idf_matrices(
            left_partition,
            right_partition
        )

        try:
            # Calculate the matches using the cosine similarity
            matches, self._true_max_n_matches = self._build_matches(
                master_matrix,
                duplicate_matrix
            )
        except OverflowError:
            master_matrix = None
            duplicate_matrix = None
            max_n_matches = self._max_n_matches
            
            def split_partition(partition):
                data_begin = begin(partition)
                data_end = end(partition)
                data_mid = data_begin + (data_end - data_begin)//2
                return [(data_begin, data_mid), (data_mid, data_end)]
            
            left_halves = split_partition(left_partition)
            right_halves = split_partition(right_partition)
            for lhalf in left_halves:
                for rhalf in right_halves:
                    self._max_n_matches = min(
                        rhalf[1] - rhalf[0],
                        max_n_matches
                    )
                    self.fit_blockwise_auto(
                        left_partition=lhalf,
                        right_partition=rhalf
                    )
            self._max_n_matches = max_n_matches
            return True

        # build match-lists from matrix
        r, c = matches.nonzero()
        d = matches.data
        (self._r, self._c, self._d) = (
            np.append(self._r, r + begin(left_partition)),
            np.append(self._c, c + begin(right_partition)),
            np.append(self._d, d)
        )

        return False

    def fit(self, n_blocks=None, force_symmetries=True):
        # initialize match-lists
        self._r = np.array([], dtype=np.int64)
        self._c = np.array([], dtype=np.int64)
        self._d = np.array([], dtype=self._config.tfidf_matrix_dtype)
        self._matches_list = pd.DataFrame()
        
        # do the matching
        if n_blocks:
            split_occurred = self.fit_blockwise_manual(n_blocks=n_blocks)
        else:
            split_occurred = self.fit_blockwise_auto()
        
        if split_occurred:
            # trim the matches to max_n_matches
            self._r, self._c, self._d = awesome_topn(
                self._r,
                self._c,
                self._d,
                self._max_n_matches,
                self._config.number_of_processes
            )
        
        # force symmetries to be respected?
        if force_symmetries:
            matrix_sz = len(self._master)
            matches = csr_matrix(
                (
                    self._d, (self._r, self._c)
                ),
                shape=(matrix_sz, matrix_sz)
            )
            # release memory
            self._r = self._c = self._d = np.array([])

            # convert to lil format for best efficiency when setting
            # matrix-elements
            matches = matches.tolil()
            # matrix diagonal elements must be exactly 1 (numerical precision
            # errors introduced by floating-point computations in
            # awesome_cossim_topn sometimes lead to unexpected results)
            matches = StringGrouper._fix_diagonal(matches)
            # the list of matches must be symmetric! 
            # (i.e., if A != B and A matches B; then B matches A)
            matches = StringGrouper._symmetrize_matrix(matches)
            matches = matches.tocsr()
            self._matches_list = self._get_matches_list(matches)
        else:
            self._matches_list = pd.DataFrame(
                {
                    key: value for key, value in zip(
                        ('master_side', 'dupe_side', 'similarity'),
                        (self._r, self._c, self._d)
                    )
                }
            )
            # release memory
            self._r = self._c = self._d = np.array([])
            
        self.is_build = True
        return self