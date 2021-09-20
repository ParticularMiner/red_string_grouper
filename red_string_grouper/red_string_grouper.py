import pandas as pd
import numpy as np
import functools
from string_grouper import StringGrouper, StringGrouperConfig
from scipy.sparse.csr import csr_matrix
from red_string_grouper.topn import awesome_topn
from red_string_grouper.sparse_dot_topn import awesome_cossim_topn
from typing import Optional


def field(field_name: str, weight=1.0, **kwargs):
    '''
    Function that returns a triple corresponding to a field:
    field_name, weight, StringGrouperConfig(**kwargs)
    :param field_name: str
    :param weight: relative priority given to this field.  Defaults to 1.0.
    :param kwargs: keyword arguments to be passed to StringGrouper
    '''
    _ = StringGrouperConfig(**kwargs)   # validate kwargs
    return field_name, weight, kwargs

def record_linkage(data_frame,
                   fields_2b_matched_fuzzily,
                   fields_2b_matched_exactly=None,
                   hierarchical=True,
                   force_symmetries=True,
                   n_blocks=None,
                   **kwargs):
    '''
    Function that combines similarity-matching results of several fields of a
    DataFrame and returns them in another DataFrame
    :param data_frame: pandas.DataFrame of strings.
    :param fields_2b_matched_fuzzily: List of tuples.  Each tuple is a triple 
        (<field name>, <weight>, <field_kwargs>) which can be input using 
        utility function field(name, weight, **kwargs).
        <field name> is the name of a field in data_frame which is to be
        matched. <weight> is a number that defines the
        **relative** importance of the field to other fields -- the field's
        contribution to the total similarity will be weighted by this number.
        <field_kwargs> is a python dict capturing any keyword arguments to be 
        passed to StringGrouper for this field.
    :param fields_2b_matched_exactly: List of tuples.  Each tuple is a pair
        (<field name>, <weight>) which can be input using 
        utility function field(name, weight).  
        <field name> is the name of a field in data_frame which is to be 
        matched exactly.  <weight> has the same meaning as in parameter 
        fields_2b_matched_fuzzily. Defaults to None.
    :param hierarchical: bool.  Determines if the output DataFrame will have a
        hierarchical column-structure (True) or not (False). Defaults to True.
    :param force_symmetries: bool. Specifies whether corrections should be
        made to the results to account for symmetry thus removing certain
        errors due to lack of numerical precision.
    :param n_blocks: Tuple[(int, int)]. This parameter is provided to boost
        performance, if possible, by splitting the dataset into n_blocks[0]
        blocks for the left operand (of the "comparison operator") and into
        n_blocks[1] blocks for the right operand before performing the
        string-comparisons blockwise.
    :param kwargs: keyword arguments to be passed to StringGrouper for all 
        "fields to be matched fuzzily".  However, any keyword arguments already
        given in fields_2b_matched_fuzzily will take precedence over those 
        given in kwargs.  
    :return: pandas.DataFrame containing matching results.
    '''
    def get_field_names(fields_tuples):
        return list(list(zip(*fields_tuples))[0])
    
    def get_field_weights(field_tuples):
        return list(list(zip(*field_tuples))[1])
    
    def get_field_value_pairs(field_names, values):
        return list(zip(field_names, values))
    
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
        for field_name, sg in fuzzy_field_grouper_pairs:
            matches = match_strings(
                df[field_name],
                red_sg=sg,
                force_symmetries=force_symmetries,
                n_blocks=n_blocks
            )
            matches.set_index(match_indexes, inplace=True)
            matches = weed_out_trivial_matches(matches)
            if hierarchical:
                merger = matches[
                    [f'left_{field_name}', 'similarity', f'right_{field_name}']
                ]
                merger.rename(
                    columns={
                        f'left_{field_name}': 'left',
                        f'right_{field_name}': 'right'
                    },
                    inplace=True
                )
            else:
                merger = matches[['similarity']]
                merger.rename(columns={'similarity': field_name}, inplace=True)
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
    
    global_config = StringGrouperConfig(**kwargs)   # validate given kwargs
    index_name_list = get_index_names(data_frame)
    match_indexes = prepend(index_name_list, prefix='left_') \
        + prepend(index_name_list, prefix='right_')
    
    # set the corpus for each fuzzy field
    fuzzy_field_grouper_pairs = []
    for field_name, _, sg_kwargs in fields_2b_matched_fuzzily:
        red_sg_kwargs = global_config._asdict()
        red_sg_kwargs.update(sg_kwargs)
        fuzzy_field_grouper_pairs += [(
            field_name,
            PersistentCorpusStringGrouper(
                data_frame[field_name],
                **red_sg_kwargs
            )
        )]
 
    fuzzy_field_names = get_field_names(fields_2b_matched_fuzzily)
    fuzzy_field_weights = get_field_weights(fields_2b_matched_fuzzily)
    
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
        exact_field_weights = get_field_weights(fields_2b_matched_exactly)
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
    
def match_strings(master, duplicates=None, red_sg=None, 
                  force_symmetries=True, n_blocks=None, **kwargs):
    if not red_sg:
        red_sg = PersistentCorpusStringGrouper(master, duplicates, **kwargs)

    red_sg._master = master
    red_sg._duplicates = duplicates
    red_sg = red_sg.fit(force_symmetries=force_symmetries, n_blocks=n_blocks)
    out = red_sg.get_matches()
    red_sg._matches_list = None
    red_sg._master = None
    return out
        

def match_most_similar(master, duplicates, red_sg=None, 
                  force_symmetries=True, n_blocks=None, **kwargs):
    kwargs['max_n_matches'] = 1
    if not red_sg:
        red_sg = PersistentCorpusStringGrouper(master, duplicates, **kwargs)

    red_sg._master = master
    red_sg._duplicates = duplicates
    red_sg = red_sg.fit(force_symmetries=force_symmetries, n_blocks=n_blocks)
    out = red_sg.get_groups()
    red_sg._matches_list = None
    red_sg._master = None
    return out
        

def group_similar_strings(master, red_sg=None, 
                  force_symmetries=True, n_blocks=None, **kwargs):
    if not red_sg:
        red_sg = PersistentCorpusStringGrouper(master, **kwargs)

    red_sg._master = master
    red_sg = red_sg.fit(force_symmetries=force_symmetries, n_blocks=n_blocks)
    out = red_sg.get_groups()
    red_sg._matches_list = None
    red_sg._master = None
    return out
        

class PersistentCorpusStringGrouper(StringGrouper):
    # This class enables StringGrouper to apply matching on different
    # successive master datasets without resetting the underlying corpus, as
    # long as each master dataset is contained in the corpus. 
    # This class inherits from StringGrouper, overwriting a few of its
    # methods: __init__(), _get_tf_idf_matrices(), _build_matches() and fit(). 
    def __init__(self, corpus, duplicates=None, **kwargs):
        # initializer is the same as before except that it now also sets the
        # corpus
        super().__init__(corpus, duplicates=duplicates, **kwargs)
        if self._config.max_n_matches is None:
            self._max_n_matches = len(self._master)
        else:
            self._max_n_matches = self._config.max_n_matches
        self._vectorizer = self._fit_vectorizer()
        self._master = None

    def _get_left_tf_idf_matrix(self, partition):
        # unlike _get_tf_idf_matrices(), _get_left_tf_idf_matrix
        # does not set the corpus but rather
        # builds a matrix using the existing corpus
        if self._duplicates is not None:
            return self._vectorizer.transform(
                self._duplicates.iloc[slice(*partition)]
            )
        else:
            return self._vectorizer.transform(
                self._master.iloc[slice(*partition)]
            )

    def _get_right_tf_idf_matrix(self, partition):
        # unlike _get_tf_idf_matrices(), _get_right_tf_idf_matrix
        # does not set the corpus but rather
        # builds a matrix using the existing corpus
        return self._vectorizer.transform(
            self._master.iloc[slice(*partition)]
        )

    def _build_matches(self, left_matrix: csr_matrix, right_matrix: csr_matrix) -> csr_matrix:
        """Builds the cossine similarity matrix of two csr matrices"""
        right_matrix = right_matrix.transpose()

        optional_kwargs = {
            'return_best_ntop': True,
            'use_threads': self._config.number_of_processes > 1,
            'n_jobs': self._config.number_of_processes
        }

        return awesome_cossim_topn(
            left_matrix, right_matrix,
            self._max_n_matches,
            self._config.min_similarity,
            **optional_kwargs
        )

    def _fit_blockwise_manual(self, n_blocks=(1, 1)):
        def divide_by(n, series):
            # mark blocks
            equal_block_sz = len(series)//n
            block_rem = len(series)%n
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
            
        if self._duplicates is not None:
            block_ranges_left = divide_by(n_blocks[0], self._duplicates)
        else:
            block_ranges_left = divide_by(n_blocks[0], self._master)
        block_ranges_right = divide_by(n_blocks[1], self._master)
        max_n_matches = self._max_n_matches
        self._true_max_n_matches = 0
        for left_block in block_ranges_left:
            left_matrix = self._get_left_tf_idf_matrix(left_block)
            for right_block in block_ranges_right:
                self._max_n_matches = min(
                    right_block[1] - right_block[0],
                    max_n_matches
                )
                right_matrix = self._get_right_tf_idf_matrix(right_block)

                # Calculate the matches using the cosine similarity
                matches, block_true_max_n_matches = self._build_matches(
                    left_matrix, right_matrix
                )
                self._true_max_n_matches = \
                    max(block_true_max_n_matches, self._true_max_n_matches)
                
                # build match-lists from matrix
                r, c = matches.nonzero()
                d = matches.data
                (self._r, self._c, self._d) = (
                    np.append(self._r, r + left_block[0]),
                    np.append(self._c, c + right_block[0]),
                    np.append(self._d, d)
                )
                
        self._max_n_matches = max_n_matches
        return max(n_blocks) > 1

    def _fit_blockwise_auto(self,
                           left_partition=(None, None),
                           right_partition=(None, None)):
        # fit() has been extended here to enable StringGrouper to handle large
        # datasets which otherwise would lead to an OverflowError
        # The handling is achieved using block matrix multiplication
        def begin(partition):
            return partition[0] if partition[0] is not None else 0

        def end(partition, left=True):
            if partition[1] is not None:
                return partition[1]
            
            if left and (self._duplicates is not None):
                return len(self._duplicates) 
            
            return len(self._master)
        
        def explicit(partition):
            return begin(partition), end(partition)

        left_matrix = self._get_left_tf_idf_matrix(left_partition)
        right_matrix = self._get_right_tf_idf_matrix(right_partition)

        try:
            # Calculate the matches using the cosine similarity
            matches, block_true_max_n_matches = self._build_matches(
                left_matrix, right_matrix)
            self._true_max_n_matches = \
                max(block_true_max_n_matches, self._true_max_n_matches)
        except OverflowError:
            left_matrix = None
            right_matrix = None
            max_n_matches = self._max_n_matches
            
            def split_partition(partition, left=True):
                data_begin = begin(partition)
                data_end = end(partition, left=left)
                data_mid = data_begin + (data_end - data_begin)//2
                return [(data_begin, data_mid), (data_mid, data_end)]
            
            left_halves = split_partition(left_partition, left=True)
            right_halves = split_partition(right_partition, left=False)
            for lhalf in left_halves:
                for rhalf in right_halves:
                    self._max_n_matches = min(
                        rhalf[1] - rhalf[0],
                        max_n_matches
                    )
                    _ = self._fit_blockwise_auto(
                        left_partition=lhalf, right_partition=rhalf)
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
        self._true_max_n_matches = 0
        
        # do the matching
        if n_blocks:
            split_occurred = self._fit_blockwise_manual(n_blocks=n_blocks)
        else:
            split_occurred = self._fit_blockwise_auto()
        
        if split_occurred:
            # trim the matches to max_n_matches
            self._r, self._c, self._d = awesome_topn(
                self._r, self._c, self._d,
                self._max_n_matches,
                n_jobs=self._config.number_of_processes
            )
        
        # force symmetries to be respected?
        if force_symmetries and not self._duplicates:
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
            self._matches_list = self._get_matches_list()
            # release memory
            self._r = self._c = self._d = np.array([])
            
        self.is_build = True
        return self

    def _get_matches_list(self,
                          matches: Optional[csr_matrix] = None
        ) -> pd.DataFrame:
        """Returns a list of all the indices of matches"""
        if matches is None:
            r, c, d = self._r, self._c, self._d
        else:
            r, c = matches.nonzero()
            d = matches.data

        return pd.DataFrame({'master_side': c,
                             'dupe_side': r,
                             'similarity': d})
