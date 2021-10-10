from builtins import isinstance
from numbers import Number
from typing import Tuple, NamedTuple, Optional
import functools
import multiprocessing
import warnings

import pandas as pd
import numpy as np
from scipy.sparse import vstack
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sparse_dot_topn_for_blocks import awesome_cossim_topn
from topn import awesome_hstack_topn
from string_grouper import StringGrouper


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


def field_pair(field_name1: str, field_name2: str, weight=1.0, **kwargs):
    '''
    Function that returns a quadruple corresponding to a field:
    field_name1, field_name2, weight, StringGrouperConfig(**kwargs)
    :param field_name1: str
    :param field_name2: str
    :param weight: relative priority given to this field-pair.  Defaults to
        1.0.
    :param kwargs: keyword arguments to be passed to StringGrouper
    '''
    _ = StringGrouperConfig(**kwargs)   # validate kwargs
    return field_name1, field_name2, weight, kwargs


def record_linkage(data_frames,
                   fields_2b_matched_fuzzily,
                   fields_2b_matched_exactly=None,
                   hierarchical=True,
                   **kwargs):
    '''
    Function that combines similarity-matching results of several fields of one
    or two DataFrames and returns them in another DataFrame.
    :param data_frames: either a pandas DataFrame or a list of two pandas
        DataFrames.
    :param fields_2b_matched_fuzzily: List of tuples.  If data_frames is a
        pandas DataFrame, then each tuple is a triple
        (<field name>, <weight>, <field_kwargs>) which can be input using
        utility function field(name, weight, **kwargs).
        <field name> is the name of a field in data_frames which is to be
        matched.
        If data_frames is a list of two pandas DataFrames, then each tuple is
        a quadruple
        (<field name1>, <field name2>, <weight>, <field_kwargs>) which can be
        input using utility function
        field_pair(name1, name2, weight, **kwargs).
        <field name1> is the name of a field in data_frame[0] which is to be
        matched. <field name2> is the name of a field in data_frame[1] which
        is to be matched.
        <weight> is a number that defines the **relative** importance of the
        field (or field-pair) to other fields (or field-pairs) -- the field's
        (or field-pair's) contribution to the total similarity will be
        weighted by this number.
        <field_kwargs> is a python dict capturing any keyword arguments to be
        passed to StringGrouper for this field (or field-pair).
    :param fields_2b_matched_exactly: List of tuples.  If data_frames is a
        pandas DataFrame, then each tuple is a pair
        (<field name>, <weight>) which can be input using
        utility function field(name, weight).
        <field name> is the name of a field in data_frames which is to be
        matched.
        If data_frames is a list of two pandas DataFrames, then each tuple is
        a triple
        (<field name1>, <field name2>, <weight>) which can be input using
        utility function field_pair(name1, name2, weight).
        <field name1> is the name of a field in data_frame[0] which is to be
        matched. <field name2> is the name of a field in data_frame[1] which
        is to be matched.
        <weight> has the same meaning as in parameter
        fields_2b_matched_fuzzily. Defaults to None.
    :param hierarchical: bool.  Determines if the output DataFrame will have a
        hierarchical column-structure (True) or not (False). Defaults to True.
    :param kwargs: keyword arguments to be passed to StringGrouper for all
        "fields to be matched fuzzily".  However, any keyword arguments already
        given in fields_2b_matched_fuzzily will take precedence over those
        given in kwargs.
    :return: pandas.DataFrame containing matching results.
    '''
    def get_field1_names(fields_tuples):
        return [n[0] for n in fields_tuples]

    def get_field2_names(fields_tuples):
        return [n[1] for n in fields_tuples]

    def get_field_names(fields_tuples):
        if isinstance(data_frames, list):
            return [f'{n1}/{n2}' for n1, n2, _, _ in fields_tuples]
        else:
            return get_field1_names(fields_tuples)

    def get_field_weights(field_tuples):
        if isinstance(data_frames, list):
            return [w for _, _, w, _ in field_tuples]
        else:
            return [w for _, w, _ in field_tuples]

    def get_field_value_pairs(field1_names, field2_names, values):
        if field2_names is None:
            return [(n1, v) for n1, v in zip(field1_names, values)]
        else:
            return [(f'{n1}/{n2}', v)
                    for n1, n2, v in zip(field1_names, field2_names, values)]

    def get_index_names(df):
        empty_df = df.iloc[0:0]
        return [field for field in empty_df.reset_index().columns
                if field not in empty_df.columns]

    def prepend(strings, prefix):
        return [f'{prefix}{i}' for i in strings]

    def horizontal_linkage(df1, df2,
                           match_indexes,
                           fuzzy_field_grouper_pairs,
                           fuzzy_field_names,
                           fuzzy_field_weights,
                           exact_field_value_pairs=None,
                           exact_field_weights=None,
                           hierarchical=True):
        horizontal_merger_list = []
        if df2 is None:
            for field_name1, sg in fuzzy_field_grouper_pairs:
                matches = sg.match_strings(df1[field_name1])
                sg.clear_data()
                matches.set_index(match_indexes, inplace=True)
                matches = weed_out_trivial_matches(matches)
                if hierarchical:
                    merger = matches[
                        [f'left_{field_name1}', 'similarity',
                         f'right_{field_name1}']]
                    merger.rename(
                        columns={
                            f'left_{field_name1}': 'left',
                            f'right_{field_name1}': 'right'},
                        inplace=True)
                else:
                    merger = matches[['similarity']]
                    merger.rename(
                        columns={'similarity': field_name1},
                        inplace=True)
                horizontal_merger_list += [merger]
        else:
            for field_name1, field_name2, sg in fuzzy_field_grouper_pairs:
                matches = sg.match_strings(df1[field_name1], df2[field_name2])
                sg.clear_data()
                matches.set_index(match_indexes, inplace=True)
                if hierarchical:
                    merger = matches[
                        [f'left_{field_name1}', 'similarity',
                         f'right_{field_name2}']]
                    merger.rename(
                        columns={
                            f'left_{field_name1}': 'left',
                            f'right_{field_name2}': 'right'},
                        inplace=True)
                else:
                    merger = matches[['similarity']]
                    merger.rename(
                        columns={'similarity': f'{field_name1}/{field_name2}'},
                        inplace=True)
                horizontal_merger_list += [merger]

        key_list = None if not hierarchical else fuzzy_field_names
        merged_df = pd.concat(
            horizontal_merger_list,
            axis=1,
            keys=key_list,
            join='inner')

        title_exact = 'Exactly Matched Fields'
        title_fuzzy = 'Fuzzily Matched Fields'
        if exact_field_value_pairs:
            exact_df = build_column_precursor_to(
                merged_df,
                exact_field_value_pairs)
            merged_df = pd.concat(
                [exact_df, merged_df],
                axis=1,
                keys=[title_exact, title_fuzzy],
                join='inner')
        totals = compute_totals(
            merged_df,
            fuzzy_field_names,
            fuzzy_field_weights,
            exact_field_weights,
            hierarchical,
            exact_field_value_pairs,
            title_fuzzy)
        return pd.concat([totals, merged_df], axis=1)

    def weed_out_trivial_matches(matches):
        num_indexes = matches.index.nlevels//2
        return matches[
            functools.reduce(
                lambda a, b: a | b,
                [
                    ((matches.index.get_level_values(i) !=
                      matches.index.get_level_values(i + num_indexes))
                     for i in range(num_indexes))
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
                totals = (merged_df[[(title_fuzzy, field, 'similarity')
                                     for field in fuzzy_field_names]]
                          .dot(fuzzy_weight_array)) + exact_field_contribution
                totals = pd.concat([totals], axis=1,
                                   keys=[('', '', title_total)])
            else:
                totals = (merged_df[[(title_fuzzy, field)
                                     for field in fuzzy_field_names]]
                          .dot(fuzzy_weight_array)) + exact_field_contribution
                totals = pd.concat([totals], axis=1, keys=[('', title_total)])
        else:
            fuzzy_weight_array /= fuzzy_weight_array.sum()
            if hierarchical:
                totals = (merged_df[[(field, 'similarity')
                                    for field in fuzzy_field_names]]
                          .dot(fuzzy_weight_array))
                totals = pd.concat([totals], axis=1, keys=[('', title_total)])
            else:
                totals = merged_df[fuzzy_field_names].dot(fuzzy_weight_array)
                totals.rename(title_total, inplace=True)
        return totals

    # Validate input data:
    if not (isinstance(data_frames, pd.DataFrame) or
            (isinstance(data_frames, list) and (len(data_frames) == 2) and
             isinstance(data_frames[0], pd.DataFrame) and
             isinstance(data_frames[1], pd.DataFrame))):
        raise ValueError('Parameter \'data_frames\' must be either a pandas '
                         'DataFrame or a list of two pandas DataFrames. ')

    if isinstance(data_frames, pd.DataFrame):
        if not(isinstance(fields_2b_matched_fuzzily, list) and
               all([(len(f) == 3 and isinstance(f[0], str) and
                     isinstance(f[1], Number) and
                     isinstance(f[2], dict))
                    for f in fields_2b_matched_fuzzily])):
            raise ValueError('When \'data_frames\' is a single DataFrame, '
                             '\'fields_2b_matched_fuzzily\' must be a list of '
                             'tuples.  Each tuple being a triple: '
                             '(<field name>, <weight>, <field_kwargs>).')
        if not(isinstance(fields_2b_matched_exactly, list) and
               all([(len(f) > 1 and isinstance(f[0], str) and
                     isinstance(f[1], Number))
                    for f in fields_2b_matched_exactly])):
            raise ValueError('When \'data_frames\' is a single DataFrame, '
                             '\'fields_2b_matched_exactly\' must be a list of '
                             'tuples.  Each tuple being a pair: '
                             '(<field name>, <weight>).')
    else:
        if not(isinstance(fields_2b_matched_fuzzily, list) and
               all([(len(f) == 4 and isinstance(f[0], str) and
                     isinstance(f[1], str) and
                     isinstance(f[2], Number) and
                     isinstance(f[3], dict))
                    for f in fields_2b_matched_fuzzily])):
            raise ValueError('When \'data_frames\' is a list of two '
                             'DataFrames, \'fields_2b_matched_fuzzily\' must '
                             'be a list of tuples.  Each tuple being a '
                             'quadruple: '
                             '(<field name1>, <field name2>, <weight>, '
                             '<field_kwargs>).')
        if not(isinstance(fields_2b_matched_exactly, list) and
               all([(len(f) > 2 and isinstance(f[0], str) and
                     isinstance(f[1], str) and isinstance(f[2], Number))
                    for f in fields_2b_matched_exactly])):
            raise ValueError('When \'data_frames\' is a list of two '
                             'DataFrames, \'fields_2b_matched_exactly\' must '
                             'be a list of tuples.  Each tuple being a '
                             'triple: '
                             '(<field name1>, <field name2>, <weight>).')

    # Validate given kwargs
    global_config = StringGrouperConfig(**kwargs)

    # Reference data sources
    if isinstance(data_frames, pd.DataFrame):
        data_frame1 = data_frames
        data_frame2 = None
    else:
        data_frame1 = data_frames[0]
        data_frame2 = data_frames[1]

    index_name_list1 = get_index_names(data_frame1)
    index_name_list2 = get_index_names(
        data_frame2) if data_frame2 is not None else index_name_list1
    match_indexes = prepend(
        index_name_list1, prefix='left_') + prepend(
            index_name_list2, prefix='right_')

    # Set the corpus for each fuzzy field (field-pair)
    fuzzy_field_grouper_pairs = []
    if data_frame2 is None:
        for field_name, _, sg_kwargs in fields_2b_matched_fuzzily:
            red_sg_kwargs = global_config._asdict()
            red_sg_kwargs.update(sg_kwargs)
            fuzzy_field_grouper_pairs += [(
                field_name,
                RedStringGrouper(
                    data_frame1[field_name],
                    **red_sg_kwargs))]
            fuzzy_field_grouper_pairs[-1][1].clear_data()
    else:
        for field_name1, field_name2, _, sg_kwargs \
                in fields_2b_matched_fuzzily:
            red_sg_kwargs = global_config._asdict()
            red_sg_kwargs.update(sg_kwargs)
            fuzzy_field_grouper_pairs += [(
                field_name1, field_name2,
                RedStringGrouper(
                    data_frame1[field_name1], data_frame2[field_name2],
                    **red_sg_kwargs))]
            fuzzy_field_grouper_pairs[-1][2].clear_data()

    fuzzy_field_names = get_field_names(fields_2b_matched_fuzzily)
    fuzzy_field_weights = get_field_weights(fields_2b_matched_fuzzily)

    if not fields_2b_matched_exactly:
        return horizontal_linkage(
            data_frame1,
            data_frame2,
            match_indexes,
            fuzzy_field_grouper_pairs,
            fuzzy_field_names,
            fuzzy_field_weights,
            hierarchical=hierarchical)
    else:
        exact_field1_names = get_field1_names(fields_2b_matched_exactly)
        groups1 = data_frame1.groupby(exact_field1_names, as_index=False)
        exact_field_weights = get_field_weights(fields_2b_matched_exactly)
        # group by exact fields:
        vertical_merger_list = []
        if data_frame2 is None:
            for group_label, group1_df in groups1:
                group_values = (list(group_label)
                                if isinstance(group_label, tuple)
                                else [group_label])
                exact_field_value_pairs = get_field_value_pairs(
                    exact_field1_names, None, group_values)
                vertical_merger_list += [
                    horizontal_linkage(
                        group1_df,
                        None,
                        match_indexes,
                        fuzzy_field_grouper_pairs,
                        fuzzy_field_names,
                        fuzzy_field_weights,
                        exact_field_value_pairs,
                        exact_field_weights,
                        hierarchical=hierarchical)]
        else:
            exact_field2_names = get_field2_names(fields_2b_matched_exactly)
            groups2 = data_frame2.groupby(exact_field2_names, as_index=False)
            # Remove field-name ambiguity by prefixing with different strings
            # for left and right operands of the following merge operation:
            left_ex_fields = prepend(exact_field1_names, 'left_')
            right_ex_fields = prepend(exact_field2_names, 'right_')
            merger = pd.merge(
                groups1.first().rename(columns={old: new for old, new in
                                                zip(exact_field1_names,
                                                    left_ex_fields)}),
                groups2.first().rename(columns={old: new for old, new in
                                                zip(exact_field2_names,
                                                    right_ex_fields)}),
                left_on=left_ex_fields, right_on=right_ex_fields)

            # iterate through the groups on both sides:
            for _, row in merger.iterrows():
                group_label = row[left_ex_fields].to_list()
                exact_field_value_pairs = get_field_value_pairs(
                    exact_field1_names, exact_field2_names,
                    group_label)
                group_label = (group_label[0] if len(group_label) == 1
                               else tuple(group_label))
                vertical_merger_list += [
                    horizontal_linkage(
                        groups1.get_group(group_label),
                        groups2.get_group(group_label),
                        match_indexes,
                        fuzzy_field_grouper_pairs,
                        fuzzy_field_names,
                        fuzzy_field_weights,
                        exact_field_value_pairs,
                        exact_field_weights,
                        hierarchical=hierarchical)]
        return pd.concat(vertical_merger_list)


DEFAULT_NGRAM_SIZE: int = 3
DEFAULT_TFIDF_MATRIX_DTYPE: type = np.float32   # (only types np.float32 and np.float64 are allowed by sparse_dot_topn)
DEFAULT_REGEX: str = r'[,-./]|\s'
DEFAULT_MAX_N_MATCHES: int = 20
DEFAULT_MIN_SIMILARITY: float = 0.8  # minimum cosine similarity for an item to be considered a match
DEFAULT_N_PROCESSES: int = multiprocessing.cpu_count() - 1
DEFAULT_IGNORE_CASE: bool = True  # ignores case by default
DEFAULT_DROP_INDEX: bool = False  # includes index-columns in output
DEFAULT_REPLACE_NA: bool = False    # when finding the most similar strings, does not replace NaN values in most
# similar string index-columns with corresponding duplicates-index values
DEFAULT_INCLUDE_ZEROES: bool = True  # when the minimum cosine similarity <=0, determines whether zero-similarity
# matches appear in the output
GROUP_REP_CENTROID: str = 'centroid'    # Option value to select the string in each group with the largest
# similarity aggregate as group-representative:
GROUP_REP_FIRST: str = 'first'  # Option value to select the first string in each group as group-representative:
DEFAULT_GROUP_REP: str = GROUP_REP_CENTROID  # chooses group centroid as group-representative by default
DEFAULT_FORCE_SYMMETRIES: bool = True  # Option value to specify whether corrections should be made to the results
# to account for symmetry thus compensating for those numerical errors that violate symmetry due to loss of
# significance
DEFAULT_N_BLOCKS: Tuple[int, int] = None  # Option value to use to split dataset(s) into roughly equal-sized blocks


class StringGrouperConfig(NamedTuple):
    r"""
    Class with configuration variables.

    :param ngram_size: int. The amount of characters in each n-gram. Default is 3.
    :param tfidf_matrix_dtype: type. The datatype for the tf-idf values of the matrix components.
    Possible values allowed by sparse_dot_topn are np.float32 and np.float64.  Default is np.float32.
    (Note: np.float32 often leads to faster processing and a smaller memory footprint albeit less precision
    than np.float64.)
    :param regex: str. The regex string used to cleanup the input string. Default is '[,-./]|\s'.
    :param max_n_matches: int. The maximum number of matching strings in master allowed per string in duplicates.
    Default is the total number of strings in master.
    :param min_similarity: float. The minimum cosine similarity for two strings to be considered a match.
    Defaults to 0.8.
    :param number_of_processes: int. The number of processes used by the cosine similarity calculation.
    Defaults to number of cores on a machine - 1.
    :param ignore_case: bool. Whether or not case should be ignored. Defaults to True (ignore case).
    :param ignore_index: whether or not to exclude string Series index-columns in output.  Defaults to False.
    :param include_zeroes: when the minimum cosine similarity <=0, determines whether zero-similarity matches
    appear in the output.  Defaults to True.
    :param replace_na: whether or not to replace NaN values in most similar string index-columns with
    corresponding duplicates-index values. Defaults to False.
    :param group_rep: str.  The scheme to select the group-representative.  Default is 'centroid'.
    The other choice is 'first'.
    :param force_symmetries: bool. In cases where duplicates is None, specifies whether corrections should be
    made to the results to account for symmetry, thus compensating for those losses of numerical significance
    which violate the symmetries. Defaults to True.
    :param n_blocks: (int, int) This parameter is provided to help boost performance, if possible, of
    processing large DataFrames, by splitting the DataFrames into n_blocks[0] blocks for the left
    operand (of the underlying matrix multiplication) and into n_blocks[1] blocks for the right operand
    before performing the string-comparisons block-wise.  Defaults to None.
    """

    ngram_size: int = DEFAULT_NGRAM_SIZE
    tfidf_matrix_dtype: int = DEFAULT_TFIDF_MATRIX_DTYPE
    regex: str = DEFAULT_REGEX
    max_n_matches: Optional[int] = None
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    number_of_processes: int = DEFAULT_N_PROCESSES
    ignore_case: bool = DEFAULT_IGNORE_CASE
    ignore_index: bool = DEFAULT_DROP_INDEX
    include_zeroes: bool = DEFAULT_INCLUDE_ZEROES
    replace_na: bool = DEFAULT_REPLACE_NA
    group_rep: str = DEFAULT_GROUP_REP
    force_symmetries: bool = DEFAULT_FORCE_SYMMETRIES
    n_blocks: Tuple[int, int] = DEFAULT_N_BLOCKS


class RedStringGrouper(StringGrouper):
    # This class enables StringGrouper to apply matching on different
    # successive master datasets without resetting the underlying corpus, as
    # long as each master dataset is contained in the corpus. 
    # This class inherits from StringGrouper, overwriting a few of its
    # methods: __init__(), _get_tf_idf_matrices(), _build_matches() and fit(). 
    def __init__(self, master: pd.Series,
                 duplicates: Optional[pd.Series] = None,
                 master_id: Optional[pd.Series] = None,
                 duplicates_id: Optional[pd.Series] = None,
                 **kwargs):
        """
        StringGrouper is a class that holds the matrix with cosine similarities between the master and duplicates
        matrix. If duplicates is not given it is replaced by master. To build this matrix the `fit` function must be
        called. It is possible to add and remove matches after building with the add_match and remove_match functions

        :param master: pandas.Series. A Series of strings in which similar strings are searched, either against itself
        or against the `duplicates` Series.
        :param duplicates: pandas.Series. If set, for each string in duplicates a similar string is searched in Master.
        :param master_id: pandas.Series. If set, contains ID values for each row in master Series.
        :param duplicates_id: pandas.Series. If set, contains ID values for each row in duplicates Series.
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig
        """
        # private members:
        self.is_build = False

        self._master: pd.DataFrame = pd.DataFrame()
        self._duplicates: Optional[pd.Series] = None
        self._master_id: Optional[pd.Series] = None
        self._duplicates_id: Optional[pd.Series] = None

        self._right_Series: pd.DataFrame = pd.DataFrame()
        self._left_Series: pd.DataFrame = pd.DataFrame()

        # After the StringGrouper is fit, _matches_list will contain the indices and similarities of the matches
        self._matches_list: pd.DataFrame = pd.DataFrame()
        # _true_max_n_matches will contain the true maximum number of matches over all strings in master if
        # self._config.min_similarity <= 0
        self._true_max_n_matches: int = 0
        self._max_n_matches: int = 0

        self._config: StringGrouperConfig = StringGrouperConfig(**kwargs)

        # initialize the members:
        self._set_data(master, duplicates, master_id, duplicates_id)
        self._set_options(**kwargs)
        self._build_corpus()

    def _set_data(self,
                  master: pd.Series,
                  duplicates: Optional[pd.Series] = None,
                  master_id: Optional[pd.Series] = None,
                  duplicates_id: Optional[pd.Series] = None):
        # Validate input strings data
        self.master = master
        self.duplicates = duplicates

        # Validate optional IDs input
        if not StringGrouper._is_input_data_combination_valid(duplicates, master_id, duplicates_id):
            raise Exception('List of data Series options is invalid')
        StringGrouper._validate_id_data(master, duplicates, master_id, duplicates_id)
        self._master_id = master_id
        self._duplicates_id = duplicates_id

        # Set some private members
        self._right_Series = self._master
        if self._duplicates is None:
            self._left_Series = self._master
        else:
            self._left_Series = self._duplicates

        self.is_build = False

    def _set_options(self, **kwargs):
        self._config = StringGrouperConfig(**kwargs)

        if self._config.max_n_matches is None:
            self._max_n_matches = len(self._master)
        else:
            self._max_n_matches = self._config.max_n_matches

        self._validate_group_rep_specs()
        self._validate_tfidf_matrix_dtype()
        self._validate_replace_na_and_drop()
        RedStringGrouper._validate_n_blocks(self._config.n_blocks)
        self.is_build = False

    def _build_corpus(self):
        self._vectorizer = TfidfVectorizer(min_df=1, analyzer=self.n_grams, dtype=self._config.tfidf_matrix_dtype)
        self._vectorizer = self._fit_vectorizer()
        self.is_build = False  # indicates if the grouper was fit or not

    def reset_data(self,
                   master: pd.Series,
                   duplicates: Optional[pd.Series] = None,
                   master_id: Optional[pd.Series] = None,
                   duplicates_id: Optional[pd.Series] = None):
        """
        Sets the input Series of a StringGrouper instance without changing the underlying corpus.
        :param master: pandas.Series. A Series of strings in which similar strings are searched, either against itself
        or against the `duplicates` Series.
        :param duplicates: pandas.Series. If set, for each string in duplicates a similar string is searched in Master.
        :param master_id: pandas.Series. If set, contains ID values for each row in master Series.
        :param duplicates_id: pandas.Series. If set, contains ID values for each row in duplicates Series.
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig
        """
        self._set_data(master, duplicates, master_id, duplicates_id)

    def clear_data(self):
        self._master = None
        self._duplicates = None
        self._master_id = None
        self._duplicates_id = None
        self._matches_list = None
        self._left_Series = None
        self._right_Series = None
        self.is_build = False

    def update_options(self, **kwargs):
        """
        Updates the kwargs of a StringGrouper object
        :param **kwargs: any StringGrouper keyword=value argument pairs
        """
        _ = StringGrouperConfig(**kwargs)
        old_kwargs = self._config._asdict()
        old_kwargs.update(kwargs)
        self._set_options(**old_kwargs)

    @property
    def master(self):
        return self._master

    @master.setter
    def master(self, master):
        if not StringGrouper._is_series_of_strings(master):
            raise TypeError('Master input does not consist of pandas.Series containing only Strings')
        self._master = master

    @property
    def duplicates(self):
        return self._duplicates

    @duplicates.setter
    def duplicates(self, duplicates):
        if duplicates is not None and not StringGrouper._is_series_of_strings(duplicates):
            raise TypeError('Duplicates input does not consist of pandas.Series containing only Strings')
        self._duplicates = duplicates

    def _fit_blockwise_manual(self, n_blocks=(1, 1)):
        # Function to compute matrix product by optionally first dividing
        # the DataFrames(s) into equal-sized blocks as much as possible.

        def divide_by(n, series):
            # Returns an array of n rows and 2 columns.
            # The columns denote the start and end of each of the n blocks.
            # Note: zero-indexing is implied.
            sz = len(series)//n
            block_rem = np.full(n, 0, dtype=np.int64)
            block_rem[:len(series) % n] = 1
            if sz > 0:
                equal_block_sz = np.full(n, sz, dtype=np.int64)
                equal_block_sz += block_rem
            else:
                equal_block_sz = block_rem[:len(series) % n]
            equal_block_sz = np.cumsum(equal_block_sz)
            equal_block_sz = np.tile(equal_block_sz, (2, 1))
            equal_block_sz[0, 0] = 0
            equal_block_sz[0, 1:] = equal_block_sz[1, :-1]
            return equal_block_sz.T

        block_ranges_left = divide_by(n_blocks[0], self._left_Series)
        block_ranges_right = divide_by(n_blocks[1], self._right_Series)

        self._true_max_n_matches = 0
        block_true_max_n_matches = 0
        vblocks = []
        for left_block in block_ranges_left:
            left_matrix = self._get_left_tf_idf_matrix(left_block)
            nnz_rows = np.full(left_matrix.shape[0], 0, dtype=np.int32)
            hblocks = []
            for right_block in block_ranges_right:
                right_matrix = self._get_right_tf_idf_matrix(right_block)
                try:
                    # Calculate the matches using the cosine similarity
                    # Note: awesome_cossim_topn will sort each row only when
                    # _max_n_matches < size of right_block or sort=True
                    matches, block_true_max_n_matches = self._build_matches(
                        left_matrix, right_matrix, nnz_rows, sort=(len(block_ranges_right) == 1)
                    )
                except OverflowError as oe:
                    import sys
                    raise (type(oe)(f"{str(oe)} Use the n_blocks parameter to split-up "
                                    f"the data into smaller chunks.  The current values"
                                    f"(n_blocks = {n_blocks}) are too small.")
                           .with_traceback(sys.exc_info()[2]))
                hblocks.append(matches)
                # end of inner loop

            self._true_max_n_matches = \
                max(block_true_max_n_matches, self._true_max_n_matches)
            if len(block_ranges_right) > 1:
                # Note: awesome_hstack_topn will sort each row only when
                # _max_n_matches < length of _right_Series or sort=True
                vblocks.append(
                    awesome_hstack_topn(
                        hblocks,
                        self._max_n_matches,
                        sort=True,
                        use_threads=self._config.number_of_processes > 1,
                        n_jobs=self._config.number_of_processes
                    )
                )
            else:
                vblocks.append(hblocks[0])
            del hblocks
            del matches
            # end of outer loop

        if len(block_ranges_left) > 1:
            return vstack(vblocks)
        else:
            return vblocks[0]

    def _fit_blockwise_auto(self,
                            left_partition=(None, None),
                            right_partition=(None, None),
                            nnz_rows=None,
                            sort=True,
                            whoami=0):
        # This is a recursive function!
        # fit() has been extended here to enable StringGrouper to handle large
        # datasets which otherwise would lead to an OverflowError
        # The handling is achieved using block matrix multiplication.
        def begin(partition):
            return partition[0] if partition[0] is not None else 0

        def end(partition, left=True):
            if partition[1] is not None:
                return partition[1]

            return len(self._left_Series if left else self._right_Series)

        left_matrix = self._get_left_tf_idf_matrix(left_partition)
        right_matrix = self._get_right_tf_idf_matrix(right_partition)

        if whoami == 0:
            # At the topmost level of recursion initialize nnz_rows
            # which will be used to compute _true_max_n_matches
            nnz_rows = np.full(left_matrix.shape[0], 0, dtype=np.int32)
            self._true_max_n_matches = 0

        try:
            # Calculate the matches using the cosine similarity
            matches, true_max_n_matches = self._build_matches(
                left_matrix, right_matrix, nnz_rows[slice(*left_partition)],
                sort=sort)
        except OverflowError:
            warnings.warn("An OverflowError occurred but is being "
                          "handled.  The input data will be automatically "
                          "split-up into smaller chunks which will then be "
                          "processed one chunk at a time.  To prevent "
                          "OverflowError, use the n_blocks parameter to split-up "
                          "the data manually into small enough chunks.")
            # Matrices too big!  Try splitting:
            del left_matrix, right_matrix

            def split_partition(partition, left=True):
                data_begin = begin(partition)
                data_end = end(partition, left=left)
                data_mid = data_begin + (data_end - data_begin)//2
                if data_mid > data_begin:
                    return [(data_begin, data_mid), (data_mid, data_end)]
                else:
                    return [(data_begin, data_end)]

            left_halves = split_partition(left_partition, left=True)
            right_halves = split_partition(right_partition, left=False)
            vblocks = []
            for lhalf in left_halves:
                hblocks = []
                for rhalf in right_halves:
                    # Note: awesome_cossim_topn will sort each row only when
                    # _max_n_matches < size of right_partition or sort=True
                    matches = self._fit_blockwise_auto(
                        left_partition=lhalf, right_partition=rhalf,
                        nnz_rows=nnz_rows,
                        sort=((whoami == 0) and (len(right_halves) == 1)),
                        whoami=(whoami + 1)
                    )
                    hblocks.append(matches)
                    # end of inner loop
                if whoami == 0:
                    self._true_max_n_matches = max(
                        np.amax(nnz_rows[slice(*lhalf)]),
                        self._true_max_n_matches
                    )
                if len(right_halves) > 1:
                    # Note: awesome_hstack_topn will sort each row only when
                    # _max_n_matches < length of _right_Series or sort=True
                    vblocks.append(
                        awesome_hstack_topn(
                            hblocks,
                            self._max_n_matches,
                            sort=(whoami == 0),
                            use_threads=self._config.number_of_processes > 1,
                            n_jobs=self._config.number_of_processes
                        )
                    )
                else:
                    vblocks.append(hblocks[0])
                del hblocks
                # end of outer loop
            if len(left_halves) > 1:
                return vstack(vblocks)
            else:
                return vblocks[0]

        if whoami == 0:
            self._true_max_n_matches = true_max_n_matches
        return matches

    def fit(self, force_symmetries=None, n_blocks=None):
        """
        Builds the _matches list which contains string-matches' indices and similarity
        Updates and returns the StringGrouper object that called it.
        """
        if force_symmetries is None:
            force_symmetries = self._config.force_symmetries
        RedStringGrouper._validate_n_blocks(n_blocks)
        if n_blocks is None:
            n_blocks = self._config.n_blocks

        # do the matching
        if n_blocks is None:
            matches = self._fit_blockwise_auto()
        else:
            matches = self._fit_blockwise_manual(n_blocks=n_blocks)

        # enforce symmetries?
        if force_symmetries and (self._duplicates is None):
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
        self.is_build = True
        return self

    def match_strings(self,
                      master: pd.Series,
                      duplicates: Optional[pd.Series] = None,
                      master_id: Optional[pd.Series] = None,
                      duplicates_id: Optional[pd.Series] = None,
                      **kwargs) -> pd.DataFrame:
        """
        Returns all highly similar strings without rebuilding the corpus.
        If only 'master' is given, it will return highly similar strings within master.
        This can be seen as an self-join. If both master and duplicates is given, it will return highly similar strings
        between master and duplicates. This can be seen as an inner-join.

        :param master: pandas.Series. Series of strings against which matches are calculated.
        :param duplicates: pandas.Series. Series of strings that will be matched with master if given (Optional).
        :param master_id: pandas.Series. Series of values that are IDs for master column rows (Optional).
        :param duplicates_id: pandas.Series. Series of values that are IDs for duplicates column rows (Optional).
        :param kwargs: All other keyword arguments are passed to StringGrouperConfig.
        :return: pandas.Dataframe.
        """
        self.reset_data(master, duplicates, master_id, duplicates_id)
        self.update_options(**kwargs)
        self = self.fit()
        return self.get_matches()

    def _get_left_tf_idf_matrix(self, partition=(None, None)):
        # unlike _get_tf_idf_matrices(), _get_left_tf_idf_matrix
        # does not set the corpus but rather
        # builds a matrix using the existing corpus
        return self._vectorizer.transform(
            self._left_Series.iloc[slice(*partition)])

    def _get_right_tf_idf_matrix(self, partition=(None, None)):
        # unlike _get_tf_idf_matrices(), _get_right_tf_idf_matrix
        # does not set the corpus but rather
        # builds a matrix using the existing corpus
        return self._vectorizer.transform(
            self._right_Series.iloc[slice(*partition)])

    def _fit_vectorizer(self) -> TfidfVectorizer:
        # if both dupes and master string series are set - we concat them to fit the vectorizer on all
        # strings
        if self._duplicates is not None:
            strings = pd.concat([self._master, self._duplicates])
        else:
            strings = self._master
        self._vectorizer.fit(strings)
        return self._vectorizer

    def _build_matches(self,
                       left_matrix: csr_matrix, right_matrix: csr_matrix,
                       nnz_rows: np.ndarray = None,
                       sort: bool = True) -> csr_matrix:
        """Builds the cossine similarity matrix of two csr matrices"""
        right_matrix = right_matrix.transpose()

        if nnz_rows is None:
            nnz_rows = np.full(left_matrix.shape[0], 0, dtype=np.int32)

        optional_kwargs = {
            'return_best_ntop': True,
            'sort': sort,
            'use_threads': self._config.number_of_processes > 1,
            'n_jobs': self._config.number_of_processes}

        return awesome_cossim_topn(
            left_matrix, right_matrix,
            self._max_n_matches,
            nnz_rows,
            self._config.min_similarity,
            **optional_kwargs)

    def _get_matches_list(self,
                          matches: csr_matrix
                          ) -> pd.DataFrame:
        """Returns a list of all the indices of matches"""
        r, c = matches.nonzero()
        d = matches.data
        return pd.DataFrame({'master_side': c.astype(np.int64),
                             'dupe_side': r.astype(np.int64),
                             'similarity': d})

    @staticmethod
    def _validate_n_blocks(n_blocks):
        errmsg = "Invalid option value for parameter n_blocks: "
        "n_blocks must be None or a tuple of 2 integers greater than 0."
        if n_blocks is None:
            return
        if not isinstance(n_blocks, tuple):
            raise Exception(errmsg)
        if len(n_blocks) != 2:
            raise Exception(errmsg)
        if not (isinstance(n_blocks[0], int) and isinstance(n_blocks[1], int)):
            raise Exception(errmsg)
        if (n_blocks[0] < 1) or (n_blocks[1] < 1):
            raise Exception(errmsg)
