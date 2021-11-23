from numbers import Number
from typing import List
import functools
import re

import pandas as pd
import numpy as np

from string_grouper import StringGrouper, StringGrouperConfig


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
        return matches[functools.reduce(
            lambda a, b: a | b,
            [(matches.index.get_level_values(i) !=
              matches.index.get_level_values(i + num_indexes))
              for i in range(num_indexes)])]

    def build_column_precursor_to(df, exact_field_value_pairs):
        exact_df = df.iloc[:, 0:0]
        exact_df = exact_df.assign(
            **{field_name: field_value
               for field_name, field_value in exact_field_value_pairs})
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
    index_name_list2 = (get_index_names(data_frame2) 
                        if data_frame2 is not None 
                        else index_name_list1)
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


class RedStringGrouper(StringGrouper):
    def n_grams(self, string: str) -> List[str]:
        """
        :param string: string to create ngrams from
        :return: list of ngrams
        """
        regex_pattern = self._config.regex
        ngram_size = self._config.ngram_size
        if isinstance(ngram_size, int):
            ngram_size = [ngram_size]
        if self._config.ignore_case and string is not None:
            string = string.lower()  # lowercase to ignore all case
        string = re.sub(regex_pattern, r'', string)
        out = []
        for sz in ngram_size:
            n_grams = zip(*[string[i:] for i in range(sz)])
            out.extend([''.join(n_gram) for n_gram in n_grams])
        return out
