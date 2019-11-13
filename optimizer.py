#! /usr/bin/env python

"""
Name: optimizer.py
Since: 11/12/2019
Author: Christen Ford
Purpose: Implements a simple query optimizer that generates all possible
execution plans for a query and picks the best execution plan.

Assumptions:
    1.) Assume that queries are stored as a list of steps in a flat file format.
    2.) Assume that a query rewriter has already rewritten a correlated query Q
    to a non-correlated query Q' if a decorrelation algorithm can be applied.
    3.) Queries, their statistics, and the associated file system statistics
    are all stored in a 'query'.json file where 'query' is the name of the
    query. This contains all of the necessary statistical information to
    compute the cost of executing a query and additionally contains the query
    itself.
"""

import math, os, sys
import simplejson as json

def permute(items):
    """Permutes a list of items.

    Arguments:
        items (list): A list of items to permute.

    Returns:
        (Generator): A generator that yields permutations of 'items' if 'items'
        is not empty, otherwise items.
    """

    def _permute(_items, _k):
        """Internal permutation method that implemented Heaps' algorithm for
        efficiently generating permutations. Permutations are performed in
        place and yielded as such. Yielded lists should not be modified and
        should be treated as read-only.

        Arguments:
            _items (list): A list of items to permute.
            _k (int): The permutation index used to control Heaps' algorithm.
        """
        if k == 1:
            yield _items
        else:
            for i in range(_k):
                yield from _permute(_items, _k-1)
                if i < _k - 1:
                    if _k % 2 == 0:
                        _items[i], _items[_k-1] = _items[_k-1], _items[i]
                    else:
                        _items[0], _items[_k-1] = _items[_k-1], _items[0]

    if not items:
        return items

    yield from _permute(items, len(items))


def read_query_file(query_file):
    """Parses a JSON query file and returns the contained JSON as a Python
    dictionary.

    Arguments:
        query_file (str): The path to the query file.

    Returns:
        (dict): A Python dictionary containing contents of the query file.
    """
    try:
        if not os.path.exists(query_file):
            raise OSError
        qdict = None
        with open(query_file) as f:
            qdict = json.load(f)
        if not isinstance(qdict, dict):
            raise OSError
        return qdict
    except OSError:
        print('OSError: There was an error parsing the query file, can not continue!')
        sys.exit(-1)


class QueryPlan(object):
    """Implements a data storage class for storing a query plan and related
    information.
    """

    def __init__(self, query_name, has_subquery, is_correlated, join_method,
        join_order, total_cost, query_plan):
        """Returns a new instace of a query plan.

        Arguments:
            query_name (str):
            has_subquery (boolean):
            is_correlated (boolean):
            join_method (str):
            join_order ():
            total_cost (int):
            query_plan ():
        """
        self.query_name = query_name
        self.has_subquery = has_subquery
        self.is_correlated = is_correlated
        self.join_method = join_method
        self.join_order = join_order
        self.total_cost = total_cost
        self.query_plan = query_plan


class QueryOptimizer(object):
    """Implements a query optimzer that expects SQL queries in a simplified
    format specific to the design of this project.
    """

    def __init__(self):
        raise NotImplementedError


    def calc_selection_cost(self, tables, selectivity, table):
        """Calculates the I/O cost for applying a selection operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            select (filter) from.
            selectivity (dict): A dictionary containing the selectivity rates
            of various predicates between the tables
            table (str): The name of the table to select.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        raise NotImplementedError


    def calc_projection_cost(self, tables, projectivity, table):
        """Calculates the I/O cost for applying a projection operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            project from.
            projectivity (float): The average percentage of records that will
            be projected via the projection operator.
            table (str): The name of the table to project.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        raise NotImplementedError


    def calc_tuple_nested_join_cost(self, tables, outer_table, inner_table):
        """Calculates the I/O cost for applying a tuple nested join operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            join.
            outer_table (str): The name of the outer table.
            inner_table (str): The name of the inner table.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        raise NotImplementedError


    def calc_page_nested_join_cost(self, tables, outer_table, inner_table):
        """Calculates the I/O cost for applying a page nested join operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            join.
            outer_table (str): The name of the outer table.
            inner_table (str): The name of the inner table.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        raise NotImplementedError


    def calc_block_nested_join_cost(self, tables, outer_table, inner_table):
        """Calculates the I/O cost for applying a block nested join operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            join.
            outer_table (str): The name of the outer table.
            inner_table (str): The name of the inner table.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        raise NotImplementedError


    def calc_sort_merge_join_cost(self, tables, left_table, right_table):
        """Calculates the I/O cost for applying a sort merge join operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            join.
            left_table (str): The name of the left table.
            right_table (str): The name of the right table.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        raise NotImplementedError


    def calc_hash_join_cost(self, tables, left_table, right_table):
        """Calculates the I/O cost for applying a hash join operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            join.
            left_table (str): The name of the left table.
            right_table (str): The name of the right table.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        raise NotImplementedError


    def generate_exec_plan(self, query_file):
        """Determines the best possible execution plan for a query assuming
        the assumptions listed at the top of this file.

        Arguments:
            (query_file): The path to a file containing a query.

        Returns:
            (QueryPlan): A QueryPlan object.
        """
        qdict = read_query_file(query_file)

        raise NotImplementedError


    @staticmethod
    def tuples_per_page(page_size, tuple_size):
        if tuple_size == 0:
            return 0
        return int(math.ceil(page_size / tuple_size))


def main():
    raise NotImplementedError


if __name__ == '__main__':
    main()
