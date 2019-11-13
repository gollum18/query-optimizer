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
        if _k == 1:
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

    def __init__(self, page_size=4096, block_size=100):
        """Returns a new instance of a QueryOptimizer.

        Arguments:
            page_size (int): The maximum page size in bytes, i.e. the number
            total amount of memory per page for holding a continguous
            collection of tuples.
            block_size (int): The maximum amount of pages that can be read into
            a single block of memory, i.e. the buffer size.
        """
        self.page_size = page_size
        self.block_size = block_size
        self.join_functions = {
            "TNL": (self.calc_tuple_nested_join_cost, None),
            "PNL": (self.calc_page_nested_join_cost, None),
            "BNJM": (self.calc_block_nested_join_cost, [50]),
            "BNJL": (self.calc_block_nested_join_cost, [30]),
            "SMJM": (self.calc_sort_merge_join_cost, [50]),
            "SMJL": (self.calc_sort_merge_join_cost, [30]),
            "HJM": (self.calc_hash_join_cost, None)
        }


    def calc_selection_cost(self, tables, table):
        """Calculates the I/O cost for applying a selection operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            select (filter) from.
            table (str): The name of the table to select.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        if not tables or not table:
            print('Selection cost operation received bad argument, cannot continue!')
            sys.exit(-1)
        if table not in tables:
            print('Selection table not found, cannot continue!')
            sys.exit(-1)
        indexed, sorted = None, None
        try:
            indexed, sorted = tables[table]['indexed'], tables[table]['sorted']
        except KeyError:
            print('Unable to determine if table is indexed and sorted, cannot continue!')
            sys.exit(-1)
        if indexed:
            raise NotImplementedError
        elif not indexed and sorted:
            return int(math.ceil(math.log2(tables[table]['num_pages'])))
        else:
            return tables[table]['num_pages']


    def calc_projection_cost(self, tables, table):
        """Calculates the I/O cost for applying a projection operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            project from.
            table (str): The name of the table to project.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        if not tables or not table:
            print('Projection cost operation received bad argument, cannot continue!')
            sys.exit(-1)
        if table not in tables:
            print('Selection table not found, cannot continue!')
            sys.exit(-1)
        indexed, sorted = None, None
        try:
            indexed, sorted = tables[table]['indexed'], tables[table]['sorted']
        except KeyError:
            print('Unable to determine if table is indexed and sorted, cannot continue!')
            sys.exit(-1)
        num_pages = tables[table]['num_pages']
        if sorted:
            return num_pages
        else:
            return num_pages * int(math.ceil(math.log2(num_pages)))


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
        if not tables or not outer_table or not inner_table:
            print('Tuple nested join operation received bad arguments, cannot continue!')
            sys.exit(-1)
        if outer_table not in tables:
            print('Outer table not found, cannot continue!')
            sys.exit(-1)
        if inner_table not in tables:
            print('Inner table not found, cannot continue!')
            sys.exit(-1)
        outer_table_pages = tables[outer_table]['num_pages']
        inner_table_pages = tables[inner_table]['num_pages']
        outer_tuples_per_page = QueryOptimizer.tuples_per_page(
            self.page_size,
            outer_table_pages
        )
        return (
            outer_table_pages +
            (outer_tuples_per_page * outer_table_pages) *
            inner_table_pages
        )


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
        if not tables or not outer_table or not inner_table:
            print('Tuple nested join operation received bad arguments, cannot continue!')
            sys.exit(-1)
        if outer_table not in tables:
            print('Outer table not found, cannot continue!')
            sys.exit(-1)
        if inner_table not in tables:
            print('Inner table not found, cannot continue!')
            sys.exit(-1)
        outer_table_pages = tables[outer_table]['num_pages']
        inner_table_pages = tables[inner_table]['num_pages']
        return outer_table_pages + outer_table_pages * inner_table_pages


    def calc_block_nested_join_cost(self, tables, outer_table, inner_table, block_size):
        """Calculates the I/O cost for applying a block nested join operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            join.
            outer_table (str): The name of the outer table.
            inner_table (str): The name of the inner table.
            block_size (int): The number of blocks to store in the buffer at
            once.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        if not tables or not outer_table or not inner_table or not block_size:
            print('Block nested join operation received bad arguments, cannot continue!')
            sys.exit(-1)
        if outer_table not in tables:
            print('Outer table not found, cannot continue!')
            sys.exit(-1)
        if inner_table not in tables:
            print('Inner table not found, cannot continue!')
            sys.exit(-1)
        if block_size <= 0:
            print('Block size invalid, cannot continue!')
            sys.exit(-1)
        outer_table_pages = tables[outer_table]['num_pages']
        inner_table_pages = tables[inner_table]['num_pages']
        outer_blocks = int(math.ceil(outer_table_pages / block_size))
        return (
            outer_table_pages +
            outer_blocks *
            inner_table_pages
        )


    def calc_sort_merge_join_cost(self, tables, left_table, right_table,
        block_size):
        """Calculates the I/O cost for applying a sort merge join operation.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to
            join.
            left_table (str): The name of the left table.
            right_table (str): The name of the right table.
            block_size (int): The number of blocks to store in the buffer at
            once.

        Returns:
            (int): The number of estimated I/O operations needed to fully apply
            the operation.
        """
        if not tables or not left_table or not right_table or not block_size:
            print('Sort merge join operation received bad arguments, cannot continue!')
            sys.exit(-1)
        if left_table not in tables:
            print('Left table not found, cannot continue!')
            sys.exit(-1)
        if right_table not in tables:
            print('Right table not found, cannot continue!')
            sys.exit(-1)
        if block_size <= 0:
            print('Block size cannot be <= 0, cannot continue!')
            sys.exit(-1)
        # TODO: Handle indexing and clustering cases
        left_table_pages = tables[left_table]['num_pages']
        right_table_pages = tables[right_table]['num_pages']
        if block_size > math.sqrt(max(left_table_pages, right_table_pages)):
            return 3 * (left_table_pages + right_table_pages)
        else:
            return (
                (left_table_pages * int(math.ceil(math.log(left_table_pages, block_size-1)))) +
                (right_table_pages * int(math.ceil(math.log(right_table_pages, block_size-1)))) +
                (left_table_pages + right_table_pages)
            )
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
        if not tables or not left_table or not right_table:
            print('Hash join operation received bad arguments, cannot continue!')
            sys.exit(-1)
        if left_table not in tables:
            print('Left table not found, cannot continue!')
            sys.exit(-1)
        if right_table not in tables:
            print('Right table not found, cannot continue!')
            sys.exit(-1)
        # TODO: Handle indexing and clustering cases
        left_table_pages = tables[left_table]['num_pages']
        right_table_pages = tables[right_table]['num_pages']
        return 3 * (left_table_pages + right_table_pages)


    def generate_exec_plan(self, query_file):
        """Determines the best possible execution plan for a query assuming
        the assumptions listed at the top of this file.

        Arguments:
            (query_file): The path to a file containing a query.

        Returns:
            (QueryPlan): A QueryPlan object.
        """
        qdict = read_query_file(query_file)
        stats = qdict['statistics']
        tables = stats['tables']
        projectivity = stats['projectivity']
        tnames = list(tables.keys())
        print(tnames)
        for key, value in self.join_functions.items():
            func = value[0]
            args = value[1]
            for p in permute(tnames):
                if args:
                    cost = func(tables, tnames[0], tnames[1], *args)
                else:
                    cost = func(tables, tnames[0], tnames[1])
                print(key, cost)


    @staticmethod
    def tuples_per_page(page_size, tuple_size):
        if tuple_size == 0:
            return 0
        return int(math.ceil(page_size / tuple_size))


def main():
    qo = QueryOptimizer()
    qo.generate_exec_plan('./q1.json')


if __name__ == '__main__':
    main()
