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
    4.) Assume that each query file contains information on whether the tables
    involved in the query are indexed, clustered, and sorted.

TODO:
    1.) All of the join methods need to also consider cases where indexing,
    sorting, and clustering are in effect.
"""

import traceback
import math, os, sys
import simplejson as json

index_type_key = 'index_type'
clustered_key = 'clustered'
indexed_key = 'indexed'
sorted_key = 'sorted'
# assume that the tuple size for an aggregate operation is 10
aggr_tuple_size = 10


def create_table(name, num_pages, tuple_size, indexed=False, index_type=None, ikey_type=None, clustered=False, sorted=False):
    """
    Creates a table for use by the query optimizer. A table is just a nested dictionary object.

    Arguments:
        name (str): The name of the table n.
        num_pages (int): The number of records in the table p | p > 0 ^ p < inf.
        tuple_size (int): The size of a single tuple in the table u | u > 0 ^ u < inf.
        indexed (boolean): Whether the data in the table is indexed or not d | d E {True, False}.
        index_type (str): The index type of the table t | t E {null, bpt_index, hash_index}.
        ikey_type (str): The key type of the index i | i E {null, primary, secondary, composite}.
        clustered (boolean): Whether the index is clustered or not c | c E {True, False}.
        sorted (boolean): Whether the table is sorted or not s | s E {True, False}.

    Returns:
        A (key, dict) pair.
    """
    return (name, {
        'num_pages': num_pages,
        'tuple_size': tuple_size,
        'indexed': indexed,
        'index_type': index_type,
        'ikey_type': ikey_type,
        'clustered': clustered,
        'sorted': sorted
    })


def permute(items):
    """Permutes a list of items.

    Arguments:
        items (list): A list of items to permute.

    Returns:
        (Generator): A generator that yields permutations of 'items' if 'items'
        is not empty, otherwise items.
    """

    def _permute(_items, _k):
        """Internal permutation method that implements Heaps' algorithm for
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
        join_order, total_cost, timestamp):
        """Returns a new instace of a query plan.

        Arguments:
            query_name (str): The name of the query.
            has_subquery (boolean): Whether the query has a subquery or not.
            is_correlated (boolean): Whether the sub-query is correlated or not.
            join_method (list): A list of join methods.
            join_order (list): A list of join orders.
            total_cost (int): The total cost of running the query plan.
        """
        self.query_name = query_name
        self.has_subquery = has_subquery
        self.is_correlated = is_correlated
        self.join_method = join_method
        self.join_order = join_order
        self.total_cost = total_cost
        self.timestamp = timestamp

    def __str__(self):
        name = "Query: {}\n\t".format(self.query_name)
        subquery = "Has Subquery: {}\n\t".format(self.has_subquery)
        correlated = "Has Correlation: {}\n\t".format(self.is_correlated)
        join_str = "Join Orders:\n\t\t"
        for i in range(len(self.join_order)):
            if i == len(self.join_order) - 1:
                join_str += "{}: {}\n\t".format(self.join_method[i], self.join_order[i])
            else:
                join_str += "{}: {}\n\t\t".format(self.join_method[i], self.join_order[i])
        total_cost = "Total Cost: {}\n\t".format(self.total_cost)
        exec_time = "Execution Time: {}".format(self.timestamp)
        return name + subquery + correlated + join_str + total_cost + exec_time


class QueryOptimizer(object):
    """Implements a query optimzer that expects SQL queries in a simplified
    format specific to the design of this project.
    """

    def __init__(self, page_size=4096, block_size=100, avg_seek_time=8, avg_latency=4):
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
        self.avg_seek_time = avg_seek_time
        self.avg_latency = avg_latency
        self.join_functions = dict()
        self._table_counter = 1

    def cost_to_timestamp(self, total_cost):
        """Converts an I/O cost metric to a timestamp containing three elements:
        hours, minutes, seconds.

        Arguments:
            cost (int): The I/O cost to convert to a timestamp.
            avg_seek_time (int): The average seek time of the disk in milliseconds.
            avg_latency (int): The average latency of the disk in milliseconds.

        Note:
            'avg_seek_time' and 'avg_latency' should be integers. Internally, they
            are both divided by 1000.
        """
        if not total_cost or not self.avg_seek_time or not self.avg_latency:
            raise ValueError
        if total_cost == 0:
            return 0, 0, 0
        seconds = total_cost
        minutes, hours = 0, 0
        while seconds >= 60:
            seconds -= 60
            minutes += 1
            if minutes == 60:
                hours += 1
                minutes = 0
        return "{:02d}h {:02d}m {:02d}s".format(hours, minutes, seconds)

    def add_join_scenario(self, key, join_function, args):
        """
        Adds a join scenario to the optimizer.
        :param key: The key for the join scenario -> uniquely identifies scenario.
        :param join_function: The join calculation function to call.
        :param args: A list of arguments to pass to join_function.
        """
        self.join_functions[key] = (join_function, args)

    def clear_join_scenarios(self):
        """
        Removes all join scenarios from the optimizer.
        """
        self.join_functions.clear()

    def delete_join_scenario(self, key):
        """
        Removes a join scenario from the optimizer.
        :param key: The key of the join scenario to remove.
        """
        if key in self.join_functions:
            del self.join_functions[key]

    def calc_aggregation_cost(self, tables, table, groupby=True):
        """ Calculates the cost of performing an aggregation in terms of Disk I/O.

        Arguments:
            tables (dict): A dictionary containing statistics on the tables to aggregate.
            table (str): The table to aggregate.
            groupby (boolean): Whether to perform a groupby or not.

        Returns:
             (int): The number of estimated I/O operations to fully apply the aggregation
             operator on the indicated table.
        """
        if not tables or not table:
            print('Aggregation operation received bad argument, cannot continue!')
            sys.exit(-1)
        if table not in tables:
            print('Aggregation table not found, cannot continue!')
            sys.exit(-1)
        try:
            sorted = tables[table]['sorted']
        except KeyError:
            print('Unable to determine if table is sorted, cannot continue!')
            sys.exit(-1)
        # if the prior step is a join with sort-merge, then there is no need to factor in sorting
        num_pages = tables[table]['num_pages']
        if sorted:
            return num_pages
        else:
            if groupby:
                return (num_pages * math.log(num_pages, 2)) + num_pages
            else:
                return num_pages

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
            print('Unable to determine if table is indexed or sorted, cannot continue!')
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
            print('Projection table not found, cannot continue!')
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
            if sorted_key in tables[left_table] and sorted_key in tables[right_table]:
                if tables[left_table][sorted_key] and tables[right_table][sorted_key]:
                    return 2 * (left_table_pages, right_table_pages)
        big_table = max(left_table_pages, right_table_pages)
        little_table = min(left_table_pages, right_table_pages)
        return 2 * (big_table + little_table) * (1 + int(math.ceil(
            math.log(big_table / block_size, block_size - 1)
        )))

    def calc_hash_join_cost(self, tables, left_table, right_table, block_size):
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
        return 2 * (left_table_pages + right_table_pages) * (1 + int(math.ceil(
            math.log(left_table_pages/(block_size-1))
        )))

    def _generate_table_name(self, id=None):
        """Generates a table name for a temporary table.

        Arguments:
            id (int): The id to use for the table.

        Returns:
             (str): A unique table name for a temporary table.
        """
        if id and not isinstance(id, int):
            raise ValueError

        if id:
            tname = 'temp{}'.format(id)
        else:
            tname = 'temp{}'.format(self._table_counter)
            self._table_counter += 1
        return tname

    def _handle_select(self, key, stats, query):
        """Handles a select operation for the query optimizer.

        Arguments:
            stats (dict): A dictionary containing statistics on tables.
            query (dict): A dictionary containing query steps.

        Returns:
            (int): An estimated cost for I/O.
        """
        if not ('select' in key):
            raise KeyError
        return self.calc_selection_cost(stats['tables'], query[key])

    def _handle_project(self, key, stats, query):
        """Handles a project operation for the query optimizer.

        Arguments:
            stats (dict): A dictionary containing statistics on tables.
            query (dict): A dictionary containing query steps.

        Returns:
            (int): An estimated cost for I/O.
        """
        if not ('project' in key):
            raise KeyError
        table = query['project']
        tname = self._generate_table_name()
        project_table = create_table(
            name=tname,
            num_pages=(
                stats['tables'][table]['num_pages'] *
                stats['projectivity']
            ),
            tuple_size=(
                stats['tables'][table]['tuple_size']
            )
        )
        stats['tables'][project_table[0]] = project_table[1]
        return self.calc_projection_cost(stats['tables'], query[key])

    def _handle_join(self, key, stats, query):
        """Handles a join operation for the query optimizer.

        Arguments:
            stats (dict): A dictionary containing statistics on tables.
            query (dict): A dictionary containing query steps.

        Returns:
        """
        if not ('join' in key):
            raise KeyError

        best_join_order = None
        best_join_method = None
        best_join_func = None
        best_join_cost = float('inf')

        join_tables = query[key][0]
        join_selectivity = query[key][1]

        for order in permute(join_tables):
            for method, params in self.join_functions.items():
                join_func = params[0]
                join_args = params[1]
                if join_args:
                    cost = join_func(stats['tables'], order[0], order[1], *join_args)
                else:
                    cost = join_func(stats['tables'], order[0], order[1])
                if cost < best_join_cost:
                    best_join_order = order
                    best_join_method = method
                    best_join_func = join_func
                    best_join_cost = cost

        tname = self._generate_table_name()
        join_table = create_table(
            name=tname,
            num_pages=(
                int(
                    math.ceil(
                        stats['tables'][best_join_order[0]]['num_pages'] *
                        stats['tables'][best_join_order[1]]['num_pages'] *
                        stats['selectivity'][join_selectivity]
                    )
                )
            ),
            tuple_size=(
                stats['tables'][best_join_order[0]]['tuple_size'] +
                stats['tables'][best_join_order[1]]['tuple_size']
            ),
            sorted=True if best_join_func == self.calc_sort_merge_join_cost else False
        )
        stats['tables'][join_table[0]] = join_table[1]

        return {
            'order': best_join_order,
            'method': best_join_method,
            'cost': best_join_cost
        }

    def _handle_correlate(self, key, stats, query):
        """Handles a correlate operation for the query optimizer.

        Note that EACH relational algebra step is performed PER tuple of
        the outer correlation table. Prior implementations of this
        algorithm only considered running join operations per tuple of the
        outer table, which is incorrect. In a correlated query, each step
        of the correlated subquery is performed per tuple of the outer table.

        Arguments:
            stats (dict): A dictionary containing statistics on tables.
            query (dict): A dictionary containing query steps.

        Returns:
            (int): An estimated cost for I/O.
        """
        if not ('correlate' in key):
            raise KeyError
        correlation = query[key]
        total_cost = 0
        join_order = correlation['join']
        outer_table_pages = stats['tables'][join_order[0]]['num_pages']
        for step in correlation['steps']:
            if step == 'select':
                total_cost += int(
                    math.ceil(
                        (
                            outer_table_pages*
                            self.calc_selection_cost(stats['tables'], correlation[step])
                        )
                    )
                )
                tname = self._generate_table_name()
                table = create_table(
                    name=tname,
                    num_pages=(
                        int(
                            math.ceil(
                                stats['tables'][step]['num_pages'] *
                                stats['selectivity']['default']
                            )
                        )
                    ),
                    tuple_size=(
                            stats['tables'][step]['tuple_size']
                    )
                )
                stats['tables'][table[0]] = table[1]
            elif step == 'join':
                join_cost = (
                    outer_table_pages *
                    self.calc_tuple_nested_join_cost(stats['tables'], *join_order)
                )
                tname = self._generate_table_name()
                table = create_table(
                    name=tname,
                    num_pages=(
                        outer_table_pages *
                        stats['tables'][join_order[0]]['num_pages'] *
                        stats['tables'][join_order[1]]['num_pages']
                    ),
                    tuple_size=(
                        stats['tables'][join_order[0]]['tuple_size'] +
                        stats['tables'][join_order[1]]['tuple_size']
                    )
                )
                for s in correlation['selectivity']:
                    table[1]['num_pages'] = table[1]['num_pages'] * stats['selectivity'][s]
                table[1]['num_pages'] = int(math.ceil(table[1]['num_pages']))
                stats['tables'][table[0]] = table[1]
                total_cost += join_cost
            elif step == 'project':
                total_cost += self.calc_projection_cost(stats['tables'], correlation[step])
                tname = self._generate_table_name()
                table = create_table(
                    name=tname,
                    num_pages=(
                        stats['tables'][correlation[step]]['num_pages']
                    ),
                    tuple_size=(
                        stats['tables'][correlation[step]]['tuple_size']
                    )
                )
                stats['tables'][table[0]] = table[1]
            elif step == 'aggr_no_groupby':
                total_cost += int(
                    math.ceil(
                        outer_table_pages *
                        self.calc_aggregation_cost(stats['tables'], correlation[step], False)
                    )
                )
            elif step == 'aggr_with_groupby':
                total_cost += int(
                    math.ceil(
                        outer_table_pages *
                        self.calc_aggregation_cost(stats['tables'], correlation[step], True)
                    )
                )
        return {'cost': total_cost, 'join_order': join_order}

    def _handle_aggregate(self, key, stats, query, group_by):
        """Handles an aggregate operation for the query optimizer.

        Arguments:
            stats (dict): A dictionary containing statistics on tables.
            query (dict): A dictionary containing query steps.

        Returns:
            (int): An estimated cost for I/O.
        """
        if not ('aggr_with_groupby' in key or not 'aggr_no_groupby' in key):
            raise KeyError
        table = query[key]
        tname = self._generate_table_name()
        if group_by:
            num_pages = (
                int(
                    math.ceil(
                        stats['tables'][table]['num_pages'] *
                        stats['projectivity']
                    )
                )
            )
        else:
            num_pages = stats['tables'][table]
        aggr_table = create_table(
            name=tname,
            num_pages=num_pages,
            tuple_size=aggr_tuple_size,
            sorted=True
        )
        stats['tables'][aggr_table[0]] = aggr_table[1]
        return self.calc_aggregation_cost(
            stats['tables'], query[key], group_by
        )

    def generate_query_plan(self, filename):
        """Determines the best possible execution plan for a query assuming
        the assumptions listed at the top of this file.

        Arguments:
            filename (str): The path to a file containing a query.

        Returns:
            (QueryPlan): A QueryPlan object.
        """
        self._table_counter = 1

        from queue import Queue

        try:
            query_file = read_query_file(filename)
            stats = query_file['statistics']
            query = query_file['query']
            query_steps = query['steps']

            step_queue = Queue()
            for qstep in query_steps:
                frame_key = qstep
                if 'select' in qstep:
                    frame_func = self._handle_select
                    frame_args = [stats, query]
                elif 'project' in qstep:
                    frame_func = self._handle_project
                    frame_args = [stats, query]
                elif 'join' in qstep:
                    frame_func = self._handle_join
                    frame_args = [stats, query]
                elif 'correlate' in qstep:
                    frame_func = self._handle_correlate
                    frame_args = [stats, query]
                elif 'aggr_no_groupby' in qstep:
                    frame_func = self._handle_aggregate
                    frame_args = [stats, query, False]
                elif 'aggr_with_groupby' in qstep:
                    frame_func = self._handle_aggregate
                    frame_args = [stats, query, True]
                else:
                    print('Error: Key \'{}\' not supported, cannot continue...'.format(qstep))
                    sys.exit(-1)
                step_queue.put_nowait({
                    'key': frame_key,
                    'func': frame_func,
                    'args': frame_args
                })

            best_join_order = []
            best_join_method = []
            corr_join_order = None
            total_cost = 0
            while not step_queue.empty():
                frame = step_queue.get_nowait()
                key = frame['key']
                func = frame['func']
                args = frame['args']
                lvalue = func(key, *args)
                if 'correlate' in key:
                    corr_join_order = lvalue['join_order']
                    total_cost += lvalue['cost']
                elif 'join' in key:
                    best_join_order.append(lvalue['order'])
                    best_join_method.append(lvalue['method'])
                    total_cost += lvalue['cost']
                else:
                    total_cost += lvalue

            total_cost = int(math.ceil(total_cost*(self.avg_latency/1000)*(self.avg_seek_time/1000)))

            return QueryPlan(
                query_name=filename.strip('.json'),
                has_subquery=True if corr_join_order else False,
                is_correlated=True if corr_join_order else False,
                join_method=best_join_method,
                join_order=best_join_order,
                total_cost=total_cost,
                timestamp=self.cost_to_timestamp(total_cost)
            )

        except KeyError as e:
            print('KeyError: Query file is malformed, cannot continue. Key {} not found...'.format(e))
            traceback.print_exc()
            sys.exit(-1)

    @staticmethod
    def tuples_per_page(page_size, tuple_size):
        if tuple_size == 0:
            return 0
        return int(math.ceil(page_size / tuple_size))


def main():
    qo = QueryOptimizer()
    join_scenarios = [
        ["TNL", qo.calc_tuple_nested_join_cost, None],
        ["PNL", qo.calc_page_nested_join_cost, None],
        ["BNJM", qo.calc_block_nested_join_cost, [50]],
        ["BNJL", qo.calc_block_nested_join_cost, [30]],
        ["SMJM", qo.calc_sort_merge_join_cost, [50]],
        ["SMJL", qo.calc_sort_merge_join_cost, [30]],
        ["HJM", qo.calc_hash_join_cost, [50]],
        ["HJL", qo.calc_hash_join_cost, [30]]
    ]
    for scenario in join_scenarios:
        qo.add_join_scenario(*scenario)
    q1_plan = qo.generate_query_plan('q1.json')
    rq1_plan = qo.generate_query_plan('rq1.json')
    print(q1_plan)
    print(rq1_plan)


if __name__ == '__main__':
    main()
