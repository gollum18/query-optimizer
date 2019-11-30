# Name: optimizer.py
# Since: 11/12/2019
# Author: Christen Ford
# Purpose: Implements a simple query optimizer that evaluates SQL-like queries
# passed in via a conformant JSON format.

# Assumptions:
#   1.) Disk I/O is converted to milliseconds that is then converted to a
#   timestamp. There was no unit on the result of converting disk I/O
#   using the formula (disk I/O * avg_seek_time * avg_latency).
#   I made the assumption that the resulting unit was in milliseconds and
#   implemented a timestamp conversion method based around this assumption.
#   That said, the timestamps reported by this function may be vastly different
#   than what was expected, but without knowing the unit for the converted disk I/O
#   nor an algorithm for converting it to a timestamp, I feel this assumption and
#   resulting output is reasonable.
#   2.) As long as the JSON schema is adhered to, this implementation allows for
#   evaluating arbitrarily nested queries. My initial implementation only handled
#   a single nested query.
#   3.) The result of an aggregate operation are currently not calculated correctly.
#   Currently aggregate operations do not correctly determine the block size and
#   tuple size of the resultant table. This should not matter in the long run as the
#   most costly operation in relational algebra is the join operation which tends
#   to dominate the other relational algebra operations.


import click
import math, os, sys
import simplejson


def cost_to_time(cost, avg_seek_time=8, avg_latency=4):
    """
    Converts a disk I/O metric to a milliseconds.
    :param cost: The disk I/O metric to convert.
    :param avg_seek_time: The average seek time in milliseconds.
    :param avg_latency: The average latency in milliseconds.
    :return: A disk I/O in milliseconds.
    """
    return int(
        math.ceil(
            cost *
            (avg_seek_time/1000) *
            (avg_latency/1000)
        )
    )


def create_table(
        name, num_pages, tuple_size, index_type="none", is_sorted=False,
        clustered=False, clustering_factor=0, primary_index=False):
    return (
        name, {
            'num_pages': num_pages,
            'tuple_size': tuple_size,
            'index_type': index_type,
            'sorted': is_sorted,
            'clustered': clustered,
            'clustering_factor': clustering_factor,
            'primary_index': primary_index
        }
    )


def create_timestamp(total_cost):
    """
    Creates a timestamp from a cost metric, average seek time and average latency.
    :param total_cost: A I/O cost metric in milliseconds.
    :return: A formatted string timestamp in the form "##h ##m ##s"
    """
    if not total_cost:
        raise ValueError
    if total_cost == 0:
        return 0, 0, 0
    seconds = (total_cost / 1000) % 60
    seconds = int(seconds)
    minutes = (total_cost / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = (total_cost / (1000 * 60 * 60)) % 24
    hours = int(hours)
    return "{:02d}h {:02d}m {:02d}s".format(hours, minutes, seconds)


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


def read_query_file(filename):
    """Parses a JSON query file and returns the contained JSON as a Python
    dictionary.

    Arguments:
        filename (str): The path to the query file.

    Returns:
        (dict): A Python dictionary containing contents of the query file.
    """
    try:
        if not os.path.exists(filename):
            raise OSError
        with open(filename) as f:
            qdict = simplejson.load(f)
        if not isinstance(qdict, dict):
            raise OSError
        return qdict
    except OSError:
        raise OSError


class QueryPlan(object):

    def __init__(self,
                 query_name,
                 join_methods,
                 join_orders,
                 total_cost,
                 timestamp,
                 has_subquery=False,
                 is_correlated=False):
        if not (isinstance(join_methods, list) and isinstance(join_orders, list)):
            raise ValueError
        if len(join_methods) != len(join_orders):
            raise ValueError
        self.query_name = query_name
        self.has_subquery = has_subquery
        self.is_correlated = is_correlated
        self.join_methods = join_methods
        self.join_orders = join_orders
        self.total_cost = total_cost
        self.timestamp = timestamp

    def __str__(self):
        name = "Query: {}\n\t".format(self.query_name)
        subquery = "Has Subquery: {}\n\t".format(self.has_subquery)
        correlated = "Has Correlation: {}\n\t".format(self.is_correlated)
        join_str = "Join Orders:\n\t\t"
        for i in range(len(self.join_orders)):
            if i == len(self.join_orders) - 1:
                join_str += "{}: {}\n\t".format(self.join_methods[i], self.join_orders[i])
            else:
                join_str += "{}: {}\n\t\t".format(self.join_methods[i], self.join_orders[i])
        total_cost = "Total Cost: {}\n\t".format(self.total_cost)
        exec_time = "Execution Time: {}".format(self.timestamp)
        return name + subquery + correlated + join_str + total_cost + exec_time


class QueryOptimizer(object):

    def __init__(self, page_size=4096, block_size=100):
        """

        :param page_size:
        :param block_size:
        """
        self.page_size = page_size
        self.block_size = block_size
        self.join_scenarios = dict()

    def add_join_scenario(self, key, join_function, args):
        """
        Adds a join scenario to the optimizer.
        :param key: The key for the join scenario -> uniquely identifies scenario.
        :param join_function: The join calculation function to call.
        :param args: A list of arguments to pass to join_function.
        """
        self.join_scenarios[key] = (join_function, args)

    def clear_join_scenarios(self):
        """
        Removes all join scenarios from the optimizer.
        """
        self.join_scenarios.clear()

    def delete_join_scenario(self, key):
        """
        Removes a join scenario from the optimizer.
        :param key: The key of the join scenario to remove.
        """
        if key in self.join_scenarios:
            del self.join_scenarios[key]

    def yield_join_scenarios(self):
        yield from self.join_scenarios.items()

    @staticmethod
    def get_matching_cost(index_type, primary, clustered, clustering_factor):
        if not primary:
            if clustered:
                clustered_cost = 1
            else:
                clustered_cost = clustering_factor
        else:
            clustered_cost = 0
        if index_type == 'bpt':
            index_cost = 4
        elif index_type == 'hash':
            index_cost = 1.2
        elif index_type == "none":
            index_cost = 0
        else:
            raise ValueError
        return index_cost + clustered_cost + 1

    def tuples_per_page(self, tuple_size):
        if tuple_size == 0:
            return 0
        return int(math.ceil(self.page_size / tuple_size))

    @staticmethod
    def calc_cartesian_product(stats, left_table_name, right_table_name):
        try:
            tables = stats['tables']
            left_table = tables[left_table_name]
            right_table = tables[right_table_name]
            left_num_pages = left_table['num_pages']
            right_num_pages = right_table['num_pages']
        except KeyError:
            print('KeyError: Table \'{}\' or \'{}\' not found! Cannot continue...'.format(
                left_table_name,
                right_table_name)
            )
            sys.exit(-1)
        return cost_to_time(left_num_pages * right_num_pages)

    def calc_tuple_nested_join_cost(self, stats, left_table_name, right_table_name):
        """

        :param stats:
        :param left_table_name:
        :param right_table_name:
        :return:
        """
        try:
            tables = stats['tables']
            left_table = tables[left_table_name]
            right_table = tables[right_table_name]
            left_num_pages = left_table['num_pages']
            left_tuple_size = left_table['tuple_size']
            right_num_pages = right_table['num_pages']
            right_tuple_size = right_table['tuple_size']
        except KeyError:
            print('KeyError: Table \'{}\' or \'{}\' not found! Cannot continue...'.format(
                left_table_name,
                right_table_name)
            )
            sys.exit(-1)
        try:
            left_clustered, right_clustered = left_table['clustered'], right_table['clustered']
            left_clustering_factor, right_clustering_factor = (
                left_table['clustering_factor'],
                right_table['clustering_factor']
            )
            left_index_type, right_index_type = left_table['index_type'], right_table['index_type']
            left_indexed, right_indexed = left_table['indexed'], right_table['indexed']
            left_primary, right_primary = left_table['primary_index'], right_table['primary_index']
        except KeyError:
            left_clustered, right_clustered = False, False
            left_clustering_factor, right_clustering_factor = 0, 0
            left_index_type, right_index_type = None, None
            left_indexed, right_indexed = False, False
            left_primary, right_primary = False, False
        if left_indexed:
            left_tuples_per_page = self.tuples_per_page(left_tuple_size)
            matching_right_cost = QueryOptimizer.get_matching_cost(
                right_index_type,
                right_primary,
                right_clustered,
                right_clustering_factor
            )
            cost = int(
                math.ceil(
                    left_tuple_size + ((left_tuple_size * left_tuples_per_page) * matching_right_cost)
                )
            )
        elif right_indexed:
            right_tuples_per_page = self.tuples_per_page(right_tuple_size)
            matching_left_cost = QueryOptimizer.get_matching_cost(
                left_index_type,
                left_primary,
                left_clustered,
                left_clustering_factor
            )
            cost = int(
                math.ceil(
                    right_tuple_size + ((right_tuple_size * right_tuples_per_page) * matching_left_cost)
                )
            )
        else:
            left_tuples_per_page = self.tuples_per_page(left_tuple_size)
            cost = int(
                math.ceil(
                    left_num_pages + left_tuples_per_page * left_tuple_size * right_num_pages
                )
            )
        return cost

    @staticmethod
    def calc_page_nested_join_cost(stats, left_table_name, right_table_name):
        """

        :param stats:
        :param left_table_name:
        :param right_table_name:
        :return:
        """
        try:
            tables = stats['tables']
            left_table = tables[left_table_name]
            right_table = tables[right_table_name]
            left_num_pages = left_table['num_pages']
            right_num_pages = right_table['num_pages']
        except KeyError:
            print('KeyError: Table \'{}\' or \'{}\' not found! Cannot continue...'.format(
                left_table_name,
                right_table_name)
            )
            sys.exit(-1)
        cost = int(
            math.ceil(
                left_num_pages + left_num_pages * right_num_pages
            )
        )
        return cost

    @staticmethod
    def calc_block_nested_join_cost(stats, outer_table_name, inner_table_name, block_size):
        """

        :param stats:
        :param outer_table_name:
        :param inner_table_name:
        :param block_size:
        :return:
        """
        try:
            tables = stats['tables']
            outer_table = tables[outer_table_name]
            inner_table = tables[inner_table_name]
            outer_num_pages = outer_table['num_pages']
            inner_num_pages = inner_table['num_pages']
        except KeyError:
            print('KeyError: Table \'{}\' or \'{}\' not found! Cannot continue...'.format(
                outer_table_name,
                inner_table_name)
            )
            sys.exit(-1)
        outer_blocks = int(
            math.ceil(
                outer_num_pages / block_size
            )
        )
        cost = outer_num_pages + (outer_blocks * inner_num_pages)
        return cost

    @staticmethod
    def calc_sort_merge_join_cost(stats, outer_table_name, inner_table_name, block_size):
        """
        Calculates the cost in I/O as time for a sort merge join.
        :param stats: A dict containing the table statistics.
        :param outer_table_name: The outer table name.
        :param inner_table_name: The inner table name.
        :param block_size: The block size.
        :return: Sort merge disk I/O cost as time.
        """
        try:
            tables = stats['tables']
            outer_table = tables[outer_table_name]
            inner_table = tables[inner_table_name]
            outer_num_pages = outer_table['num_pages']
            inner_num_pages = inner_table['num_pages']
            outer_table_sorted, inner_table_sorted = outer_table['sorted'], inner_table['sorted']
        except KeyError:
            print('KeyError: Table \'{}\' or \'{}\' not found! Cannot continue...'.format(
                outer_table_name,
                inner_table_name)
            )
            sys.exit(-1)
        max_num_pages = max(outer_num_pages, inner_num_pages)
        if outer_table_sorted and inner_table_sorted:
            cost = outer_num_pages + inner_num_pages
        elif block_size > math.sqrt(max_num_pages):
            cost = 3 * (outer_num_pages + inner_num_pages)
        else:
            cost = int(
                2 * (outer_num_pages + inner_num_pages) * (
                    1 + math.ceil(
                        math.log(
                            outer_num_pages/block_size,
                            block_size-1
                        )
                    )
                )
            )
        return cost

    @staticmethod
    def calc_hash_join_cost(stats, outer_table_name, inner_table_name, block_size):
        """
        Calculates the has join disk I/O cost as time.
        :param stats: A dict containing table stats.
        :param outer_table_name: The outer table name.
        :param inner_table_name: The inner table name.
        :param block_size: The block size.
        :return: Hash join disk I/O cost as time.
        """
        try:
            tables = stats['tables']
            outer_table = tables[outer_table_name]
            inner_table = tables[inner_table_name]
            outer_num_pages = outer_table['num_pages']
            inner_num_pages = inner_table['num_pages']
        except KeyError:
            print('KeyError: Table \'{}\' or \'{}\' not found! Cannot continue...'.format(
                outer_table_name,
                inner_table_name)
            )
            sys.exit(-1)
        if block_size > math.sqrt(outer_num_pages):
            cost = 3 * (outer_num_pages + inner_num_pages)
        else:
            cost = int(
                2 * (outer_num_pages + inner_num_pages) * (
                    1 + math.ceil(
                        math.log(
                            outer_num_pages / (block_size - 1),
                            block_size - 1
                        )
                    )
                ) + (outer_num_pages + inner_num_pages)
            )
        return cost

    def calc_join_cost(self, stats, join):
        """
        Calculates the cost for a join operation in disk I/O as time.
        :param stats: A dict containing table stats.
        :param join: The join operation to run.
        :return: A dict containing the join cost, join order, join method, and join function.
        """
        tables = join['tables']
        best_join_method = None
        best_join_order = None
        best_join_function = None
        best_join_cost = float('inf')
        for order in permute(tables):
            for method, value in self.yield_join_scenarios():
                function, args = value[0], value[1]
                if args is not None:
                    temp_cost = function(stats, *tables, *args)
                else:
                    temp_cost = function(stats, *tables)
                if temp_cost < best_join_cost:
                    best_join_cost = temp_cost
                    best_join_method = method
                    best_join_order = order
                    best_join_function = function
        return {
            "total_cost": cost_to_time(best_join_cost),
            "join_order": best_join_order,
            "join_method": best_join_method,
            "join_function": best_join_function
        }

    @staticmethod
    def calc_selection_cost(stats, select):
        """

        :param stats:
        :param select:
        :return:
        """
        try:
            table_name = select['table']
            table = stats[table_name]
            num_pages = table['num_pages']
        except KeyError as e:
            print('KeyError: Selection expected key {} but not found! Cannot continue...'.format(e))
            sys.exit(-1)
        try:
            table_sorted = table['sorted']
        except KeyError:
            table_sorted = False
        if table_sorted:
            cost = int(
                math.ceil(
                    math.log2(
                        num_pages
                    )
                )
            )
        else:
            cost = num_pages
        return cost_to_time(cost)

    @staticmethod
    def calc_projection_cost(stats, project):
        """

        :param stats:
        :param project:
        :return:
        """
        try:
            table_name = project['table']
            table = stats['tables'][table_name]
            num_pages = table['num_pages']
        except KeyError as e:
            print('KeyError: Projection expected key {} but not found! Cannot continue...'.format(e))
            sys.exit(-1)
        try:
            table_sorted = table['sorted']
        except KeyError:
            table_sorted = False
        if table_sorted:
            cost = num_pages
        else:
            cost = int(
                math.ceil(
                    num_pages * math.log2(
                        num_pages
                    )
                )
            )
        return cost_to_time(cost)

    @staticmethod
    def calc_aggregation_cost(stats, aggregate):
        """

        :param stats:
        :param aggregate
        :return:
        """
        try:
            table_name = aggregate['table']
            groupby = aggregate['groupby']
            table = stats['tables'][table_name]
            num_pages = table['num_pages']
        except KeyError as e:
            print('KeyError: Aggregation expected key {} but not found! Cannot continue...'.format(e))
            sys.exit(-1)
        try:
            table_sorted = table['sorted']
        except KeyError:
            table_sorted = False
        if groupby:
            if table_sorted:
                cost = num_pages
            else:
                cost = int(
                    math.ceil(
                        math.log2(num_pages) + num_pages
                    )
                )
        else:
            cost = num_pages
        return cost_to_time(cost)

    def calc_best_exec_plan(self, filename):
        """

        :param filename:
        :return:
        """
        def exec_query(query, is_subquery=False):
            try:
                query_steps = query['steps']
                if is_subquery:
                    is_correlated = query['is_correlated']
                else:
                    is_correlated = False
            except KeyError as e:
                print('KeyError: Key \'{}\' not found in Query, cannot continue...'.format(e))
                sys.exit(-1)
            query_cost = 0
            query_join_methods = []
            query_join_orders = []
            query_join_functions = []
            for step in query_steps:
                if 'select' in step:
                    select = query[step]
                    table = select['table']
                    query_cost += self.calc_selection_cost(stats, select)
                    num_pages = stats[table]['num_pages']
                    temp_name = select['result']
                    for rate in select['select_rates']:
                        num_pages *= stats['select_rates'][rate]
                    tuple_size = stats[table]['tuple_size']
                    is_sorted = stats[table]['sorted']
                    temp_table = create_table(name=temp_name,
                                              num_pages=num_pages,
                                              tuple_size=tuple_size,
                                              is_sorted=is_sorted)
                    stats['tables'][temp_table[0]] = temp_table[1]
                elif 'project' in step:
                    project = query[step]
                    table = project['table']
                    query_cost += self.calc_projection_cost(stats, project)
                    num_pages = stats['tables'][table]['num_pages']
                    temp_name = project['result']
                    for rate in project['project_rates']:
                        num_pages *= stats['project_rates'][rate]
                    tuple_size = stats['tables'][table]['tuple_size']
                    is_sorted = stats['tables'][table]['sorted']
                    temp_table = create_table(name=temp_name,
                                              num_pages=num_pages,
                                              tuple_size=tuple_size,
                                              is_sorted=is_sorted)
                    stats['tables'][temp_table[0]] = temp_table[1]
                elif 'aggregate' in step:
                    aggregate = query[step]
                    table = aggregate['table']
                    query_cost += self.calc_aggregation_cost(stats, aggregate)
                    num_pages = stats['tables'][table]['num_pages']
                    temp_name = aggregate['result']
                    tuple_size = stats['tables'][table]['tuple_size']
                    # TODO: FIGURE OUT WHAT PAGES AND TUPLE SIZE SHOULD BE
                    temp_table = create_table(name=temp_name,
                                              num_pages=num_pages,
                                              tuple_size=tuple_size)
                    stats['tables'][temp_table[0]] = temp_table[1]
                elif 'subquery' in step:
                    subquery = query[step]
                    table = subquery['table']
                    subquery_result = exec_query(subquery, True)
                    if subquery["is_correlated"]:
                        corr_cost = subquery_result['total_cost']
                        join_order = query_join_orders[-1]
                        num_pages = stats['tables'][join_order[0]]['num_pages']
                        query_cost += num_pages * corr_cost
                    else:
                        query_cost += subquery_result['total_cost']
                    query_join_methods.extend(subquery_result['join_methods'])
                    query_join_orders.extend(subquery_result['join_orders'])
                    temp_name = subquery['result']
                    stats['tables'][temp_name] = stats['tables'][table]
                elif 'join' in step:
                    join = query[step]
                    if is_correlated:
                        try:
                            tnj_tables = join['tables']
                        except KeyError as e:
                            print('KeyError: Correlated join expected {} key but not found, cannot continue!'.format(e))
                            sys.exit(-1)
                        corr_join_cost = float('inf')
                        corr_join_order = None
                        for order in permute(tnj_tables):
                            temp_cost = self.calc_tuple_nested_join_cost(stats, *tnj_tables)
                            corr_join_order = order
                            corr_join_cost = temp_cost
                        query_cost += stats['tables'][corr_join_order[0]]['num_pages'] * corr_join_cost
                        query_join_orders.append(corr_join_order)
                        query_join_methods.append('CORR_TNJ')
                        query_join_functions.append(self.calc_tuple_nested_join_cost)
                    else:
                        join_result = self.calc_join_cost(stats, join)
                        query_cost += join_result['total_cost']
                        query_join_orders.append(join_result['join_order'])
                        query_join_methods.append(join_result['join_method'])
                        query_join_functions.append(join_result['join_function'])
                    temp_name = join['result']
                    join_order = query_join_orders[-1]
                    num_pages = (
                        stats['tables'][join_order[0]]['num_pages'] *
                        stats['tables'][join_order[1]]['num_pages']
                    )
                    for select_rate in query[step]['select_rates']:
                        num_pages = int(
                            math.ceil(
                                num_pages *
                                stats['select_rates'][select_rate]
                            )
                        )
                    tuple_size = (
                        stats['tables'][join_order[0]]['tuple_size'] +
                        stats['tables'][join_order[1]]['tuple_size']
                    )
                    join_function = query_join_functions[-1]
                    is_sorted = True if join_function == self.calc_sort_merge_join_cost else False
                    temp_table = create_table(
                        name=temp_name,
                        num_pages=num_pages,
                        tuple_size=tuple_size,
                        is_sorted=is_sorted
                    )
                    stats['tables'][temp_table[0]] = temp_table[1]
            return {
                'total_cost': query_cost,
                'join_orders': query_join_orders,
                'join_methods': query_join_methods,
                'is_correlated': is_correlated
            }

        try:
            query_file = read_query_file(filename)
        except OSError:
            print('IOError: Unable to open query file! Cannot continue...')
            sys.exit(-1)
        try:
            stats = query_file['stats']
            root_query = query_file['query']
        except KeyError:
            print('Error: Unable to parse required information for building execution plan, cannot continue!')
            sys.exit(-1)
        query_name = filename.split('/')[-1].replace('.json', '')
        result = exec_query(root_query)
        total_cost = result['total_cost']
        join_orders = result['join_orders']
        join_methods = result['join_methods']
        timestamp = create_timestamp(total_cost)
        return QueryPlan(
            query_name=query_name,
            total_cost=total_cost,
            join_orders=join_orders,
            join_methods=join_methods,
            has_subquery=root_query['has_subquery'],
            is_correlated=root_query['is_correlated'],
            timestamp=timestamp
        )


@click.command()
@click.option('-ps', '--page-size', type=int, default=4096)
@click.option('-bs', '--block-size', type=int, default=100)
@click.argument('filepaths', nargs=-1, default=['q1.json', 'rq1.json'])
def main(page_size, block_size, filepaths):
    """
    Runs the query optimzer with a preconfigured set of join operations.
    :param page_size: The page size in bytes.
    :param block_size: The number of pages per block.
    :param filepaths: A list containing paths to queries to evaluate -- variadic.
    """
    qo = QueryOptimizer(page_size, block_size)
    join_scenarios = [
        ["TNJ", qo.calc_tuple_nested_join_cost, None],
        ["PNJ", qo.calc_page_nested_join_cost, None],
        ["BNJM", qo.calc_block_nested_join_cost, [50]],
        ["BNJL", qo.calc_block_nested_join_cost, [30]],
        ["SMJM", qo.calc_sort_merge_join_cost, [50]],
        ["SMJL", qo.calc_sort_merge_join_cost, [30]],
        ["HJM", qo.calc_hash_join_cost, [50]],
        ["HJL", qo.calc_hash_join_cost, [30]]
    ]
    for scenario in join_scenarios:
        qo.add_join_scenario(*scenario)
    try:
        for filepath in filepaths:
            print(qo.calc_best_exec_plan(filepath))
    except OSError:
        print('No plan generated for path: {}, path not valid!')


if __name__ == '__main__':
    main(filepaths=['q1.json', 'rq1.json'])
