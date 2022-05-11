from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import re
import math
import torch
import random
import sympy

from math import pi
from sympy.parsing.sympy_parser import parse_expr

from egraph import exp, log, sqrt, sign
from egraph import sin, cos, tan, cot
from egraph import sinh, cosh, tanh, coth
from egraph import asin, acos, atan, acot
from egraph import asinh, acosh, atanh
from egraph import E, PI
from egraph import expression, EGraph, Node, Rule, apply_rules

from .envs.char_sp import UnknownSymPyOperator


def replace_y(prefix_str):
    return prefix_str.replace("Y'", 'z').replace('Y', 'y')


def replace_log(prefix_str):
    return prefix_str.replace('ln', 'log')


def replace_pi(prefix_str):
    return prefix_str.replace('pi', 'PI')


class ODEEgraphDataset(torch.utils.data.Dataset):
    def __init__(self, ode_dataset, env, programs_per_egraph=1000, max_length=512, training=True):
        self.ode_dataset = ode_dataset
        self.env = env
        self.max_length = max_length
        self.training = training
        self.programs_per_egraph = programs_per_egraph
        self.build_infix_list()
        self.num_egraphs = 1 + len(self.infix_list) // self.programs_per_egraph
        self.eg_list = [EGraph() for i in range(self.num_egraphs)]

        for i, item in enumerate(self.infix_list):
            self.eg_list[i // self.programs_per_egraph].add_node(
                expression(eval(f'lambda x, y, z: {item}')))
        self.rules = [
            Rule(lambda x, y, z: ((x * y) / z, x * (y / z))), # reassociate
            Rule(lambda x: (x / x, Node(1,()))), # simplify
            Rule(lambda x: (x * 1, x)), # simplify
            Rule(lambda x, y: (x + y, y + x)), # additive commutative
            Rule(lambda x, y: (x * y, y * x)), # multiplication commutative
            Rule(lambda x, y, z: ((x + y) * z, x * z + y * z)), # multiplication distributive
            Rule(lambda x, y, z: ((x + y) + z, x + (y + z))), # reassociate
            Rule(lambda x, y: (x + y, x - y + 2*y)),
            #Rule(lambda x, y: (x - y, x + y - 2*y)),
            Rule(lambda x, y: (x * y, x / (1 / y))),
            Rule(lambda x, y: (x / y, x * (1 / y))),
        ]
        for eg in self.eg_list:
            apply_rules(eg, self.rules)
        self.goal_nodes = [
            self.eg_list[i // self.programs_per_egraph].add_node(
                expression(eval(f'lambda x, y, z: {item}')))
            for i, fn in enumerate(self.infix_list)
        ]
        self.reversed_graphs = [self.build_reversed_graph(eg) for eg in self.eg_list]

    def odestr_to_infix(self, input_str):
        processed_prefix = replace_y(input_str)
        infix = self.env.prefix_to_infix(processed_prefix.split())
        infix = infix.replace('Abs', 'abs')
        infix = replace_log_e(infix)
        return infix

    def build_infix_list(self):
        self.infix_list = []
        self.infix_errors = []
        self.interested_item = []
        for index, item in enumerate(self.ode_dataset.data):
            processed_prefix = replace_y(item[0])
            #infix = self.env.prefix_to_infix(processed_prefix.split())
            #infix = infix.replace('Abs', 'abs')
            #infix = replace_log(infix)
            #self.infix_list.append(infix)
            #if 'E' in infix:
            #    self.interested_item.append((index, item))
            try:
                infix = self.env.prefix_to_infix(processed_prefix.split())
                infix = infix.replace('Abs', 'abs')
                infix = replace_log(infix)
                self.infix_list.append(infix)
                if 'E' in infix:
                    self.interested_item.append((index, item))
            except:
                self.infix_errors.append((index, item))
                continue

        success_infix_list = []
        self.success_to_original = []
        self.egraph_errors = []
        for index, item in enumerate(self.infix_list):
            eg = EGraph()
            lambda_fn = f'lambda x, y, z: {item}'
            try:
                node = eg.add_node(expression(eval(lambda_fn)))
                success_infix_list.append(item)
                self.success_to_original.append(index)
            except:
                self.egraph_errors.append((index, lambda_fn))
        self.infix_list = success_infix_list

    def build_reversed_graph(self, egraph):
        reversed_graph = {}
        for node, incoming_node_list in egraph.eclasses().items():
            for enode in incoming_node_list:
                reversed_graph[node] = [
                    (incoming_node.args, incoming_node.key)
                    for incoming_node in incoming_node_list if len(incoming_node.args) > 0]
        return reversed_graph

    def sample_path(self, node, eg, reversed_graph):
        visited = {node: 0 for node in reversed_graph.keys()}
        stack = [node]
        path = []
        while len(stack) > 0:
            curr = stack.pop(0)
            visited[curr] += 1
            enode_list = eg.eclasses()[curr]
            enode_input_lens = list(filter(lambda x: len(x.args) == 0, enode_list))
            if len(enode_input_lens) > 0:
                path.append((curr, enode_input_lens[0].key))
                continue
            incoming_edges = reversed_graph[curr]

            valid_incoming_edges = []
            for edge in incoming_edges:
                arguments_list = [inputs.find() for inputs in edge[0]]
                if curr not in arguments_list:
                    valid_incoming_edges.append(edge)

            if len(incoming_edges) > 0:
                arguments, key = random.choice(valid_incoming_edges)
                if len(arguments) > 1:
                    path.append((curr, key, arguments[0].find(), arguments[1].find()))
                    stack.append(arguments[0].find())
                    stack.append(arguments[1].find())
                elif len(arguments) == 1:
                    path.append((curr, key, arguments[0].find()))
                    stack.append(arguments[0].find())
        return path

    def transform_path_to_infix(self, goal_node, sampled_path):
        pending_variables = [goal_node]
        results = str(goal_node)
        for node_parents in sampled_path:
            if len(node_parents) == 4:
                curr_node, ops, p1, p2 = node_parents
                #results = results.replace(str(curr_node), f'({p1}{ops}{p2})')
                results = re.sub(f'{curr_node}(?!\d)', f'({p1}{ops}{p2})', results)
                pending_variables.remove(curr_node)
                pending_variables.extend([p1, p2])
            elif len(node_parents) == 2:
                curr_node, value = node_parents
                #results = results.replace(str(curr_node), f'({value})')
                #results = re.sub(f'{curr_node}(?!\d)', f'{value}', results)
                results = re.sub(f'{curr_node}(?!\d)', f'({value})', results)
                pending_variables.remove(curr_node)
            elif len(node_parents) == 3:
                curr_node, ops, p1 = node_parents
                #results = results.replace(str(curr_node), f'({ops}({p1}))')
                results = re.sub(f'{curr_node}(?!\d)', f'({ops}({p1}))', results)
                pending_variables.remove(curr_node)
                pending_variables.append(p1)
        return results

    def __len__(self):
        return len(self.infix_list)

    def test_get(self, idx):
        x = self.infix_list[idx]
        original_x, y = self.ode_dataset.data[self.success_to_original[idx]]
        eg = self.eg_list[idx // self.programs_per_egraph]
        goal_node = eg.add_node(expression(eval(f'lambda x, y, z: {x}')))
        reversed_graph = self.reversed_graphs[idx // self.programs_per_egraph]

        sampled_path = self.sample_path(goal_node, eg, reversed_graph)
        sampled_infix = self.transform_path_to_infix(goal_node, sampled_path)
        while sampled_infix == x:
            sampled_path = self.sample_path(goal_node, eg, reversed_graph)
            sampled_infix = self.transform_path_to_infix(goal_node, sampled_path)

        # Replace back Y, Y', ln and E.
        #sampled_infix = sampled_infix.replace('z', "Y'").replace('y', 'Y')
        sampled_infix = sampled_infix.replace('math.e', 'E').replace('log', 'ln')
        x = x.replace('math.e', 'E').replace('log', 'ln')
        replace_dict = {'z': "Y'", 'y': 'Y', 'log': 'ln', 'math.e': 'E'}
        import pdb; pdb.set_trace()
        sympy_code = parse_expr(x, evaluate=False, local_dict=self.env.local_dict)
        sampled_sympy_code = parse_expr(
            sampled_infix, evaluate=False, local_dict=self.env.local_dict)
        prefix_code = self.env.sympy_to_prefix(sympy_code)
        sampled_prefix = self.env.sympy_to_prefix(sampled_sympy_code)

        replace_map = lambda x: replace_dict[x] if x in replace_dict else x
        adjust_sampled_prefix = list(map(replace_map, sampled_prefix))

        original_x = original_x.split()
        y = y.split()
        return {
            'original_x': original_x,
            'x': x,
            'prefix': prefix_code,
            'y': y,
            'sampled_infix': sampled_infix,
            'sampled_prefix': sampled_prefix,
            'adjust_sampled_prefix': adjust_sampled_prefix
        }

    def __getitem__(self, idx):
        try:
            x = self.infix_list[idx]
            original_x, y = self.ode_dataset.data[self.success_to_original[idx]]
            eg = self.eg_list[idx // self.programs_per_egraph]
            goal_node = eg.add_node(expression(eval(f'lambda x, y, z: {x}')))
            reversed_graph = self.reversed_graphs[idx // self.programs_per_egraph]

            sampled_path = self.sample_path(goal_node, eg, reversed_graph)
            sampled_infix = self.transform_path_to_infix(goal_node, sampled_path)
            #while sampled_infix == x:
            #    sampled_path = self.sample_path(goal_node, eg, reversed_graph)
            #    sampled_infix = self.transform_path_to_infix(goal_node, sampled_path)

            # Replace back Y, Y', ln and E.
            sampled_infix = sampled_infix.replace('math.e', 'E').replace('log', 'ln')
            x = x.replace('math.e', 'E').replace('log', 'ln')
            replace_dict = {'z': "Y'", 'y': 'Y', 'log': 'ln', 'math.e': 'E'}

            #print(f'======================={idx}==========================')
            #print(f'original x, {original_x}')

            sympy_code = parse_expr(x, evaluate=False, local_dict=self.env.local_dict)

            #print(f'original infix, {sympy_code}')

            sampled_sympy_code = parse_expr(sampled_infix, evaluate=False, local_dict=self.env.local_dict)
            prefix_code = self.env.sympy_to_prefix(sympy_code)
            sampled_prefix = self.env.sympy_to_prefix(sampled_sympy_code)

            replace_map = lambda x: replace_dict[x] if x in replace_dict else x
            adjust_sampled_prefix = list(map(replace_map, sampled_prefix))

            original_x = original_x.split()
            y = y.split()

            #print(f'sampled_infix, {sampled_sympy_code}')
            #print(f'sampled_prefix, {adjust_sampled_prefix}')

            if len(adjust_sampled_prefix) > 512:
                return original_x, y, original_x
            else:
                return original_x, y, adjust_sampled_prefix
        except UnknownSymPyOperator:
            return None
        #return {
        #    'original_x': original_x,
        #    'x': x,
        #    'prefix': prefix_code,
        #    'y': y,
        #    'sampled_infix': sampled_infix,
        #    'sampled_prefix': sampled_prefix,
        #    'adjust_sampled_prefix': adjust_sampled_prefix
        #}

    def collate_fn(self, elements):
        elements = list(filter(lambda x: x is not None, elements))
        x, y, sampled_x = zip(*elements)
        nb_ops = [sum(int(word in self.env.OPERATORS) for word in seq) for seq in x]
        # for i in range(len(x)):
        #     print(self.env.prefix_to_infix(self.env.unclean_prefix(x[i])))
        #     print(self.env.prefix_to_infix(self.env.unclean_prefix(y[i])))
        #     print("")
        x = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in y]
        sampled_x = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in sampled_x]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        sampled_x, sampled_x_len = self.env.batch_sequences(sampled_x)
        return (x, x_len), (y, y_len), (sampled_x, sampled_x_len), torch.LongTensor(nb_ops)
