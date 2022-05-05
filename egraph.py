from graphviz import Digraph
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
import regex as re
import math
from abc import ABC

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
EPS = np.finfo(np.float32).eps

import sympy
from sympy import Symbol, symbols


def sin(node):
    if isinstance(node, Node):
        return Node("sin", tuple([node]))
    else:
        return math.sin(node)


def cos(node):
    if isinstance(node, Node):
        return Node("cos", tuple([node]))
    else:
        return math.cos(node)


def tan(node):
    if isinstance(node, Node):
        return Node("tan", tuple([node]))
    else:
        return math.tan(node)


def cot(node):
    if isinstance(node, Node):
        return Node("cot", tuple([node]))
    else:
        return 1 / math.tan(node)


def asin(node):
    if isinstance(node, Node):
        return Node("asin", tuple([node]))
    else:
        return math.asin(node)


def acos(node):
    if isinstance(node, Node):
        return Node("acos", tuple([node]))
    else:
        return math.acos(node)


def atan(node):
    if isinstance(node, Node):
        return Node("atan", tuple([node]))
    else:
        return math.atan(node)


def acot(node):
    if isinstance(node, Node):
        return Node("acot", tuple([node]))
    else:
        return math.atan(1 / node)


def sinh(node):
    if isinstance(node, Node):
        return Node("sinh", tuple([node]))
    else:
        return math.sinh(node)


def cosh(node):
    if isinstance(node, Node):
        return Node("cosh", tuple([node]))
    else:
        return math.cosh(node)


def tanh(node):
    if isinstance(node, Node):
        return Node("tanh", tuple([node]))
    else:
        return math.tanh(node)


def coth(node):
    if isinstance(node, Node):
        return Node("coth", tuple([node]))
    else:
        return math.cosh(node) / math.sinh(node)


def asinh(node):
    if isinstance(node, Node):
        return Node("asinh", tuple([node]))
    else:
        return math.asinh(node)


def acosh(node):
    if isinstance(node, Node):
        return Node("acosh", tuple([node]))
    else:
        return math.acosh(node)


def atanh(node):
    if isinstance(node, Node):
        return Node("atanh", tuple([node]))
    else:
        return math.atanh(node)


def exp(node):
    if isinstance(node, Node):
        return Node("exp", tuple([node]))
    else:
        return math.exp(node)


def log(node):
    if isinstance(node, Node):
        return Node("log", tuple([node]))
    else:
        return math.log(node)


def sqrt(node):
    if isinstance(node, Node):
        return Node("sqrt", tuple([node]))
    else:
        return math.sqrt(node)


def sign(node):
    if isinstance(node, Node):
        return Node("sign", tuple([node]))
    else:
        return math.copysign(1, node)


def mysquare(node):
    if isinstance(node, Node):
        return Node("square", tuple([node]))
    else:
        return math.pow(node, 2)


def myroot(node):
    if isinstance(node, Node):
        return Node("root", tuple([node]))
    else:
        return math.pow(node, 0.5)


def mymin(*args):
  return Node('min', tuple(arg if isinstance(arg, Node) else Node(arg, ()) for arg in args))


def mymax(*args):
  return Node('max', tuple(arg if isinstance(arg, Node) else Node(arg, ()) for arg in args))


# overload some operators to allow us to easily construct these
def _mk_op(key):
  return lambda *args: Node(key, tuple(arg if isinstance(arg, Node) else Node(arg, ()) for arg in args))

def _mk_op_reverse(key):
  return lambda *args: Node(key, tuple(arg if isinstance(arg, Node) else Node(arg, ()) for arg in reversed(args)))


class Node(ABC):

  def __init__(self, key: 'Either[str, int, float]', args: 'Tuple[Node,...]'):
    if key == math.pi:
      self.key = Symbol('pi')
      #self.key = Symbol('pi')
    elif key == math.e:
      self.key = Symbol('e')
    else:
      self.key = key
    self.args = args
  #key: 'Either[str, int]' # use int to represent int literals,
  #args: 'Tuple[Node,...]'

  __add__ = _mk_op('+')
  __mul__ = _mk_op('*')
  __lshift__ = _mk_op('<<')
  __truediv__ = _mk_op('/')
  __pow__ = _mk_op('**')
  __sub__ = _mk_op('-')
  __lt__ = _mk_op('<')
  __le__ = _mk_op('<=')
  __eq__ = _mk_op('==')
  __ne__ = _mk_op('!=')
  __gt__ = _mk_op('>')
  __ge__ = _mk_op('>=')
  __mod__ = _mk_op('%')
  __abs__ = _mk_op('abs')

  __radd__ = _mk_op_reverse('+')
  __rmul__ = _mk_op_reverse('*')
  __rlshift__ = _mk_op_reverse('<<')
  __rtruediv__ = _mk_op_reverse('/')
  __rpow__ = _mk_op_reverse('**')
  __rsub__ = _mk_op_reverse('-')
  __rmod__ = _mk_op_reverse('%')
  #object.__radd__(self, other)
  #object.__rsub__(self, other)
  #object.__rmul__(self, other)
  #object.__rdiv__(self, other)
  #object.__rtruediv__(self, other)
  #object.__rfloordiv__(self, other)
  #object.__rmod__(self, other)
  #object.__rdivmod__(self, other)
  #object.__rpow__(self, other)
  #object.__rlshift__(self, other)
  #object.__rrshift__(self, other)
  #object.__rand__(self, other)
  #object.__rxor__(self, other)
  #object.__ror__(self, other)

  def __neg__(self):
      # -(Node(1, ()) + Node(2, ()))
    #if isinstance(self.key, str):
    #  if self.key.startswith('-'):
    #    key = self.key[1:]
    #  else:
    #    key = '-' + self.key
    #elif isinstance(self.key, int):
    #  key = -self.key
    #return Node(key, self.args)
    return Node('-', tuple([self]))

  #def __add__(self):
  #  return lambda *args: Node('+', tuple(arg if isinstance(arg, Node) else Node(arg, ()) for arg in args))

  # print it out like an s-expr
  def __repr__(self):
    if self.args:
      return f'({self.key} {" ".join(str(arg) for arg in self.args)})'
    else:
      return str(self.key)


def expression(fn):
  c = fn.__code__
  args = [Node(c.co_varnames[i], ()) for i in range(c.co_argcount)]
  return fn(*args)


class ENode(NamedTuple):
    key: str
    args: 'Tuple[EClassID, ...]'

    def canonicalize(self):
      return ENode(self.key, tuple(arg.find() for arg in self.args))

class EClassID:
    def __init__(self, eg, id):
        self.id = id #  just for debugging
        self.parent : 'Optional[EClassID]' = None

        # A list of enodes that use this eclass and the eclassid of that use.
        # This is only set on a canonical eclass (parent == None) and
        # will be used to repair the graph on merges.
        self.uses : Optional[List[Tuple['ENode', 'EClassID']]] = []

    def __repr__(self):
        return f'e{self.id}'

    def __lt__(self, other):
        return self.id < other.id

    # union-find's 'find' operation, that finds the canonical object for this set
    def find(self):
        if self.parent is None:
            return self
        r = self.parent.find()
        # path compression, makes find cheaper on subsequent calls.
        self.parent = r
        return r


#ENode.canonicalize = lambda self: ENode(self.key, tuple(arg.find() for arg in self.args))
Env = Dict[str, EClassID]


class EGraph:
    def __init__(self):
        self.i = 0 # id counter for debugging EClassIDs
        self.version = 0 # increments every time we mutate the EGraph, so that we
                         # can quickly see when we haven't changed one

        # this map, from ENodes (canonical) to EClassID (not canonical), is maintained
        # so that we check if we have already defined a particular ENode
        self.hashcons : Dict['ENode', 'EClassID'] = {}

        # a list of EClasses that have mutated by a merge operation, and whose users must
        # be updated to a new canonical version, we will later define a `rebuild` function
        # that processes this worklist.
        self.worklist : List['EClassID'] = []

    def add_enode(self, enode: 'ENode'):
        enode = enode.canonicalize()
        eclass_id = self.hashcons.get(enode, None)
        if eclass_id is None:
            # we don't already have this node, so create one
            self.version += 1
            eclass_id = self._new_singleton_eclass()
            for arg in enode.args:
                # we need to update the uses lists of each arg,
                # since we are adding this enode as a new use.
                arg.uses.append((enode, eclass_id))
            self.hashcons[enode] = eclass_id
        # hashcons's rhs is not canonicalized, so canonicalize it here:
        return eclass_id.find()

    def _new_singleton_eclass(self):
        r = EClassID(self, self.i)
        self.i += 1
        return r

    # helper for translating our normal Nodes into the egraph
    def add_node(self, node: 'Node'):
       return self.add_enode(ENode(node.key, tuple(self.add_node(n) for n in node.args)))

    def add_lambda_code_list_as_node(self, lambda_code_list):
      for lambda_fn in lambda_code_list:
          if isinstance(lambda_fn, Callable):
            self.add_node(expression(lambda_fn))
          elif isinstance(lambda_fn, str):
            self.add_node(expression(eval(lambda_fn)))

    # extract the actual eclasses
    def eclasses(self) -> Dict['EClassID', List['ENode']]:
        r = {}
        for node, eid in self.hashcons.items():
            eid = eid.find()
            if eid not in r:
                r[eid] = [node]
            else:
                r[eid].append(node)
        return r

    def get_ops_list(self):
        ops_list = []
        for k, v in self.eclasses().items():
            for enode in v:
                if len(enode.args) > 0:
                    ops_list.append(enode.key)
        return ops_list

    # return dict where key is the eclass id, the value is the string repr of its enode.
    def estrings(self):
        sorted_eclass = sorted(self.eclasses().items())
        node_list = sorted(self.eclasses().keys())
        i = 0
        str_list = []
        for node in node_list:
            node_id = node.id
            while node.id > i:
                str_list.append('empty')
                i += 1
            enode_list = self.eclasses()[node]
            if len(enode_list) > 1:
                enode_strs = [enode.__repr__() for enode in enode_list]
                str_list.append(' and '.join(enode_strs))
            else:
                str_list.append(enode_list[0].__repr__())
            i += 1
        return str_list

    # Find node without parent.
    def find_goal_node(self):
        appeared_in_args = set()
        for enode_list in self.eclasses().values():
            for enode in enode_list:
                for node in enode.args:
                    appeared_in_args.add(node)
        node_list = set(self.eclasses().keys())
        return node_list - appeared_in_args

    def build_digraph(self):
      def format_record(x):
        if isinstance(x, list):
          return '{' + '|'.join(format_record(e) for e in x) + '}'
        else:
          return x
      def escape(x):
        return str(x).replace('<', '\<').replace('>', '\>')
      g = Digraph(node_attr={'shape': 'record', 'height': '.1'})
      for eclass, enodes in self.eclasses().items():
          g.node(f'{eclass.id}', label=f'e{eclass.id}', shape='circle')

          for enode in enodes:
            enode_id = str(id(enode))
            #g.edge(f'{eclass.id}', enode_id)
            g.edge(enode_id, f'{eclass.id}')

            record = [escape(enode.key)]
            for i, arg in enumerate(enode.args):
              #g.edge(f'{enode_id}:p{i}', f'{arg.id}')
              g.edge(f'{arg.id}', f'{enode_id}:p{i}')
              record.append(f'<p{i}>')
            g.node(enode_id, label='|'.join(record))

      return g

    # tell the graph that 'a' and 'b' calculate the same value.
    def merge(self, a: 'EClassID', b: 'EClassID'):
      a = a.find()
      b = b.find()
      if a is b:
          return a
      self.version += 1
      a.parent = b
      # maintain the invariant that uses are only
      # recorded on the top level EClassID
      b.uses += a.uses
      a.uses = None
      # we have updated eclass b, so nodes in the hashcons
      # might no longer be canonical and we might discover that two
      # enodes are actually the same value. We will repair this later, by
      # remember what eclasses changed:
      self.worklist.append(b)

    # ensure we have a de-duplicated version of the EGraph
    def rebuild(self):
      while self.worklist:
          # deduplicate repeated calls to repair the same eclass
          todo = {eid.find(): None for eid in self.worklist}
          self.worklist = []
          for eclass_id in todo:
              if eclass_id.parent is None:
                self.repair(eclass_id)
              else:
                # TODO(ywenxu): Ignore for now, fix this later.
                continue

    def repair(self, eclass_id: 'ENodeID'):
      assert eclass_id.parent is None
      # reset the uses of this eclass, we will fill them in again at the end
      uses, eclass_id.uses = eclass_id.uses, []
      # any of the uses in the hashcons might no longer be canonical, so re-canonicalize it
      for p_node, p_eclass in uses:
          if p_node in self.hashcons:
              del self.hashcons[p_node]
          p_node = p_node.canonicalize()
          self.hashcons[p_node] = p_eclass.find()
      # because we merged classes, some of the enodes that are uses might now be the same expression,
      # meaning we can merge further eclasses
      new_uses = {}
      for p_node, p_eclass in uses:
          p_node = p_node.canonicalize()
          if p_node in new_uses:
              self.merge(p_eclass, new_uses[p_node])
          new_uses[p_node] = p_eclass.find()
      # note the find, it is possible that eclass_id got merged
      # and uses should only be attributed to the eclass representative
      eclass_id.find().uses += new_uses.items()

    def subst(self, pattern: Node, env: Env) -> EClassID:
      if not pattern.args and not isinstance(pattern.key, int):
          return env[pattern.key]
      else:
          enode = ENode(pattern.key, tuple(self.subst(arg, env) for arg in pattern.args))
          return self.add_enode(enode)


class Rule:
    def __init__(self, fn):
      self.lhs, self.rhs = expression(fn)
    def __repr__(self):
        return f"{self.lhs} -> {self.rhs}"
rules = [
    Rule(lambda x: (x * 2, x << 1)), # times is a shift
    Rule(lambda x, y, z: ((x * y) / z, x * (y / z))), # reassociate
    Rule(lambda x: (x / x, Node(1,()))), # simplify
    Rule(lambda x: (x * 1, x)), # simplify
]


def ematch(eclasses: 'Dict[EClassID, List[ENode]]', pattern: Node) -> 'List[Tuple[EClassID, Env]]':
    def match_in(p: 'Node', eid: 'EClassID', env: Env) -> 'Tuple[Bool, Env]':
        def enode_matches(p: Node, enode: ENode, env: Env) -> 'Tuple[Bool, Env]':
            if enode.key != p.key:
                return False, env
            new_env = env
            for arg_pattern, arg_eid in zip(p.args, enode.args):
                matched, new_env = match_in(arg_pattern, arg_eid, new_env)
                if not matched:
                    return False, env
            return True, new_env
        if not p.args and not isinstance(p.key, int):
            # this is a leaf variable like x, match it with the environment
            id = p.key
            if id not in env:
                env = {**env} # this is expensive but can be optimized
                env[id] = eid
                return True, env
            else:
                # check this value matches to the same thing
                return env[id] is eid, env
        else:
            # does one of the ways to define this class match the pattern?
            for enode in eclasses[eid]:
                matches, new_env = enode_matches(p, enode, env)
                if matches:
                    return True, new_env
            return False, env

    matches = []
    for eid in eclasses.keys():
        match, env = match_in(pattern, eid, {})
        if match:
            matches.append((eid, env))
    return matches


def apply_rules(eg: 'EGraph', rules: 'List[Rule]'):
    eclasses = eg.eclasses()
    matches = []
    for rule in rules:
        for eid, env in ematch(eclasses, rule.lhs):
            matches.append((rule, eid, env))
    #print(f"VERSION {eg.version}")
    for rule, eid, env in matches:
        new_eid = eg.subst(rule.rhs, env)
        #if eid is not new_eid:
        #  print(f'{eid} MATCHED {rule} with {env}')
        eg.merge(eid, new_eid)
    eg.rebuild()
    return eg


def find_matches(eg: 'EGraph', rules: 'List[Rule]'):
    eclasses = eg.eclasses()
    matches = []
    for rule in rules:
        for eid, env in ematch(eclasses, rule.lhs):
            matches.append((rule, eid, env))
    return matches
    #print(f"VERSION {eg.version}")


def subst(matches, eg):
    for rule, eid, env in matches:
        new_eid = eg.subst(rule.rhs, env)
        if eid is not new_eid:
          print(f'{eid} MATCHED {rule} with {env}')
        eg.merge(eid, new_eid)
    eg.rebuild()
    return eg


def construct_lambda_from_code_old(code):
    import regex as re
    #input_pattern = '[a-zA-Z][0-9]+'
    #matches = re.findall(input_pattern, dsl_code)
    input_pattern = 'n\d+'
    matches = re.findall(input_pattern, code)
    matches = [m.strip() for m in matches]
    input_variables = list(sorted(set(matches)))

    code_lines = code.split('\n')
    raw_code_lines = [line for line in code_lines if line != '' and not line.startswith('import')]

    # Remove comments in the code.
    code_lines = []
    comment_pattern = '#(\s)*(\w+\s*)+$'
    for line in raw_code_lines:
        comment_match = re.finditer(comment_pattern, line)
        for m in comment_match:
            line = line.replace(line[m.start(): m.end()], '')
            line = line.strip()
        code_lines.append(line)

    assignment_lines = code_lines[:len(input_variables)]
    code_lines = code_lines[len(input_variables):]

    input_var_dict = {}
    #for var, line in zip(input_variables, assignment_lines):
    for line in assignment_lines:
        left, right = line.split('=')
        input_var_dict[left.strip()] = right.strip()

    var_dict = {}
    pattern = '[a-zA-Z][0-9]+'
    for line in code_lines:
        left, right = line.split('=')
        left = left.strip()
        right = right.strip()
        #rhs_variables = sorted(re.findall(pattern, right))
        #for var in rhs_variables:
        #    if var not in input_variables:
        #        right = right.replace(var, '({})'.format(var_dict[var]))
        rhs_variables_indices = re.finditer(pattern, right)
        prev_pointer = 0
        pieces = []

        for m in rhs_variables_indices:
            pieces.append(right[prev_pointer:m.start()])
            var = right[m.start():m.end()]
            if var not in input_variables:
                pieces.append('({})'.format(var_dict[var]))
            else:
                pieces.append(var)
            prev_pointer = m.end()
        pieces.append(right[prev_pointer:])
        right = ''.join(pieces)
        var_dict[left] = right
    var_dict.update(input_var_dict)
    return input_variables, var_dict


def construct_lambda_from_code(code):
    import regex as re
    input_pattern = r"n\d+"
    matches = re.findall(input_pattern, code)
    matches = [m.strip() for m in matches]
    input_vars  = list(sorted(set(matches)))

    code_lines = code.split('\n')
    raw_code_lines = [line for line in code_lines if line != '' and not line.startswith('import')]

    # Remove comments in the code.
    code_lines = []
    comment_pattern = r"#(\s)*(\w+\s*)+$"
    for line in raw_code_lines:
        comment_match = re.finditer(comment_pattern, line)
        for m in comment_match:
            line = line.replace(line[m.start(): m.end()], '')
            line = line.strip()
        code_lines.append(line)

    assignment_lines = code_lines[:len(input_vars)]
    code_lines = code_lines[len(input_vars):]

    input_var_dict = {}
    #for var, line in zip(input_variables, assignment_lines):
    for line in assignment_lines:
        left, right = line.split('=')
        input_var_dict[left.strip()] = right.strip()

    var_dict = {}
    pattern = r"[a-zA-Z][0-9]+"
    for line in code_lines:
        left, right = line.split('=')
        left = left.strip()
        right = right.strip()
        rhs_variables_indices = re.finditer(pattern, right)
        prev_pointer = 0
        pieces = []

        for m in rhs_variables_indices:
            pieces.append(right[prev_pointer:m.start()])
            var = right[m.start():m.end()]
            if var not in input_vars:
                pieces.append('({})'.format(var_dict[var]))
            else:
                pieces.append(var)
            prev_pointer = m.end()
        pieces.append(right[prev_pointer:])
        right = ''.join(pieces)
        var_dict[left] = right
    var_dict.update(input_var_dict)

    answer = var_dict['answer']
    const_pattern = '(?<!([a-z_]|[a-z]\d+))(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?' # Match numbers like 1e-5
    match_result = re.finditer(const_pattern, answer)
    match_indices = [(m.start(), m.end()) for m in match_result]
    raw_matches = [answer[start:end] for (start, end) in match_indices]

    matches = [m[1:] if m.startswith('(') else m for m in raw_matches]
    matches = set([m.strip() for m in matches])
    matches = list(matches)
    match_to_const = {m: 'const_{}'.format(k) for k, m in enumerate(matches)}
    var_dict.update({v: k for k, v in match_to_const.items()})
    input_vars.extend(list(match_to_const.values()))
    # Handle math.pi
    if 'math.pi' in answer:
        match_to_const['math.pi'] = 'const_{}'.format(len(matches))
        input_vars.append('const_{}'.format(len(matches)))

    prev_pointer = 0
    code_pieces = []
    for (start, end) in match_indices:
        code_pieces.append(answer[prev_pointer:start])
        match_string = answer[start:end]
        if match_string.startswith('(') or match_string.startswith(' '):
            key = match_string[1:]
            code_pieces.append(match_string.replace(key, match_to_const[key]))
        else:
            key = match_string
            code_pieces.append(match_to_const[key])
        prev_pointer = end
    code_pieces.append(answer[prev_pointer:])
    answer = ''.join(code_pieces)

    if 'math.pi' in answer:
        answer = answer.replace('math.pi', match_to_const['math.pi'])

    input_vars_string = ', '.join(input_vars)
    lambda_code = 'lambda {}: {}'.format(input_vars_string, answer)

    return lambda_code


def compute_answer(lambda_list, var_dict_list):
    lambda_answers = []
    code_answers = []
    for lambda_code, var_dict in zip(lambda_list, var_dict_list):
        code = lambda_code[1]
        ldict = {}
        try:
            exec(code, globals(), ldict)
            code_answers.append(ldict['answer'])
        except Exception as e:
            code_answers.append('Error: {}'.format(e))

        lambda_func = lambda_code[0]
        input_vars_str = lambda_func.split(':')[0][6:]
        input_vars = input_vars_str.split(',')
        input_vars = [var.strip() for var in input_vars]
        lambda_input_dict = {}
        for var in input_vars:
            lambda_input_dict[var] = float(var_dict[var])
        try:
            result = eval(lambda_func)(**lambda_input_dict)
            lambda_answers.append(result)
        except Exception as e:
            lambda_answers.append('Error: {}'.format(e))
    return lambda_answers, code_answers


def build_egraph_mathqa():
    from APPSBaseDataset import MathQADataset
    import transformers
    import regex as re
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    test_data = MathQADataset('data/mathqa', tokenizer, test=True)

    lambda_1 = construct_lambda_from_code(test_data.examples[5]['code'])
    print(lambda_1)
    lambda_2 = construct_lambda_from_code(test_data.examples[12]['code'])
    print(lambda_2)
    lambda_list = [lambda_1, lambda_2]
    for idx in [43, 255, 1400]:
        lambda_list.append(construct_lambda_from_code(test_data.examples[idx]['code']))
    print('\n'.join(lambda_list))

    eg = EGraph()
    for lambda_fn in lambda_list:
        eg.add_node(expression(eval(lambda_fn)))
    rules = [
        Rule(lambda x, y, z: ((x * y) / z, x * (y / z))), # reassociate
        Rule(lambda x: (x / x, Node(1,()))), # simplify
        Rule(lambda x: (x * 1, x)), # simplify
        Rule(lambda x, y: (x + y, y + x)), # additive commutative
        Rule(lambda x, y: (x * y, y * x)), # multiplication commutative
        Rule(lambda x, y, z: ((x + y) * z, x * z + y * z)), # multiplication distributive
        Rule(lambda x, y, z: ((x + y) + z, x + (y + z))), # reassociate
    ]
    apply_rules(eg, rules)

    edge_matrix_1 = []
    edge_matrix_2 = []
    edge_type = []
    node_list = []
    for k, v in eg.eclasses().items():
        node_list.append(k)
        for connection in v:
            if connection.args is not None:
                incoming_nodes = connection.args
            if len(incoming_nodes) == 2:
                edge_matrix_1.append([incoming_nodes[0].id, k.id])
                edge_matrix_2.append([incoming_nodes[1].id, k.id])
                edge_type.append(connection.key)
            elif len(incoming_nodes) == 1:
                edge_matrix_1.append([incoming_nodes[0].id, k.id])
                edge_matrix_2.append([incoming_nodes[0].id, k.id])
                edge_type.append(connection.key)

    print(node_list)
    print(edge_matrix_1)
    print(edge_matrix_2)
    print(edge_type)
    print(len(edge_type))
    print(len(edge_matrix_1))
    print(len(edge_matrix_2))

    operation_types = set(edge_type)
    features = {
        '%': torch.normal(0, 1, size=[768]),
        '**': torch.normal(0, 1, size=[768]),
        '+': torch.normal(0, 1, size=[768]),
        '-': torch.normal(0, 1, size=[768]),
        '*': torch.normal(0, 1, size=[768]),
        '/': torch.normal(0, 1, size=[768]),
    }
    edge_feat = [features[ops] for ops in edge_type]
    edge_feat = torch.stack(edge_feat)
    node_initialized_feature = torch.normal(0, 1, size=[768])
    node_feat = torch.stack([node_initialized_feature] * 100)
    edge_1 = torch.tensor(edge_matrix_1)
    edge_2 = torch.tensor(edge_matrix_2)
    print(edge_feat.size())
    print(edge_1.size())
    print(node_feat.size())

    ops_to_edge = {
        '**': [],
        '%': [],
        '+': [],
        '-': [],
        '*': [],
        '/': []
    }

    edge_dict = {
        '**': [],
        '%': [],
        '+': [],
        '-': [],
        '*': [],
        '/': []
    }

    ops_to_idx = {
        '**': [],
        '%': [],
        '+': [7],
        '-': [6],
        '*': [0],
        '/': [17]
    }
    for ops, edge, edge_feature in zip(edge_type, edge_1, edge_feat):
        ops_to_edge[ops].append(edge)
        edge_dict[ops].append(edge_feature)
    #return (node_feat, edge_feat, edge_1, edge_2, ops_to_edge, eg, lambda_list)
    #edge_feature_embeddings = [
    #    edge_feat[7], edge_feat[6], edge_feat[0], edge_feat[17],
    #]

    adjacency_lists = []
    edge_feature_embeddings = []
    for ops in sorted(['**', '%', '+', '-', '*', '/']):
        if len(ops_to_edge[ops]) > 1:
            edge_pairs = torch.stack(ops_to_edge[ops])
            adjacency_lists.append(tuple([edge_pairs[:, 0], edge_pairs[:, 1]]))
            edge_feature_embeddings.append(torch.stack(edge_dict[ops]))
        else:
            adjacency_lists.append([])
            edge_feature_embeddings.append([])
    return (node_feat, edge_feature_embeddings, adjacency_lists, eg, lambda_list)


if __name__ == "__main__":
    from APPSBaseDataset import MathQADataset
    import transformers
    import regex as re
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    test_data = MathQADataset('data/mathqa', tokenizer, test=True)

    lambda_1 = construct_lambda_from_code(test_data.examples[5]['code'])
    print(lambda_1)
    lambda_2 = construct_lambda_from_code(test_data.examples[12]['code'])
    print(lambda_2)
    lambda_list = [lambda_1, lambda_2]
    for idx in [43, 255, 1400]:
        lambda_list.append(construct_lambda_from_code(test_data.examples[idx]['code']))
    print('\n'.join(lambda_list))

    eg = EGraph()
    for lambda_fn in lambda_list:
        eg.add_node(expression(eval(lambda_fn)))
    rules = [
        Rule(lambda x, y, z: ((x * y) / z, x * (y / z))), # reassociate
        Rule(lambda x: (x / x, Node(1,()))), # simplify
        Rule(lambda x: (x * 1, x)), # simplify
        Rule(lambda x, y: (x + y, y + x)), # additive commutative
        Rule(lambda x, y: (x * y, y * x)), # multiplication commutative
        Rule(lambda x, y, z: ((x + y) * z, x * z + y * z)), # multiplication distributive
        Rule(lambda x, y, z: ((x + y) + z, x + (y + z))), # reassociate
    ]
    apply_rules(eg, rules)

    edge_matrix_1 = []
    edge_matrix_2 = []
    edge_type = []
    node_list = []
    for k, v in eg.eclasses().items():
        node_list.append(k)
        for connection in v:
            if connection.args is not None:
                incoming_nodes = connection.args
            if len(incoming_nodes) == 2:
                edge_matrix_1.append([incoming_nodes[0].id, k.id])
                edge_matrix_2.append([incoming_nodes[1].id, k.id])
                edge_type.append(connection.key)
            elif len(incoming_nodes) == 1:
                edge_matrix_1.append([incoming_nodes[0].id, k.id])
                edge_matrix_2.append([incoming_nodes[0].id, k.id])
                edge_type.append(connection.key)

    print(node_list)
    print(edge_matrix_1)
    print(edge_matrix_2)
    print(edge_type)
    print(len(edge_type))
    print(len(edge_matrix_1))
    print(len(edge_matrix_2))

    operation_types = set(edge_type)
    features = {
        '*': torch.normal(0, 1, size=[512]),
        '+': torch.normal(0, 1, size=[512]),
        '-': torch.normal(0, 1, size=[512]),
        '/': torch.normal(0, 1, size=[512]),
    }
    edge_feat = [features[ops] for ops in edge_type]
    edge_feat = torch.stack(edge_feat)
    node_initialized_feature = torch.normal(0, 1, size=[512])
    node_feat = torch.stack([node_initialized_feature] * len(node_list))
    edge_1 = torch.tensor(edge_matrix_1)
    edge_2 = torch.tensor(edge_matrix_2)
    print(edge_feat.size())
    print(edge_1.size())
    print(node_feat.size())

    #gnn = GNN(
    #    msg_dim = 512,
    #    node_state_dim = 512,
    #    edge_feat_dim = 512
    #)
    import pdb; pdb.set_trace()

    #test_data = MathQADataset('data/mathqa', tokenizer, test=False)
    #lambda_code_list = []
    #import_error_tuple = []
    #error_tuple = []
    #error_code = []
    #exp_list = []
    #var_dict_list = []
    #for i in range(len(test_data.examples)):
    #    code, dsl_code = test_data.examples[i]['code'], test_data.examples[i]['dsl_code']
    #    if code.startswith('import'):
    #        import_error_tuple.append(('Index {}:'.format(i), code, dsl_code))
    #        continue
    #    try:
    #        input_vars, var_dict = construct_lambda_from_code(code)
    #        answer = var_dict['answer']
    #        #const_pattern = '([^a-z{()}]\d+(?:\.\d+)?)'
    #        #matches = re.findall(const_pattern, answer)
    #        #const_pattern = '[^a-z_]([0-9]*[.])?[0-9]+'
    #        #const_pattern = '(?<![a-z_])([0-9]*[.])?[0-9]+'
    #        #const_pattern = '(?<!([a-z_]|[a-z]\d+))([0-9]*[.])?[0-9]+' # Don't match the variable such as n10.
    #        const_pattern = '(?<!([a-z_]|[a-z]\d+))(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?' # Match numbers like 1e-5
    #        match_result = re.finditer(const_pattern, answer)
    #        match_indices = [(m.start(), m.end()) for m in match_result]
    #        raw_matches = [answer[start:end] for (start, end) in match_indices]

    #        # The case where the match starts with '('
    #        #matches = []
    #        #for m in raw_matches:
    #        #    if m.startswith('('):
    #        #        matches.append(m[1:])
    #        #    else:
    #        #        matches.append(m)
    #        matches = [m[1:] if m.startswith('(') else m for m in raw_matches]
    #        matches = set([m.strip() for m in matches])
    #        matches = list(matches)
    #        match_to_const = {m: 'const_{}'.format(k) for k, m in enumerate(matches)}
    #        var_dict.update({v: k for k, v in match_to_const.items()})
    #        input_vars.extend(list(match_to_const.values()))
    #        # Handle math.pi
    #        if 'math.pi' in answer:
    #            match_to_const['math.pi'] = 'const_{}'.format(len(matches))
    #            input_vars.append('const_{}'.format(len(matches)))

    #        #integer_pattern = '^[0-9]+'
    #        #int_match_indices = re.finditer(integer_pattern, answer)
    #        #int_matches = [answer[m.start():m.end()] for m in match_indices]j
    #        #int_matches = set([m.strip() for m in int_matches])j
    #        #matches = list(matches.union(int_matches))j
    #        #existing_consts = []
    #        #for j, m in enumerate(matches):
    #        #    answer = answer.replace(m, 'const_{}'.format(j))
    #        #    input_vars.append('const_{}'.format(j))
    #        #    var_dict['const_{}'.format(j)] = m

    #        prev_pointer = 0
    #        code_pieces = []
    #        for (start, end) in match_indices:
    #            code_pieces.append(answer[prev_pointer:start])
    #            match_string = answer[start:end]
    #            if match_string.startswith('(') or match_string.startswith(' '):
    #                key = match_string[1:]
    #                code_pieces.append(match_string.replace(key, match_to_const[key]))
    #            else:
    #                key = match_string
    #                code_pieces.append(match_to_const[key])
    #            prev_pointer = end
    #        code_pieces.append(answer[prev_pointer:])
    #        answer = ''.join(code_pieces)

    #        if 'math.pi' in answer:
    #            answer = answer.replace('math.pi', match_to_const['math.pi'])

    #        var_dict_list.append(var_dict)
    #        input_vars_string = ', '.join(input_vars)
    #        lambda_code = 'lambda {}: {}'.format(input_vars_string, answer)
    #        lambda_code_list.append((lambda_code, code, dsl_code))

    #    except Exception as e:
    #        error_tuple.append(('Index {}:'.format(i), code, dsl_code, 'Error message: {}'.format(e)))

    #    try:
    #        func = eval(lambda_code)
    #        exp_list.append(exp(func))
    #    #except NameError as ne:
    #    #    import pdb; pdb.set_trace()
    #    except Exception as e:
    #        error_code.append(('Index {}:'.format(i), lambda_code, 'Error message: {}'.format(e)))
    #        #import pdb;pdb.set_trace()

    #print(len(test_data))
    #print(len(error_tuple))
    #print(len(error_code))
    #lambda_answers, code_answers = compute_answer(lambda_code_list, var_dict_list)

    #not_match_index = []
    #indi = []
    #for k, (aa,bb) in enumerate(zip(lambda_answers,code_answers)):
    #    indi.append(aa==bb)
    #    if not (aa == bb):
    #        not_match_index.append(k)
