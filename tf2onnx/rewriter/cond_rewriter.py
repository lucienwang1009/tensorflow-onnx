# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter.loop_rewriter_base
"""

from __future__ import division
from __future__ import print_function
import copy
import logging
import os
from collections import deque, defaultdict, OrderedDict
from onnx import helper
from tf2onnx import utils
from tf2onnx.graph import Graph
from tf2onnx.rewriter.rnn_utils import is_loopcond_op, is_tensor_array_op, is_tensor_array_write_op
from tf2onnx.rewriter.rnn_utils import REWRITER_RESULT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.cond_rewriter_base")
# log.setLevel(logging.DEBUG)

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

def remove_duplicate_list(l):
    return list(set(l))

def union_list(a, b):
    return list(set(a).union(set(b)))

def intersection_list(a, b):
    return list(set(a).intersection(set(b)))

class CondBodyGraph:
    def __init__(self):
        self.output = []
        self.nodes = []
        self.input = []

    def union(self, cond_body_graph):
        self.output = union_list(self.output, cond_body_graph.output)
        self.nodes = union_list(self.nodes, cond_body_graph.nodes)

    def intersection(self, cond_body_graph):
        intersect_node = [
            n.name for n in intersection_list(self.nodes, cond_body_graph.nodes)
        ]
        intersect_output = intersection_list(self.output, cond_body_graph.output)
        return intersect_node + intersect_output

    def remove_duplicate(self):
        self.output = remove_duplicate_list(self.output)
        self.nodes = remove_duplicate_list(self.nodes)
        self.input = remove_duplicate_list(self.input)


class CondContext:
    def __init__(self,
                 cond_scope,
                 pred_input,
                 true_body_graph,
                 false_body_graph,
                 switchs,
                 merges):
        self.cond_scope = cond_scope
        self.pred_input = pred_input
        self.true_body_graph = true_body_graph
        self.false_body_graph = false_body_graph
        self.switchs = switchs
        self.merges = merges
        # output: true output, false output
        self.true_output_mapping = {}
        self.false_output_mapping = {}
        # remove duplicate
        self.remove_duplicate()

    def remove_duplicate(self):
        # remove duplicate
        self.true_body_graph.remove_duplicate()
        self.false_body_graph.remove_duplicate()
        self.switchs = remove_duplicate_list(self.switchs)
        self.merges = remove_duplicate_list(self.merges)


class CondGraphMeta:
    def __init__(self,
                 true_graph_input,
                 true_output,
                 false_graph_input,
                 false_output):
        self.true_graph_input = true_graph_input
        self.true_output = true_output
        self.false_graph_input = false_graph_input
        self.false_output = false_output
        self.if_node = None


class CondGraphDict:
    # NOTE: keep the order for nest if
    COND_CONTEXT_DICT = OrderedDict()


class CondRewriter:
    def __init__(self, g):
        self.g = g
        self._if_input = {}

    def _common_name_scope(self, *nodes):
        return os.path.commonpath([n.name for n in nodes])

    def _output_info_from_output(self, output):
        """create a output info from output"""
        output_info = []
        for o in output:
            dtype = self.g.get_dtype(o)
            utils.make_sure(dtype, "the dtype for {} is None".format(o))
            shape = self.g.get_shape(o)
            utils.make_sure(shape is not None, "the shape for {} is None".format(o))
            output_info.append(utils.make_onnx_inputs_outputs(o, dtype, shape))
        return output_info

    def _trace_back(self, name_scope, merge_nodes):
        """trace back to the switch from merge nodes"""
        log.debug("trace back from [%s]", ",".join(n.name for n in merge_nodes))
        stack = [m for m in merge_nodes]
        downstream_graph = defaultdict(CondBodyGraph)
        # take down output info
        for merge_node in merge_nodes:
            for i in merge_node.input:
                input_node = self.g.get_node_by_output(i)
                downstream_graph[input_node].output.append(i)
        true_body_graph = CondBodyGraph()
        false_body_graph = CondBodyGraph()
        switchs = set()
        while stack:
            node = stack.pop()
            # NOTE: input, control input and input of added if node
            inputs = node.input + node.control_input + self._if_input.get(node, [])
            for inp in inputs:
                input_node = self.g.get_node_by_output(inp)
                # control input
                if not input_node:
                    log.debug("control input: %s", inp)
                    input_node = self.g.get_node_by_name(inp)
                    # downstream_graph[input_node].control_dependent = True
                utils.make_sure(input_node, "cannot find node {}".format(inp))
                if not input_node.name.startswith(name_scope):
                    raise ValueError(
                        "{} has different scope name from {}".format(input_node.name, name_scope)
                    )
                if self._is_merge(input_node):
                    raise ValueError("nest merge at {} in {}".format(input_node.name, name_scope))
                # stop at the first switch
                if self._is_switch(input_node):
                    log.debug("encounter the first switch: %s", input_node.name)
                    # false
                    if input_node.output[0] == inp:
                        false_body_graph.union(downstream_graph[node])
                        # NOTE: in case switch connects to merge directly
                        false_body_graph.union(downstream_graph[input_node])
                        false_body_graph.input.append(inp)
                        log.debug("=================false body graph===============")
                        log.debug(false_body_graph.nodes)
                        log.debug(false_body_graph.output)
                        log.debug(false_body_graph.input)
                    # true
                    else:
                        true_body_graph.union(downstream_graph[node])
                        true_body_graph.union(downstream_graph[input_node])
                        true_body_graph.input.append(inp)
                        log.debug("=================true body graph===============")
                        log.debug(true_body_graph.nodes)
                        log.debug(true_body_graph.output)
                        log.debug(true_body_graph.input)
                    switchs.add(input_node)
                else:
                    downstream_graph[input_node].nodes.append(input_node)
                    downstream_graph[input_node].union(downstream_graph[node])
                    stack.append(input_node)
        # one node cannot belong to both true and false graph
        intersection = true_body_graph.intersection(false_body_graph)
        if intersection:
            raise ValueError("true graph and false graph intersect at [{}]".format(
                ",".join(intersection)
            ))
        return true_body_graph, false_body_graph, list(switchs)

    def _workaround_for_placeholder(self, output):
        output_node = self.g.get_node_by_output(output)
        if output_node.type == "Placeholder" or \
                output_node.type == "Const":
            placeholder_id = self.g.make_node(
                "Identity",
                [output]
            )
            self._copy_dtype_and_shape(output, placeholder_id.output[0])
            self.g.replace_all_inputs(self.g.get_nodes(), output, placeholder_id.output[0])
            output = placeholder_id.output[0]
            all_nodes = self.g.get_nodes()
            all_nodes.append(placeholder_id)
            self.g.set_nodes(all_nodes)
        return output

    def _parse_cond(self, name_scope, merge_nodes):
        """parse condition subgraph for these merge nodes"""
        true_body_graph, false_body_graph, switchs = self._trace_back(name_scope, merge_nodes)
        # find pred output from switch
        pred_input = switchs[0].input[1]
        cond_context = CondContext(
            name_scope,
            pred_input,
            true_body_graph,
            false_body_graph,
            switchs,
            merge_nodes
        )
        return cond_context

    def _pair_output_with_merge(self, true_output, false_output, merges):
        """pair output according to the order of merges"""
        log.debug(
            "pair ture and false output according to the order of merge"
        )
        log.debug("true outuput: %s", true_output)
        log.debug("false output: %s", false_output)
        log.debug("merges: %s", [m.input for m in merges])
        utils.make_sure(
            len(true_output)==len(merges),
            "output info in true branch cannot match merges"
        )
        utils.make_sure(
            len(false_output)==len(merges),
            "output info in false branch cannot match merges"
        )
        paired_false_output = []
        paired_true_output = []
        for merge in merges:
            f_input = None
            if merge.input[0] in true_output:
                paired_true_output.append(merge.input[0])
                f_input = merge.input[1]
            elif merge.input[1] in true_output:
                paired_true_output.append(merge.input[1])
                f_input = merge.input[0]
            else:
                raise ValueError(
                    "No output info in true branch outputs to merge node: {}".format(merge.name)
                )
            if f_input in false_output:
                paired_false_output.append(f_input)
            else:
                raise ValueError(
                    "No output info in false branch outputs to merge node: {}".format(merge.name)
                )
        return paired_true_output, paired_false_output, merges

    def _copy_dtype_and_shape(self, old_output, new_output):
        self.g.copy_dtype(old_output, new_output)
        self.g.copy_shape(old_output, new_output)

    def _create_if_node(self, cond_context):
        """create a if node without graph attribute"""
        log.debug("create if node")
        if_node = self.g.make_node(
            "If",
            [cond_context.pred_input],
            op_name_scope=cond_context.cond_scope,
            outputs=[m.output[0] for m in cond_context.merges],
            skip_conversion=True
        )
        # take down inputs of if node for nest cond parsing
        self._if_input[if_node] = union_list(
            cond_context.true_body_graph.input,
            cond_context.false_body_graph.input

        )
        # take down output mapping
        paired_true_output, paired_false_output, _ = self._pair_output_with_merge(
            cond_context.true_body_graph.output,
            cond_context.false_body_graph.output,
            cond_context.merges
        )
        true_output_mapping = {}
        false_output_mapping = {}
        for i in range(len(cond_context.merges)):
            true_output_mapping[if_node.output[i]] = paired_true_output[i]
            false_output_mapping[if_node.output[i]] = paired_false_output[i]
        cond_context.true_output_mapping = true_output_mapping
        cond_context.false_output_mapping = false_output_mapping
        return if_node

    def _cut_off_connection(self, cond_context):
        """cut off switchs and merges"""
        nodes_to_remove = cond_context.switchs + cond_context.merges
        nodes_to_add = []
        log.debug("cut off switch connection")
        # replace switch with identity node
        for switch in cond_context.switchs:
            false_switch_id = self.g.make_node(
                "Identity",
                [switch.input[0]],
                outputs=[switch.output[0]],
                op_name_scope=cond_context.cond_scope
            )
            true_switch_id = self.g.make_node(
                "Identity",
                [switch.input[0]],
                outputs=[switch.output[1]],
                op_name_scope=cond_context.cond_scope
            )
            nodes_to_add.extend([false_switch_id, true_switch_id])
        # replace merge with if node
        log.debug("cut off merge connection")
        if_node = self._create_if_node(cond_context)
        nodes_to_add.append(if_node)
        cond_context.if_node = if_node
        return nodes_to_add, nodes_to_remove

    def _insert_id_on_input(self, inputs, op_name_scope=None):
        node_mapping = {}
        for inp in inputs:
            inp_id = self.g.make_node("Identity", [inp], op_name_scope=op_name_scope)
            self.g.replace_all_inputs(self.g.get_nodes(), inp, inp_id.output[0])
            node_mapping[inp] = inp_id
            self._copy_dtype_and_shape(inp, inp_id.output[0])
        return node_mapping

    def _add_id_on_input_output(self, cond_context):
        """add identities nodes to input and output as a marker for post rewriting"""
        # NOTE: input is the output of switch (identity after cutting off)
        # it's already a marker that don't need to be changed
        ids = []
        # ids.extend(
        #     self._insert_id_on_input(
        #         cond_context.true_body_graph.input,
        #         cond_context.cond_scope
        #     ).values()
        # )
        # ids.extend(
        #     self._insert_id_on_input(
        #         cond_context.false_body_graph.input,
        #         cond_context.cond_scope
        #     ).values()
        # )
        node_mapping = self._insert_id_on_input(
            cond_context.true_body_graph.output,
            cond_context.cond_scope
        )
        ids.extend(node_mapping.values())
        # update cond output mappings
        for outter_out, inner_out in cond_context.true_output_mapping.items():
            cond_context.true_output_mapping[outter_out] = \
                node_mapping[inner_out].output[0]
        cond_context.true_body_graph.output = cond_context.true_output_mapping.values()
        node_mapping = self._insert_id_on_input(
            cond_context.false_body_graph.output,
            cond_context.cond_scope
        )
        ids.extend(node_mapping.values())
        # update cond output mappings
        for outter_out, inner_out in cond_context.false_output_mapping.items():
            cond_context.false_output_mapping[outter_out] = \
                node_mapping[inner_out].output[0]
        cond_context.false_body_graph.output = cond_context.false_output_mapping.values()
        return ids

    def pre_run(self):
        # find all merge Op, merge ops with the same name scope belong to the same condition
        name_scope_merges = defaultdict(list)
        all_nodes = self.g.get_nodes()
        for n in all_nodes:
            if self._is_merge(n):
                name_scope = "/".join(n.name.split("/")[:-1])
                name_scope_merges[name_scope].append(n)
        # sort by name_scope, the longer the inner
        name_scope_merges = OrderedDict(sorted(name_scope_merges.items(), key=lambda x: x[0], reverse=True))
        # check if need rewrite
        if len(name_scope_merges.keys()) == 0:
            return all_nodes

        # list keep the order so that we can handle nest cond
        for name_scope, merge_nodes in name_scope_merges.items():
            cond_context = self._parse_cond(name_scope, merge_nodes)
            nodes_to_add, nodes_to_remove = self._cut_off_connection(cond_context)
            nodes_to_add.extend(self._add_id_on_input_output(cond_context))
            self._update_nodes(nodes_to_add, nodes_to_remove)
            # for post rewriter
            cond_context.remove_duplicate()
            CondGraphDict.COND_CONTEXT_DICT[cond_context.if_node] = cond_context
        log.debug("cond pre rewrite done")

        return self.g.get_nodes()

    def pre_rewrite(self):
        log.debug("enter cond pre rewrite")
        try:
            return self.pre_run()
        except Exception as ex:
            self._empty_cond_context_dict()
            raise ex

    def _update_nodes(self, nodes_to_add=[], nodes_to_remove=[]):
        all_nodes = self.g.get_nodes()
        nodes_to_add = remove_duplicate_list(nodes_to_add)
        nodes_to_remove = remove_duplicate_list(nodes_to_remove)
        for n in nodes_to_remove:
            if n in all_nodes:
                all_nodes.remove(n)
        all_nodes.extend(nodes_to_add)
        self.g.set_nodes(all_nodes)

    def _extract_cond_graph(self, cond_context, true_or_false=True):
        cond_body_graph = None
        if true_or_false:
            cond_body_graph = cond_context.true_body_graph
        else:
            cond_body_graph = cond_context.false_body_graph
        log.debug(
            "extract graph from %s to %s",
            cond_body_graph.output,
            cond_body_graph.input
        )
        branch_nodes = self.g.extract_sub_graph_nodes(
            cond_body_graph.output,
            cond_body_graph.input
        )
        if len(branch_nodes) == 0:
            # NOTE: since we added identities on input and output of each branch,
            # if we cannot find any node in one branch, there must be something wrong
            raise ValueError("{} branch is empty".format(true_or_false))
        return branch_nodes

    def _make_cond_graph(self, nodes, output, name):
        onnx_nodes = []
        output_shapes = {}
        dtypes = {}
        for n in nodes:
            for output_name in n.output:
                output_shapes[output_name] = self.g.get_shape(output_name)
                dtypes[output_name] = self.g.get_dtype(output_name)
        body_g = Graph(
            [],
            output_shapes=output_shapes,
            dtypes=dtypes
        )
        body_g.set_nodes(nodes)
        body_g.outputs = output
        body_g.topological_sort(body_g.get_nodes())
        return body_g.make_graph("tf cond sub graph", name)

    def _get_body_graph_output(self, output_mapping, node_output):
        body_graph_output = []
        for output in node_output:
            if output not in output_mapping:
                raise ValueError("cannot find mapping for {}".format(output))
            body_graph_output.append(output_mapping[output])
        return body_graph_output

    def _empty_cond_context_dict(self):
        while CondGraphDict.COND_CONTEXT_DICT:
            CondGraphDict.COND_CONTEXT_DICT.popitem()

    def post_rewrite(self):
        log.debug("enter cond pre rewrite")
        try:
            return self.post_run()
        except Exception as ex:
            self._empty_cond_context_dict()
            raise ex

    def post_run(self):
        log.debug("enter cond post rewriter")
        for if_node, cond_context in CondGraphDict.COND_CONTEXT_DICT.items():
            log.debug("post rewrite %s", cond_context.cond_scope)
            log.debug("===============post true graph=================")
            log.debug(cond_context.true_body_graph.output)
            log.debug(cond_context.true_body_graph.input)
            log.debug("===============post false graph=================")
            log.debug(cond_context.false_body_graph.output)
            log.debug(cond_context.false_body_graph.input)

            true_branch_nodes = self._extract_cond_graph(cond_context, True)
            false_branch_nodes = self._extract_cond_graph(cond_context, False)
            nodes_to_remove = true_branch_nodes + false_branch_nodes

            true_output = self._get_body_graph_output(
                cond_context.true_output_mapping,
                if_node.output
            )
            false_output = self._get_body_graph_output(
                cond_context.false_output_mapping,
                if_node.output
            )
            log.debug("true graph: %s", true_branch_nodes)
            log.debug("true input: %s", cond_context.true_body_graph.input)
            log.debug("true output: %s", true_output)
            log.debug("false graph: %s", false_branch_nodes)
            log.debug("false input: %s", cond_context.false_body_graph.input)
            log.debug("false output: %s", false_output)
            true_onnx_graph = self._make_cond_graph(
                true_branch_nodes,
                true_output,
                utils.make_name("{}_true_graph".format(cond_context.cond_scope))
            )
            false_onnx_graph = self._make_cond_graph(
                false_branch_nodes,
                false_output,
                utils.make_name("{}_false_graph".format(cond_context.cond_scope))
            )
            if_node.set_attr("then_branch", true_onnx_graph)
            if_node.set_attr("else_branch", false_onnx_graph)
            self._update_nodes(nodes_to_remove=nodes_to_remove)
        self._empty_cond_context_dict()
        return self.g.get_nodes()


    def _is_switch(self, node):
        return node.type == "Switch"

    def _is_merge(self, node):
        return node.type == "Merge"

    def _is_pack(self, node):
        return node.type == "Pack"


def rewrite_cond(g, ops):
    return CondRewriter(g).pre_rewrite()

def rewrite_cond_body_graph(g, ops):
    return CondRewriter(g).post_rewrite()
