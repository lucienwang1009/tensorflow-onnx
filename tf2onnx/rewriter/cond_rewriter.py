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
from tf2onnx.rewriter.loop_rewriter_base import LoopRewriterBase

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tf2onnx.rewriter.cond_rewriter_base")
# log.setLevel(logging.DEBUG)

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test

class CondGraphContext:
    def __init__(self):
        self.output = set()
        self.nodes = set()

    def union(self, cond_body_graph):
        self.output |= cond_body_graph.output
        self.nodes |= cond_body_graph.nodes

    def intersection(self, cond_body_graph):
        intersect_node = [
            n.name for n in self.nodes.intersection(cond_body_graph.nodes)
        ]
        intersect_output = self.output.intersection(cond_body_graph.output)
        return list(intersect_node) + list(intersect_output)

    def remove_duplicate(self):
        self.output = remove_duplicate_list(self.output)
        self.nodes = remove_duplicate_list(self.nodes)


class CondContext:
    def __init__(self, cond_scope, pred_input, true_graph_context,
                 false_graph_context, switchs, merges):
        self.cond_scope = cond_scope
        self.pred_input = pred_input
        self.true_graph_context = true_graph_context
        self.false_graph_context = false_graph_context
        self.switchs = set(switchs)
        # list of Merge in order
        self.merges = merges
        self.if_node = None
        # output: true output, false output
        self.true_graph = None
        self.false_graph = None


class CondRewriter:
    def __init__(self, g):
        self.g = g

    def _trace_back(self, name_scope, merge_nodes):
        """trace back to the switch from merge nodes"""
        log.debug("trace back from [%s]", ",".join(n.name for n in merge_nodes))
        stack = [m for m in merge_nodes]
        downstream_graph = defaultdict(CondGraphContext)
        true_graph_context = CondGraphContext()
        false_graph_context = CondGraphContext()
        # take down output info
        for merge_node in merge_nodes:
            for i in merge_node.input:
                input_node = self.g.get_node_by_output(i)
                downstream_graph[input_node].output.add(i)
                # if switch connects to merge directly
                if self._is_switch(input_node):
                    if i == input_node.output[0]:
                        false_graph_context.output.add(i)
                    else:
                        true_graph_context.output.add(i)
        switchs = set()
        while stack:
            node = stack.pop()
            # NOTE: input and input of added if node
            inputs = node.input + node.get_implicit_inputs() # + node.control_input
            for inp in inputs:
                input_node = self.g.get_node_by_output(inp)
                # control input
                if not input_node:
                    log.debug("control input: %s", inp)
                    input_node = self.g.get_node_by_name(inp)
                print(inp)
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
                        false_graph_context.union(downstream_graph[node])
                        log.debug("=================false body graph===============")
                        log.debug(false_graph_context.nodes)
                        log.debug(false_graph_context.output)
                    # true
                    else:
                        true_graph_context.union(downstream_graph[node])
                        log.debug("=================true body graph===============")
                        log.debug(true_graph_context.nodes)
                        log.debug(true_graph_context.output)
                    switchs.add(input_node)
                    self._workaround_for_placeholder(input_node.input[0])
                else:
                    downstream_graph[input_node].nodes.add(input_node)
                    downstream_graph[input_node].union(downstream_graph[node])
                    stack.append(input_node)
        # one node cannot belong to both true and false graph
        intersection = true_graph_context.intersection(false_graph_context)
        if intersection:
            raise ValueError("true graph and false graph intersect at [{}]".format(
                ",".join(intersection)
            ))
        return true_graph_context, false_graph_context, switchs

    def _workaround_for_placeholder(self, output):
        output_node = self.g.get_node_by_output(output)
        if output_node.type == "Placeholder" or \
                output_node.type == "Const":
            placeholder_id = self.g.make_node(
                "Identity",
                [output],
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
        true_graph_context, false_graph_context, switchs = self._trace_back(name_scope, merge_nodes)
        # find pred output from any switch
        pred_input = list(switchs)[0].input[1]
        return pred_input, true_graph_context, false_graph_context, switchs

    def _pair_output_with_merge(self, true_output, false_output, merges):
        """pair output according to the order of merges"""
        log.debug(
            "pair ture and false output according to the order of merge"
        )
        log.debug("true outuput: %s", true_output)
        log.debug("false output: %s", false_output)
        log.debug("merges: %s", [m.input for m in merges])
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

    def _set_branch_graph(self, cond_context):
        """set body graph for each branch"""
        log.debug("set graph for if branchs")
        paired_true_output, paired_false_output, _ = self._pair_output_with_merge(
            cond_context.true_graph_context.output,
            cond_context.false_graph_context.output,
            cond_context.merges
        )
        cond_context.true_graph = utils.construct_graph_from_nodes(
            self.g,
            list(cond_context.true_graph_context.nodes),
            paired_true_output,
            [self.g.get_shape(out) for out in paired_true_output],
            [self.g.get_dtype(out) for out in paired_true_output]
        )
        cond_context.false_graph = utils.construct_graph_from_nodes(
            self.g,
            list(cond_context.false_graph_context.nodes),
            paired_false_output,
            [self.g.get_shape(out) for out in paired_false_output],
            [self.g.get_dtype(out) for out in paired_false_output]
        )
        cond_context.if_node.set_body_graph_as_attr("then_branch", cond_context.true_graph)
        cond_context.if_node.set_body_graph_as_attr("else_branch", cond_context.false_graph)

    def _cut_off_connection(self, cond_context):
        """cut off switchs and merges"""
        nodes_to_remove = list(cond_context.switchs) + list(cond_context.merges)
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
            cond_context.false_graph_context.nodes.add(false_switch_id)
            true_switch_id = self.g.make_node(
                "Identity",
                [switch.input[0]],
                outputs=[switch.output[1]],
                op_name_scope=cond_context.cond_scope
            )
            cond_context.true_graph_context.nodes.add(true_switch_id)
            nodes_to_add.extend([false_switch_id, true_switch_id])
        # replace merge with if node
        log.debug("cut off merge connection")
        cond_context.if_node = self.g.make_node(
            "If",
            [cond_context.pred_input],
            op_name_scope=cond_context.cond_scope,
            outputs=[m.output[0] for m in cond_context.merges],
            skip_conversion=False
        )
        nodes_to_add.append(cond_context.if_node)
        return nodes_to_add, nodes_to_remove

    def run(self):
        """tf.cond rewriter"""
        # find all merge Op, merge ops with the same name scope belong to the same condition
        name_scope_merges = defaultdict(list)
        all_nodes = self.g.get_nodes()
        for n in all_nodes:
            if self._is_merge(n):
                name_scope = utils.tf_name_scope(n.name)
                name_scope_merges[name_scope].append(n)
        # sort by name_scope, the longer the inner
        name_scope_merges = OrderedDict(sorted(name_scope_merges.items(), key=lambda x: x[0], reverse=True))
        # check if need rewrite
        if len(name_scope_merges.keys()) == 0:
            return all_nodes

        # list keep the order so that we can handle nest cond
        for name_scope, merge_nodes in name_scope_merges.items():
            pred_input, true_graph_context, false_graph_context, switchs = \
                self._parse_cond(name_scope, merge_nodes)
            cond_context = CondContext(
                name_scope,
                pred_input,
                true_graph_context,
                false_graph_context,
                switchs,
                merge_nodes
            )
            nodes_to_add, nodes_to_remove = self._cut_off_connection(cond_context)
            self._set_branch_graph(cond_context)
            nodes_to_remove.extend(
                list(cond_context.true_graph_context.nodes) + \
                list(cond_context.false_graph_context.nodes)
            )
            self._update_nodes(nodes_to_add, nodes_to_remove)
        log.debug("cond pre rewrite done")

        return self.g.get_nodes()

    def rewrite(self):
        log.debug("enter cond pre rewrite")
        return self.run()

    def _update_nodes(self, nodes_to_add=[], nodes_to_remove=[]):
        all_nodes = self.g.get_nodes()
        nodes_to_add = list(set(nodes_to_add))
        nodes_to_remove = list(set(nodes_to_remove))
        for n in nodes_to_remove:
            if n in all_nodes:
                all_nodes.remove(n)
        all_nodes.extend(nodes_to_add)
        self.g.set_nodes(all_nodes)

    def _is_switch(self, node):
        return node.type == "Switch"

    def _is_merge(self, node):
        return node.type == "Merge"


def rewrite_cond(g, ops):
    return CondRewriter(g).rewrite()
