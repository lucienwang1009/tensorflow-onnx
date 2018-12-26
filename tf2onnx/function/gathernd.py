# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.tf2onnx - gathernd op conversion
"""
import sys
import numpy as np
from onnx import helper, onnx_pb
from onnx.onnx_pb import TensorProto
from tf2onnx import utils


# pylint: disable=useless-return,broad-except,logging-not-lazy,unused-argument,missing-docstring
def make_gathernd_inner_loop(ctx, params, index, dtype):
    # gather_cur = params
    # for (int i=0; i<size(index); i++)
    #   gather_res = gather(gather_cur, index[i])
    nodes = []
    trip_node = ctx.make_node("Size", [index.output[0]])
    nodes.append(trip_node.op)
    cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=np.bool))
    body_inputs = [helper.make_tensor_value_info("inner_i", onnx_pb.TensorProto.INT64, []),
                   helper.make_tensor_value_info("inner_cond", onnx_pb.TensorProto.BOOL, []),
                   helper.make_tensor_value_info("gathernd_cur", dtype, [])]
    body_outputs = [helper.make_tensor_value_info("inner_cond_out", onnx_pb.TensorProto.BOOL, [],),
                    helper.make_tensor_value_info("inner_res", dtype, [])]
    body_nodes = []
    index_i = ctx.make_node("Gather", [index.output[0], "inner_i"], attr={"axis": 0})
    gather = ctx.make_node("Gather", ["gathernd_cur", index_i.output[0]], attr={"axis": 0})
    squeeze = ctx.make_node("Squeeze", [gather.output[0]], attr={"axes": [0]}, outputs=["inner_res"])
    body_nodes.extend([index_i.op, gather.op, squeeze.op])
    body_nodes.append(utils.make_onnx_identity("inner_cond", "inner_cond_out"))
    body_graph = helper.make_graph(body_nodes, utils.make_name("gathernd_inner_body"), body_inputs, body_outputs)
    inner_loop = ctx.make_node("Loop", [trip_node.output[0],
                                        cond_const.output[0],
                                        params],
                               attr={"body": body_graph})
    nodes.append(inner_loop.op)
    return nodes, inner_loop

def gathernd_op(ctx, node, name, args):
    """GatherNd op."""
    # T output = GatherNd(T Input, INT32/INT64 indices)
    nodes = []
    params = node.input[0]
    indices = node.input[1]
    # same as the attr Tparams
    node_dtype = ctx.get_dtype(params)
    utils.make_sure(node_dtype, "Dtype of {} is None".format(indices))
    # reshape indices into [sum(indices[:-1]), indices[-1]]
    indices_shape = ctx.make_node("Shape", [indices], dtypes=[TensorProto.INT64])
    outter_shape = ctx.make_node("Slice",
                                 [indices_shape.output[0]],
                                 attr={"axes": [0], "ends": [-1], "starts": [0]},
                                 dtypes=[TensorProto.INT64])
    inner_shape = ctx.make_node("Slice",
                                [indices_shape.output[0]],
                                attr={"axes": [0], "ends": [sys.maxsize], "starts": [-1]},
                                dtypes=[TensorProto.INT64])
    outter_shape_sum = ctx.make_node("ReduceSum",
                                     [outter_shape.output[0]],
                                     attr={"axes": [0], "keepdims": 1},
                                     dtypes=[TensorProto.INT64])
    flatten_shape = ctx.make_node("Concat",
                                  [outter_shape_sum.output[0], inner_shape.output[0]],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    flatten_indices = ctx.make_node("Reshape", [indices, flatten_shape.output[0]])
    nodes.extend([indices_shape, outter_shape, inner_shape, outter_shape_sum, flatten_shape, flatten_indices])
    # outter loop for each index
    # for (int i=0; i<outter_shape_sum; i++) inner_loop(params, flatten_indices[i])
    cond_const = ctx.make_const(utils.make_name("cond"), np.ones((), dtype=np.bool))
    dummy_const = ctx.make_const(utils.make_name("dummy"), np.ones((), dtype=np.int64))
    body_inputs = [helper.make_tensor_value_info("i", onnx_pb.TensorProto.INT64, []),
                   helper.make_tensor_value_info("cond", onnx_pb.TensorProto.BOOL, []),
                   helper.make_tensor_value_info("params", node_dtype, [])]
    body_outputs = [helper.make_tensor_value_info("cond_out", onnx_pb.TensorProto.BOOL, [],),
                    helper.make_tensor_value_info("params_out", node_dtype, []),
                    helper.make_tensor_value_info("result", node_dtype, [])]
    body_nodes = []
    index = ctx.make_node("Gather", [flatten_indices.output[0], "i"], attr={"axis": 0})
    index_squeeze = ctx.make_node("Squeeze", [index.output[0]], attr={"axes": [0]})
    # inner loop to gather result
    inner_loop_nodes, inner_loop = make_gathernd_inner_loop(ctx, "params", index_squeeze, node_dtype)
    body_nodes.extend([index.op, index_squeeze.op])
    body_nodes.extend(inner_loop_nodes)
    body_nodes.append(utils.make_onnx_identity("cond", "cond_out"))
    body_nodes.append(utils.make_onnx_identity("params", "params_out"))
    body_nodes.append(utils.make_onnx_identity(inner_loop.output[0], "result"))
    body_graph = helper.make_graph(body_nodes, utils.make_name("gathernd_body"), body_inputs, body_outputs)
    gathernd_loop = ctx.make_node("Loop",
                                  [outter_shape_sum.output[0], cond_const.output[0], params],
                                  output_count=2,
                                  attr={"body": body_graph})
    nodes.append(gathernd_loop)
    # reshape to target shape
    # output shape of gathernd: indices.shape[:-1] + gathernd_output.shape[1:]
    inner_loop_shape = ctx.make_node("Shape", [gathernd_loop.output[1]], dtypes=[TensorProto.INT64])
    # workaround in case gathernd_loop is 1-dimensional
    one_const = ctx.make_const(utils.make_name("one"), np.array([1], dtype=np.int64))
    inner_loop_shape_ = ctx.make_node("Concat",
                                      [inner_loop_shape.output[0], one_const.output[0]],
                                      attr={"axis": 0},
                                      dtypes=[TensorProto.INT64])
    output_inner_shape = ctx.make_node("Slice",
                                       [inner_loop_shape_.output[0]],
                                       attr={"axes": [0], "ends": [sys.maxsize], "starts": [1]},
                                       dtypes=[TensorProto.INT64])
    output_shape_ = ctx.make_node("Concat",
                                  [outter_shape.output[0], output_inner_shape.output[0]],
                                  attr={"axis": 0},
                                  dtypes=[TensorProto.INT64])
    output_shape = ctx.make_node("Slice",
                                 [output_shape_.output[0]],
                                 attr={"axes": [0], "ends": [-1], "starts": [0]},
                                 dtypes=[TensorProto.INT64])
    output_reshape = ctx.make_node("Reshape",
                                   [gathernd_loop.output[1], output_shape.output[0]],
                                   outputs=[node.output[0]])
    nodes.extend([inner_loop_shape,
                  inner_loop_shape_,
                  output_inner_shape,
                  output_shape_,
                  output_shape,
                  output_reshape])
    return nodes
