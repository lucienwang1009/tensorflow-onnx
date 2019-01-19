# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Unit Tests for custom rnns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops
from backend_test_base import Tf2OnnxBackendTestBase

# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test
# pylint: disable=abstract-method,arguments-differ

class CondTests(Tf2OnnxBackendTestBase):

    def test_cond_replicate_output(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        res = tf.cond(x[0] < y[0], lambda: [x, x], lambda: [y, y], name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_const(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        true_const = tf.constant(True, name="true_const", dtype=tf.bool)
        def cond_graph():
            with tf.name_scope("cond_graph", "cond_graph", [x, y]):
                b = tf.constant(np.array([2,1,3], dtype=np.float32), name="b", dtype=tf.float32)
                return b
        res = tf.cond(true_const, lambda: x+y, cond_graph, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_complex(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        def cond_graph():
            b = tf.constant(np.array([1], dtype=np.float), dtype=tf.float32)
            z = tf.pow(x, b)
            z1 = tf.abs(x + y)
            return tf.cond(x[0] > y[0], lambda: z[1], lambda: z1[2])
        res = x[2] * tf.cond(x[0]<y[0], cond_graph, cond_graph, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"] #, "output1:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        res = tf.cond(x[0]<y[0], lambda: x+y, lambda: x-y, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"] #, "output1:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_split_graph(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        def cond_graph():
            with tf.name_scope("cond_graph", "cond_graph", [x, y]):
                b = tf.constant(10, name="b", dtype=tf.float32)
                return b
        res = tf.cond(x[0]<y[0], cond_graph, cond_graph, name="test_cond")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"] #, "output1:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_multi_merge(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        res = tf.cond(x[0]<y[0], lambda: [x, x+y], lambda: [x, x-y], name="test")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"] #, "output1:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_case_multi_merge(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        res = tf.case([(tf.reduce_all(x<1), lambda: [x+y, x-y]), (tf.reduce_all(y>0), lambda: [tf.abs(x), tf.square(y)])],
                      default=lambda: [x, y], name="test_case")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"] #, "output1:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_case(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        res = tf.case([(tf.reduce_all(x<1), lambda: x+y), (tf.reduce_all(y>0), lambda: tf.square(y))],
                      default=lambda: x, name="test_case")
        _ = tf.identity(res, name="output")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"] #, "output1:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_case_temp(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        z_val = np.array(5, dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        # b = tf.constant(5, name="b", dtype=tf.float32)
        # true_const = tf.constant(True, name="true_const", dtype=tf.bool)
        def cond_graph():
            with tf.name_scope("cond_graph", "cond_graph", [x]):
                a = tf.constant(10, name="a", dtype=tf.float32)
                return x + a
                # z = tf.identity(x, name="x_id")
                # return tf.cond(x[1] < y[1], lambda: x, lambda: y, name="test_inner")
        res = tf.case([(tf.reduce_all(x<1), lambda: x), (tf.reduce_all(y>0), lambda: x+y)])
        # res = tf.cond(a<b, lambda: tf.add(x,y), lambda: tf.square(y))
        # res1 = tf.cond(tf.reduce_any(x<y), cond_graph, lambda: tf.square(y))
        _ = tf.identity(res, name="output")
        #_ = tf.identity(res1, name="output1")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"] #, "output1:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)

    def test_cond_temp(self):
        x_val = np.array([1,2,3], dtype=np.float32)
        y_val = np.array([4,5,6], dtype=np.float32)
        z_val = np.array(5, dtype=np.float32)
        x = tf.placeholder(tf.float32, x_val.shape, name="input_1")
        y = tf.placeholder(tf.float32, y_val.shape, name="input_2")
        # b = tf.constant(5, name="b", dtype=tf.float32)
        true_const = tf.constant(True, name="true_const", dtype=tf.bool)
        def cond_graph():
            with tf.name_scope("cond_graph", "cond_graph", [x, y]):
                a = tf.constant(True, name="a", dtype=tf.bool)
                b = tf.constant(10, name="b", dtype=tf.float32)
                # z = tf.identity(x, name="x_id")
                # return tf.cond(x[1] < y[1], lambda: x, lambda: y, name="test_inner")
                # return tf.cond(a, lambda: b, lambda: b)
                return b
        res = tf.cond(x[0]<y[0], lambda: b, lambda: b, name="test")
        # res = tf.cond(a<b, lambda: tf.add(x,y), lambda: tf.square(y))
        # res1 = tf.cond(tf.reduce_any(x<y), cond_graph, lambda: tf.square(y))
        _ = tf.identity(res, name="output")
        #_ = tf.identity(res1, name="output1")

        feed_dict = {"input_1:0": x_val, "input_2:0": y_val}
        input_names_with_port = ["input_1:0", "input_2:0"]
        output_names_with_port = ["output:0"] #, "output1:0"]
        self.run_test_case(feed_dict, input_names_with_port, output_names_with_port)


if __name__ == '__main__':
    Tf2OnnxBackendTestBase.trigger(CondTests)
