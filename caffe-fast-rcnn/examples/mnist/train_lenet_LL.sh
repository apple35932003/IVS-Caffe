#!/usr/bin/env sh

./build/tools/caffe train -gpu 3 --solver=examples/mnist/lenet_solver_LL.prototxt -weights=examples/mnist/lenet.caffemodel
