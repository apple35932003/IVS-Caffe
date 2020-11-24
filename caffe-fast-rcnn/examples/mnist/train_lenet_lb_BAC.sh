#!/usr/bin/env sh

./build/tools/caffe train -gpu 2 --solver=examples/mnist/lenet_solver_lb_BAC.prototxt -weights=examples/mnist/lenet.caffemodel
