#!/usr/bin/env sh

./build/tools/caffe train -gpu 3 --solver=examples/mnist/lenet_solver_elb_BAC.prototxt -weights=examples/mnist/lenet.caffemodel
