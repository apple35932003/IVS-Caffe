#!/usr/bin/env sh

./build/tools/caffe train -gpu 2 --solver=examples/mnist/lenet_solver_IVS.prototxt 
