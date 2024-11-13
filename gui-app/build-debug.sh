#!/bin/bash
THIS_SCRIPT_DIR=$(dirname $(readlink -f $0))
cd $THIS_SCRIPT_DIR
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug 
cmake --build build --config Debug