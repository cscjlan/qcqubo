#!/bin/bash

set -exu
flags="-xhip --std=c++20 -O3"

if ! command -v CC &> /dev/null
then
    CC=hipcc
else
    CC=CC
fi

$CC $flags -o qubo -Isrc src/main.cpp
$CC $flags -o tests -Isrc src/tests.cpp
