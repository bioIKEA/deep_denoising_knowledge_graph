#!/bin/bash

# case 1
python testRun.py 0 1 0;

# case 2
# (case 1 and 2) check the difference of mask in cost function
python testRun.py 1 1 0;

# case 3
# (case 1, 3) check the difference of more positive edges
python testRun.py 0 1 1

# case 4
# (case 3, 4) check the difference of weight
python testRun.py 1 0.1 1
