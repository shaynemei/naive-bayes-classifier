#!/bin/sh

if [ -z "$1" ]; then
	echo "Usage: build_NB2.sh training_data test_data prior_delta cond_prob_delta model_file sys_output > acc_file"
else
	/opt/python-3.6/bin/python3 build_NB2.py $1 $2 $3 $4 $5 $6
	#python3 build_NB2.py $1 $2 $3 $4 $5 $6
fi