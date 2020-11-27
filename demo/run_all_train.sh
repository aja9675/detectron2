#!/bin/bash

# no changes
python our_demo.py --activation=relu --loss=BCE --run_type=train
# just activation
python our_demo.py --activation=sps --loss=BCE --run_type=train
# just loss
python our_demo.py --activation=relu --loss=reg --run_type=train
# both used
python our_demo.py --activation=sps --loss=reg --run_type=train
