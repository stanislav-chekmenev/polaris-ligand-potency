#!/bin/bash

# create env from yaml
micromamba env create -f env.yml

# Remove the old version of torch
micromamba run -n polaris-env pip uninstall -y torch torchvision
micromamba run -n polaris-env pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Print the final message
echo "Environment 'polaris-env' created successfully!"
