#!/bin/bash
#
# Script Name: geo-nlp-env.sh
# Description: This script sets up the environment geo-nlp
# Date: 2025-10-21
# Version: 1.0
# Usage: ./env.sh

# Create env
mamba create -n geo-nlp --file requirements.txt
 # Alternative:
 # mamba update --name geo-nlp --file requirements.txt
mamba activate geo-nlp
# https://stackoverflow.com/questions/76722680/what-is-the-best-way-to-combine-conda-with-standard-python-packaging-tools-e-g/76722681