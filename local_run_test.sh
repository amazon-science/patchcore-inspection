#!/bin/bash

# To be safe, switch into the folder that contains this script.
cd "$( cd "$( dirname "$0" )" && pwd )"

env PYTHONPATH=./src python3 -m pytest -v
