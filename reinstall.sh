#!/bin/bash

rm -r build/
rm -r diff_gaussian_voxelization.egg-info/
pip uninstall diff-gaussian-voxelization -y && pip install .
