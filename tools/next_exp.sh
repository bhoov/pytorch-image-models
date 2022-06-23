#!/bin/bash

scancel $1
rm *.err *.out
sbatch lightning_sbatch.sbatch