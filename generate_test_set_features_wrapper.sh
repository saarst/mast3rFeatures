#!/bin/bash
#SBATCH --gres=gpu:A40:1          # Request 1 gpu type A40
#SBATCH --mail-user=[user]@campus.technion.ac.il
#SBATCH --mail-type=ALL           # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="generate_test_set"
#SBATCH -o ./logs/out_job%A_%a.txt        # stdout goes to out_job.txt
#SBATCH -e ./logs/err_job%A_%a.txt        # stderr goes to err_job.txt


# Run your Python script with the specified delta and dynamically set exp_name
conda run -n mast3r python generate_test_set_features.py 


