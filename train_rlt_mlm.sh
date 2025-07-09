#!/bin/bash

# --- SLURM Preamble: Requesting Resources ---

#SBATCH -J rlt_mlm_mini    # A descriptive name for your job
#SBATCH -o jobs_outs/rlt_mlm_mini.out
#SBATCH -e jobs_outs/rlt_mlm_mini.err
#! Which project should be charged:
#SBATCH --account=TKNOWLES-SL2-GPU
#! How many nodes and tasks:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#! Request a GPU. The partition determines the type of GPU.
#SBATCH -p ampere
#SBATCH --gres=gpu:1


#! How much wallclock time will be required? (HH:MM:SS)
#SBATCH --time=12:00:00 # Request 12 hours. Adjust based on your expected run time.

#! What types of email messages do you wish to receive?
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end
#SBATCH --mail-user=vr375@cam.ac.uk

# --- Start of commands ---

echo "======================================================="
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "======================================================="

# 1. Set up the environment
. /etc/profile.d/modules.sh      # Enables the module command
module purge                     # Clear all loaded modules
module load rhel8/default-amp    # Load basic environment for the ampere partition
module load cuda/11.8            # Load the required CUDA version

# 2. Activate your Conda environment
# (Assuming your .bashrc sets up conda correctly)

source /home/vr375/rds/hpc-work/lobster-env/bin/activate


# 4. Navigate to your project directory
export REPO_DIR=/home/vr375/code/lobster # Use an environment variable for clarity
cd $REPO_DIR
echo "Current directory: $(pwd)"
echo "Running training script..."

# 5. Execute your Python training script
#    This command runs the final, complete training pipeline we designed.
uv run --active lobster_train

echo "======================================================="
echo "Job Finished: $(date)"
echo "======================================================="
