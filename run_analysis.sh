#!/bin/bash
#SBATCH --job-name=MovieAnalysis     # Job name
#SBATCH --output=movie_analysis_%j.out   # Standard output file
#SBATCH --error=movie_analysis_%j.err    # Standard error log
#SBATCH --partition=gpu_a100              # Use the A100 GPU partition
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=55GB                   # Memory per node (55GB should be sufficient for 4-bit 70B model + system)
#SBATCH --gres=gpu:1                 # Using 1 GPU card (Correct syntax from CityU guide)
#SBATCH --time=04:00:00              # Wall clock time limit (4 hours)

# Load necessary modules
module purge
# Load CUDA module with its components. This might be dependent on HPC configuration.
module load cuda/12.1 cuda/blas/12.1 cuda/fft/12.1

# --- CONDA INITIALIZATION ---
# This directly sources the Conda initialization script.
source /gpfs1/home/share/shared/common_software_stack/packages/miniconda3/25/etc/profile.d/conda.sh
# Activate your Conda environment. This should now work as conda is initialized.
conda activate movie_env

# --- PROXY SETTINGS REMOVED ---
# Internet access is not allowed on compute nodes. Model is downloaded locally.
# No need for HTTP_PROXY, HTTPS_PROXY, NO_PROXY

# Set Hugging Face token for model authentication and download (still good practice)
export HF_TOKEN="hf_YOUR_ACTUAL_HUGGING_FACE_TOKEN" # <<< IMPORTANT: REPLACE THIS WITH YOUR TOKEN

# Navigate to your project directory (absolute path for robustness)
cd /home/tcwong383/movie_value_analysis/

# Run your Gradio application
echo "Starting Gradio application (app.py)..."
python app.py