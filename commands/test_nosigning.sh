#!/bin/bash
#SBATCH --job-name=full5           # Job name
#SBATCH --mail-user=hannah.bull@limsi.fr                # Where to send mail
#SBATCH --cpus-per-task=40          # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Requesting X GPU(s)
#SBATCH --mem=90G     # Job memory request
#SBATCH --time=99:00:00                   # Time limit hrs:min:sec
#SBATCH --partition=gpu

export PATH=/users/hbull/.vscode-server/bin/899d46d82c4c95423fb7e10e68eba52050e30ba3/bin:/users/hbull/.vscode-server/bin/899d46d82c4c95423fb7e10e68eba52050e30ba3/bin:/users/hbull/miniconda3/envs/slalign/bin:/usr/condabin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/users/hbull/.local/bin:/users/hbull/bin:/users/hbull/.local/bin:/users/hbull/bin

source /users/hbull/miniconda3/etc/profile.d/conda.sh; conda activate slalign
cd /scratch/shared/beegfs/hbull/repos/sign_language_train

python main.py \
--gpu_id 0 \
--n_workers 32 \
--batch_size 64 \
--model 'transformerencoder' \
--dataset 'densefeatures' \
--trainer 'densetrainer' \
--train_ids 'data/bobsl_align_train.txt' \
--train_data_loc '/scratch/shared/beegfs/shared-datasets/bsltrain/features/bobsl/featurise_c8697_pltp1_0.5_a_d8hasentsyn_m8prajhasent_swin-s_pretkinetics-v0-stride0.0625/filtered' \
--train_labels_loc '/scratch/shared/beegfs/hbull/repos/sign_finder/no_signing_annotations/no_signing.pkl' \
--val_ids 'data/bobsl_align_val.txt' \
--val_data_loc '/scratch/shared/beegfs/shared-datasets/bsltrain/features/bobsl/featurise_c8697_pltp1_0.5_a_d8hasentsyn_m8prajhasent_swin-s_pretkinetics-v0-stride0.0625/filtered' \
--val_labels_loc '/scratch/shared/beegfs/hbull/repos/sign_finder/no_signing_annotations/no_signing.pkl' \
--test_ids 'data/bobsl_full.txt' \
--test_data_loc '/scratch/shared/beegfs/shared-datasets/bsltrain/features/bobsl/featurise_c8697_pltp1_0.5_a_d8hasentsyn_m8prajhasent_swin-s_pretkinetics-v0-stride0.0625/filtered' \
--test_labels_loc '/scratch/shared/beegfs/hbull/repos/sign_finder/no_signing_annotations/no_signing.pkl' \
--test_only \
--resume 'checkpoints_nosign/checkpoints/model_0000028128.pt' \
--stride 4 \
--input_len 25 