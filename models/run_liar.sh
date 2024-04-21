!/bin/bash
# The interpreter used to execute the script

#SBATCH” directives that convey submission options:

#SBATCH --job-name=demo
#SBATCH --account=eecs595w24_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:03:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=bert_train.out

# The application(s) to execute along with its input arguments and options:

/bin/hostname
echo "starting job"
nvidia-smi
echo "base model"
python3 liar_fine_tuning.py --train_data_path='../data/liar_dataset/train.tsv' --eval_data_path='../data/liar_dataset/valid.tsv'
echo "pretrained model"
# FILL IN MORE DETAILS LATER
python3 liar_fien_tuning.py --train_data_path='../data/liar_dataset/train.tst' --eval_data_path='../data/liar_dataset/valid.tsv' --model_checkpoint=''