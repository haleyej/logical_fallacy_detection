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
python3 evaluate_misinfo_detection.py --data_path='../data/liar_dataset/test.tsv' --model_type='base'
echo "pretrained model"
#FILL IN LATER
python3 evaluate_misinfo_detection.py --data_path='../data/liar_dataset/test.tsv' --model_type='pretrained'