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
python3 snli_fine_tuning.py --train_data_path='../data/snli/snli_1.0_train.txt' --eval_data_path='../data/snli/snli_1.0_dev.txt' --output_dir='snli_logging' --logging-dir='snli_outputs'