import os
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from typing import Literal

sns.set_theme(context = 'paper', style = 'darkgrid')

def plot_loss(path:str, 
              data_type:Literal['evaluation', 'train'] = 'evaluation', 
              save_path:str = None, 
              n_steps:int = 5000) -> None:
    
    df = pd.read_csv(path)
    df.columns = ['step', 'loss', 'loss_min', 'loss_max']
    df['step'] = df['step'] * n_steps

    f  = sns.lineplot(data = df, x = 'step', y = 'loss')
    f.set(title = f"{data_type.title()} Loss by Training Step", 
          xlabel = 'Training Step', 
          ylabel = 'Loss')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{data_type}_loss.png'))
    plt.show()


def plot_metric(path:str, 
                metric:Literal['f1', 'accuracy'], 
                save_path:str = None, 
                n_steps:int = 5000) -> None:

    df = pd.read_csv(path)
    df.columns = ['step', metric, f"{metric}_min", f"{metric}_max"]
    df['step'] = df['step'] * n_steps

    f = sns.lineplot(df, x = 'step', y = metric)
    f.set(title = f"{metric.title()} by Training Step", 
          xlabel = 'Training Step', 
          ylabel = metric.title())

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{metric}.png'))
    plt.show()


def plot_all_metrics(f1_path:str, 
                     accuracy_path:str, 
                     save_path:str = None) -> None:

    f1 = pd.read_csv(f1_path)
    f1.columns = ['step', 'metric_val', 'min', 'max']
    f1['metric'] = 'F1'

    accuracy = pd.read_csv(accuracy_path)
    accuracy.columns = ['step', 'metric_val', 'min', 'max']
    accuracy['metric'] = 'Accuracy'

    df = pd.concat([f1, accuracy])
    f = sns.lineplot(data = df, x = 'step', y = 'metric_val', hue = 'metric')
    f.set(title = 'Evaluation Macro F1 and Accuracy by Epoch', 
          xlabel = 'Training Step', 
          ylabel = 'Value')

    f.legend(title = 'Metric')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'all_metrics.png'))
    
    plt.show()


def compare_losses(df_1_path:str, 
                   df_2_path:str, 
                   df_1_type:str='Logical Inference', 
                   df_2_type:str='DistilBERT',
                   n_steps:int=5000,
                   save_path:str=None) -> None:
    
    df_1 = pd.read_csv(df_1_path)
    df_2 = pd.read_csv(df_2_path)

    df_1['model'] = df_1_type
    df_2['model'] = df_2_type 

    df_1.columns = ['step', 'loss', 'loss_min', 'loss_max', 'model']
    df_2.columns = ['step', 'loss', 'loss_min', 'loss_max', 'model']

    df = pd.concat([df_1, df_2], axis = 0)

    df['step'] = df['step'] * n_steps 

    f = sns.lineplot(df, x = 'step', y = 'loss', hue = 'model')
    f.set(title = 'Evaluation Loss by Model', xlabel = 'Step', ylabel = 'Loss')
    f.legend(title = 'Model')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'comparitive_loss.png'))

    plt.show()
compare_losses


def plot_all_runs(runs:list[str]) -> None:
    for run in runs:
        plot_loss(f'evaluation/runs/{run}/loss.csv', save_path=f'evaluation/runs/{run}')
        plot_metric(f'evaluation/runs/{run}/f1.csv', 'f1', save_path=f'evaluation/runs/{run}')
        plot_metric(f'evaluation/runs/{run}/accuracy.csv', 'accuracy', save_path=f'evaluation/runs/{run}')
        plot_all_metrics(f'evaluation/runs/{run}/f1.csv', f'evaluation/runs/{run}/accuracy.csv', save_path=f'evaluation/runs/{run}')

def main():
    runs = ['snli', 'liar_snli', 'liar_bert']
    #plot_all_runs(runs)

    compare_losses('evaluation/runs/liar_snli/loss.csv', 'evaluation/runs/liar_bert/loss.csv', save_path='evaluation/runs')

if __name__ == "__main__":
    main()