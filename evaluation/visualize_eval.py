import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

sns.set_theme(context = 'paper', style = 'darkgrid')

def load_data(path_1:str, path_2:str, path_1_type:str='DistilBERT Base', path_2_type:str='SNLI Finetuning') -> pd.DataFrame:
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)

    df1['Model'] = path_1_type
    df2['Model'] = path_2_type 

    df = pd.concat([df1, df2])
    return df 


def visualize_accuracy(df:pd.DataFrame, save_path:str=None) -> None:
    accuracy = df[df['metric'] == 'accuracy']
    f = sns.barplot(accuracy, x = 'Model', y = 'value')
    f.set(title = 'Test Set Accuracy by Model', xlabel = '', ylabel = 'Accuracy')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'accuracy_by_model.png'))
    plt.show()


def visualize_f1(df:pd.DataFrame, save_path:str=None) -> None:
    f1 = df[df['metric'] != 'accuracy']
    f = sns.barplot(f1, x = 'Model', y = 'value', hue = 'metric')
    f.legend(loc = 'lower left', title = 'Metric')
    f.set(title = 'Test Set F1 Score by Model', xlabel = '', ylabel = 'F1')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'f1_scores.png'))
    plt.show()




def main():
    df = load_data('evaluation/misinfo_test/base_evaluation.csv', 'evaluation/misinfo_test/pretrained_evaluation.csv') 
    visualize_accuracy(df, 'evaluation/misinfo_test')
    visualize_f1(df, 'evaluation/misinfo_test')

if __name__ == "__main__":
    main()