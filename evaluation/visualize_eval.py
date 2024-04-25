import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

def visualize_eval(path_1:str, path_2:str, path_1_type:str='DistilBERT Base', path_2_type:str='SNLI Finetuning'):
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)

    df1['Model'] = path_1_type
    df2['Model'] = path_2_type 

    df = pd.concat([df1, df2])

    accuracy = df[df['metric'] == 'accuracy']

    f = sns.barplot(df, x = 'Model', y = 'value')
    plt.show()


def main():
    visualize_eval('evaluation/misinfo_test/base_evaluation.csv', 'evaluation/misinfo_test/pretrained_evaluation.csv') 

if __name__ == "__main__":
    main()