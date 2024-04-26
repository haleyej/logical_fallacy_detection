from typing import Literal

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

from snli_fine_tuning import load_snli


def load_liar(path:str) -> tuple[list[str]]:
    labels_map = {
            'true': 1, 
            'mostly-true': 1, 
            'half-true': 1, 
            'barely-true': 1 , 
            'false': 0, 
            'pants-fire': 0
        }

    texts = []
    labels = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("\t")
            texts.append(line[2])
            labels.append(labels_map[line[1]])
    
    return texts, labels


def get_baseline_scores(train_data:tuple[list], 
                        test_data:tuple[list], 
                        baseline_type:Literal['nb', 'dummy']) -> tuple[float]:
    enc = CountVectorizer()
    train_texts, train_labels = train_data
    test_texts, test_labels = test_data 

    train_vectors = enc.fit_transform(train_texts)
    test_vectors = enc.transform(test_texts)

    if baseline_type == 'nb': 
        gnb = GaussianNB()
        gnb.fit(train_vectors.toarray(), train_labels)
        predictions = gnb.predict(test_vectors.toarray())
    elif baseline_type == 'dummy':
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(train_vectors.toarray(), train_labels)
        predictions = clf.predict(test_vectors.toarray())

    accuracy = accuracy_score(predictions, test_labels)
    f1 = f1_score(predictions, test_labels)

    return accuracy, f1 


def main():
    snli_train = load_snli('data/snli/snli_1.0_train.txt')
    snli_test = load_snli('data/snli/snli_1.0_test.txt')

    snli_dummy = get_baseline_scores(snli_train, snli_test, 'dummy')
    snli_nb = get_baseline_scores(snli_train, snli_test, 'nb')

    liar_train = load_liar('data/liar_dataset/train.tsv')
    liar_test = load_liar('data/liar_dataset/test.tsv')

    liar_dummy = get_baseline_scores(liar_train, liar_test, 'dummy')
    liar_nb = get_baseline_scores(liar_train, liar_test, 'dummy')


    print('SNLI DATASET')
    print(f"NAIVE BAYES: {snli_nb}")
    print(f"DUMMY CLASSIFIERS: {snli_dummy}")
    print('')
    print('-------------')
    print('LIAR DATASET')
    print(f"NAIVE BAYES: {liar_nb}")
    print(f"DUMMY CLASSIFIERS: {liar_dummy}")


if __name__ == "__main__":
    main()