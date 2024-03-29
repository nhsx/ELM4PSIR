# Datasets

Here are some notes of the classification datasets.

It will be much easier to preprocess and load data via [torchtext](https://github.com/pytorch/text), check out its documentation [here](https://pytorch.org/text/).

Here are statistics of some popular classification datasets:

| Dataset                | Classes | Train Samples | Test Samples | Total     | Download                                                     |
| ---------------------- | ------- | ------------- | ------------ | --------- | ------------------------------------------------------------ |
| AG News                | 4       | 120,000       | 7,600        | 127,600   | [Goole Drive](https://drive.google.com/file/d/0Bz8a_Dbh9QhbUDNpeUdjb0wxRms/view?usp=sharing) |
| IMDB                   | 2       | 25,000        | 25,000       | 50,000    | [Link](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) |
| SST-2                  | 2       | /             | /            | 94.2k     | [Link](http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip) |
| SST-5                  | 5       | /             | /            | 56.4k     | [Link](http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip) |
| TREC                   | 6 / 50  | 5,452         | 500          | 5,952     | [Link](https://cogcomp.seas.upenn.edu/Data/QA/QC/)           |


&nbsp;

## Text Classification

All of the following datasets can be downloaded [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) (Google Drive). They are proposed and described in this paper:

[**Character-level Convolutional Networks for Text Classification.**](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) *Xiang Zhang, et al.* NIPS 2015.

- **AG News**

  News articles, original data are from [here](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html).

  **4 Classes:** 0: World, 1: Sports, 2: Business, 3: Sci/Tech


&nbsp;

## Sentiment Analysis

- [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/)

  Proposed in paper:

  [**Learning Word Vectors for Sentiment Analysis.**](https://www.aclweb.org/anthology/P11-1015.pdf) *Andrew L. Maas, et al.* ACL 2011.

  **2 Classes:** Negative, Positive

  **samples:** train: 25,000, test: 25,000

  **Description:** Movie reviews, the ratings range from 1-10. A negative review has a score ≤ 4, and a positive review has a score ≥ 7.

- [SST](https://nlp.stanford.edu/sentiment/)

  Movie reviews.

  - SST-5 (Fine-grained)

    **5 Classes:** Very Negative, Negative, Neutral, Positive, Very Positive

    **samples:**  94.2k

  - SST-2 (Binary)

    **2 Classes:** Negative, Positive

    **samples:**  56.4k

    **Description:**  Same as SST-5 but with neutral reviews removed and binary labels.


&nbsp;

## Question Classification

- [TREC](https://cogcomp.seas.upenn.edu/Data/QA/QC/)

  A dataset for classifying questions into semantic categories.

  **samples:** train: 5,452, test: 500

  - TREC-6

    **6 Classes:** Abbreviation, Description, Entity, Human, Location, Numeric Value

  - TREC-50 (Fine-grained)

    **50 Classes**
