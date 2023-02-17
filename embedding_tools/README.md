# Interactive comparisons of text embeddings

This is a collection of notebooks for creative visualisations of text embeddings/vectors produced by NLP models, in particular transformer based pre-trained language models (PLMs) such as BERT. This is not an exhaustive exploration of this space, and there exists many other routes to take for exploring embeddings.

## Pre-requisites

The majority of the packages will be covered by the root directories [requirements.txt](../requirements.txt)

### Notebooks
Each notebook is fairly self contained, but as a general overview:

#### Compare_PLM_embeddings
[`compare_PLM_embeddings.ipynb`](./compare_PLM_embeddings.ipynb) looks at the word and sentence embeddings for differently trained PLMs with a focus on patient safety report examples. This is primarily done through cosine similarity calculations and is adapted from [here](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial).

#### 2D and 3D visualisations of word embeddings
[`plotly_visualise_word_embeddings.ipynb`](./plotly_visualise_word_embeddings.ipynb) looks at generating contextualised and static word embeddings for different PLMs given a set of words. It then plots these embeddings in a 2D or 3D space by using a dimensionality reduction technique called multi-dimensional scaling.

#### 2D and 3D visualisations of word and sentence embeddings
[`Sentence_Embeddings_RoBERTa.ipynb`](./Sentence_Embeddings_RoBERTa.ipynb) is a notebook from [here](https://github.com/nidharap/Notebooks) and in its current form has not been edited, this is merely an example to pivot to your own models and dataset.
