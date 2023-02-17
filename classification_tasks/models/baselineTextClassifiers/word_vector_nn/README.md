> ### This section of the project codebase is an reworking of [Renovamen/Text-Classification](https://github.com/Renovamen/Text-Classification/)

> #### See the original code base for more dataset implementations.

---

# Text Classification
PyTorch re-implementation of some text classificaiton models.


## Supported Models

Train the following models by editing `model_name` item in config files ([here](https://github.com/Renovamen/Text-Classification/tree/master/configs) are some example config files). Click the link of each for details.

- [**Hierarchical Attention Networks (HAN)**](https://github.com/Renovamen/Text-Classification/tree/master/models/HAN) (`han`)

    **Hierarchical Attention Networks for Document Classification.** *Zichao Yang, et al.* NAACL 2016. [[Paper]](https://www.aclweb.org/anthology/N16-1174.pdf)

- [**fastText**](https://github.com/Renovamen/Text-Classification/tree/master/models/fastText) (`fasttext`)

    **Bag of Tricks for Efficient Text Classification.** *Armand Joulin, et al.* EACL 2017. [[Paper]](https://www.aclweb.org/anthology/E17-2068.pdf) [[Code]](https://github.com/facebookresearch/fastText)

- [**Bi-LSTM + Attention**](https://github.com/Renovamen/Text-Classification/tree/master/models/AttBiLSTM) (`attbilstm`)

    **Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.** *Peng Zhou, et al.* ACL 2016. [[Paper]](https://www.aclweb.org/anthology/P16-2034.pdf)

- [**TextCNN**](https://github.com/Renovamen/Text-Classification/tree/master/models/TextCNN) (`textcnn`)

    **Convolutional Neural Networks for Sentence Classification.** *Yoon Kim.* EMNLP 2014. [[Paper]](https://www.aclweb.org/anthology/D14-1181.pdf) [[Code]](https://github.com/yoonkim/CNN_sentence)

- [**Transformer**](https://github.com/Renovamen/Text-Classification/tree/master/models/Transformer) (`transformer`)

    **Attention Is All You Need.** *Ashish Vaswani, et al.* NIPS 2017. [[Paper]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [[Code]](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)


## Requirements

First, make sure your environment is installed with:

- Python >= 3.5

Then install requirements:

```bash
pip install -r requirements.txt
```
N.B. We have included the specific requirements of the orignal repository here, and it is benefitial to make a seperate environment for this section.

## Dataset

Currently, the following dataset is supported:

- **Incident report binary severity prediction**

Set the path to the create binary severity classification dataset (`dataset_path`) in your config files.

We also include the code and configs for the following tasks:

- AG News (Click [here](./notes/datasets.md) for details of these datasets)

You should download and unzip it first, then set their path (`dataset_path`) in your config files. If you would like to use other datasets, they may have to be stored in the same format as the above mentioned datasets.


## Pre-trained Word Embeddings

If you would like to use pre-trained word embeddings (like [GloVe](https://github.com/stanfordnlp/GloVe)), just set `emb_pretrain` to `True` and specify the path to pre-trained vectors (`emb_folder` and `emb_filename`) in your config files. You could also choose to fine-tune word embeddings or not with by editing `fine_tune_embeddings` item.

Or if you want to randomly initialize the embedding layer's weights, set `emb_pretrain` to `False` and specify the embedding size (`embed_size`).


## Preprocess

The preprocessing of the data is done manually and stored locally first (where `configs/test.yaml` is the path to your config file):

```bash
python preprocess.py --config configs/example.yaml
```

Then load data dynamically using PyTorch's Dataloader when training (see [`datasets/dataloader.py`](datasets/dataloader.py)).
 This may takes a little time, but in this way, the training can occupy less memory (which means we can have a large batch size) and take less time.


## Train

To train a model, just run:

```bash
python train.py --config configs/example.yaml
```

If you have enabled the tensorboard (`tensorboard: True` in config files), you can visualize the losses and accuracies during training by:

```bash
tensorboard --logdir=<your_log_dir>
```

## Test

Test a checkpoint and compute accuracy on test set:

```bash
python test.py --config configs/example.yaml
```

## Classify

To predict the category for a specific sentence:

First edit the following items in [`classify.py`](classify.py):

```python
checkpoint_path = 'str: path_to_your_checkpoint'

# pad limits
# only makes sense when model_name == 'han'
sentence_limit_per_doc = 15
word_limit_per_sentence = 20
# only makes sense when model_name != 'han'
word_limit = 200
```

Then, run:

```bash
python classify.py
```

## Acknowledgement

This project codebase was based on [Renovamen/Text-Classification](https://github.com/Renovamen/Text-Classification/) and [sgrvinod/a-PyTorch-Tutorial-to-Text-Classification](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification).
