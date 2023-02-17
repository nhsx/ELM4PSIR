# Instructions

After following the steps provided in the root of this repo to generate a LM training/test dataset you can then train some language models with either gensim word2vec or transformers.

But generally speaking all that is required for language modelling is a large enough text dataset, ideally a collection of decent length documents for which a language modelling objective can be applied.


## Transformers
```
cd ./transformers/
```

This script will take the provided LM training data and convert it into a dataset suitable for the Transformers (huggingfaces) trainer class which will handle a great deal of the LM architecture and training setup.

There is only really one argument that can change how the LM training will change and that is the LM pre-training objective, which in this case is either Masked Language Modelling or next-word-prediction/autoregression. This is set by the `--mlm` flag.

As we are likely training a model initialised from another PLM, such as "roberta-base", the training objective has already somewhat become fixed to whatever that initial model was trained on. The Trainer class will handle any discepancies though and it should fail if you try MLM with GPT for instance.


To run training with bert or roberta based models with MLM:

```{bash}
python transformers/run_lm_training --hf_model_name {bert/roberta of choice} --max_steps {maximum_training_steps} --save_every_steps {how often to save ckpt} --eval_every_steps {how often to evaluate on validation set} --mlm

```

To run with GPT based models do the same but remove the `--mlm` flag


## word2vec

This is presented as a jupyter notebook with self contained instructions and is based on <http://mccormickml.com/2016/04/27/word2vec-resources/>.


## DeCLUTR

Deep Contrastive Learning for Unsupervised Textual Representations (DeCLUTR) is very much an independent and rather large codebase which attempts to implement the approach first introduced in this [research-paper](https://aclanthology.org/2021.acl-long.72.pdf) and accompnaying [github-repo](https://github.com/JohnGiorgi/DeCLUTR). They deserve all the credit really and this codebase was merely adapted to the incident reports dataset and subtle tweaks to work on
a Windows OS better.

The DeCLUTR modelling procedure works by sampling spans from a document to create anchors and positive samples for contrastive loss. The original paper found that using 2 anchors per document was best, and they opted to use span lengths of 512 tokens. The algorithms they use require documents to be a length specified by this formula below, where $AN$ refers to the number of anchors per document and $MSL$ refers to the maximum span length:

$$2 * AN * MSL$$

The original paper thus restricted their documents for training to be of a minimum length of 2048 tokens, which is very long when compared to the lengths of incident reports (averaging around 60 tokens per document). Thus we had to play around with different span lengths.


⚠️ **NOTICE:** The DeCLUTR work in its current form is heavily reliant on the [allennlp-package](https://github.com/allenai/allennlp) and whilst the code could be written to work with more native pytorch or transformers libraries, it was not worth the time nor effort. The major caveat here is that allennlp is no longer maintained as an active codebase as of 2022 and was made specifically to work on linux machines, with some components breaking on windows machines.

### Pre-processing

The preprocessing involves a minor cleaning step and tokenization, before restricting the text datasets to a minimum span based on the formula outlined above. At present, for simplicity more than anything, this will throw away any documents that are $< minimum\_span$.

 The preprocessing scripts will use simple white space tokenization if a specific pretrained tokenizer is not provided. To stay inline with the use of RoBERTa base as the staple model, we will tokenizer using the roberta-base tokenizer.

```
python ./scripts/pre_process_custom_text_dataset.py --input_file_path {data text file path} --save_directory {where to save the processed data} --min_length {length of span based on above formula} --max_instance {maximum number of samples to save} --pretrained_model_name_or_path {tokenizer model}
```


### Training models

The allennlp framework utilies configs to call their trainer classes - which whilst efficient can be a pain if you are not familar....

The configs basically just contain all required parameters to train a model - this will be dataset and environment specific, but examples of configs from the original DeCLUTR repo are provided in:
```
./training_configs
```
You really need to look at these and get an idea of what parameters are required/changed per experiment.

With an experimental config created, allennlp train can be called like below:

```
# run declutr-base with min 64 span| 2 anchor | 2 positives
allennlp train {training_config.jsonnet} --serialization-dir {save_directory} --include-package "declutr" -f

```

To then save this model in the transformers format run:
```
python ./scripts/save_pretrained_hf.py --archive_file {serialization-dir given to training script} --save_directory {save_directory}
```
