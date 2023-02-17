# Classification Tasks
## Overview

The `classification_tasks` folder contains code for the training and evaluation of various models on a selection of [GLUE Benchmark](https://gluebenchmark.com/) datasets as a general NLP benchmark, plus two patient safety incident specific classification *pseudo*-tasks using available categorical variables:
- **patient safety incident type (IN05)**
- **degree of harm (PD09)**

We call these tasks "psuedo" as we are using them as a measure of our model performance, even though they aren't an explicit downstream tasks that we are required to model.

**NOTE:** the baseline word2vec and scikit-learn models were only applied to the **PD09** task.

## Preparing Datasets
### Severity Classification (Degree of Harm, PD09)
For now, the only classification dataset setup is the incidence severity prediction.  This uses the incident severity labels (range 1-5) associated to each of the free text reports.

Our overall dataset is very large, and thus we opt to create a much smaller and balanced binary classification dataset, by first binning incidence labels into two groups: `0` representing lower (1-3) and `1` representing higher (4-5) severity.

Further, the class distribution in the total dataset is heavily skewed toward class 0 i.e. lower harm (thankfully) - this skew can be problematic for large neural networks, as they can often overfit to the majority class, and can struggle to find a better solution as predicting the majority class gives locally 'good' results.

We could look to various methods to try to produce a more balanced dataset but we instead opted for a somewhat crude approach that is dataset specific.  We create a sub-sample of the dataset by randomly pulling out `N` samples per label.

This was primarily done to allow us to select a sample size of our choosing which is large enough to give a useful result, but small enough to avoid training on all available data points (as that is around 2 million documents and would take considerable time).

Creating a "balanced" severity dataset with approximately 7k samples per class can be acheived by running the following from the `classification_tasks` folder:

```bash
python .\data_utils\create_fewshot_dataset.py --data_dir {directory_containing_training_data} --save_dir {directory_for_saving_created_dataset} --dataset severity --binary_class_transform --few_shot_n 7000
```

__N.B.__ This task is simply a multi-class problem converted to a binary classification problem via binning, so can be easily adapted to any such problem.

### Incident Category Classification (Patient Safety Incident Type, IN05)
This pertains to label of incident category which can take one of 15 possible values e.g. self harm.

This task has only been implemented with the transformer-based classification models and the dataset which feeds that pipeline is created with instructions inside the repo main [README.md](../README.md)

### Model training
Instructions for each model type/pipeline are given in their own respective folders `README.md` file:

- [BoW/Tf-idf vectors with random forest (or any scikit-learn classifier)](./models/baselineTextClassifiers/sklearn_models/)

- [Word vector with different neural network architectures](./models/baselineTextClassifiers/word_vector_nn/)

- [Transformer based PLMs with classification heads](./models/transformer_plms/)

 ### Baseline GLUE tasks

The [Baseline GLUE tasks](./baseline_tasks/Text_Classification_on_GLUE.ipynb) setup has been adapted from a nicely constructed notebook [huggingface-classification-notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb) courtesy of `HuggingFace` - the changes mainly focus on saving out logs and checkpoints locally.  Note that this notebook requires the installation of [`evaluate`](https://github.com/huggingface/evaluate) which we did not include in the main `requirements.txt` as it was easier to run in a seperate environment at this time - see the above link for installation instructions.

## Model Explainability Exploration

Notebooks have been created to implement *explainbility* or *interpretability* methods for classification models, such as those used in this repo within the [explainability_exploration](./explainability_exploration/) folder.

This is a large active research space and we do not endevour to present a thorough investigation, rather highlight some useful tools that exist and align well with `transformer` based models.

**Note:** most of the libraries work best with transformer based models that have been trained for a classification task using the [AutoModelForSequenceClassification class](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html). If the classification head is a custom one, with different naming conventions, certain features may not work as expected.

These notebooks are based on the following libraries, and we encourage anyone who wants to know more to visit the original repos:

- [Ferret](https://github.com/g8a9/ferret) provides an easy to use package for benchmarking interpretability techniques with a clear focus on integrating with the [huggingface transformers](https://huggingface.co/docs/transformers/installation) library - see [ferret_explainability.md](./explainability_exploration/ferret_explainability.md) for more details.

- [Truelens](https://github.com/truera/trulens) is a slightly more involved and deep-diving library including cross-framework functionality - see [truelens_explainability.md](./explainability_exploration/truelens_explainability.md) for more details.

- [SHAP](https://shap.readthedocs.io/en/latest/index.html) is a library dedicated to using Shapley based explanations - see [shap_transformers_explainability.md](./explainability_exploration/shap_transformers_explainability.md) for more details.

Both Ferret and Trulens cover a variety of techniques such as: LIME, gradient based, integrated gradients.
