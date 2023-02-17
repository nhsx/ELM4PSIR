# ELM4PSIR - Exploring Language Modelling for (NHS) Patient Safety Incident Reports
## NHS England -  Digital Analytics and Research Team (DART) - PhD Internship Project

### About the Project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

ELM4PSIR presents code to train, evaluate, and explore various Language Models (LM) applied to patient safety incident data in the NHS from the National Reporting and Learning System (NLRS) with the goal of creating better models to aid in various downstream tasks.

This work was conducted as part of an NHS England DART PhD Internship project by [Niall Taylor](https://github.com/NtaylorOX) for around five months between June - November 2022.  Further information on the original project proposal can be found [here](https://nhsx.github.io/nhsx-internship-projects/incident-language-model/).

The associated report can be found in the [reports](./reports) folder.

_**Note:** No data, public or private are shared in this repository._

### Project Stucture

ELM4PSIR is made up of multiple "strands" or pipelines revolving around generating meaningful numerical or embedding representations of text data.
```
├── checklist_testing     # Example notebook using CheckList with patient safety style data
├── classification_tasks  # Multiple pipelines for training/evaluating classifiers
├── embedding_tools       # Code and notebooks for comparing and visualising LM embeddings
├── language_modelling    # Code for training/evaluating various language models
├── reports               # Project reports
├── topic_modelling       # A folder containing avenues for topic modelling with PLMs
├── utils                 # Scripts for creating training and test patient safety datasets
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENCE
├── OPEN_CODE_CHECKLIST.md
├── README.md
└── requirements.txt
```
#### `checklist_testing`
An example implementation of *behavioural* testing of trained NLP models with a patient safety example. This utilises the `CheckList` framework outlined at [checklist-repo](https://github.com/macrotcr/checklist). Further details provided in the [checklist_testing](./checklist_testing/) folder.

#### `classification_tasks`
A quite large set of pipelines for training and evaluating various text classification models on selected downstream pseudo-tasks related to patient safety reports, with more details in the [report](./reports).

#### `embedding_tools`
A set of scripts and notebooks for comparing and visualising contextualised embeddings produced by the pretrained language models produced by this repo.

#### `language_modelling`
Contains language modelling pipelines for word2vec training with gensim, and transformer based pre-trained language models (PLMs) using huggingface transformers.

#### `topic_modelling`
An attempt to highlight and direct users to a range of topic modelling approaches to work with the modelling approaches outlined in this repo, in particular the approaches that work with embeddings produced by pretrained language models

### Built With

The majority of this codebase was developed in [![Python v3.8](https://img.shields.io/badge/python-v3.8-blue.svg)](https://www.python.org/downloads/release/python-380/).  The work is mostly undertaken in [`PyTorch`](https://pytorch.org/) and heavily utilises the [`Transformers`](https://huggingface.co/docs/transformers/index) package for the vast majority language model handelling.

Further, we have included modified copies of various scripts and codebases within the repository (with references) where appropriate or needed.  The most sizable inclusions are the [DeCLUTR](./language_modelling/DeCLUTR/) codebase which was modified to work on our hardware/OS setup, and the [Word Vector models](./classification_tasks/models/baselineTextClassifiers/word_vector_nn/) used in the baselining approaches.  We give thanks to the authors of all components incorporated for making such useful and resuable projects.

⚠️ Warning ⚠️

[![Python v3.7](https://img.shields.io/badge/python-v3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) was used for the DeCLUTR model training c.f. [`language_modelling/DeCLUTR`](./language_modelling/DeCLUTR/). It is highly recommended that a separate virtual environment is used for DeCLUTR, which has its own setup instructions found in the `language_modelling` folder [README.md](./language_modelling/DeCLUTR/README.md).

### Getting Started

#### Installation

See `./requirements.txt` for package versions - installation in a virtual environment is recommended:
```{bash}
conda create --name elm4psir python=3.8
conda activate elm4psir
python -m pip install -r requirements.txt
```

When training on GPU machines, the appropriate PyTorch bundle should be installed - for more info: `https://pytorch.org/get-started/locally/`

```{bash}
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

#### spaCy NLP

A few of the projects will benefit from using [spaCy](https://spacy.io/). To use spacy you have to:

```{bash}
python -m pip install -U spacy
```

followed by downloading their pretrained models via:

```{bash}
python -m spacy download en_core_web_sm
```

as an example, refer to their website for more details.


#### GPU Support

Training with GPU is recommended. Single-GPU training has been tested with:
- `NVIDIA Tesla T4`
- `cuda 11.3`
- `Windows Server 2019`

### Usage and Datasets

__NOTE__ The vast majority of the language modelling code in this repo can be used out of the box with any decent sized text dataset, all that is required is care with how the directories are setup and attention to how these different files are saved.

The repository has multiple different pipelines and models to train for various tasks - to enable the training of the language models and classification tasks will require some initial preprocessing steps as follows:

#### Create training and held out data splits
We create a 90%:10% train/held out datasets from the original raw reports data. We prepare the held-out dataset such that it is available for independent evaluation.

The script then uses the 90% training data as the basis for creating a further 90%:10% train/test split for LM training and other downstream tasks.

Run the following script to create these datasets, all `csv` files will be stored at the provided `save_path`

```{bash}
python utils/create_lm_data_split.py --raw_data_file {raw_data_path} --save_path {directory_to_save_new_data} --hold_out_percentage 0.10
```

Once complete we have saved new `training_data/held_out_data` and `lm_training/lm_test` files in the `{directory_to_save_new_data}` in a `csv` format.

#### Preprocessing and cleaning the LM training and test datasets

We apply some fairly light-touch cleaning with some simple regex and removal of tabs/whitespace, etc. for the LM training/test data. This will create a new data folder for with "cleaned data".

One can also opt to process/create a sample of the training and test data, which given the size of the training data can be useful for development/debugging etc.

e.g. run the following to clean and create training data for 10k training notes and 2k test notes

```{bash}
python utils/prepare_notes_for_lm.py --training_notes_path {full_path_to_lm_training_data} --test_notes_path {full_path_to_lm_test_data} --save_path {directory_to_save_cleaned_data} --sample --train_sample_size 10000 --test_sample_size 2000
```
or if instead using all the data available
```{bash}
python utils/prepare_notes_for_lm.py --training_notes_path {full_path_to_lm_training_data} --test_notes_path {full_path_to_lm_test_data} --save_path {directory_to_save_cleaned_data}
```

#### Setup pseudo-classification datasets

We have given the user the possibility to create pseudo-classification tasks with the categorical variables provided alongside the incident report data - *pseudo* as we are using them as a method to evaluate LMs ability to embed the structure present in the data.

There is the ability to process all possible category/task datasets from available categorical variables given and output them all together in one `csv`, or to process each individually and save to their own respective folders.

__NOTE__  We have have only implemented downstream models for the following two categorical variables provided with the NHS patient safety incident reports data:
- Incident Category Classification (Patient Safety Incident Type, IN05)
- Severity Classification (Degree of Harm, PD09)

with further details provided inside the `classification_tasks` folder [README.md](./classification_tasks/README.md).

To create the datasets for each category individually run the following:

```{bash}
python utils/prepare_classification_datasets.py --data_path {path_to_training_data} --save_path {path_to_save_data} --process_individually
```

To combine all tasks into one, the following without the `--process_individually` flag:

```{bash}
python utils/prepare_classification_datasets.py --data_path {path_to_training_data} --save_path {path_to_save_data}
```

#### Alternative data sources
##### MIMIC-III data

Much of the work presented here is targeted at developing and adapting language modelling techniques for a niche clinical domain, and whilst our work focused on patient safety incident reports, the codebase is largely data agnostic and can be applied to any domain.

A popular, accessible clinical dataset which could be used instead is MIMIC-III, refer to [physionet](physionet.org/conent/mimiciii/1.4) for details on how to access.

From a high-level, the **NOTEEVENTS.csv** would provide a suitable dataset for this repository. Further, there is a wealth of research that has used these data for NLP models for a variety of language modelling and downstream tasks. A good starting point for pre-processing and curating a suitable dataset for this repo would be the following [github-repo](https://jamesmullenbach/caml-mimic).

#### Outputs
The associated report can be found in the [reports](./reports) folder.

### Roadmap

See the repo [Issues](./Issues/) for a list of proposed features (and known problems).

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The project specific documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contact

To find out more about [DART](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch [here](mailto:analytics-unit@nhsx.nhs.uk).

<!-- ### Acknowledgements -->
