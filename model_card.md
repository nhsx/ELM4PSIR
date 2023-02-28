# Model Card: ELM4PSIR

This repository was created as part of an NHS England PhD internship project and contains code to train, evaluate, and explore various Language Models (LM) with the goal of creating better models to aid in various useful downstream tasks.  The focus of the project was patient safety incident data from the National Reporting and Learning System (NLRS).

## Model Details
## Model Use
### Intended Use
This repository is intended for use in experimenting with different langauge modelling approaches by further pre-training, fine-tuning, or applying minor modifications for domain specific use-cases.

### Out-of-Scope Use Cases
Models produced from this repository are not considered suitable to use in language modelling tasks in a production environment without further testing and assurance.

## Training Data
Experiments in this repository were run against a corpus taken from nationally collected free-text documents relating to all types of patient safety incidents in the NHS from the National Reporting and Learning System (NRLS).  For more details on this collection, please see the official website - [NHS England - National Patient Safety Incident Reports](https://www.england.nhs.uk/patient-safety/national-patient-safety-incident-reports/). We worked with a sub-sample of approximately 2.3 million de-identified reports produced in the financial year 2019/2020.

The repository also gives adivce on how the approaches contained within could be applied to other datasets in the root [README.md](./README.md).

## Performance and Limitations
Much of this repository uses open-access pre-trained language models as the base for various experiments.  These models include variants of [`RoBERTa`](https://arxiv.org/abs/1907.11692) and [`DeCLUTR`](https://arxiv.org/abs/2006.03659), with more information available in the associated [report](./reports/ELM4PSIR_NT.pdf).

The models are sensitive to hyperparameter choices. Given the size of the models and the pre-training options, extensive hyperparameter tuning was not possible within the timelines of this project.

The models have not been explicitly tested for bias and fairness. In particular, an evaluation of model performance by age, gender and race has not been performed. Additional performance differences may also be found between common and rare events.  We do however explore how behavioural testing could be used to help in this arena within the repository - further information can be found in the [Checklist Testing](./checklist_testing/) folder.
