# Trasformer Pre-trained Language Models - Instructions

This repository utilises a combination of the HuggingFace (HF) Transformers package and PyTorch Lightning - which generally works well but can be problematic at times when making customisations to the transformer architectures as these will sometimes lead to incompatibility with other frameworks that are expecting the base Transformer classes.

In particular the approach to creating a PLM + classification head can be done with custom code or with Transformers `AutoModelForSequenceClassification`. The latter is by far the easiest and will allow easy integration to other frameworks that use Transformers off the shelf, but does limit the customisability a little.

For now we will just explore the approach which sticks to using Transformers classes though. We have added a couple of functions to the training process, including the ability to selectively freeze parts of the model during training - this is primarily to allow the freezing of all PLM parameters and only finetune the additional classification head parameters to investigate the PLMs knowledge from pre-training alone.

# Patient Safety Incident Report Tasks
## Binary Severity Classification (Degree of Harm, PD09)

One task we have implemented so far is the binary severity classification task as described in the main folder `classification_tasks`. The encoder model argument should either be the name of a HF pre-trained model or the path to locally trained BERT or RoBERTa model.

The dataset we have available to train with is large, and realistically we do not need to train on all of the data, so we provide a `FewShotSampling` class to enable generate smaller samples of data whilst attempting to maintain as well a balanced dataset as possible. Further, the larger the sample size the more likely our class imbalance will increase.

For our experiments we chose to sample 7k samplers per binary label for the training data, resulting in ~14k training samples. We also set max epochs to 5 for now.

**Note** the `./pl_trainer.py` script will create and use the same dataset created by `../data_utils/create_fewshot_dataset.py`

### With frozen PLM

**Note** we use a function to freeze N layers of the transformer PLM - for our experiments we chose to freeze all layers using the argument 'nr_frozen_epochs' = -1. Feel free to select a different number of layers to freeze, keeping in mind different PLMs have varying number of layers.

```{bash}
python ./pl_trainer.py --model_type autoforsequence --encoder_model {PLM_OF_CHOICE}  --dataset severity --binary_class_transform --training_size fewshot --few_shot_n 7000 --eval_few_shot_n 7000 --nr_frozen_epochs 5 --nr_frozen_layers -1 --max_epochs 5
```

### Finetune everything
```{bash}
python ./pl_trainer.py --model_type autoforsequence --encoder_model {PLM_OF_CHOICE}  --dataset severity --binary_class_transform --training_size fewshot --few_shot_n 7000 --eval_few_shot_n 7000 --max_epochs 5
```

## Incident Category Classification (Patient Safety Incident Type, IN05)
The second task we have implemented is the classification of incident category - IN05.

For our experiments we chose to sample 2k samples per class label for the training data, resulting in ~25-30k training samples. We chose to only evaluate on 200 samples per class. We also set max epochs to 5 for now.

### With frozen PLM
```{bash}
python ./pl_trainer.py --model_type autoforsequence --encoder_model {PLM_OF_CHOICE}  --dataset type --binary_class_transform --training_size fewshot --few_shot_n 2000 --eval_few_shot_n 200 --nr_frozen_epochs 30 --max_epochs 5
```

### Finetune everything
```{bash}
python ./pl_trainer.py --model_type autoforsequence --encoder_model {PLM_OF_CHOICE}  --dataset severity --binary_class_transform --training_size fewshot --few_shot_n 2000 --eval_few_shot_n 200 --max_epochs 5
```
