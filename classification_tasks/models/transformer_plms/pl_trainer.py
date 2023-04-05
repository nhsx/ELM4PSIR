import argparse
import json
import os

# add the sys path for data utils etc
import sys
import warnings
from datetime import datetime

import pandas as pd
import torch
import yaml
from hf_transformer_classifier import IncidentDataModule, IncidentModel
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

sys.path.append("../..")


from data_utils.dataset_processing import (  # noqa: E402
    FewShotSampler,
    convert_to_binary_classes,
    count_trainable_model_parameters,
    encode_classes,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

"""
Script to run training with a argument specified Transformer model as the pre-trained
encoder for document classification.

#### With AutoModelForSequenceClassification class ######
Example cmd usage:

python pl_trainer.py --model_type autoforsequence --encoder_model
emilyalsentzer/Bio_ClinicalBERT --batch_size 4 --gpu_idx 0 --max_epochs 10
--dataset severity --few_shot_n 7000 --eval_few_shot_n 7000

# for running with frozen language model

python pl_trainer.py --model_type autoforsequence --encoder_model roberta-base
--batch_size 16 --gpu_idx 0 --training_size fewshot --few_shot_n 7000
--eval_few_shot_n 7000  --dataset severity --binary_class_transform
--nr_frozen_epochs 99 --max_epochs 30

# for running with locally trained model

python pl_trainer.py --model_type autoforsequence
--encoder_model <MODEL_LOCATION> --batch_size 8 --gpu_idx 0 --training_size fewshot
--few_shot_n 7000 --eval_few_shot_n 7000 --dataset severity --binary_class_transform
--nr_frozen_epochs 99 --max_epochs 30


#### With custom classifier on top #####


python pl_trainer.py --model_type customclassifier --encoder_model
emilyalsentzer/Bio_ClinicalBERT --batch_size 4 --gpu_idx 0 --max_epochs 10
--dataset severity

# for running with frozen language model

python pl_trainer.py --model_type customclassifier --encoder_model roberta-base
--batch_size 16 --gpu_idx 0 --training_size fewshot --few_shot_n 7000
--dataset severity --binary_class_transform --nr_frozen_epochs 99 --max_epochs 30

# for running with locally trained model

`python pl_trainer.py --model_type customclassifier --encoder_model <MODEL_LOCATION>
--batch_size 8 --gpu_idx 0 --training_size fewshot --few_shot_n 7000
--dataset severity --binary_class_transform --nr_frozen_epochs 99 --max_epochs 30`

"""


# TODO - edit below to handle all tasks

# classes are imbalanced - lets calculate class weights for loss


def get_class_weights(train_df, label_col):
    """
    Function to compute class weights for cross entropy loss in the case of imbalanced
    datasets i.e. to penalize model that overfits to majority class
    """
    classes = list(train_df[label_col].unique())
    class_dict = {}
    nSamples = []
    for c in classes:
        class_dict[c] = len(train_df[train_df[label_col] == c])
        nSamples.append(class_dict[c])

    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    return torch.FloatTensor(normedWeights)


def read_csv(data_dir, filename):
    return pd.read_csv(f"{data_dir}{filename}", index_col=None)


def main():
    parser = argparse.ArgumentParser()

    # TODO - add an argument to specify whether using balanced data then update
    # directories based on that

    # Required parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data path to the directory containing the notes and referral data files",
    )
    parser.add_argument(
        "--cache_dir",
        default="./hf_models/",
        type=str,
        help="data path to the directory containing the notes and referral data files",
    )

    parser.add_argument(
        "--training_file",
        default="train.csv",
        type=str,
        help="data path to the directory containing the notes and referral data files",
    )
    parser.add_argument(
        "--validation_file",
        default="valid.csv",
        type=str,
        help="The default name of the training file",
    )
    parser.add_argument(
        "--test_file",
        default="test.csv",
        type=str,
        help="The default name of hte test file",
    )

    parser.add_argument(
        "--pretrained_models_dir",
        default="",
        type=str,
        help="The data path to the directory containing local pretrained models",
    )

    parser.add_argument(
        "--text_col",
        default="text",
        type=str,
        help="col name for the column containing the text",
    )

    parser.add_argument(
        "--log_save_dir",
        type=str,
        help="The data path to save tb log files to",
    )
    parser.add_argument(
        "--ckpt_save_dir",
        type=str,
        help="The data path to save trained ckpts to",
    )

    parser.add_argument(
        "--reinit_n_layers",
        default=0,
        type=int,
        help=(
            "number of pretrained final bert encoder layers to reinitialize for "
            "stabilisation"
        ),
    )
    parser.add_argument(
        "--max_tokens", default=512, type=int, help="Max tokens to be used in modelling"
    )
    parser.add_argument(
        "--max_epochs", default=10, type=int, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="batch size for training"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="number of batches to accumlate before optimization step",
    )
    parser.add_argument(
        "--balance_data",
        action="store_true",
        help="Whether not to balance dataset based on least sampled class",
    )
    parser.add_argument(
        "--binary_class_transform",
        action="store_true",
        help="Whether not to balance dataset based on least sampled class",
    )
    parser.add_argument(
        "--binary_severity_split_value",
        default=3,
        type=int,
        help="The severity value ranging from 0 - N severity to split by",
    )
    parser.add_argument(
        "--class_weights",
        action="store_true",
        help="Whether not to apply ce_class_weights for cross entropy loss function",
    )

    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
        help=(
            "gpu device to use e.g. 0 for cuda:0, or for more gpus use comma "
            "separated e.g. 0,1,2"
        ),
    )

    # 'allenai/biomed_roberta_base',
    # #'simonlevine/biomed_roberta_base-4096-speedfix',
    # # 'bert-base-uncased',
    # # emilyalsentzer/Bio_ClinicalBERT'

    parser.add_argument(
        "--encoder_model",
        default="bert-base-uncased",
        type=str,
        help="Encoder model to be used.",
    )

    parser.add_argument(
        "--max_tokens_longformer",
        default=4096,
        type=int,
        help="Max tokens to be considered per instance..",
    )

    parser.add_argument(
        "--encoder_learning_rate",
        default=1e-05,
        type=float,
        help="Encoder specific learning rate.",
    )
    parser.add_argument(
        "--classifier_learning_rate",
        default=1e-05,
        type=float,
        help="Classification head learning rate.",
    )
    parser.add_argument(
        "--classifier_hidden_dim",
        default=768,
        type=int,
        help="Size of hidden layer in bert classification head.",
    )

    parser.add_argument(
        "--nr_frozen_epochs",
        default=0,
        type=int,
        help="Number of epochs we want to keep the encoder model frozen.",
    )
    parser.add_argument(
        "--nr_frozen_layers",
        default=0,
        type=int,
        help="Number of epochs we want to keep the encoder model frozen.",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout value for classifier head.",
    )

    parser.add_argument(
        "--dataset",
        default="severity",
        type=str,
        help="name of dataset",
    )

    parser.add_argument(
        "--model_type",
        default="autoforsequence",
        type=str,
        help="This will alter the architecture and forward pass used by IncidentModel",
    )

    parser.add_argument(
        "--label_col",
        default="label",  # label column of dataframes provided - should be label if
        # using the dataprocessors from utils
        type=str,
        help="string value of column name with the int class labels",
    )

    parser.add_argument(
        "--loader_workers",
        default=24,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that \
            the data will be loaded in the main process.",
    )

    # Early Stopping
    parser.add_argument(
        "--monitor",
        default="monitor_balanced_accuracy",
        type=str,
        help="Quantity to monitor.",
    )

    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=4,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--fast_dev_run",
        default=False,
        type=bool,
        help="Run for a trivial single batch and single epoch.",
    )

    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        help="Optimization algorithm to use e.g. adamw, adafactor",
    )

    parser.add_argument(
        "--training_size",
        default="full",
        type=str,
        help="full training used, fewshot, or zero",
    )

    parser.add_argument("--few_shot_n", type=int, default=128)
    parser.add_argument("--eval_few_shot_n", type=int, default=128)

    parser.add_argument(
        "--sensitivity",
        default=False,
        type=bool,
        help=(
            "Run sensitivity trials - investigating the influence of classifier hidden "
            "dimension on performance in frozen plm setting."
        ),
    )

    parser.add_argument(
        "--optimized_run",
        default=False,
        type=bool,
        help="Run the optimized frozen model after hp search ",
    )

    # TODO - add an argument to specify whether using balanced data then update
    # directories based on that
    args = parser.parse_args()

    print(f"arguments provided are: {args}")
    # set up parameters
    # data_dir = args.data_dir
    # log_save_dir = args.log_save_dir
    # ckpt_save_dir = args.ckpt_save_dir
    # pretrained_dir = args.pretrained_models_dir
    pretrained_model_name = args.encoder_model
    cache_dir = args.cache_dir
    max_tokens = args.max_tokens
    n_epochs = args.max_epochs
    batch_size = args.batch_size
    reinit_n_layers = args.reinit_n_layers
    # accumulate_grad_batches = args.accumulate_grad_batches
    model_type = args.model_type

    # TODO  if we are loading a locally trained language model as the base - need to
    # adjust the model name to only be the model name - not full path to its folder

    if "saved_models" in args.encoder_model:
        if "declutr" in args.encoder_model:
            pretrained_model_name = args.encoder_model.split("/")[5]
        else:
            pretrained_model_name = args.encoder_model.split("/")[3]
    else:
        pretrained_model_name = args.encoder_model.split("/")[-1]

    # very ugly but we change the file paths based on whether full training provided
    if args.training_size == "full":
        args.few_shot_n = "all"

    time_now = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    # set up the ckpt and logging dirs

    # update ckpt and logs dir based on the dataset etc
    if args.sensitivity:
        logger.warning("Performing sensitivity analysis!")
        ckpt_dir = (
            f"{args.ckpt_save_dir}/sensitivity/{args.dataset}/{args.training_size}_"
            f"{args.few_shot_n}_classhidden_{args.classifier_hidden_dim}/"
            f"{pretrained_model_name}/{model_type}/version_{time_now}"
        )
        # log_dir = f"./logs/{args.dataset}/"
        log_dir = (
            f"{args.log_save_dir}/sensitivity/{args.dataset}/{args.training_size}_"
            f"{args.few_shot_n}_classhidden_{args.classifier_hidden_dim}/"
        )

    elif args.optimized_run:
        logger.warning("Performing optimized run based on hp search!")
        ckpt_dir = (
            f"{args.ckpt_save_dir}/optimized/{args.dataset}/"
            f"{args.training_size}_{args.few_shot_n}_classhidden_"
            f"{args.classifier_hidden_dim}/{pretrained_model_name}/{model_type}"
            f"/version_{time_now}"
        )
        # log_dir = f"./logs/{args.dataset}/"
        log_dir = (
            f"{args.log_save_dir}/optimized/{args.dataset}/{args.training_size}_"
            f"{args.few_shot_n}_classhidden_{args.classifier_hidden_dim}/"
        )

    elif args.binary_class_transform:
        ckpt_dir = (
            f"{args.ckpt_save_dir}/{args.dataset}/binary_class/{args.training_size}_"
            f"{args.few_shot_n}_classhidden_{args.classifier_hidden_dim}/"
            f"{pretrained_model_name}/{model_type}/version_{time_now}"
        )
        # log_dir = f"./logs/{args.dataset}/"
        log_dir = (
            f"{args.log_save_dir}/{args.dataset}/binary_class/{args.training_size}_"
            f"{args.few_shot_n}_classhidden_{args.classifier_hidden_dim}/"
        )
    else:
        ckpt_dir = (
            f"{args.ckpt_save_dir}/{args.dataset}/{args.training_size}_"
            f"{args.few_shot_n}/{pretrained_model_name}/{model_type}"
            f"/version_{time_now}"
        )
        # log_dir = f"./logs/{args.dataset}/"
        log_dir = (
            f"{args.log_save_dir}/{args.dataset}/{args.training_size}_"
            f"{args.few_shot_n}/"
        )

    # update ckpt and logs dir based on whether plm (encoder) was frozen during training

    if args.nr_frozen_epochs > 0:
        logger.warning(f"Freezing the encoder/plm for {args.nr_frozen_epochs} epochs")

        if args.sensitivity:
            logger.warning("Performing frozen sensitivity analysis")
            ckpt_dir = (
                f"{args.ckpt_save_dir}/sensitivity/{args.dataset}/{args.training_size}"
                f"_{args.few_shot_n}_classhidden_{args.classifier_hidden_dim}"
                f"/frozen_plm_{args.nr_frozen_layers}_layers/{pretrained_model_name}/"
                f"{model_type}/version_{time_now}"
            )
            log_dir = f"{log_dir}/frozen_plm"
        elif args.optimized_run:
            ckpt_dir = (
                f"{args.ckpt_save_dir}/optimized_run/{args.dataset}/"
                f"{args.training_size}_{args.few_shot_n}_classhidden_"
                f"{args.classifier_hidden_dim}/frozen_plm_{args.nr_frozen_layers}_layers/"
                f"{pretrained_model_name}/{model_type}/version_{time_now}"
            )
            log_dir = f"{log_dir}/frozen_plm"
        elif args.binary_class_transform:
            ckpt_dir = (
                f"{args.ckpt_save_dir}/{args.dataset}/binary_class/{args.training_size}"
                f"_{args.few_shot_n}_classhidden_{args.classifier_hidden_dim}"
                f"/frozen_plm_{args.nr_frozen_layers}_layers/{pretrained_model_name}/{model_type}/"
                f"version_{time_now}"
            )
            log_dir = f"{log_dir}/frozen_plm"
        else:
            ckpt_dir = (
                f"{args.ckpt_save_dir}/{args.dataset}/{args.training_size}_"
                f"{args.few_shot_n}/frozen_plm_{args.nr_frozen_layers}_layers/"
                f"{pretrained_model_name}/{model_type}/version_{time_now}"
            )
            log_dir = f"{log_dir}/frozen_plm"

    logger.warning(f"Logs will be saved at: {log_dir} and ckpts at: {ckpt_dir}")

    # load tokenizer - important to use the args.encoder model as this will be the full
    # path for local models rather than just the extracted model name
    # i.e. the pretrained_model_name
    print(f"loading tokenizer : {args.encoder_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        f"{args.encoder_model}", cache_dir=cache_dir
    )

    # TODO update the dataloading to use the custom dataprocessors from data_utils in
    # this folder

    # for now we are obtaining all data from one training/test file. So we can load in
    # now and do subsetting for actual datasets
    all_training_data = pd.read_csv(f"{args.data_dir}/{args.training_file}")
    all_validation_data = pd.read_csv(f"{args.data_dir}/{args.validation_file}")
    all_test_data = pd.read_csv(f"{args.data_dir}/{args.test_file}")

    # first we may need to do some task specific preprocessing
    if args.dataset == "severity":
        logger.warning(f"Using the following dataset: {args.dataset} ")

        # first remove all data with label 6 as this is a meaningless label and it
        # only applies to more than 1 or two documents
        all_training_data = all_training_data[all_training_data[args.dataset] < 6]

        train_df = all_training_data.copy()

        # are we doing any downsampling or balancing etc
        # class_weights = args.class_weights
        # balance_data = args.balance_data

        # First we work on training data to generate class_labels which will come in
        # handly for certain plots etc. assign to label column and substract 1 from
        # all label values to 0 index and drop rest
        train_df["label"] = train_df[args.dataset] - 1
        train_df = train_df[["text", "label"]]

        # now create the val/test dfs
        valid_df = all_validation_data.copy()
        valid_df["label"] = (
            valid_df[args.dataset] - 1
        )  # the original labels were 1:N. But most neural network loss functions
        # expect 0:N
        valid_df = valid_df[["text", "label"]]

        test_df = all_test_data.copy()
        test_df["label"] = (
            test_df[args.dataset] - 1
        )  # the original labels were 1:N. But most neural network loss functions
        # expect 0:N
        test_df = test_df[["text", "label"]]

        # if binary_transform - convert labels from range 0-5 to 0/1 (low/high) severity
        if args.binary_class_transform:
            logger.warning("Converting to binary classification problem")
            # update save dir

            train_df = convert_to_binary_classes(
                df=train_df, split_value=args.binary_severity_split_value
            )
            valid_df = convert_to_binary_classes(
                df=valid_df, split_value=args.binary_severity_split_value
            )
            test_df = convert_to_binary_classes(
                df=test_df, split_value=args.binary_severity_split_value
            )

        # get class label encodings based on training data
        class_labels, idx_to_class, class_to_idx = encode_classes(
            df=train_df, meta_df=None, label_col="label"
        )
        logger.warning(
            (
                f"Class labels: {class_labels}\n\nidx_to_class:{idx_to_class}\n\n"
                f"class_to_idx:{class_to_idx}"
            )
        )
    elif args.dataset == "type":
        logger.warning(f"Using the following dataset: {args.dataset} ")
        train_df = all_training_data.copy()

        # are we doing any downsampling or balancing etc
        # class_weights = args.class_weights
        # balance_data = args.balance_data

        # First we work on training data to generate class_labels which will come in
        # handly for certain plots etc. assign to label column and substract 1 from all
        # label values to 0 index and drop rest
        train_df["label"] = (train_df[args.dataset] - 1).astype("int")
        train_df = train_df[["text", "label"]]

        # NOTE we need to convert the labels to within the range 0-N classes
        # get class label encodings based on training data
        class_labels, idx_to_class, class_to_idx = encode_classes(
            df=train_df, meta_df=None, label_col="label"
        )

        # NOTE with this datset, where the int labels are not ordered we want to
        # re-assign the class labels to the new indexing
        class_labels = list(class_to_idx.values())
        logger.warning(
            (
                f"Class labels: {class_labels}\n\nidx_to_class:{idx_to_class}\n\n"
                f"class_to_idx:{class_to_idx}"
            )
        )
        train_df["label"] = train_df["label"].map(class_to_idx)

        # now create the val/test dfs
        valid_df = all_validation_data.copy()
        valid_df["label"] = (valid_df[args.dataset] - 1).astype(
            "int"
        )  # the original labels were 1:N. But most neural network loss functions
        # expect 0:N
        valid_df = valid_df[["text", "label"]]
        # map int classes to their idx to keep within the 0-Nclass range
        valid_df["label"] = valid_df["label"].map(class_to_idx)

        test_df = all_test_data.copy()
        test_df["label"] = (test_df[args.dataset] - 1).astype(
            "int"
        )  # the original labels were 1:N. But most neural network loss functions
        # expect 0:N
        test_df = test_df[["text", "label"]]
        # map int classes to their idx to keep within the 0-Nclass range
        test_df["label"] = test_df["label"].map(class_to_idx)

    else:
        # TODO implement other datasets
        raise NotImplementedError
    # if doing few shot sampling - apply few shot sampler

    if args.training_size == "fewshot":
        logger.warning("Will be performing few shot learning!")
        # initialise the sampler
        train_support_sampler = FewShotSampler(
            num_examples_per_label=args.few_shot_n,
            also_sample_dev=False,
            label_col=args.label_col,
        )
        # now apply to each dataframe but convert to dictionary in records form first
        train_df = train_support_sampler(train_df.to_dict(orient="records"), seed=1)

        # with big datasets we will want to sample fewer val and test sets
        eval_support_sampler = FewShotSampler(
            num_examples_per_label=args.eval_few_shot_n,
            also_sample_dev=False,
            label_col=args.label_col,
        )
        # do we actually want to resample the val and test sets - probably not?
        valid_df = eval_support_sampler(valid_df.to_dict(orient="records"), seed=1)
        test_df = eval_support_sampler(test_df.to_dict(orient="records"), seed=1)

    logger.warning(
        f"train_df shape: {train_df.shape} and train_df cols:{train_df.columns}"
    )

    # get number labels or classes as length of class_labels
    n_labels = len(class_labels)

    logger.info(f"Number of classes : {n_labels}")
    # push data through pipeline
    # instantiate datamodule
    data_module = IncidentDataModule(
        train_df,
        valid_df,
        test_df,
        tokenizer,
        batch_size=batch_size,
        max_token_len=max_tokens,
    )
    # set up some parameters
    steps_per_epoch = len(train_df) // batch_size
    total_training_steps = steps_per_epoch * n_epochs
    # warmup_steps = total_training_steps // 5
    warmup_steps = 100
    warmup_steps, total_training_steps

    # get some class specific loss weights - only needed if doing some form of
    # weighted cross entropy with ubalanced classes
    ce_class_weights = get_class_weights(data_module.train_df, args.label_col)

    # save the hyperparams and arguments to a config file
    # ensure the save path folder has been created
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with open(f"{ckpt_dir}/config.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    with open(f"{ckpt_dir}/config.yaml", "w") as f:
        yaml.dump(args.__dict__, f)

    # set up model
    model = IncidentModel(
        model=args.encoder_model,
        num_labels=n_labels,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        nr_frozen_epochs=args.nr_frozen_epochs,
        nr_frozen_layers=args.nr_frozen_layers,
        ce_class_weights=ce_class_weights,
        weight_classes=args.class_weights,
        reinit_n_layers=reinit_n_layers,
        class_labels=class_labels,
        classifier_hidden_dim=args.classifier_hidden_dim,
        encoder_learning_rate=args.encoder_learning_rate,
        classifier_learning_rate=args.classifier_learning_rate,
        optimizer=args.optimizer,
        dropout=args.dropout,
        cache_dir=args.cache_dir,
        model_type=args.model_type,
    )

    print(model)
    # do not save ckpts for few shot learners
    # setup checkpoint and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{ckpt_dir}",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor=args.monitor,
        mode=args.metric_mode,
        save_last=False,
    )

    tb_logger = TensorBoardLogger(
        save_dir=f"{log_dir}/{pretrained_model_name}",
        version="version_" + time_now,
        name=f"{model_type}",
    )

    # early stopping based on validation metric of choice
    early_stopping_callback = EarlyStopping(
        monitor=args.monitor, mode=args.metric_mode, patience=args.patience
    )

    # log the number of trainable params
    logger.info(
        f"Number of trainable parameters: {count_trainable_model_parameters(model)}"
    )

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        accelerator="gpu",
        devices=[args.gpu_idx],
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
        ],  # usually we want the early stopping callback in here
        max_epochs=args.max_epochs,
        default_root_dir="./",
    )
    # ------------------------
    # 6 START TRAINING
    # ------------------------

    trainer.fit(model, data_module)

    # test
    trainer.test(
        ckpt_path=f"{ckpt_dir}/best-checkpoint.ckpt",
        dataloaders=data_module.test_dataloader(),
    )

    logger.warning(f"Model type is: {model_type} with class: {type(model_type)}")
    # and finally save the model in the same ckpt dir but with the save_pretrained
    # function
    if model_type == "autoforsequence":
        # reload the best ckpt
        logger.info(
            (
                "Reloading the best checkpoint and saving in the pretrained format for "
                "easier reloading into transformers frameworks"
            )
        )
        best_classification_model = IncidentModel.load_from_checkpoint(
            f"{ckpt_dir}/best-checkpoint.ckpt"
        )

        # make sure to unfreeze all parameters
        best_classification_model.unfreeze_encoder()
        logger.info(
            (
                "After unfreezing the model the trainable params are:"
                f"{count_trainable_model_parameters(best_classification_model)}"
            )
        )

        # extract the automodelforseqeuence classification component
        best_auto_model = best_classification_model.model

        # TODO adapt below to create a label2id and id2label for the class labels
        # # set label2index
        # auto_model.config.label2id = class_labels_dict

        # # we need to assign the label indices to their string counter part - need to
        # switch key/values from class_labels_idx
        # auto_model.config.id2label = {
        #   class_labels_dict[k]:k for k in class_labels_dict
        # }

        # save the model
        best_auto_model.save_pretrained(f"{ckpt_dir}/pretrained_format/")

        # also save the tokenizer
        tokenizer.save_pretrained(f"{ckpt_dir}/pretrained_format/")


# run script
if __name__ == "__main__":
    main()
