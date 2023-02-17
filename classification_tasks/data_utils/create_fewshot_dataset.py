import argparse
import os
import warnings

import pandas as pd
from dataset_processing import FewShotSampler, convert_to_binary_classes, encode_classes
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


def main():
    """
    Script to run training with a argument specified BERT model as the pre-trained
    encoder for instance classification.

    Example cmd usage:
    `python ./data_utils/create_fewshot_data.py --{data_file}`
    """

    parser = argparse.ArgumentParser()

    # TODO - add an argument to specify whether using balanced data then update
    # directories based on that

    # Required parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        help=(
            "The data path to the directory containing the text data csv file with "
            "labels"
        ),
    )

    parser.add_argument(
        "--training_file",
        default="train.csv",
        type=str,
        help="The filename of the training file containing the text and labels",
    )
    parser.add_argument(
        "--validation_file",
        default="valid.csv",
        type=str,
        help="The default name of the validation file",
    )
    parser.add_argument(
        "--test_file",
        default="test.csv",
        type=str,
        help="The default name of the test file",
    )

    parser.add_argument(
        "--text_col",
        default="text",
        type=str,
        help="col name for the column containing the text",
    )

    parser.add_argument(
        "--save_dir", type=str, help="The data path to save the created dataset"
    )

    parser.add_argument(
        "--balance_data",
        action="store_true",
        help="Whether not to balance dataset based on least sampled class",
    )
    parser.add_argument(
        "--binary_class_transform",
        action="store_true",
        help="Whether not to convert a multi-class problem into a binary one",
    )  # this is fairly specific to a dataset we developed
    parser.add_argument(
        "--binary_severity_split_value",
        default=3,
        type=int,
        help=(
            "The mid point value ranging from 0 - N_classes to split into a binary "
            "set of classification i.e. 0/1"
        ),  # this is fairly specific to a dataset we developed
    )

    parser.add_argument(
        "--dataset",
        default=None,  # or:
        type=str,
        help="name of dataset",
    )

    parser.add_argument(
        "--label_col",
        default="label",  # label column of dataframes provided
        type=str,
        help="string value of column name with the int class labels",
    )

    parser.add_argument(
        "--loader_workers",
        default=4,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that \
            the data will be loaded in the main process.",
    )

    parser.add_argument("--few_shot_n", type=int, default=7000)

    args = parser.parse_args()

    print(f"arguments provided are: {args}")

    # for now we are obtaining all data from one training/test file. So we can load in
    # now and do subsetting for actual datasets

    all_training_data = pd.read_csv(f"{args.data_dir}/{args.training_file}")
    all_validation_data = pd.read_csv(f"{args.data_dir}/{args.validation_file}")
    all_test_data = pd.read_csv(f"{args.data_dir}/{args.test_file}")

    # first we may need to do some task specific preprocessing
    if args.dataset == "severity":
        logger.warning(f"Using the following dataset: {args.dataset} ")

        # first remove all data with label 6 as this is a meaningless label and it only
        # applies to more than 1 or two documents
        all_training_data = all_training_data[all_training_data[args.dataset] < 6]

        train_df = all_training_data.copy()

        # First we work on training data to generate class_labels which will come in
        # handly for certain plots etc assign to label column and substract 1 from all
        # label values to 0 index and drop rest
        train_df["label"] = train_df[args.dataset] - 1
        train_df = train_df[["text", "label"]]

        # now create the val/test dfs
        val_df = all_validation_data.copy()
        val_df["label"] = val_df[args.dataset] - 1
        # the original labels were 1-N. But most neural network loss functions
        # expect 0-N
        val_df = val_df[["text", "label"]]

        test_df = all_test_data.copy()
        test_df["label"] = test_df[args.dataset] - 1
        # the original labels were 1-N. But most neural network loss functions
        # expect 0-N
        test_df = test_df[["text", "label"]]

        # if binary_transform - convert labels from range 0-5 to 0/1 (low/high)
        # severity
        if args.binary_class_transform:
            logger.warning("Converting to binary classification problem")
            # update save dir

            train_df = convert_to_binary_classes(
                df=train_df, split_value=args.binary_severity_split_value
            )
            val_df = convert_to_binary_classes(
                df=val_df, split_value=args.binary_severity_split_value
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

        logger.warning("Will be performing few shot learning!")
        # initialise the sampler
        support_sampler = FewShotSampler(
            num_examples_per_label=args.few_shot_n,
            also_sample_dev=False,
            label_col=args.label_col,
        )
        # now apply to each dataframe but convert to dictionary in records form first
        train_df = support_sampler(train_df.to_dict(orient="records"), seed=1)

        # do we actually want to resample the val and test sets - probably not?
        val_df = support_sampler(val_df.to_dict(orient="records"), seed=1)
        test_df = support_sampler(test_df.to_dict(orient="records"), seed=1)

        # now write these to file
        # set save dir
        dataset_save_path = f"{args.save_dir}/{args.dataset}/balanced/"

        if args.binary_class_transform:
            dataset_save_path = f"{dataset_save_path}/binary_class/"

        # create if it doesn't exist
        if not os.path.exists(f"{dataset_save_path}"):
            os.makedirs(f"{dataset_save_path}")

        # now write each dataframe to file
        train_df.to_csv(f"{dataset_save_path}/train.csv", index=None)
        val_df.to_csv(f"{dataset_save_path}/valid.csv", index=None)
        test_df.to_csv(f"{dataset_save_path}/test.csv", index=None)

    else:
        # TODO implement los and mimic readmission
        raise NotImplementedError
    # if doing few shot sampling - apply few shot sampler


if __name__ == "__main__":
    main()
