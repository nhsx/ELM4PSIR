import argparse
import os
import re
import time
import warnings

import pandas as pd
from loguru import logger
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

"""
Script to prepare incident notes data for different classification tasks based on the
available categorical variables including: location, severity, type.
The datasets can be processed and stored individually in their own respective folder,
or all together in one big dataset. These just make certain downstream tasks easier to
organise.


Example usage from cmd line for sampling:

`python prepare_classification_datasets.py --sample --sample_size 10`

or for all data

`python prepare_classification_datasets.py`

"""


class LMTextData:
    def __init__(
        self,
        data_path: str = None,
        test_notes_path: str = None,
        save_path=None,
        admin_language=None,
        replacement_map=None,
        remove_punctuation=True,
        sample=True,
        sample_size=500,
        seed=41,
        text_col=None,
        dataset_names_map={
            "Key.PD09": "severity",
            "Key.IN05": "type",
            "Key.RP02": "location",
        },
        process_individually=False,
        binary_transform=False,
    ):

        self.admin_language = admin_language
        self.sample = sample
        self.sample_size = sample_size
        self.data_path = data_path
        self.test_notes_path = test_notes_path
        self.save_path = save_path
        self.admin_language = admin_language
        self.seed = seed
        self.text_col = text_col
        self.remove_punctuation = remove_punctuation
        self.replacement_map = replacement_map
        self.dataset_names_map = dataset_names_map
        self.process_individually = process_individually
        self.binary_transform = binary_transform

        # make save_dirs if not exist
        for dir in [self.save_path]:
            if not os.path.exists(f"{dir}"):
                os.makedirs(f"{dir}")

    def clean_data(
        self, admin_language, notes_df: pd.DataFrame, remove_punctuation=True
    ) -> pd.DataFrame:

        """
         TODO - add function to allow mapping of known acronyms to full words

         remove redundant information from free text using some common NLP techniques
             and heuristic rules

         Args:
             text_col: string -> name of column with text in
             notes_df: pd.DataFrame -> dataframe with the notes in a text column
             remove_punctuation: bool -> False will not remove punctuationm, true will
             use regex to strip certain punctuation


        Returns: pd.DataFrame: notes_df
        """

        logger.info("Cleaning data!")

        text_col = self.text_col
        filtered_df = notes_df.copy()

        # remove some punctuation
        if remove_punctuation:
            filtered_df[text_col] = filtered_df[text_col].replace(
                r"\[.*?\]", "", regex=True
            )

        # print("Cleaning and applying some regex rules!")
        # remove refundnant new lines etc
        # nan_value = float("NaN")

        # heuristic rules
        for original, replacement in tqdm(self.replacement_map):
            # drop if text is none
            filtered_df = filtered_df[filtered_df[text_col].notna()]

            # in case dataframe is empty after drop
            if filtered_df.empty:
                return filtered_df

            filtered_df[text_col] = filtered_df[text_col].str.replace(
                original, replacement
            )

        # if admin language is not none, try remove?
        if admin_language is not None:
            for admin_token in admin_language:
                filtered_df[text_col] = filtered_df[text_col].str.replace(
                    admin_token, " "
                )
        # strip white space and lower case
        filtered_df[text_col] = filtered_df[text_col].str.strip()
        filtered_df[text_col] = filtered_df[text_col].str.lower()

        # found white spaces still persist - so now double remove them
        filtered_df[text_col] = [re.sub(r"\\s+", " ", x) for x in filtered_df[text_col]]

        # drop na

        return filtered_df.dropna()

    def create_classification_datasets(self):

        """
        Slightly crude function to run over the dataframe and create each
        classification dataset we have

        """

        # read in whole df - which will be slit into train/val/test sets
        start = time.time()

        text_col = self.text_col

        if self.sample:
            logger.info("Will be sampling the dataset")

            notes_temp = pd.read_csv(self.data_path, nrows=self.sample_size)
            save_path = f"{self.save_path}/sample_{self.sample_size}/"
        else:
            notes_temp = pd.read_csv(self.data_path)
            save_path = self.save_path
        # first clean whole dataframe

        # clean notes
        cleaned_all_df = self.clean_data(
            admin_language=self.admin_language,
            remove_punctuation=self.remove_punctuation,
            notes_df=notes_temp,
        )

        logger.warning(f"Shape of all cleaned data is: {cleaned_all_df.shape}")

        # TODO

        # - Add a more appropriate string label mapping to allow better plots etc

        if self.process_individually:
            # now work on each dataset in turn

            for key, dataset in self.dataset_names_map.items():
                logger.info(f"Working on {dataset}")

                # drop any nas for that dataset class
                dataset_df = cleaned_all_df[cleaned_all_df[key].notna()]
                logger.warning(f"Shape of non-NA dataset df is: {dataset_df.shape}")

                # convert labels to int
                dataset_df[key] = dataset_df[key].astype(int)

                # assign to label column
                dataset_df["label"] = dataset_df[key]

                dataset_df = dataset_df[[text_col, "label"]]

                # create train,val, test splits

                df_train, df_valid, df_test = self.train_val_test_split(dataset_df)

                logger.info(
                    (
                        f"Shape of training data: {df_train.shape}\n\nShape of "
                        f"validation data: {df_valid.shape}\n\nShape of test "
                        f"data: {df_test.shape}"
                    )
                )

                # save to the save_dir with a new subfolder
                dataset_save_path = f"{save_path}/{dataset}/"
                # make save_dirs if not exist
                if not os.path.exists(f"{dataset_save_path}"):
                    os.makedirs(f"{dataset_save_path}")

                # save each dataframe to file
                df_train.to_csv(f"{dataset_save_path}/train.csv", index=None)
                df_valid.to_csv(f"{dataset_save_path}/valid.csv", index=None)
                df_test.to_csv(f"{dataset_save_path}/test.csv", index=None)
        else:
            # or just create one all together
            # save to the save_dir with a new subfolder
            dataset_save_path = f"{save_path}/all_together/"
            # make save_dirs if not exist
            if not os.path.exists(f"{dataset_save_path}"):
                os.makedirs(f"{dataset_save_path}")

            dataset_df = cleaned_all_df.rename(columns=self.dataset_names_map).copy()

            # create train,val, test splits

            df_train, df_valid, df_test = self.train_val_test_split(dataset_df)

            logger.info(
                (
                    f"Shape of training data: {df_train.shape}\n\nShape of validation "
                    f"data: {df_valid.shape}\n\nShape of test data: {df_test.shape}"
                )
            )

            # save each dataframe to file
            df_train.to_csv(f"{dataset_save_path}/train.csv", index=None)
            df_valid.to_csv(f"{dataset_save_path}/valid.csv", index=None)
            df_test.to_csv(f"{dataset_save_path}/test.csv", index=None)

        end = time.time()

        print(f"That took a total of: {end - start} seconds")

    def train_val_test_split(self, dataset):
        df_train = dataset.sample(frac=0.8, random_state=self.seed)
        df_test = dataset.drop(df_train.index)
        df_val = df_train.sample(frac=0.25, random_state=self.seed)
        df_train = df_train.drop(df_val.index)
        return df_train, df_val, df_test


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_path",
        type=str,
        help="The data path to the file containing patient cohort data file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="The directory to save processed and cleaned data files",
    )

    parser.add_argument(
        "--admin_language",
        default=[
            "FINAL REPORT",
            "Date/Time",
            "Phone",
            "Date of Birth",
            "DoB",
            "Completed by",
            "Dictated by",
            "name of assessor:",
            "assessed by",
            "private and confidential",
            "\t",
        ],
        type=list,
        help=(
            "User defined list of strings to replace during cleaning using "
            "regular expression"
        ),
    )
    parser.add_argument(
        "--replacement_map",
        default=[
            ("\n", " "),
            ("\r\n\r", " "),
            ("\r", " "),
            ("\t", " "),
            ("w/", "with"),
            ("_", " "),
            ("*", " "),
            ("  ", " "),
            ('"', ""),
            ("-", " "),
            ("pt", "patient"),
        ],
        type=list,
        help=(
            "set of tuples to act as replacement maps. The first element of each "
            "tuple will be replaced by the second iteratively",
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether or not to process a sub sample of the data",
    )
    parser.add_argument(
        "--sample_size",
        default=10,
        type=int,
        help="The sample size to use when subsetting training data",
    )

    parser.add_argument(
        "--text_col",
        default="IN07",
        type=str,
        help="The name of the column with the text data in",
    )
    parser.add_argument(
        "--dataset_names_map",
        default={"Key.PD09": "severity", "Key.IN05": "type", "Key.RP02": "location"},
        type=dict,
        help="The mapping from key to a more generic class name",
    )
    parser.add_argument(
        "--process_individually",
        action="store_true",
        help=(
            "Whether or not to process and save each class/dataset separately into "
            "own subfolders or put all in one"
        ),
    )

    args = parser.parse_args()

    print(f"vars args: {vars(args)}")

    # instantiate class with all arguments provided
    text_dataset = LMTextData(**vars(args))

    # now run the read_write_all_text function for training data
    text_dataset.create_classification_datasets()


if __name__ == "__main__":
    main()
