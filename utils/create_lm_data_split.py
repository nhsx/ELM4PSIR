import argparse
import os

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


def create_holdout_set(
    raw_data_file: str = None,
    save_path: str = None,
    hold_out_percentage: float = 0.10,
    seed: int = 42,
):

    """
    Function to read in all raw data and save a holdout set. Leaving a working training
    and holdout set for later use.


    Args:
        data_path: string -> absolute path to file containing all raw data
        save_path: string -> aboslute path to directory to save newly created train and
        test file
        lm_training_split: bool -> if splitting the already split lm training data, set
        to True to alter the save paths
        hold_out_percentage: float -> percentage of all data to be held out


    Example usage for creating initial training:held out sets with a 90:10 split :

    python create_lm_data_split.py --raw_data_file ./data/raw_data.csv
    --save_path ./data/lm_data/ --hold_out_percentage 0.10

    """

    logger.warning(
        (
            "Creating a train/held-out data split with a "
            f"{(1-hold_out_percentage)}:{hold_out_percentage} split!"
        )
    )
    # save to file
    # ----------- Process raw file with all notes -------------------------------
    # read in file
    logger.info(
        (
            "Splitting the raw data into training and hold out set from the raw data "
            f"file: {raw_data_file}"
        )
    )

    all_data = pd.read_csv(f"{raw_data_file}", index_col=None)

    # create the splits
    train_data, hold_data = train_test_split(
        all_data, test_size=hold_out_percentage, random_state=seed
    )
    logger.info(
        (
            f"Size of training data: {train_data.shape}"
            f"\nSize of hold data: {hold_data.shape}"
        )
    )
    train_data.to_csv(f"{save_path}/training_data.csv", index=False)
    hold_data.to_csv(f"{save_path}/held_out_data.csv", index=False)

    # - Create a new LM training and test set based on the above create training split -

    logger.info(
        (
            "Splitting the raw training data into training and test data for LM and "
            "downstream tasks!"
        )
    )
    # create the splits
    lm_train_data, lm_test_data = train_test_split(
        train_data, test_size=hold_out_percentage, random_state=seed
    )
    logger.info(
        (
            f"Size of LM training data: {lm_train_data.shape}"
            f"\nSize of LM test data: {lm_test_data.shape}"
        )
    )
    lm_train_data.to_csv(f"{save_path}/lm_training_data.csv", index=False)
    lm_test_data.to_csv(f"{save_path}/lm_test_data.csv", index=False)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--raw_data_file",
        type=str,
        help="The data path to the directory containing the raw data",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        help="The path of the folder to save the newly created train and test datasets",
    )

    parser.add_argument(
        "--hold_out_percentage",
        default=0.10,
        type=float,
        help=(
            "The decimal percentage of data to be held out i.e. the size of the held "
            "out dataset"
        ),
    )

    args = parser.parse_args()

    # assert that hold_out_percentage is under 1 i.e. it should be a decimal percentage
    assert (
        args.hold_out_percentage < 1
    ), "sample percentage should be provided as a decimal percentage i.e. 0.10 for 10 %"
    # make save_dir if not exist
    if not os.path.exists(f"{args.save_path}"):
        os.makedirs(f"{args.save_path}")

    # run the split function
    create_holdout_set(**vars(args))


if __name__ == "__main__":
    main()
