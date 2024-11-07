# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import numpy as np
from typing import Any, Optional, Tuple
from functools import reduce

import matplotlib.pyplot as plt
from downloaders import (
    download_github_sources,
    download_pdf_sources,
    download_wikipedia_sources,
    download_huggingface_sources,
)
from utils import (
    CodeLineCountFilter,
    TextLineCountFilter,
    clean_and_unify,
    dedupe,
    filter_code,
    filter_text,
    redact_code,
)

import nemo_curator as nc
from nemo_curator import ExactDuplicates, Modify, ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import RepeatingTopNGramsFilter, WordCountFilter
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import (
    get_all_files_paths_under,
    separate_by_metadata,
)
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")


def download_sources(hf_limit: Optional[int] = None,
) -> str:
    """
    Downloads all the dataset sources and converts them to the JSONL format.
    Args:
        wikipedia_limit (int): Maximum number of wiki urls to be downloaded
        github_limit (int): Maximum number of github repos to be downloaded
        pdf_limit (int): Maximum number of pdf to be downloaded
        hf_limit (int): Maximum number of huggingface datasets to be downloaded

    Returns:
        tuple: the list of text files and the list of code files.
    """

    hf_dir = download_huggingface_sources("sources/huggingface_urls.jsonl", limit = hf_limit)
    hf_files = get_all_files_paths_under(hf_dir)

    # delete original file once extracted into four files
    code_file = ""
    text_file = []
    for file in hf_files:
        if "code_out.jsonl" in file:
            code_file += file
        elif "MG-Verilog.jsonl" in file:
            os.remove(file)
            print("removed original download")
        else:
            text_file.append(file)
    return text_file, code_file

def run_curation_pipeline(args: Any, text_files: list, code_files: str) -> None:
    """
    Run the curation pipeline on the Wiki+Arxiv+Github datasets.

    Args:
        args (Any): Command-line arguments.
        jsonl_dir (str): Directory path where the JSONL files are stored.
    """
    print("Running the curation pipeline...")
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))
    
    # Overwrite existing files in the curated directory.
    out_path = os.path.join(DATA_DIR, "curated")
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # Define data curation steps for text and pdf files
    curation_steps_text = Sequential(
        [
            clean_and_unify,
            ScoreFilter(
                TextLineCountFilter(), text_field="file_type_count", score_type=bool
            ),
            filter_text,
            dedupe,
        ]
    )

    # Define data curation steps for code files
    curation_steps_code = Sequential(
        [
            clean_and_unify,
            ScoreFilter(
                CodeLineCountFilter(), text_field="file_type_count", score_type=bool
            ),
            filter_code,
            dedupe,
            redact_code,
        ]
    )

    # keep record of all the entries that pass curation, only keep the entries of the unions of them
    all_passed_ids = []
    for text_file in text_files:
        orig_dataset_text = DocumentDataset.read_json(text_file, add_filename=True)
        # create a field combining fields file type and line count
        orig_dataset_text.df["file_type_count"] = (
            orig_dataset_text.df["file_type"]
            + " : "
            + orig_dataset_text.df["line_count"].astype(str)
        )

        dataset_text = curation_steps_text(orig_dataset_text)
        dataset_text = dataset_text.persist()

        print(f"Original dataset length for text files: {len(orig_dataset_text.df)}")
        print(f"After data curation: {len(dataset_text.df)}")
        all_passed_ids.append(dataset_text.df["id"].values.compute())
        #dataset_text.to_json(out_path, write_to_filename=True)
    print("finish curating text files!")
    
    
    # curate code
    print("start curating code")
    orig_dataset_code = DocumentDataset.read_json(code_files, add_filename=True)
    orig_dataset_code.df["file_type_count"] = (
        orig_dataset_code.df["file_type"]
        + " : "
        + orig_dataset_code.df["line_count"].astype(str)
    )
    dataset_code = curation_steps_code(orig_dataset_code)
    dataset_code = dataset_code.persist()
    print(f"Original dataset length for code files: {len(orig_dataset_code.df)}")
    print(f"After dataprep: {len(dataset_code.df)}")
    all_passed_ids.append(dataset_code.df["id"].values.compute())
    #dataset_code.to_json(out_path, write_to_filename=True)


    selected_ids = list(reduce(np.intersect1d, tuple(all_passed_ids)))
    print("length of selected entries: ", len(selected_ids))

    print("deleting filtered out entries from original text and code")
    
    '''
    separated_data_text = separate_by_metadata(
        dataset_text.df, out_path, "category"
        ).compute()
    separated_data_code = separate_by_metadata(
        dataset_code.df, out_path, "category"
    ).compute()
    '''
    
    client.close()


def blend_and_shuffle(
    args: Any, dataset_paths: list, dataset_weights: list, target_size: int
) -> None:
    """
    Blend and shuffle curated data based on file paths for continued pre-training

    Args:
        args (Any): Command-line arguments.
        dataset_paths (list): List containing directory paths where the different JSONL files are stored.
        dataset_weights (list): List setting weights for each directory path
        target_size (int): Target number of data samples after blending
    """
    root_path = os.path.join(DATA_DIR, "curated")
    output_path = root_path + "/data_blended"
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Blend the datasets
    datasets = [DocumentDataset.read_json(path) for path in dataset_paths]
    blended_dataset = nc.blend_datasets(target_size, datasets, dataset_weights)

    shuffle = nc.Shuffle(seed=42)
    blended_dataset = shuffle(blended_dataset)

    # Save the blend
    blended_dataset.to_json(output_path)


'''
function to split data into train val test
# +
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this is the main script to do supervised fine tuning on custom datasets
import json
import random
data_folder = "databricks-dolly-15k/"

def write_file(data, filename):
    assert data and len(data) != 0, "data provided is empty, did not save!"
    with open(filename, "w") as f:
        for line in data:
            f.write(line.strip() + "\n")
    print("finished writing file: ", filename)

# define marco to be used 
input_file = data_folder + "databricks-dolly-15k-output.jsonl"
training_output_file = data_folder + "train.jsonl"
validation_output_file = data_folder + "validation.jsonl"
test_output_file = data_folder + "test.jsonl"

# specify proportion of data for training and validation
train_proportion = 0.80
validation_proportion = 0.15

assert train_proportion > 0.0 and validation_proportion > 0.0 and train_proportion + validation_proportion < 1.0, "either train or validation proportion is not right!"

# read and shuffle JSON file objects
with open(input_file, "r") as f:
    lines = f.readlines()
    random.shuffle(lines)

# calculate split indices
total_lines = len(lines)
train_index = int(total_lines * train_proportion)
val_index = int(total_lines * (train_proportion + validation_proportion))

# distribute JSON objects into train, validation, tests sets
train_data = lines[:train_index]
validation_data = lines[train_index:val_index]
test_data = lines[val_index:]

# write JSON objects to files
write_file(train_data, training_output_file)
write_file(validation_data, validation_output_file)
write_file(test_data, test_output_file)
print("finish generating train, validation and test data")

'''

def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)
    print("Args: ", args)

    # Download all the sources and get the list of text and code files.
    text_files, code_file = download_sources(1)
    run_curation_pipeline(args, text_files, None)
    
    '''
    # blend and shuffle datasets
    root_path = os.path.join(DATA_DIR, "curated")
    dataset_paths = [
        root_path + "/CPP",
        root_path + "/VerilogVHDL",
        root_path + "/text",
        root_path + "/Python",
    ]
    dataset_weights = [1.0, 4.0, 4.0, 1.0]
    target_size = 20
    blend_and_shuffle(args, dataset_paths, dataset_weights, target_size)

    ### TODO: add step to split data into train validation test
    '''


if __name__ == "__main__":
    main()
