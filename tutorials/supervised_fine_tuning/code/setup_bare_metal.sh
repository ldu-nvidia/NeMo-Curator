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

# please read the README file first for prerequisites

# create virtual environment for dependency isolation
cd ~/
python3 -m venv test
source test/bin/activate

# install NeMo framework: from source
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo/requirements
pip3 install -r requirements.txt
#pip3 install cython
#pip3 install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"

# install NeMo with pip3
#pip3 install Cython packaging
#pip3 install nemo_toolkit['all']

# install huggingface cli

# read hf access token from token.env
source token.env
echo "installing huggingface hub"
pip3 install -U "huggingface_hub[cli]==0.23.2"
pip3 install -U datasets==2.1.0

# shell script to log into huggingface-hub with token
echo "login to huggingface cli"
huggingface-cli login --token $HF_ACCESS_TOKEN --add-to-git-credential

# clone the nemo-curator repo
echo "pulling git curator repo"
git clone https://github.com/ldu-nvidia/NeMo-Curator.git
echo "installing dependency for NeMo-Curator"
pip3 install cython
pip3 install --extra-index-url https://pypi.nvidia.com "./NeMo-Curator[all]"

cd NeMo-Curator/tutorials/supervised_fine_tuning/code/
echo "install packages needed for SFT playbook"
pip3 install -r requirements.txt

# create directory for storing model and download model from hf
cd ..
mkdir mistral-7B-hf
echo "downloading mistral model checkpoint into folder"
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir mistral-7B-hf
echo "downloading model checkpoint, this will take a while..."

echo "convert nemo model from .hf format to .nemo format, this will take a while..."
# TODO have issue converting
pip3 install transformers --upgrade
pip3 install torch 
python3 ~/NeMo/scripts/checkpoint_converters/convert_mistral_7b_hf_to_nemo.py --input_name_or_path=./mistral-7B-hf/ --output_path=mistral.nemo
echo "model format conversion finished!"

# might need this for data curation to work
#pip3 install -U datasets
# eiter way should work: pulling and run a prebuilt container or install NeMo framework from the source
#echo "pull and run nemo training container"
#docker run --gpus device=1 --shm-size=2g --net=host --ulimit memlock=-1 --rm -it -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results nvcr.io/nvidia/nemo:24.07 bash





python3 data_curation.py

"""# verify the size and integrity of the file
du -sh databricks-dolly-15k/databricks-dolly-15k.jsonl;
sha256sum databricks-dolly-15k/databricks-dolly-15k.jsonl

echo "preprocess data sources to follow correct format"

# could be skipped if the format is correct already
python3 /opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/preprocess.py --input databricks-dolly-15k/databricks-dolly-15k.jsonl

# sanity check for the downloaded data
echo "checking if jsonl files exist!"
ls databricks-dolly-15k/
echo "check first three examples in the output jsonl file!"
head -n 3 databricks-dolly-15k/databricks-dolly-15k-output.jsonl

# generate data and sanity check
echo "generating train validation test dataset"

echo "check train val test data are generated"
cd databricks-dolly-15k/
echo "check train, val, test data are generated"
ls"""