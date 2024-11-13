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
python3 -m venv sft
source sft/bin/activate

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
pip3 install -U datasets

# shell script to log into huggingface-hub with token
echo "login to huggingface cli"
huggingface-cli login --token $HF_ACCESS_TOKEN --add-to-git-credential

# clone the nemo-curator repo


# create directory for storing model and download model from hf
mkdir mistral-7B-hf
echo "downloading mistral model checkpoint into folder"
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir mistral-7B-hf
echo "downloading model checkpoint, this will take a while..."

# eiter way should work: pulling and run a prebuilt container or install NeMo framework from the source
#echo "pull and run nemo training container"
#docker run --gpus device=1 --shm-size=2g --net=host --ulimit memlock=-1 --rm -it -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results nvcr.io/nvidia/nemo:24.07 bash


