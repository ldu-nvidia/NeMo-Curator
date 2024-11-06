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

# install huggingface cli
echo "installing huggingface hub"
pip3 install -U "huggingface_hub[cli]"

# read token from secure location
HF_TOKEN= ""
# shell script to log into huggingface-hub with token
echo "login to huggingface cli"
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
pip3 install -U datasets

# create directory for storing model and download model from hf
mkdir mistral-7B-hf
echo "downloading mistral model checkpoint into folder"
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir mistral-7B-hf



