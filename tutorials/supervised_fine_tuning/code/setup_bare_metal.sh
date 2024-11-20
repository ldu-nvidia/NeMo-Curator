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



## docker command to start NeMo:24.09 container, this is NeMo2.0 version
#docker run --gpus all -it --rm -p 8885:8885  -v ~/:/workspace nvcr.io/nvidia/nemo:24.09 

docker run -it -p 8080:8080 -p 8088:8088 --rm --gpus '"device=1,2,4"' --ipc=host --network host -v $(pwd):/workspace nvcr.io/nvidia/nemo:24.09

# uninstall nemo-curator installed by the container and install the latest version instead
pip uninstall nemo-curator -y
rm -r /opt/NeMo-Curator
git clone https://github.com/NVIDIA/NeMo-Curator.git /opt/NeMo-Curator
python -m pip install --upgrade pip
pip install --extra-index-url https://pypi.nvidia.com "/opt/NeMo-Curator[all]"

# install huggingface cli
# read hf access token from token.env
source token.env
echo "installing huggingface hub"
pip install -U "huggingface_hub[cli]"
#pip3 install -U datasets


# create directory for storing model and download model from hf
echo "login to huggingface cli"
huggingface-cli login --token $HF_ACCESS_TOKEN
#mkdir Llama-3.1-8B
mkdir mistral-7b-v0.3
echo "downloading llama3 model checkpoint into folder"
#huggingface-cli download meta-llama/Llama-3.1-8B --local-dir Llama-3.1-8B
huggingface-cli download mistralai/mistral-7b-v0.3 --local-dir mistral-7b-v0.3/
echo "downloading model checkpoint, this will take a while..."

echo "convert nemo model from .hf format to .nemo format, this will take a while..."
python3 /opt/NeMo/scripts/checkpoint_converters/convert_mistral_7b_hf_to_nemo.py --input_name_or_path=./mistral-7b-v0.3/ --output_path=mistral-7b.nemo
echo "model format conversion finished!"


### everything works till here!

# install dependency to run DAPT/SFT
git clone https://github.com/ldu-nvidia/NeMo-Curator/tree/sft_playbook_development
cd NeMo-Curator
git checkout sft_playbook_development
cd tutorials/supervised_fine_tuning/code/
echo "install packages needed for SFT playbook"
python -m pip install --upgrade pip
pip install -r requirements.txt

# might have issue with opencv version 
pip install qgrid
pip uninstall --yes $(pip list --format=freeze | grep opencv)
# might not need this
#rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
pip install opencv-python-headless
# would be able to run through curate_data.py at this point
python3 data_curation.py


# pending is to run actual sft

# might need this for data curation to work
#pip3 install -U datasets
# eiter way should work: pulling and run a prebuilt container or install NeMo framework from the source
#echo "pull and run nemo training container"
#docker run --gpus device=1 --shm-size=2g --net=host --ulimit memlock=-1 --rm -it -v ${PWD}:/workspace -w /workspace -v ${PWD}/results:/results nvcr.io/nvidia/nemo:24.07 bash

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