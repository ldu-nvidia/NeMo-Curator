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

docker run -it -p 8080:8080 -p 8088:8088 --rm --gpus '"device=0,2"' --ipc=host --network host -v $(pwd):/workspace nvcr.io/nvidia/nemo:24.09

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



# set environment variables
MODEL="../mistral-7b.nemo"
TRAIN_DS=["data/merged/MG-Verilog_block_summary_in_out_train.jsonl"]
VALID_DS=["data/merged/MG-Verilog_block_summary_in_out_validation.jsonl"]
TEST_DS=["data/merged/MG-Verilog_block_summary_in_out_test.jsonl"]
CONCAT_SAMPLING_PROBS="[1.0]"

# set tensor and pipeline parallel size, TP_SIZE*PP_SIZE == number of available GPUs
TP_SIZE=2
PP_SIZE=1

# now run SFT command by appropriately setting the values for the parameters needed to run the job
echo "running supervised fine tuning step..."
torchrun --nproc_per_node=2 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
   trainer.precision=bf16 \
   trainer.devices=2 \
   trainer.num_nodes=1 \
   trainer.val_check_interval=0.1 \
   trainer.max_steps=5 \
   model.restore_from_path=${MODEL} \
   model.micro_batch_size=1 \
   model.global_batch_size=128 \
   model.tensor_model_parallel_size=${TP_SIZE} \
   model.pipeline_model_parallel_size=${PP_SIZE} \
   model.megatron_amp_O2=True \
   model.sequence_parallel=True \
   model.activations_checkpoint_granularity=selective \
   model.activations_checkpoint_method=uniform \
   model.optim.name=distributed_fused_adam \
   model.optim.lr=1e-6 \
   model.answer_only_loss=True \
   model.peft.peft_scheme=none \
   model.data.train_ds.file_names=${TRAIN_DS} \
   model.data.validation_ds.file_names=${VALID_DS} \
   model.data.test_ds.file_names=${TEST_DS} \
   model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
   model.data.train_ds.max_seq_length=2048 \
   model.data.validation_ds.max_seq_length=2048 \
   model.data.train_ds.micro_batch_size=1 \
   model.data.train_ds.global_batch_size=128 \
   model.data.validation_ds.micro_batch_size=1 \
   model.data.validation_ds.global_batch_size=128 \
   model.data.test_ds.micro_batch_size=1 \
   model.data.test_ds.global_batch_size=256 \
   model.data.train_ds.num_workers=0 \
   model.data.validation_ds.num_workers=0 \
   model.data.test_ds.num_workers=0 \
   model.data.validation_ds.metric.name=loss \
   model.data.test_ds.metric.name=loss \
   exp_manager.create_wandb_logger=False \
   exp_manager.explicit_log_dir=/results \
   exp_manager.resume_if_exists=True \
   exp_manager.resume_ignore_no_checkpoint=True \
   exp_manager.create_checkpoint_callback=True \
   exp_manager.checkpoint_callback_params.monitor=validation_loss \
   exp_manager.checkpoint_callback_params.save_best_model=False \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.mode=min \
   ++cluster_type=BCP
echo "finished supervised fine tuning, results saved into /results/checkpoint/"


# next is to test the sft model
# after the SFT step, we evaluate the model using megatron_gpt_generate.py script
PATH_TO_TRAINED_MODEL=/results/checkpoints/megatron_gpt_peft_none_tuning.nemo
echo "performing model testing after sft"
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    model.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=1 \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.test_ds.names=['MG-Verilog_block_summary_in_out_test'] \
    model.data.test_ds.global_batch_size=16 \
    model.data.test_ds.micro_batch_size=2 \
    model.data.test_ds.tokens_to_generate=20 \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    inference.greedy=True \
    model.data.test_ds.output_file_path_prefix=/results/sft_results \
    model.data.test_ds.write_predictions_to_file=True
echo "finished testing model, here are some sample results: "
tail -n 5 /results/sft_results.jsonl
