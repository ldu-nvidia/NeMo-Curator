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

# if compute uses slurm, request compute
srun -G 8 --exclusive -p dgxa100_80g_2tb --pty -t 24:00:00 bash

docker run -it -p 8080:8080 -p 8088:8088 --rm --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host --network host -v $(pwd):/workspace nvcr.io/nvidia/nemo:24.09

python -m pip install --upgrade pip
source token.env
echo "installing huggingface hub"
pip install -U "huggingface_hub[cli]"

huggingface-cli login --token $HF_ACCESS_TOKEN

## download Llama3.1-8b model from HF and convert the format

mkdir Llama-3.1-8b/
echo "downloading llama3 model checkpoint into folder"
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir Llama-3.1-8b
echo "finished downloading Llama3.1-8b model from huggingface"

echo "convert nemo model from .hf format to .nemo format, this will take a while..."
python3 /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path=./Llama-3.1-8b/ --output_path=Llama-3.1-8b.nemo
# check if the converted file exist
if [ -f "Llama-3.1-8b.nemo" ]; then
    echo "model format conversion finished, delete huggingface model file"
    rm -rf Llama-3.1-8b/
else 
    echo "format conversion failed, exit"
    exit
fi


##### training script for actual sft
cd /workspace/Documents/Repos/NeMo-Curator/tutorials/supervised_fine_tuning/code
MODEL="/workspace/results_lr20e-6_mb32/checkpoints/megatron_gpt_peft_none_tuning.nemo"
TRAIN_DS=["data/merged/MG-Verilog_high_level_global_summary_in_out_train.jsonl"]
VALID_DS=["data/merged/MG-Verilog_high_level_global_summary_in_out_validation.jsonl"]
TEST_DS=["data/merged/MG-Verilog_high_level_global_summary_in_out_test.jsonl"]
CONCAT_SAMPLING_PROBS="[1.0]"

# set tensor and pipeline parallel size, TP_SIZE*PP_SIZE == number of available GPUs
TP_SIZE=8
PP_SIZE=1
SCHEME="lora"
LR=2e-5
BATCH_SIZE=32
OUTPUT_DIR="/workspace/results_LR_"+"$LR"+"_BATCH_SIZE_"+"$BATCH_SIZE/"
echo "output directory is " + $OUTPUT_DIR


# now run SFT command by appropriately setting the values for the parameters needed to run the job
echo "running supervised fine tuning step..."
torchrun --nproc_per_node=8 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
   trainer.precision=bf16 \
   trainer.devices=8 \
   trainer.num_nodes=1 \
   trainer.val_check_interval=0.1 \
   trainer.max_steps=20 \
   model.restore_from_path=${MODEL} \
   model.micro_batch_size=32 \
   model.global_batch_size=128 \
   model.tensor_model_parallel_size=${TP_SIZE} \
   model.pipeline_model_parallel_size=${PP_SIZE} \
   model.megatron_amp_O2=True \
   model.sequence_parallel=True \
   model.activations_checkpoint_granularity=selective \
   model.activations_checkpoint_method=uniform \
   model.optim.name=distributed_fused_adam \
   model.optim.lr=4e-5 \
   model.answer_only_loss=True \
   model.peft.peft_scheme=none \
   model.data.train_ds.file_names=${TRAIN_DS} \
   model.data.validation_ds.file_names=${VALID_DS} \
   model.data.test_ds.file_names=${TEST_DS} \
   model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
   model.data.train_ds.max_seq_length=2048 \
   model.data.validation_ds.max_seq_length=2048 \
   model.data.train_ds.micro_batch_size=32 \
   model.data.train_ds.global_batch_size=128 \
   model.data.validation_ds.micro_batch_size=32 \
   model.data.validation_ds.global_batch_size=128 \
   model.data.test_ds.micro_batch_size=32 \
   model.data.test_ds.global_batch_size=128 \
   model.data.train_ds.num_workers=8 \
   model.data.validation_ds.num_workers=8 \
   model.data.test_ds.num_workers=8 \
   model.data.validation_ds.metric.name=loss \
   model.data.test_ds.metric.name=loss \
   exp_manager.create_wandb_logger=False \
   exp_manager.explicit_log_dir=/workspace/results_lr_4e-5_B_32_12_24th \
   exp_manager.resume_if_exists=True \
   exp_manager.resume_ignore_no_checkpoint=True \
   exp_manager.create_checkpoint_callback=True \
   exp_manager.checkpoint_callback_params.monitor=validation_loss \
   exp_manager.checkpoint_callback_params.save_best_model=True \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.mode=min \
   ++cluster_type=BCP



cd /workspace/Documents/Repos/NeMo-Curator/tutorials/supervised_fine_tuning/code
# this is the original model
MODEL="/workspace/Llama-3.1-8b.nemo"
TEST_DS=["data/merged/MG-Verilog_high_level_global_summary_in_out_test.jsonl"] 
TEST_NAMES="[testingsftperformance]"

TP_SIZE=8
PP_SIZE=1

#  this is the model after sft
PATH_TO_TRAINED_MODEL="/workspace/results_lr20e-6_mb32/checkpoints/megatron_gpt_peft_none_tuning.nemo"

# The generation run will save the generated outputs over the test dataset in a file prefixed like so
OUTPUT_PREFIX="sft_output"

python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    model.restore_from_path=${MODEL} \
    model.peft.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=8 \
    trainer.num_nodes=1 \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.test_ds.names=${TEST_NAMES} \
    model.data.test_ds.global_batch_size=128 \
    model.data.test_ds.micro_batch_size=32 \
    model.data.test_ds.tokens_to_generate=400 \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    inference.greedy=True  \
    model.data.test_ds.output_file_path_prefix=${OUTPUT_PREFIX} \
    model.data.test_ds.write_predictions_to_file=True \
    model.data.test_ds.truncation_field="null" \
    model.data.test_ds.add_bos=False \
    model.data.test_ds.add_eos=True \
    model.peft.peft_scheme=none \
    model.data.test_ds.add_sep=False \
    model.data.test_ds.label_key="output" \
    model.data.test_ds.prompt_template="\{input\}\ \{output\}"

