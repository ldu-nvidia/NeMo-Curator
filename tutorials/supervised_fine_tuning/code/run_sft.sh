
# set environment variables
MODEL="mistral.nemo"
TRAIN_DS="[databricks-dolly-15k/train.jsonl]"
VALID_DS="[databricks-dolly-15k/validation.jsonl]"
TEST_DS="[databricks-dolly-15k/test.jsonl]"
VALID_NAMES="[databricks-dolly-15k]"
CONCAT_SAMPLING_PROBS="[1.0]"

# set tensor and pipeline parallel size, TP_SIZE*PP_SIZE == number of available GPUs
TP_SIZE=8
PP_SIZE=1

# now run SFT command by appropriately setting the values for the parameters needed to run the job
echo "running supervised fine tuning step..."
torchrun --nproc_per_node=8 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
   trainer.precision=bf16 \
   trainer.devices=8 \
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
