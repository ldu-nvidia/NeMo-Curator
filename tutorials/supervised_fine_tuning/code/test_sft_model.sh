# after the SFT step, we evaluate the model using megatron_gpt_generate.py script
PATH_TO_TRAINED_MODEL=/results/checkpoints/megatron_gpt_peft_none_tuning.nemo
TEST_DS=["code/source/test.jsonl"]
echo "performing model testing after sft"
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    model.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=1 \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.test_ds.names=['dolly-15k_test'] \
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