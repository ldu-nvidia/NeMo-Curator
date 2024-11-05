# firstly convert the foundation model from .hf to .nemo format
# convert model from .hf format to .nemo format
echo "convert nemo model from .hf format to .nemo format, this will take a while..."
python3 /opt/NeMo/scripts/checkpoint_converters/convert_mistral_7b_hf_to_nemo.py --input_name_or_path=./mistral-7B-hf/ --output_path=mistral.nemo
echo "model format conversion finished!"

# run data curation pipeline
python3 data_curation.py


"""# verify the size and integrity of the file
du -sh databricks-dolly-15k/databricks-dolly-15k.jsonl;
sha256sum databricks-dolly-15k/databricks-dolly-15k.jsonl

echo "preprocess data sources to follow correct format"
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