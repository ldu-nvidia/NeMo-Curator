Supervised-Fine-Tuning: Fine Tune foundation Model for Domain and Task Specific Applications (GTC2025)

**DLI Course ID:** x-fx-81-v1

* Author: Leo Du ldu@nvidia.com
* Maintainer: Josh Wyatt jwyatt@nvidia.com

---

**Hardware Requirements**

---

Since supervised fine tuning adjust all parameters of the model, it is a compute intensive job. This playbook can run on a compute node with at least 8 GPUs for parallel computing purposes.

---

**Software Version**

---

This tutorial uses container `nemo:24.12` please make sure you are able to pull this container from `nvcr.io/nvidia/nemo24:12`

---

Walk through

---

In this section, we will walk you through the domain adapted supervised fine tuning steps

* Step 0: you need an API to download pretrained LLM model from huggingface hub [how to create HF hub access token](https://huggingface.co/docs/hub/en/security-tokenshttps:/) place and copy paste your token to the `$HF_TOKEN`variable in token.env file you created
* Step 1: convert model format into `.nemo` format
* Step 2: perform data curation step for open source verilog dataset
* Step 4: generate train, validation and test dataset
* Step 5: conduct supervised fine tuning and generated fine tuned model checkpoint, this will take a while
* Step 6: evaluate the fine tuned model by loss

---

Usage

---

1. setup correct nvcr.io access
2. setup huggingface-hub API access key
3. run `bash curate_data.sh` to curate training data (optional, the curated data is already provided in `/data/merged`)
4. run `bash run_sft.sh` to pull foundation model, convert checkpoint format and conduct sft training
5. training log will be inside the `/logs` folder
