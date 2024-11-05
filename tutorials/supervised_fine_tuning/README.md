ChipNeMo - Supervised fine tuning on foundation model with NeMo Framework

---

**Introduction**

---

[ChipNeMo](https://arxiv.org/pdf/2311.00176) is a chip design domain adapted LLM. Instead of directly deploying off-the shelf commercial or open-source pretrained LLMs, the paper adopted the following domain adaptation techniques

1. domain adaptive continued pretraining
2. model alignment with domain-specific instructions
3. domain adapted retrieval models

Specifically, LLama 2 foundation model was continually pre-trained with more than 20B plus tokens on domain-specific chip design data including code, documents, etc. and then fine-tuned with instruction datasets from design data as well as external sources.

Evaluations on the domain adapted ChipNeMo model demonstrates that domain adapted pretraining of LLM can lead to superior performance in domain related downstream tasks compared to their base Llama 2 model without degenerations in generic LLM capabilities, following is a general flow of domain adaptation technique used in the paper.

---

**Hardware Requirements**

---

Since supervised fine tuning adjust all parameters of the model, it is a compute intensive job. This playbook can run on a compute node with at least 8 GPUs for parallel computing purposes.

---

**Software Version**

---

NeMo framework is subject to constant updates and sometimes using a different version could result in error, hence it is suggested to use the exact same version. This tutorial uses container `nemo:2407` and the nemo framework used is `2.0.0rc1`, please pull this branch if you are installing NeMo from source

---

Walk through

---

In this section, we will walk you through the domain adapted supervised fine tuning steps

* Step 0: you need an API to download pretrained LLM model from huggingface hub [how to create HF hub access token](https://huggingface.co/docs/hub/en/security-tokenshttps:/) place and copy paste your token to the `$HF_TOKEN`variable inside either setup script `./code/setup_bare_metal.sh` or `./code/setup_container.sh`
* Step 1: convert model format into `.nemo` format
* Step 2: download and verify dolly-15k dataset from huggingface
* Step 3: preprocess downloaded dataset
* Step 4: generate train, validation and test dataset
* Step 5: actually conduct supervised fine tuning and generated fine tuned model checkpoint, this will take a while
* Step 6: evaluate the fine tuned model using test data

---

Usage

---

**users could refer to the `demo.ipynb` notebook for expected output of each bash scripts**

The playbook works for both bare metal and containered environment. Go to the `./code` directory with `cd code/`

* **For bare metal**: additional step like creating virtual environment, pulling and running container `nvcr.io/nvidia/nemo:24.07` are needed. Run `bash setup_bare_metal.sh` in this case, although setting up your own environment cause overhead but you get to specify the exact version of NeMo you will use hence provide reproducibility.
* **Inside running `nemo:24.07` container**: if you are already inside the running container, simply run `bash setup_container.sh` will download model checkpoint only without setting up environment, although this approach is simpler, NeMo version packaged into the container can not be controlled explicitly, possibly resulting in future issues due to code change.
* Next, run `bash run_sft.sh` which will execute from step 1 till step 6

---

TODO

---

1. fix bug in the test script run
2. how to evaluate the sft model?
