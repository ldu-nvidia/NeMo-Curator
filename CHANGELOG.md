# Changelog

## NeMo Curator 0.5.0

### Highlights

- Image Curation
  - Image Embedding Creation
  - Aesthetic Classifier
  - NSFW Classifier
  - Semantic Deduplication
- Text Curation
  - Quality Classifier
  - Aegis Classifier
  - FineWeb-Edu Classifier

**Full Changelog**: <https://github.com/NVIDIA/NeMo-Curator/commits/v0.5.0>

## NeMo Curator 0.4.1

## What's Changed

* Add spacy<3.8 pin to r0.4.1 by @ayushdg in <https://github.com/NVIDIA/NeMo-Curator/pull/279>

**Full Changelog**: <https://github.com/NVIDIA/NeMo-Curator/compare/v0.4.0...v0.4.1>

## NeMo Curator 0.4.0

## Highlights

- Semantic Deduplication
- Resiliparse for Text Extraction
- Improve Distributed Data Classification - Domain classifier is 1.55x faster through intelligent batching
- Synthetic data generation for fine-tuning

## What's Changed

* Update README by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/6
* [Tutorials] Add a readme file for the TinyStories tutorial by @Maghoumi in https://github.com/NVIDIA/NeMo-Curator/pull/5
* Add workflow for running cpu pytests by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/13
* Add pre-commit style checks by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/14
* Add citation by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/15
* Fix Noisy CUDA Shutdown by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/20
* Bump Python and RAPIDS versions by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/16
* Add batched decorator by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/18
* Add issue templates by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/22
* Add dependency to fix justext by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/24
* Fix metadata inference with pandas and dask by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/35
* Disable PyTorch Compile Multiprocessing by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/34
* Improve speed of AddId module by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/36
* Make GPU dependencies optional by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/27
* Fix failing GPU tests with latest pandas bump by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/41
* Adds Nemo Curator K8s example by @terrykong in https://github.com/NVIDIA/NeMo-Curator/pull/40
* Move common dedup utils and remove unused code by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/42
* Fix lang id example by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/37
* Add dataset blending tool by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/32
* High level fuzzy duplicates module by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/46
* Fix indexing in PII Modifier by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/55
* Disable string conversion globally by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/56
* Fix issue #43 (empty files creation) and improve reading/writing speed by @miguelusque in https://github.com/NVIDIA/NeMo-Curator/pull/57
* [Tutorials] Add a tutorial for PEFT data curation by @Maghoumi in https://github.com/NVIDIA/NeMo-Curator/pull/45
* Only import PII constants during Curator import by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/61
* Align `extract_partitioning_index` logic with upstream shuffling by @rjzamora in https://github.com/NVIDIA/NeMo-Curator/pull/60
* [REVIEW] Switch Models to use Crossfit by @VibhuJawa in https://github.com/NVIDIA/NeMo-Curator/pull/58
* Remove argparse from get_client function signature by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/12
* Fuzzy Dedup:  Use text_field instead of hardcoded text column by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/74
* Add pull request template by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/78
* Add jupyter notebook tutorial for single node mulilingual dataset by @nicoleeeluo in https://github.com/NVIDIA/NeMo-Curator/pull/30
* Update issue templates by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/81
* Fix #91 - Incorrect reference to domain_classifier_example.py by @miguelusque in https://github.com/NVIDIA/NeMo-Curator/pull/92
* Fix #63. Add --input-meta parameter to explicitly specify the jsonl field dtypes by @miguelusque in https://github.com/NVIDIA/NeMo-Curator/pull/75
* Update readme by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/93
* Update documentation for new version by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/83
* Update requirements documentation. by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/98
* Make sure query-planning is disabled for now by @rjzamora in https://github.com/NVIDIA/NeMo-Curator/pull/97
* Applying SEO Best Pratices by @aschilling-nv in https://github.com/NVIDIA/NeMo-Curator/pull/104
* Shuffle CC result on group before writing out by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/110
* Added tutorials to index.rst by @jgerh in https://github.com/NVIDIA/NeMo-Curator/pull/113
* Pin to numpy<2 to avoid spacy compat issues by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/119
* Fix #116. Fix broken links by @miguelusque in https://github.com/NVIDIA/NeMo-Curator/pull/117
* Update index.rst by @aschilling-nv in https://github.com/NVIDIA/NeMo-Curator/pull/129
* Fix nemo_curator import in CPU only environment when GPU packages are installed. by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/123
* Improve Common Crawl download by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/82
* Update README.md by @Maghoumi in https://github.com/NVIDIA/NeMo-Curator/pull/126
* Allow multiple filenames per partition when separating by metadata by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/99
* [REVIEW] Add Resiliparse option for text extraction by @sarahyurick in https://github.com/NVIDIA/NeMo-Curator/pull/128
* Fix 69 - Refactor how arguments are added to scripts by @miguelusque in https://github.com/NVIDIA/NeMo-Curator/pull/102
* Stricter check for query planning. by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/107
* Add DataFrame example to Distributed Data Classification tutorial by @sarahyurick in https://github.com/NVIDIA/NeMo-Curator/pull/137
* Enable Sem-dedup by @VibhuJawa in https://github.com/NVIDIA/NeMo-Curator/pull/130
* Remove lxml installation by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/140
* Nemotron 340 SDG Pipeline Tutorial by @chrisalexiuk-nvidia in https://github.com/NVIDIA/NeMo-Curator/pull/144
* Add Synthetic Data Generation Module by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/136
* Skip explicit comms shuffle for dask-cuda 24.06 by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/147
* Add support for NeMo SDK by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/131
* [REVIEW] Fix SemDedup bugs by @VibhuJawa in https://github.com/NVIDIA/NeMo-Curator/pull/151
* [pre-commit.ci] pre-commit suggestions by @pre-commit-ci in https://github.com/NVIDIA/NeMo-Curator/pull/135
* Fix bug with torch rmm and nemo by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/155

## New Contributors
* @ayushdg made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/13
* @terrykong made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/40
* @rjzamora made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/60
* @nicoleeeluo made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/30
* @aschilling-nv made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/104
* @pre-commit-ci made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/135

**Full Changelog**: https://github.com/NVIDIA/NeMo-Curator/commits/v0.4.0s

## NeMo Curator 0.3.0

## What's Changed

* Update README by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/6
* [Tutorials] Add a readme file for the TinyStories tutorial by @Maghoumi in https://github.com/NVIDIA/NeMo-Curator/pull/5
* Add workflow for running cpu pytests by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/13
* Add pre-commit style checks by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/14
* Add citation by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/15
* Fix Noisy CUDA Shutdown by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/20
* Bump Python and RAPIDS versions by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/16
* Add batched decorator by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/18
* Add issue templates by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/22
* Add dependency to fix justext by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/24
* Fix metadata inference with pandas and dask by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/35
* Disable PyTorch Compile Multiprocessing by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/34
* Improve speed of AddId module by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/36
* Make GPU dependencies optional by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/27
* Fix failing GPU tests with latest pandas bump by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/41
* Adds Nemo Curator K8s example by @terrykong in https://github.com/NVIDIA/NeMo-Curator/pull/40
* Move common dedup utils and remove unused code by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/42
* Fix lang id example by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/37
* Add dataset blending tool by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/32
* High level fuzzy duplicates module by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/46
* Fix indexing in PII Modifier by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/55
* Disable string conversion globally by @ryantwolf in https://github.com/NVIDIA/NeMo-Curator/pull/56
* Fix issue #43 (empty files creation) and improve reading/writing speed by @miguelusque in https://github.com/NVIDIA/NeMo-Curator/pull/57
* [Tutorials] Add a tutorial for PEFT data curation by @Maghoumi in https://github.com/NVIDIA/NeMo-Curator/pull/45
* Only import PII constants during Curator import by @ayushdg in https://github.com/NVIDIA/NeMo-Curator/pull/61
* Align `extract_partitioning_index` logic with upstream shuffling by @rjzamora in https://github.com/NVIDIA/NeMo-Curator/pull/60

## New Contributors
* @Maghoumi made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/5
* @terrykong made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/40
* @miguelusque made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/57
* @rjzamora made their first contribution in https://github.com/NVIDIA/NeMo-Curator/pull/60

**Full Changelog**: https://github.com/NVIDIA/NeMo-Curator/commits/v0.3.0

## PyPi
https://pypi.org/project/nemo-curator/0.3.0/
