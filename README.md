# Knowledge Editing with Dynamic Knowledge Graphs for Multi-Hop Question Answering(KEDKG)([AAAI 2025](https://arxiv.org/abs/2412.13782))：

## Project Run Guide

### 1. Environment Setup

```bash
conda create -n Kedkg python=3.9
conda activate Kedkg
pip install -r requirements.txt
python -m spacy_entity_linker "download_knowledge_base"
```

### 2. Directory Structure

```
Knowledge_Editing
	- datasets // Dataset folder
	
	- en_core_web_md // Spacy English core model for NLP tasks
	
	- model
		- rebel-large // Relation extraction model
		- distilbert-base-cased // BERT classifier
		
	- output // Temporary processing files needed for training the classifier
	
	- prompts // Prompts required for the workflow
	
	- test
		- dynamic_exp.py // Main experiment for Kedkg
		- fastchat_api // Example file to run the model in API format
		- rebel.py // Uses the relation extraction model to extract knowledge triples that require editing (for training purposes)
		
	- train
		- results // Trained relation judge model
		- results_entity_judge // Trained entity judge model
		- datasets_divide.json // Dataset for training the decomposition model
		- datasets_entity_judge.json // Dataset for the entity judge model
		- datasets_relation_judge.json // Dataset for the relation judge model
		- generate_divide.py // Code file for generating the decomposition model training dataset
		- generate_entity_judge.py // Code file for generating the entity judge model training dataset
		- generate_relation_judge.py // Code file for generating the relation judge model training dataset
		- train.py // Code file for training relation and judge models
	
	- requirements.txt // Project dependencies

```

### 3. Notes

（1）**Model Deployment**: Our method uses an API format to deploy the model. Refer to the documentation at [FastChat](https://github.com/lm-sys/FastChat).

（2）**Pre-trained Models**: You need to download the `distilbert-base-cased` and `rebel-large` models from HuggingFace and place them in the project directory:


- [distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased)

- [rebel-large](https://huggingface.co/Babelscape/rebel-large)


（3）**Datasets and Pre-trained Models**: We provide all the datasets and generation code for training the entity judge and relation judge models. However, the decomposition model training requires LLaMA-factory. You will need to use `datasets_divide.json` to train the decomposition model. The weights for the entity and relation judge models can be found on [Google Drive](https://drive.google.com/drive/folders/14xr7ruFZdmqCJ6_thbgirmTIVeP1QWHk?usp=sharing)。

（4）**Parameter Adjustments in** `dynamic_exp.py`:

- Update `openai.api_base` to match your model's endpoint. If you are using GPT-3.5, you need to provide an API for the `run_llm_answer(query)` and `run_llm_divide(query)` functions.

- Modify **Line 424** and **Line 430** to adjust the dataset and the total data volume.

（5）**Run the Experiment**:
Use the following command to execute the experiment:

```
python dynamic_exp.py --edit [N]
```

Here, `N` represents the batch size for editing.

## Reference
Please cite our paper if you use our models in your works:

```
@inproceedings{kedkg_aaai_25,
	title={Knowledge Editing with Dynamic Knowledge Graphs for Multi-Hop Question Answering},
	author = {Yifan Lu and Yigeng Zhou and Jing Li and Yequan Wang and Xuebo Liu and Daojing He and Fangming Liu and Min Zhang},
	booktitle = {The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI)},
	year={2025}
}
```