# 代码运行指南：

### 1. 安装环境

```bash
conda create -n Kedkg python=3.9
conda activate Kedkg
pip install -r requirements.txt
python -m spacy_entity_linker "download_knowledge_base"
```

### 2. 目录结构

```
Knowledge_Editing
	- datasets // 数据集
	
	- en_core_web_md // spacy英文核心模型，用于处理NLP任务
	
	- model
		- rebel-large // 关系抽取模型
		- distilbert-base-cased // BERT分类器
		
	- output // 训练分类器所需的临时处理文件
	
	- prompts // 流程所需的prompts
	
	- test
		- dynamic_exp.py // Kedkg的主实验
		- fastchat_api // 以api格式运行模型的示例文件
		- rebel.py // 使用关系抽取模型抽取数据集中需要编辑的知识三元组（训练用）
		
	- train
		- results //训练好的 relation judge 模型
		- results_entity_judge //训练好的 entity judge 模型
		- datasets_divide.json // 用于训练分解模型的数据集
		- datasets_entity_judge.json // 用于实体判别模型的数据集
		- datasets_entity_judge.json // 用于关系判别模型的数据集
		- generate_divide.py // 用于生成训练分解模型的数据集的代码文件
		- generate_entity_judge.py // 用于生成实体判别模型的数据集的代码文件
		- generate_relation_judge.py // 用于生成关系判别模型的数据集的代码文件
		- train.py // 用于训练关系模型和判别模型的代码文件
	
	- requirements.txt // 项目依赖
```

### 3. 注意事项

（1）我们的方法使用API格式来部署模型，参考文档 https://github.com/lm-sys/FastChat

（2）您需要从huggingface上自行下载distilbert-base-cased和rebel-large模型，放到项目目录中。

https://huggingface.co/distilbert/distilbert-base-cased

https://huggingface.co/Babelscape/rebel-large


（3）我们提供了训练所需的所有数据集和实体判别模型、关系判别模型的数据集文件和生成代码，但分解模型的训练是使用llama-factory训练的，代码在我的co-author那里，因此您需要使用datasets_divide.json 对问题分解模型进行训练。实体判别模型、关系判别模型的权重文件可以在[Google Drive]https://drive.google.com/drive/folders/1TxxeuL0LPNi5K3aampXiMcPE33LPAk_S?usp=sharing, https://drive.google.com/drive/folders/1aM3XPccAAF_GzsFHT6kacQrOJIp2eGeq?usp=sharing 找到。

（4）在dynamic_exp.py文件中可能需要修改如下参数：

​	  （a）openai.api_base = "https://api.openai.com/v1" 这个需要改成你的模型端口，如果使用gpt3.5需要提供API（run_llm_answer(query) 和 run_llm_divide(query)这两个函数）

​	  （b）Line 424， Line 430 修改使用的数据集和数据总量。

（5）使用

```
python dynamic_exp.py --edit [N]
```

来运行，N是指编辑的批次大小。