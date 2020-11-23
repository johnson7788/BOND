# BOND
此repo包含我们的代码和论文的distantly/weakly labeled预处理数据 [BOND: BERT-Assisted Open-Domain Name Entity Recognition with Distant Supervision (KDD2020)](https://arxiv.org/abs/2006.15509)

## BOND

![BOND-Framework](docs/bond.png)

## Benchmark
结果(实体级别的F1 Score)总结如下：

| Method | CoNLL03 | Tweet | OntoNote5.0 | Webpage | Wikigold |
| ------ | ------- | ----- | ----------- | ------- | -------- |
| Full Supervision  | 91.21 | 52.19 | 86.20 | 72.39 | 86.43 |
| Previous SOTA | 76.00 | 26.10 | 67.69 | 51.39 | 47.54 |
| BOND | 81.48 | 48.01 | 68.35 | 65.74 | 60.07 |

- *Full Supervision*: Roberta Finetuning/BiLSTM CRF
- *Previous SOTA*: BiLSTM-CRF/AutoNER/LR-CRF/KALM/CONNET

## 依赖
```buildoutcfg
python 3.7
pip install -r requirements.txt
```

## 目录
```buildoutcfg
├── LICENSE
├── README.md
├── data_utils.py
├── dataset
├── docs
├── eval.py
├── model_utils.py
├── modeling_roberta.py
├── outputs/
    └── conll03  模型输出
├── pretrained_model  保存transformers下载的预训练模型
├── requirements.txt   python依赖
├── run_ner.py
├── run_self_training_ner.py
├── scripts
└── venv   #环境文件
```

## Data
我们在这里发布了五个开放域的远距离/弱标签的NER数据集: [dataset](dataset)
```buildoutcfg
一条数据的格式
"str_words": ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."], 
"words": [0, 15473, 0, 590, 8, 3848, 0, 6233, 2], 
"chars": [[31, 48], [6, 0, 52, 0, 12, 2, 7], [42, 0, 6, 14, 1, 3], [12, 1, 9, 9], [2, 5], [21, 5, 19, 12, 5, 2, 2], [36, 6, 4, 2, 4, 7, 11], [9, 1, 14, 21], [18]], 
"tags": [2, 0, 0, 0, 0, 0, 0, 0, 0], 
"defs": [null, null, null, null, null, null, null, null, null]}


tag_to_id.json
{"O": 0, "B-LOC": 1, "B-ORG": 2, "B-PER": 3, "B-MISC": 4, "I-PER": 5, "I-MISC": 6, "I-ORG": 7, "I-LOC": 8, "<START>": 9, "<STOP>": 10}
```
## Training & Evaluation

我们提供了所有五个开放域的远距离/弱标签NER数据集的训练脚本 [scripts](scripts). E.g., 对CoNLL03进行BOND训练和评估
```
cd BOND
./scripts/conll_self_training.sh
```
对于CoNLL03的第一阶段训练和评估
```
cd BOND
./scripts/conll_baseline.sh
```
测试结果(实体水平F1 Score)总结如下：

| Method | CoNLL03 | Tweet | OntoNote5.0 | Webpage | Wikigold |
| ------ | ------- | ----- | ----------- | ------- | -------- |
| Stage I| 75.61   | 46.61 | 68.11       | 59.11   | 52.15    |
| BOND   | 81.48   | 48.01 | 68.35       | 65.74   | 60.07    |


## Citation

Please cite the following paper if you are using our datasets/tool. Thanks!

```
@inproceedings{liang2020bond,
  title={BOND: Bert-Assisted Open-Domain Named Entity Recognition with Distant Supervision},
  author={Liang, Chen and Yu, Yue and Jiang, Haoming and Er, Siawpeng and Wang, Ruijia and Zhao, Tuo and Zhang, Chao},
  booktitle={ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2020}
}
```
