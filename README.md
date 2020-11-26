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
对中文化妆品词识别和分类
使用的robert模型是:https://github.com/ymcui/Chinese-BERT-wwm

数据集: dataset/cosmetics

注意： 使用BertTokenizer以及BertModel加载，请勿使用RobertaTokenizer/RobertaModel！
```buildoutcfg
#训练
run_self_training_ner.py --data_dir dataset/cosmetics/ --model_type roberta --model_name_or_path hfl/chinese-roberta-wwm-ext --learning_rate 1e-5 --weight_decay 1e-4 --adam_epsilon 1e-8 --adam_beta1 0.9 --adam_beta2 0.98 --num_train_epochs 50 --warmup_steps 200 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 32 --logging_steps 100 --save_steps 100000 --do_train --do_eval --do_predict --evaluate_during_training --output_dir outputs/cosmetics --cache_dir pretrained_model --seed 0 --max_seq_length 128 --overwrite_output_dir --self_training_reinit 0 --self_training_begin_step 1 --self_training_label_mode soft --self_training_period 1 --self_training_hp_label 5.9

```


## Training & Evaluation

我们提供了所有五个开放域的远距离/弱标签NER数据集的训练脚本 [scripts](scripts). 
E.g., 对CoNLL03进行BOND训练和评估
```
cd BOND
./scripts/conll_self_training.sh
```

对于CoNLL03的第一阶段训练和评估: baseline
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
