# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, hp_labels):
        """构建一个输入的样本.

        Args:
            guid: 样本的唯一ID.
            words: list. 单词组成的序列.
            labels: (Optional) list. 序列中每个单词的标签。这应该是指定用于训练和开发样本，但不用于测试样本.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.hp_labels = hp_labels

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, full_label_ids, hp_label_ids):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.full_label_ids = full_label_ids
        self.hp_label_ids = hp_label_ids

def read_examples_from_file(data_dir, mode):
    """
    读取文件
    :param data_dir:  eg:  'dataset/conll03_distant/'
    :param mode: eg: train or test
    :return:
    """
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    #样本索引从1开始
    guid_index = 1
    #用于保存所有样本
    examples = []

    with open(file_path, 'r') as f:
        data = json.load(f)
        #迭代每条数据
        for item in data:
            words = item["str_words"]
            labels = item["tags"]
            # 是否存在高精度的high precision label
            if "tags_hp" in labels:
                hp_labels = item["tags_hp"]
            else:
                # 如果不存在，就设为None
                hp_labels = [None]*len(labels)
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, hp_labels=hp_labels))
            guid_index += 1
    
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    show_exnum = 5,
):
    """
    把examples转换成 features
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    :param examples: 由InputExample组成的样本列表
    :param label_list:   eg:['O', 'B-LOC', 'B-ORG', 'B-PER', 'B-MISC', 'I-PER', 'I-MISC', 'I-ORG', 'I-LOC', '<START>', '<STOP>']
    :param max_seq_length:  eg: 128
    :param tokenizer:  eg: 加载好的tokenzier
    :param cls_token_at_end: cls是否在末尾，还是在开头
    :param cls_token:  使用的CLS token标识符，eg: [CLS] 或 '<s>'
    :param cls_token_segment_id:  默认cls token使用的id  eg: 0
    :param sep_token:  默认是 [SEP] eg: '</s>'
    :param sep_token_extra:  bool eg: True
    :param pad_on_left: 从左面开始padding
    :param pad_token:   用padding的值， eg: 1
    :param pad_token_segment_id:   pad 的segment的id，属于哪个段落 eg:0
    :param pad_token_label_id:   # pad token 对应的label id  eg:-100
    :param sequence_a_segment_id:   序列a的segment 的id  eg:0
    :param mask_padding_with_zero:
    :param show_exnum: 显示几条样本, 默认显示前5条
    :return:
    """
    features = []
    # 记录超过最大长度的样本格式
    extra_long_samples = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("已经写入10000条样本 %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        full_label_ids = []
        hp_label_ids = []
        for word, label, hp_label in zip(example.words, example.labels, example.hp_labels):
            # 单词tokenzier, 有的单词会被拆分，例如 Blackburn --> ["Black", "burn"]， 那么full_label_ids就会记录这个问题
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # 对word的第一个字使用真实的label id，后面其它的字，使用pad id. eg: ["Black", "burn"] --> [5, -100]
            label_ids.extend([label] + [pad_token_label_id] * (len(word_tokens) - 1))
            # 如果hp_label是None，那么就使用pad_token_label_id, 否则就使用hp_label的一个真实id，然后其它用pad id  eg: ["Black", "burn"] -->[-100, -100]
            hp_label_ids.extend([hp_label if hp_label is not None else pad_token_label_id] + [pad_token_label_id] * (len(word_tokens) - 1))
            # 那么完全label: ["Black", "burn"] --> [5 5]
            full_label_ids.extend([label] * len(word_tokens) )

        # 特殊token等于3,如果sep_token_extra为True，因为roberta的特殊token是3个， Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        # 如果实际的token个数大于序列最大长度减去特殊长度，Roberta的特殊token一共有3个
        if len(tokens) > max_seq_length - special_tokens_count:
            # 开始截断tokens
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            hp_label_ids = hp_label_ids[: (max_seq_length - special_tokens_count)]
            full_label_ids = full_label_ids[: (max_seq_length - special_tokens_count)]
            extra_long_samples += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        hp_label_ids += [pad_token_label_id]
        full_label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta 使用额外的分开的b/w 句子对, roberta uses an extra separator b/w pairs of sentences
            # eg: ['EU', 're', 'ject', 's', 'German', 'call', 'to', 'boy', 'cott', 'British', 'lam', 'b', '.', '</s>', '</s>']
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            hp_label_ids += [pad_token_label_id]
            full_label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            hp_label_ids += [pad_token_label_id]
            full_label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            hp_label_ids = [pad_token_label_id] + hp_label_ids
            full_label_ids = [pad_token_label_id] + full_label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
        #token 转换成id,  eg: [-100, 2, 0, -100, -100, 0, 0, 0, 0, -100, 0, 0, -100, 0, -100, -100]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # mask的真实签为1，填充标签为0。仅关注真实token.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad 到最大序列长度
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            hp_label_ids = ([pad_token_label_id] * padding_length) + hp_label_ids
            full_label_ids = ([pad_token_label_id] * padding_length) + full_label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            hp_label_ids += [pad_token_label_id] * padding_length
            full_label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(hp_label_ids) == max_seq_length
        assert len(full_label_ids) == max_seq_length

        if ex_index < show_exnum:
            logger.info(f"*** Example {ex_index} ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("hp_label_ids: %s", " ".join([str(x) for x in hp_label_ids]))
            logger.info("full_label_ids: %s", " ".join([str(x) for x in full_label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids, full_label_ids=full_label_ids, hp_label_ids=hp_label_ids)
        )
    logger.info("超过最大序列长度的样本个数有%d，总样本数有%d", extra_long_samples, len(examples))
    return features


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, evaluate=False):
    """
    :param args: argparse参数
    :param tokenizer:  初始化的tokenizer
    :param labels:  eg: ['O', 'B-LOC', 'B-ORG', 'B-PER', 'B-MISC', 'I-PER', 'I-MISC', 'I-ORG', 'I-LOC', '<START>', '<STOP>']
    :param pad_token_label_id:  eg : -100
    :param mode:  eg: train or test
    :return:
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # cache文件名字
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("从cache文件中加载features %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("未发现cache文件，开始从源文件创建features %s", args.data_dir)
        # 读取样本
        examples = read_examples_from_file(args.data_dir, mode)
        #样本转换成features, 一条样本包含 guid, tokens, input_ids, input_mask, segment_ids, 可选 label_ids, hp_label_ids, full_label_ids
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # 所有样本转换成tensor
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_full_label_ids = torch.tensor([f.full_label_ids for f in features], dtype=torch.long)
    all_hp_label_ids = torch.tensor([f.hp_label_ids for f in features], dtype=torch.long)
    # 样本的id，eg: [1,2,3,...,14040]
    all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_full_label_ids, all_hp_label_ids, all_ids)
    return dataset

def get_labels(path = None):
    """
    获取所有labels， 如果存在tag_to_id.json,那么读取后获取，否则直接返回设置好的
    :param path: 数据集目录： eg: 'dataset/conll03_distant/'
    :return:
    """
    if path and os.path.exists(path + "tag_to_id.json"):
        labels = []
        with open(path + "tag_to_id.json", "r") as f:
            data = json.load(f)
            for l, _ in data.items():
                labels.append(l)
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-LOC", "B-ORG", "B-PER", "B-MISC", "I-PER", "I-MISC", "I-ORG", "I-LOC"]

def tag_to_id(path = None):
    if path and os.path.exists(path + "tag_to_id.json"):
        with open(path + "tag_to_id.json", 'r') as f:
            data = json.load(f)
        return data
    else:
        return {"O": 0, "B-LOC": 1, "B-ORG": 2, "B-PER": 3, "B-MISC": 4, "I-PER": 5, "I-MISC": 6, "I-ORG": 7, "I-LOC": 8}

def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []

    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks


if __name__ == '__main__':
    save(args)