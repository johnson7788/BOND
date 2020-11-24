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

import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from modeling_roberta import RobertaForTokenClassification_v2
from data_utils import load_and_cache_examples, get_labels
from model_utils import multi_source_label_refine, soft_frequency, mt_update, get_mt_loss, opt_grad
from eval import evaluate

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    ),
    (),
)
# 这里roberta使用了自定义的model
MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification_v2, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, model_class, config, t_total, epoch):
    """
    初始化模型
    :param args:
    :param model_class: # huggface的model 类别，用于加载模型
    :param config: 模型配置
    :param t_total: 总的steps
    :param epoch: 训练的epoch
    :return:  model, optimizer, scheduler
    """
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # 模型到GPU
    model.to(args.device)

    # 模型参数， Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    #优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
    # 学习率scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # 如果已经存在optimizer or scheduler，那么加载已存在的
    if epoch == 0:
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    #混合精度训练
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # 多GPU，Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    #清空梯度
    model.zero_grad()
    return model, optimizer, scheduler

def train(args, train_dataset, model_class, config, tokenizer, labels, pad_token_label_id):
    """
    训练模型
    :param args: argparse参数
    :param train_dataset:  训练集Dataset
    :param model_class: 加载好的model
    :param config: model配置
    :param tokenizer: 加载好的tokenizer
    :param labels:  所有的labels, eg: ['O', 'B-LOC', 'B-ORG', 'B-PER', 'B-MISC', 'I-PER', 'I-MISC', 'I-ORG', 'I-LOC', '<START>', '<STOP>']
    :param pad_token_label_id: pad token对应的label的id eg:-100
    :return:
    """

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir,'tfboard'))
    #计算batch_size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 随机采样的方式
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # 定义Dataloader，设置采样方式和batch_size
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #计算总的steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model, optimizer, scheduler = initialize(args, model_class, config, t_total, 0)
    # Train!
    logger.info("***** 开始训练 *****")
    logger.info("  样本总数 = %d", len(train_dataset))
    logger.info("  Epochs总数 = %d", args.num_train_epochs)
    logger.info("  每个GPU的Batch size = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  梯度累积步数 = %d", args.gradient_accumulation_steps)
    logger.info("  总步数 = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # 检查是否从一个checkpoint继续训练,  重新设置global_step,epochs_trained,steps_trained_in_current_epoch
    # 需要你把自动加载model_name_or_path里面的模型
    if os.path.exists(args.model_name_or_path):
        # 从model_name_or_path获取global_step
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        #计算已经训练了多少epochs
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        #计算当前是第多少个训练epoch
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    #总的Epoch进度条
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, best_test = [0, 0, 0], [0, 0, 0]
    if args.mt:
        teacher_model = model
    self_training_teacher_model = model

    for epoch in train_iterator:
        # 每个epoch的进度条
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            #设置模型为train
            model.train()
            # 放到GPU
            batch = tuple(t.to(args.device) for t in batch)
            # 在一定步骤之后定期更新label
            if global_step >= args.self_training_begin_step:

                # 定期更新一个新的teacher模型
                delta = global_step - args.self_training_begin_step
                if delta % args.self_training_period == 0:
                    # 满足更新条件，开始更新，拷贝一个模型作为教师模型
                    self_training_teacher_model = copy.deepcopy(model)
                    #教师模型设置为评估
                    self_training_teacher_model.eval()
                    
                    # 获得新teacher后，重新初始化student模型
                    if args.self_training_reinit:
                        model, optimizer, scheduler = initialize(args, model_class, config, t_total, epoch)

                # 使用当前的teacher更新label
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                with torch.no_grad():
                    # outputs:  (loss), logits, final_embedding, (hidden_states), (attentions)
                    outputs = self_training_teacher_model(**inputs)
                label_mask = None
                if args.self_training_label_mode == "hard":
                    #直接用最大值位置索引作为硬标签
                    pred_labels = torch.argmax(outputs[0], axis=2)
                    pred_labels, label_mask = multi_source_label_refine(args,batch[5],batch[3],pred_labels,pad_token_label_id,pred_logits=outputs[0])
                elif args.self_training_label_mode == "soft":
                    #计算软标签
                    pred_labels = soft_frequency(logits=outputs[0], power=2)
                    # combined_labels 用的是真实的labels, 根据self_training_hp_label 计算 pred_labels, label_mask
                    pred_labels, label_mask = multi_source_label_refine(args=args,hp_labels=batch[5],combined_labels=batch[3],pred_labels=pred_labels,pad_token_label_id=pad_token_label_id)
                # 使用teacher模型的输出pred_labels和label_mask作为我们模型的输入
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": pred_labels, "label_mask": label_mask}
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            # 如果不是distilbert，那么需要使用segment_ids，这里是token_type_ids作为key
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                ) 
            #输入到模型
            outputs = model(**inputs)
            # 损失，logits和final_embeds，final_embeds是transformers 的roberta的输出
            loss, logits, final_embeds = outputs[0], outputs[1], outputs[2] # model outputs are always tuple in pytorch-transformers (see doc)
            mt_loss, vat_loss = 0, 0

            # Mean teacher training scheme, 使用mean teacher的方法
            if args.mt and global_step % args.mt_updatefreq == 0:
                update_step = global_step // args.mt_updatefreq
                if update_step == 1:
                    teacher_model = copy.deepcopy(model)
                    teacher_model.train(True)
                elif update_step < args.mt_rampup:
                    alpha = args.mt_alpha1
                else:
                    alpha = args.mt_alpha2
                mt_update(teacher_model.named_parameters(), model.named_parameters(), args.mt_avg, alpha, update_step)

            if args.mt and update_step > 0:
                with torch.no_grad():
                    teacher_outputs = teacher_model(**inputs)
                    teacher_logits, teacher_final_embeds = teacher_outputs[1], teacher_outputs[2]

                _lambda = args.mt_lambda
                if args.mt_class != 'smart':
                    _lambda = args.mt_lambda * min(1,math.exp(-5*(1-update_step/args.mt_rampup)**2))

                if args.mt_loss_type == "embeds":
                    mt_loss = get_mt_loss(final_embeds, teacher_final_embeds.detach(), args.mt_class, _lambda)
                else:
                    mt_loss = get_mt_loss(logits, teacher_logits.detach(), args.mt_class, _lambda)

            # Virtual adversarial training, 使用VAT的方法
            if args.vat:

                if args.model_type in ["roberta", "camembert", "xlmroberta"]:
                    word_embed = model.roberta.get_input_embeddings()
                elif args.model_type == "bert":
                    word_embed = model.bert.get_input_embeddings()
                elif args.model_type == "distilbert":
                    word_embed = model.distilbert.get_input_embeddings()

                if not word_embed:
                    print("Model type not supported. Unable to retrieve word embeddings.")
                else:
                    embeds = word_embed(batch[0])
                    vat_embeds = (embeds.data.detach() + embeds.data.new(embeds.size()).normal_(0, 1)*1e-5).detach()
                    vat_embeds.requires_grad_()

                    vat_inputs = {"inputs_embeds": vat_embeds, "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet"] else None
                        )  # XLM and RoBERTa don"t use segment_ids

                    vat_outputs = model(**vat_inputs)
                    vat_logits, vat_final_embeds = vat_outputs[1], vat_outputs[2]

                    if args.vat_loss_type == "embeds":
                        vat_loss = get_mt_loss(vat_final_embeds, final_embeds.detach(), args.mt_class, 1)
                    else:
                        vat_loss = get_mt_loss(vat_logits, logits.detach(), args.mt_class, 1)
                    # 优化梯度
                    vat_embeds.grad = opt_grad(vat_loss, vat_embeds, optimizer)[0]
                    norm = vat_embeds.grad.norm()

                    if (torch.isnan(norm) or torch.isinf(norm)):
                        print("Hit nan gradient in embed vat")
                    else:
                        adv_direct = vat_embeds.grad / (vat_embeds.grad.abs().max(-1, keepdim=True)[0]+1e-4)
                        vat_embeds = vat_embeds + args.vat_eps * adv_direct
                        vat_embeds = vat_embeds.detach()

                        vat_inputs = {"inputs_embeds": vat_embeds, "attention_mask": batch[1], "labels": batch[3]}
                        if args.model_type != "distilbert":
                            inputs["token_type_ids"] = (
                                batch[2] if args.model_type in ["bert", "xlnet"] else None
                            )  # XLM and RoBERTa don"t use segment_ids

                        vat_outputs = model(**vat_inputs)
                        vat_logits, vat_final_embeds = vat_outputs[1], vat_outputs[2]
                        if args.vat_loss_type == "embeds":
                            vat_loss = get_mt_loss(vat_final_embeds, final_embeds.detach(), args.mt_class, args.vat_lambda) \
                                    + get_mt_loss(final_embeds, vat_final_embeds.detach(), args.mt_class, args.vat_lambda)
                        else:
                            vat_loss = get_mt_loss(vat_logits, logits.detach(), args.mt_class, args.vat_lambda) \
                                    + get_mt_loss(logits, vat_logits.detach(), args.mt_class, args.vat_lambda)
            # 可以mt和vat一起使用，然后计算损失，也可以都不用
            loss = loss + args.mt_beta * mt_loss + args.vat_beta * vat_loss
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            #混合精度
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # 把损失取出来，计算总损失
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 满足一定step，就记录日志
                    if args.evaluate_during_training:

                        logger.info("***** Entropy loss: %.4f, mean teacher loss : %.4f; vat loss: %.4f *****", \
                            loss - args.mt_beta * mt_loss - args.vat_beta * vat_loss, \
                            args.mt_beta * mt_loss, args.vat_beta * vat_loss)

                        results, _, best_dev, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_dev, mode="dev", prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        results, _, best_test, is_updated  = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_test, mode="test", prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)
                        for key, value in results.items():
                            tb_writer.add_scalar("test_{}".format(key), value, global_step)

                        output_dirs = []
                        if args.local_rank in [-1, 0] and is_updated:
                            updated_self_training_teacher = True
                            output_dirs.append(os.path.join(args.output_dir, "checkpoint-best"))

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            output_dirs.append(os.path.join(args.output_dir, "checkpoint-{}".format(global_step)))

                        if len(output_dirs) > 0:
                            for output_dir in output_dirs:
                                logger.info("Saving model checkpoint to %s", args.output_dir)
                                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                                # They can then be reloaded using `from_pretrained()`
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)
                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
            # 判断是否迭代完成
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        #判断epoch是否迭代完成
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model, global_step, tr_loss / global_step, best_dev, best_test

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="输入数据目录。应该包含CoNLL-2003 NER任务的训练文件",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="列表中选择的模型类型: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="列表中选择的预训练模型或快捷方式名称的路径: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="输出目录, 将在其中写入模型预测和checkpoint",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="预训练的配置名称或路径(如果与model_name不同)"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="预训练的tokenizer名称或路径(如果与model_name不同)",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="您想在哪里存储从s3下载的预训练模型",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="tokenization后的最大总输入序列长度。长度大于此长度的序列将被截断，较短的序列将被填充。",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="是否在每个日志记录step的训练期间进行评估.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="如果使用的是uncased的模型，请设置此标志"
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="训练时每个GPU / CPU的批次大小。")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="评估时每个GPU / CPU的批次大小。"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="在执行向后/更新过程之前要梯度累积的更新步骤数。",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="BETA2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="要执行的训练epoch总数。"
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: 设置要执行的训练步骤总数。覆盖num_train_epochs。",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    # mean teacher
    parser.add_argument('--mt', type = int, default = 0, help = 'mean teacher. 是否使用mean teacher')
    parser.add_argument('--mt_updatefreq', type=int, default=1, help = 'mean teacher update frequency')
    parser.add_argument('--mt_class', type=str, default="kl", help = 'mean teacher class, choices:[smart, prob, logit, kl(default), distill].')
    parser.add_argument('--mt_lambda', type=float, default=1, help= "trade off parameter of the consistent loss.")
    parser.add_argument('--mt_rampup', type=int, default=300, help="rampup iteration.")
    parser.add_argument('--mt_alpha1', default=0.99, type=float, help="moving average parameter of mean teacher (for the exponential moving average).")
    parser.add_argument('--mt_alpha2', default=0.995, type=float, help="moving average parameter of mean teacher (for the exponential moving average).")
    parser.add_argument('--mt_beta', default=10, type=float, help="coefficient of mt_loss term.")
    parser.add_argument('--mt_avg', default="exponential", type=str, help="moving average method, choices:[exponentail(default), simple, double_ema].")
    parser.add_argument('--mt_loss_type', default="logits", type=str, help="subject to 衡量模型差异, choices:[embeds, logits(default)].")

    # virtual adversarial training
    parser.add_argument('--vat', type = int, default = 0, help = 'virtual adversarial training.')
    parser.add_argument('--vat_eps', type = float, default = 1e-3, help = 'perturbation size for virtual adversarial training.')
    parser.add_argument('--vat_lambda', type = float, default = 1, help = 'trade off parameter for virtual adversarial training.')
    parser.add_argument('--vat_beta', type = float, default = 1, help = 'coefficient of the virtual adversarial training loss term.')
    parser.add_argument('--vat_loss_type', default="logits", type=str, help="subject to measure model difference, choices = [embeds, logits(default)].")

    # self-training
    parser.add_argument('--self_training_reinit', type = int, default = 0, help = '如果teacher模型已更新，是否重新初始化student模型。0表示重启重新初始化，1表示不初始化')
    parser.add_argument('--self_training_begin_step', type = int, default = 900, help = '开始步骤(通常在第一个epoch之后)开始self-training。')
    parser.add_argument('--self_training_label_mode', type = str, default = "hard", help = '伪标签类型. choices:[hard(default), soft]. 软标签是一个teacher模型预测出来的，类似logits的概率值，是浮点数，硬标签直接就是整数，就是对应概率最大的位置的索引，例如soft是0.82, hard就是1')
    parser.add_argument('--self_training_period', type = int, default = 878, help = 'the self-training period., 每训练多少个step后，更新一下teacher模型')
    parser.add_argument('--self_training_hp_label', type = float, default = 0, help = 'use high precision label.')
    parser.add_argument('--self_training_ensemble_label', type = int, default = 0, help = 'use ensemble label.')

    args = parser.parse_args()

    # 决定是否覆盖已有的output目录
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # 如果outputs目录不存在，那么创建
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)
    logger.warning(
        "处理的 rank: %s, device: %s, n_gpu: %s, 是否分布式训练: %s, 是否 16-bits 训练 : %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    # 获取这个数据的所有labels.  eg: ['O', 'B-LOC', 'B-ORG', 'B-PER', 'B-MISC', 'I-PER', 'I-MISC', 'I-ORG', 'I-LOC', '<START>', '<STOP>']
    labels = get_labels(args.data_dir)
    num_labels = len(labels)
    # 使用交叉熵, 忽略索引作为padding label ID，以便以后真实标签ID去计算损失, eg: pad_token_label_id = -100
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # 加载预训练模型
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("训练/评估 参数 %s", args)

    # 开始训练
    if args.do_train:
        #加载数据集
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        #开始训练模型
        model, global_step, tr_loss, best_dev, best_test = train(args, train_dataset, model_class, config, tokenizer, labels, pad_token_label_id)
        #打印日志
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # 保存last-practice：如果您使用模型的默认名称，则可以使用from_pretrained()重新加载它
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("保存模型 checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    #评估模型
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("评估如下 checkpoints: %s", checkpoints)

        if not best_dev:
            best_dev = [0, 0, 0]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _, best_dev, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_dev, mode="dev", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)

        if not best_test:
            best_test = [0, 0, 0]
        result, predictions, _, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, best=best_test, mode="test")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test.json"), "r") as f:
                example_id = 0
                data = json.load(f)
                for item in data:
                    output_line = str(item["str_words"]) + " " + predictions[example_id].pop(0) + "\n"
                    writer.write(output_line)
                    example_id += 1

    return results


if __name__ == "__main__":
    main()
