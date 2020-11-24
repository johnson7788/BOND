from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaForTokenClassification_v2(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        """
        :param config:  模型的配置
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        # 加载模型配置，初始化一个roberta模型
        self.roberta = RobertaModel(config)
        # 初始化一个dropout和自定义分类层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 调用transformers的自对应初始化权重函数
        self.init_weights()

    def forward( self,input_ids=None,attention_mask=None,token_type_ids=None,position_ids=None,
                 head_mask=None, inputs_embeds=None, labels=None, label_mask=None,):
        """
        :param input_ids: 输入的id
        :param attention_mask:
        :param token_type_ids: segment id
        :param position_ids: 模型使用position_id来识别哪个token在哪个位置
        :param head_mask:
        :param inputs_embeds:
        :param labels:
        :param label_mask:
        :return:
        """
        #首先调用原始的roberta的模型，得到输出, 返回 [last_hidden_states, pooled_output, hidden_states, attentions]
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        #获取last_hidden_states作为特征
        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)
        logits = self.classifier(sequence_output)
        # 输出为 [logits, final_embedding, hidden_states, attentions]
        outputs = (logits, final_embedding,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # 训练模式
            # 只计算我们关注的token的损失,
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & label_mask.view(-1)
                # 取到未mask的logits，进行下一步计算损失
                active_logits = logits.view(-1, self.num_labels)[active_loss]
            # 判断形状相同， eg: labels.shape torch.Size([16, 128])  [batch_size,seq_length]   logits.shape: torch.Size([16, 128, 11])  [batch_size,seq_length, num_class]
            if labels.shape == logits.shape:
                #散度损失. 有mask，就用mask的计算损失，否则计算所有损失
                loss_fct = KLDivLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    # 一个批次计算损失 eg: active_logits.shape: torch.Size([485, 11]),  active_labels.shape: 485
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            #把损失也加入到outputs
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, final_embedding, (hidden_states), (attentions)
