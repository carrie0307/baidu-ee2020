import torch.nn as nn
import torch
from transformers import *
from torchcrf import CRF
from utils import sequence_mask
from torch.autograd import Variable

class ConvNet(nn.Module):
    def __init__(self, num_labels, batch_size, device):
        super(ConvNet, self).__init__()

        # 卷积后尺寸：(W-F+2P) / S + 1
        # 池化后尺寸：(W-F)/S + 1
        self.num_labels = num_labels
        self.device = device
        self.bert_embedding = BertModel.from_pretrained(pretrained_model_name_or_path='/home/cuishiyao/roberta').to(device)
        self.bert_dim = 768
        self.cnn_out_dim = 200
        self.use_crf = True

        self.num_elabels = 131
        self.num_rlabels = 243


        self.cnn_layer1 = nn.Sequential(nn.Conv1d(in_channels=self.bert_dim,
                                                  out_channels=self.cnn_out_dim,
                                                  kernel_size=5,
                                                  stride=1,
                                                  padding=2),
                                        nn.ReLU()).to(device)
        self.cnn_layer2 = nn.Sequential(nn.Conv1d(in_channels=self.bert_dim,
                                                  out_channels=self.cnn_out_dim,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1),
                                        nn.ReLU()).to(device)

        self.batch_size = batch_size
        self.trigger_fc = nn.Linear(self.cnn_out_dim * 2, self.num_elabels).to(device)
        self.role_fc = nn.Linear(self.cnn_out_dim * 4, self.num_rlabels).to(device)
        if self.use_crf:
            self.trigger_crf = CRF(self.num_elabels, batch_first=True).to(device)
            self.role_crf = CRF(self.num_rlabels, batch_first=True).to(device)

    def forward(self, input_ids, trigger_labels, golden_triggers, argument_labels, lens, text_ids, mode='training'):
        batch_size = input_ids.shape[0]
        bert_embed = self.bert_embedding(input_ids)[0].float()  # [batch, seq, bert-dim]
        bert_embed = bert_embed[:, 1:-1, :].permute(0, 2, 1).contiguous()  # [batch, bert-dim, seq]

        cnn_out1 = self.cnn_layer1(bert_embed)
        cnn_out2 = self.cnn_layer2(bert_embed)
        cnn_out = torch.cat([cnn_out1, cnn_out2], dim=-2)
        cnn_out = cnn_out.permute(0, 2, 1).contiguous() # [batch, seq, cnn_dim * 2]
        ed_logits = self.trigger_fc(cnn_out)

        # decode predict triggers
        mask = sequence_mask(lens, self.device, 128)
        if mode == 'training':
            ed_loss = -self.trigger_crf(ed_logits, trigger_labels, mask, reduction='token_mean')
        else:
            ed_loss = None
        ed_tag_seq = self.trigger_crf.decode(ed_logits) # [batch, seq]

        if mode != 'training':
            triggers, counter = [], 0
            for a_tag_seq in ed_tag_seq:  # [seq]
                instance_triggers, starting = [], False
                for i, pred_id in enumerate(a_tag_seq):
                    if pred_id < 130:
                        if pred_id % 2 == 0:  # B-EventType
                            starting = True
                            start, end = i, i + 1
                            event_type = int(pred_id)
                            instance_triggers.append((start, end, event_type))
                        elif starting:
                            instance_triggers[-1] = (start, end+1, event_type)
                        else:
                            starting = False
                    else:
                        starting = False
                counter += len(instance_triggers)
                triggers.append(instance_triggers)
        else:
            triggers = golden_triggers

        assert len(triggers) == cnn_out.shape[0]
        assert len(triggers) == batch_size

        ae_hidden, ae_keys, ae_masks = [], [], []
        seq, dim = cnn_out.shape[1], cnn_out.shape[-1]
        for i, instance_triggers in enumerate(triggers):
            # instance_triggers[(start, end, type), ...]
            if instance_triggers:
                for (start, end, e_type) in instance_triggers:
                    instance_rep = cnn_out[i, :, :].unsqueeze(0)  # [1, seq, dim*2]
                    trigger_rep = instance_rep[:, start:end, :]  # [1, num, dim*2]
                    trigger_rep = torch.mean(trigger_rep, dim=1).expand(seq, dim)  # [seq, dim]
                    ae_instance_rep = torch.cat([instance_rep.squeeze(0), trigger_rep], dim=-1)  # [seq, dim*2*2]
                    ae_hidden.append(ae_instance_rep)
                    ae_masks.append(mask[i])
        if mode == 'training':
            assert len(ae_hidden) == len(argument_labels)
        else:
            assert len(ae_hidden) == counter

        if ae_hidden:
            ae_hidden = torch.stack(ae_hidden) # [n, seq, dim*2]
            ae_masks = torch.stack(ae_masks)
            ae_logits = self.role_fc(ae_hidden)


        if mode == 'training':
            ae_loss = - self.role_crf(ae_logits, argument_labels, ae_masks, reduction='token_mean')
        else:
            ae_loss = None
        ae_tag_seq = self.role_crf.decode(ae_logits) # [batch, seq]

        if mode == 'training':
            return ed_loss, ae_loss, ed_tag_seq, ae_tag_seq
        else:
            return triggers, ae_tag_seq










