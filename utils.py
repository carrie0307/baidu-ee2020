import json
import torch
from torch.nn import functional as F
from torch.autograd import Variable

def load_answers(filename):
    D = {}
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            arguments = []
            for event in l['event_list']:
                event_type = event['event_type']
                for argument in event['arguments']:
                    arguments.append((event_type + "-" + argument['role'], argument['argument']))
            D[l["id"]] = arguments
    return D

def evaluate(pred_file, gold_file):

    target = load_answers(gold_file)
    predict_res = load_answers(pred_file)
    text_ids = predict_res.keys()
    # print("t*********************ext_ids: ", text_ids)
    predict_num, correct_pred_num, real_num = 0.00001, 0, 0.00001

    for uid in text_ids:
        pred_res = predict_res[uid]  # [{}, {}, ...]
        gold_res = target[uid] #  [{}, {}, ]

        predict_num += len(pred_res)
        real_num += len(gold_res)
        for item in pred_res:
            if item in gold_res:
                correct_pred_num += 1

    p = correct_pred_num / predict_num
    r = correct_pred_num / real_num
    f1 = (2 * p * r) / (p + r + 0.00001)
    return p, r, f1

def sequence_mask(sequence_length, device, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    # seq_range = torch.range(0, max_len - 1).long()
    # 注意这里的修改
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand).to(device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def new_masked_cross_entropy(logits, target, length, num_labels, device, label_weights=None):
    # length = Variable(torch.LongTensor(length)).cuda()
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    # print("logits: ", logits.shape)
    max_batch_length = logits.shape[1]
    batch = logits.shape[0]

    mask = sequence_mask(sequence_length=length, device=device, max_len=max_batch_length)  # mask: (batch, max_len)
    # print("mask: ", mask.shape)
    target = target[:, :max_batch_length].contiguous()
    target = target.masked_select(mask.to(device)) # [N]

    mask = mask.view(batch, max_batch_length, -1).contiguous() # [batch, len, 1]
    logits = logits.masked_select(mask.to(device))
    logits = logits.view(-1, num_labels) # [N, 35]

    loss = F.cross_entropy(logits, target, weight=label_weights)
    return loss

def convert_ids2char(ids, id2char):
    chars = []
    for iid in ids:
        chars.append(id2char[iid])
    return chars

def decode(outputs):
    batch_predict_ids = []
    for instance in outputs:  # instance: [seq, dim]
        _, instance_label_ids = torch.max(instance, dim=-1)
        batch_predict_ids.append(instance_label_ids.cpu().numpy().tolist())
    return batch_predict_ids

def extract_pred_res(trigger_id2label, role_id2label, tokenizer, batch_token_ids, pred_triggers, pred_arguments, lens, text_ids):
    batch_size = len(lens)
    all_res = []
    counter = 0
    for b_index in range(batch_size):
        instance_res = {'id': text_ids[b_index], 'event_list': []}
        length = lens[b_index]
        instance_tokens_ids = batch_token_ids[b_index][1: length + 1] # remove csl and sep
        instance_tokens = tokenizer.convert_ids_to_tokens(instance_tokens_ids)

        for j, (trigger_start, trigger_end, event_id) in enumerate(pred_triggers[b_index]):
            # event_type
            if event_id != 130:
                event_type = trigger_id2label[str(event_id)]
                assert event_type.startswith('B-')
                event_type = event_type[2:]

                # if pred_arguments:
                pred_argument_ids = pred_arguments[counter + j][:length]  # [seq]
                assert len(pred_argument_ids) == len(instance_tokens)
                arguments, starting = [], False
                for k, pred_id in enumerate(pred_argument_ids):
                    if pred_id < 242:
                        if pred_id % 2 == 0:  # B-EventType
                            starting = True
                            role_type = role_id2label[str(pred_id)]
                            assert role_type.startswith('B-')
                            role_type = role_type[2:]
                            char = instance_tokens[k][2:] if instance_tokens[k].startswith("##") else instance_tokens[k]
                            arguments.append({'role': role_type, 'argument': char})
                        elif starting:
                            char = instance_tokens[k][2:] if instance_tokens[k].startswith("##") else instance_tokens[k]
                            arguments[-1]['argument'] = arguments[-1]['argument'] + char
                        else:
                            starting = False
                    else:
                        starting = False
                instance_res['event_list'].append({"event_type": event_type, "arguments": arguments})
        counter += len(pred_triggers[b_index])
        # print("instance_res: ", instance_res)
        all_res.append(instance_res)
    return all_res

def write_file(all_res, out_file):

    fw = open(out_file, 'w', encoding='utf-8')
    for instance_res in all_res:
        l = json.dumps(instance_res, ensure_ascii=False)
        fw.write(l + '\n')
    fw.close()

def load_data(filename, training=True):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            if training:
                arguments = []
                triggers = []
                for event in l['event_list']:
                    triggers.append((event['event_type'], event['trigger']))
                    temp = []
                    for argument in event['arguments']:
                        temp.append((argument['role'], argument['argument']))
                    arguments.append(temp)
                assert len(triggers) == len(arguments)
                D.append((l['text'], arguments, triggers, l['id']))
            else:
                D.append((l['text'], None, None, l['id']))
    D.sort(key=lambda i: len(i[0]), reverse=True)
    return D


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def tokenize_encode(text, char2id):
    li = []
    for char in text:
        li.append(char2id[char])
    return li

def data_generator(data, tokenizer, argument_label2id, trigger_label2id, batch_size, training=True):
    max_len = 128
    batch_token_ids, batch_attn_mask, batch_trigger_labels, batch_argument_labels, batch_triggers, lens, text_ids = [], [], [], [], [], [], []
    for (text, arguments, triggers, text_id) in data:
        text_ids.append(text_id)
        tokenize_text = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokenize_text)
        attention_mask = [1] * len(token_ids)
        if training:
            trigger_labels = [130] * len(token_ids)
            instance_argument_labels = []
            triggers_index = []

            i = 0
            while i < len(triggers):
                # trigger
                event_type, trigger = triggers[i]
                tokenize_trigger = tokenizer.tokenize(trigger)
                a_token_ids_trigger = tokenizer.convert_tokens_to_ids(tokenize_trigger)
                start_index = search(a_token_ids_trigger, token_ids)
                if start_index < 0 or start_index >= max_len:
                    triggers.remove((event_type, trigger))
                    continue
                triggers_index.append((start_index, start_index + len(a_token_ids_trigger), int( int(trigger_label2id['B-'+event_type]))))
                if start_index != -1:
                    trigger_labels[start_index] = int(trigger_label2id['B-'+event_type])
                    for j in range(1, len(a_token_ids_trigger)):
                        trigger_labels[start_index + j] = int(trigger_label2id['I-'+event_type])

                event_arguments = arguments[i]
                argument_labels = [242] * len(token_ids)
                for (role_type, argument) in event_arguments:
                    tokenize_argument = tokenizer.tokenize(argument)
                    a_token_ids_argument = tokenizer.convert_tokens_to_ids(tokenize_argument)
                    start_index_argument = search(a_token_ids_argument, token_ids)
                    if start_index_argument != -1:
                        argument_labels[start_index_argument] = int(argument_label2id['B-' + role_type])
                        for k in range(1, len(a_token_ids_argument)):
                            argument_labels[start_index_argument + k] = int(argument_label2id['I-' + role_type])
                i += 1

                instance_argument_labels.append(argument_labels)
            # print("!!:",  instance_argument_labels)
            # print(text_id, triggers, len(triggers), len(instance_argument_labels))
            assert len(instance_argument_labels) == len(triggers)
            assert len(triggers) == len(triggers_index)

        length = len(token_ids)
        if length < max_len:
            lens.append(length)
            padding_len = max_len - length
            token_ids.extend([0] * padding_len)
            attention_mask.extend([0] * padding_len)  # [0] for [PAD] in bert
            if training:
                trigger_labels.extend([0] * padding_len)
                for k, argument_labels in enumerate(instance_argument_labels):
                    instance_argument_labels[k].extend([0] * padding_len)
        else:
            lens.append(max_len)
            token_ids = token_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            if training:
                trigger_labels = trigger_labels[:max_len]
                for k, argument in enumerate(instance_argument_labels):
                    instance_argument_labels[k] = instance_argument_labels[k][:max_len]

        batch_token_ids.append([101] + token_ids + [102])
        batch_attn_mask.append(attention_mask)
        if training:
            batch_trigger_labels.append(trigger_labels)
            batch_argument_labels.append(instance_argument_labels)
            batch_triggers.append(triggers_index)
        if len(batch_token_ids) == batch_size:
            batch_arguments = []
            if training:
                for instance_argument_labels in batch_argument_labels:
                    batch_arguments.extend(instance_argument_labels)
            yield [batch_token_ids, batch_attn_mask, batch_trigger_labels, batch_arguments, batch_triggers, lens, text_ids]
            batch_token_ids, batch_attn_mask, batch_trigger_labels, batch_argument_labels, batch_triggers, lens, text_ids = [], [], [], [], [], [], []
    if batch_token_ids:
        batch_arguments = []
        for instance_argument_labels in batch_argument_labels:
            batch_arguments.extend(instance_argument_labels)
        yield [batch_token_ids, batch_attn_mask, batch_trigger_labels, batch_arguments, batch_triggers, lens, text_ids]


# p,r,f = evaluate('./devres/1_dev_bertcnn.json', './data/dev.json')
# print(p,r,f)

if __name__ == '__main__':

    from transformers import *

    role_label2id, role_id2label = {}, {}
    with open('./dict/vocab_roles_label_map.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label, iid = line.split()
            role_label2id[label] = iid
            role_id2label[iid] = label
    print(role_label2id, role_id2label)

    trigger_label2id, trigger_id2label = {}, {}
    with open('./dict/vocab_trigger_label_map.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label, iid = line.split()
            trigger_label2id[label] = iid
            trigger_id2label[iid] = label
    print(trigger_label2id, trigger_id2label)

    D = load_data('./data/dev.json')
    tokenizer = BertTokenizer.from_pretrained('../roberta-large/vocab.txt',do_lower_case=False)
    for batch_token_ids, batch_attn_mask, batch_trigger_labels, batch_argument_labels, batch_triggers, lens, text_ids in data_generator(D, tokenizer, role_label2id, trigger_label2id, 2, training=True):
        print("token_ids: ", batch_token_ids)
        print("trigger_labels: ", batch_trigger_labels)
        print("argument_labels: ", batch_argument_labels)
        print("batch_triggers: ", batch_triggers)