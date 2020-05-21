from tqdm import tqdm
import argparse, os, random
import numpy as np
from transformers import *
from utils import *
from cnn import ConvNet as Model
from transformers import AdamW, get_linear_schedule_with_warmup
import pickle

def train_epoch(train_data, epoch, model, scheduler, optimizer, batch_size, device):

    print("trainging ...")
    model.train()
    total_ed_loss, total_ae_loss, total_loss = 0, 0, 0
    with tqdm(total=11958) as pbar:
        for batch_token_ids, batch_attn_mask, batch_trigger_labels, batch_argument_labels, batch_triggers, lens, text_ids in data_generator(
                train_data, tokenizer, role_label2id, trigger_label2id, batch_size, training=True):

            assert len(batch_token_ids) == len(batch_triggers)
            optimizer.zero_grad()
            input_ids = torch.LongTensor(batch_token_ids).to(device)
            attn_mask = torch.FloatTensor(batch_attn_mask).to(device)
            trigger_labels = torch.LongTensor(batch_trigger_labels).to(device)
            argument_labels = torch.LongTensor(batch_argument_labels).to(device)
            ed_loss, ae_loss, ed_tag_seq, ae_tag_seq = model(input_ids, trigger_labels, batch_triggers, argument_labels,
                                                             torch.LongTensor(lens).to(device), text_ids, mode='training')
            loss = ed_loss + ae_loss

            back_loss = ed_loss + args.aealpha * ae_loss
            back_loss.backward()
            requires_grad_params = list(filter(lambda p: p.requires_grad, model.parameters()))
            torch.nn.utils.clip_grad_norm_(requires_grad_params, args.maxnorm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_ed_loss += ed_loss.item()
            total_ae_loss += ae_loss.item()
            pbar.update(batch_size)

    total_loss = total_loss / (11958 / batch_size)
    ed_loss = total_ed_loss / (11958 / batch_size)
    ae_loss = total_ae_loss / (11958 / batch_size)
    print("epoch: {}, total_loss: {}, ed_loss: {}, ae_loss: {}".format(epoch, total_loss, ed_loss, ae_loss))





def dev_epoch(valid_data, model, epoch, batch_size, device):

    model.eval()
    all_res = []
    with tqdm(total=1498) as pbar:
        for batch_token_ids, batch_attn_mask, batch_trigger_labels, batch_argument_labels, batch_triggers, lens, text_ids in data_generator(
                valid_data, tokenizer, role_label2id, trigger_label2id, batch_size, training=True):

            input_ids = torch.LongTensor(batch_token_ids).to(device)  # Batch size 1
            attn_mask = torch.FloatTensor(batch_attn_mask).to(device)
            trigger_labels = torch.LongTensor(batch_trigger_labels).to(device)
            argument_labels = torch.LongTensor(batch_argument_labels).to(device)
            pred_triggers, pred_arguments = model(input_ids, None, None, None,
                                                 torch.LongTensor(lens).to(device), text_ids, mode='dev')

            assert len(pred_triggers) == len(text_ids)
            batch_res = extract_pred_res(trigger_id2label, role_id2label, tokenizer, batch_token_ids, pred_triggers, pred_arguments, lens, text_ids)
            all_res.extend(batch_res)

            pbar.update(batch_size)

        write_file(all_res, "./devres/" + args.log_name + "_" + str(epoch) + "_dev_bertcnn.json")
        p, r, f1= evaluate("./devres/"  + args.log_name + "_" + str(epoch) + "_dev_bertcnn.json", '/home/cuishiyao/baidu/data/dev.json')
        print("epoch: {}, p:{}, r:{}, f1: {}".format(epoch, p, r, f1))
        return f1


def test_epoch(test_data, model, batch_size, device):

    model.eval()
    all_res = []
    with tqdm(total=1496) as pbar:
        for batch_token_ids, batch_attn_mask, batch_trigger_labels, batch_argument_labels, batch_triggers, lens, text_ids in data_generator(
                    test_data, tokenizer, role_label2id, trigger_label2id, batch_size, training=False):

            input_ids = torch.LongTensor(batch_token_ids).to(device)  # Batch size 1
            attn_mask = torch.FloatTensor(batch_attn_mask).to(device)
            trigger_labels = torch.LongTensor(batch_trigger_labels).to(device)
            argument_labels = torch.LongTensor(batch_argument_labels).to(device)
            pred_triggers, pred_arguments = model(input_ids, None, None, None,
                                                  torch.LongTensor(lens).to(device), text_ids, mode='test')

            assert len(pred_triggers) == len(text_ids)
            batch_res = extract_pred_res(trigger_id2label, role_id2label, tokenizer, batch_token_ids, pred_triggers,
                                         pred_arguments, lens, text_ids)
            all_res.extend(batch_res)
            pbar.update(batch_size)

    write_file(all_res, "./res/" + log_name + "_test_bertcnn.json")



parser = argparse.ArgumentParser(description='baidu competition')
parser.add_argument('--seed', default=1023, type=int)
parser.add_argument('--log_name', default='test', type=str)
parser.add_argument('--crf', default=1, type=int)
parser.add_argument('--epoch', default=5, type=int)
parser.add_argument('--bertlr', default=2e-5,type=float)
parser.add_argument('--crflr', default=2e-3,type=float)
parser.add_argument('--cnnlr', default=2e-3,type=float)
parser.add_argument('--cuda', default=1,type=int)
parser.add_argument('--batch_size', default=30, type=int)
parser.add_argument('--maxnorm', default=3, type=float)
parser.add_argument('--aealpha', default=1.5, type=float)
parser.add_argument('--warm_rate', default=0.1, type=float)
args = parser.parse_args()
log_name = args.log_name

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 读取数据
train_data = load_data('/home/cuishiyao/baidu/data/train.json', training=True)
valid_data = load_data('/home/cuishiyao/baidu/data/dev.json', training=True)
test_data = load_data('/home/cuishiyao/baidu/data/test1.json', training=False)

role_label2id, role_id2label = {}, {}
with open('./dict/vocab_roles_label_map.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        label, iid = line.split()
        role_label2id[label] = iid
        role_id2label[iid] = label
# print(role_label2id, role_id2label)

trigger_label2id, trigger_id2label = {}, {}
with open('./dict/vocab_trigger_label_map.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        label, iid = line.split()
        trigger_label2id[label] = iid
        trigger_id2label[iid] = label
# print(trigger_label2id, trigger_id2label)

batch_size = args.batch_size
device = torch.device(args.cuda)
tokenizer = BertTokenizer.from_pretrained('../roberta/vocab.txt',do_lower_case=False)
# bert_embeddings = np.load("bert_embeddings.npy")
model = Model(1, batch_size, device)

parameters_to_optimize = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
parameters_to_optimize = [
    {'params': [p for n, p in parameters_to_optimize
        if not any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.01},
    {'params': [p for n, p in parameters_to_optimize
        if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0},
    {'params': [p for n,p in parameters_to_optimize if 'bert' not in n and 'crf' not in n],'lr':args.cnnlr},
    {'params': [p for n,p in parameters_to_optimize if 'bert' not in n and 'crf' in n],'lr':args.crflr}
    ]
optimizer = AdamW(parameters_to_optimize, lr=args.bertlr, correct_bias=False)
total_steps = int(11958 // args.batch_size) * args.epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_rate * total_steps), num_training_steps=total_steps)
max_f1 = 0
for epoch in range(5):
    train_epoch(train_data, epoch+1, model, scheduler, optimizer, batch_size, device)
    f1 = dev_epoch(valid_data, model, epoch+1, batch_size, device)
    if f1 > max_f1:
       torch.save(model.state_dict(), "./models/{}.pkl".format(log_name))

e_model = Model(1, batch_size, device)
e_model.load_state_dict(torch.load("./models/" + log_name + ".pkl"))
print("model loaded ...")
test_epoch(test_data, e_model, batch_size, device)













