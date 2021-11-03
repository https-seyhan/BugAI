#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
# Bug prediction of Software codes using Advanced Deep Learning XLNet model.
# The aim is to build a AI model of software bug prediction using XLNet.
import os
import pandas as pd
import math
import numpy as np
#pytorch
import torch
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm,trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#Pytorch XLNet module
from pytorch_transformers import (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)

os.chdir('/home/saul/bugpred')
data_file_address = "bug-metrics.csv"
#df_data = pd.read_csv(data_file_address,sep=",",encoding="utf-8",names=['labels','texts'])
df_data = pd.read_csv(data_file_address,sep=",",encoding="utf-8")

df_data.criticalBugs.head(n=2)
df_data.criticalBugs.unique()

# Analyse the labels distribution
df_data.criticalBugs.value_counts()

#Parser data into document structure
# Get sentence data

sentences = df_data.classname.str.replace('::', ' ').to_list()
sentences[0]

# Get tag labels data and convert to list
labels = df_data.criticalBugs.to_list()
os.chdir('/home/saul/deeplearning/xlnet')

#Make TAG name into index for training
# Set a dict for mapping id to tag name
#tag2idx = {t: i for i, t in enumerate(tags_vals)}

# Recommend to set it by manual define, good for reusing
# 0:no bug, 1: bug Binary classification
tag2idx={'0': 0,
 '1': 1}
tag2idx
# Mapping index to name
tag2name={tag2idx[key] : key for key in tag2idx.keys()}

#Make tranning data
#Make raw data into trainable data for XLNet, including:
#    Set gpu environment
#    Load tokenizer and tokenize
#    Set 3 embedding, token embedding, mask word embedding, segmentation embedding
#    Split data set into train and validate, then send them to dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

#Load tokenizer
# Manual define vocabulary address, if you download the model in local
# The vocabulary can download from "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-spiece.model"
vocabulary = 'xlnet-base-cased-spiece.model'

# Len of the sentence must be the same as the training model
# See model's 'max_position_embeddings' = 512
#max_len  = 64
# With cased model, set do_lower_case = False
#use pre-trained weight vectors

tokenizer = XLNetTokenizer(vocab_file=vocabulary,do_lower_case=False)

#Set text input embedding
#    token id embedding
#    mask embedding
#    segment embedding

#The Embedding process was referred to XLNet official repo
#This process is very different from BERT

#max_len  = 64
max_len  = 16
# Set batch num
#batch_num = 32
batch_num = 16

full_input_ids = []
full_input_masks = []
full_segment_ids = []

SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

UNK_ID = tokenizer.encode("<unk>")[0]
CLS_ID = tokenizer.encode("<cls>")[0]
SEP_ID = tokenizer.encode("<sep>")[0]
MASK_ID = tokenizer.encode("<mask>")[0]
EOD_ID = tokenizer.encode("<eod>")[0]

for i,sentence in enumerate(sentences):
    # Tokenize sentence to token id list
    tokens_a = tokenizer.encode(sentence)
    # Trim the len of text
    if(len(tokens_a)>max_len-2):
        tokens_a = tokens_a[:max_len-2]
     
    tokens = []
    segment_ids = []
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
  
    # Add <sep> token
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)
    # Add <cls> token
    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)
    input_ids = tokens
    # The mask has 0 for real tokens and 1 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [0] * len(input_ids)
    # Zero-pad up to the sequence length at forn

    if len(input_ids) < max_len:
        delta_len = max_len - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids
     
    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(segment_ids) == max_len
    full_input_ids.append(input_ids)
    full_input_masks.append(input_mask)
    full_segment_ids.append(segment_ids)
  
    if 3 > i:
        print("No.:%d"%(i))
        print("sentence: %s"%(sentence))
        print("input_ids:%s"%(input_ids))
        print("attention_masks:%s"%(input_mask))
        print("segment_ids:%s"%(segment_ids))
        print("\n")
      
#Set label embedding
# Make label into id
tags = [tag2idx[str(lab)] for lab in labels]
print(tags[0])

#Split data into train and validate
#70% for training, 30% for validation
#Split all data
tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs = train_test_split(full_input_ids, tags,full_input_masks,full_segment_ids, 
                                                            random_state=4, test_size=0.4)
len(tr_inputs),len(val_inputs),len(tr_segs),len(val_segs)
#Set data into tensor

#Not recommend tensor.to(device) at this process, since it will run out of GPU memory
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
tr_segs = torch.tensor(tr_segs)
val_segs = torch.tensor(val_segs)

#Copy data into data loader
# Set token embedding, attention embedding, segment embedding

train_data = TensorDataset(tr_inputs, tr_masks,tr_segs, tr_tags)
print(train_data.tensors)
train_sampler = RandomSampler(train_data)
# Drop last can make batch training better for the last one
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)
valid_data = TensorDataset(val_inputs, val_masks,val_segs, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)

#Train model
#Load XLNet model
# In this document, contain confg(txt) and weight(bin) files
#model_file_address = 'xlnet-base-cased'
model_file_address = '/home/saul/deeplearning/xlnet'
# Will load config and weight with from_pretrained()
# Recommand download the model before using
# Download model from "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin"
# Download model from "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json" 
model = XLNetForSequenceClassification.from_pretrained(model_file_address,num_labels=len(tag2idx))

# Set model to GPU,if you are using GPU machine
model.to(device)

# Add multi GPU support
if n_gpu >1:
    model = torch.nn.DataParallel(model)

# Set epoch and grad max num
epochs = 10
#epochs = 3
max_grad_norm = 1.0

# Calcuate train optimiazaion num
num_train_optimization_steps = int( math.ceil(len(tr_inputs) / batch_num) / 1) * epochs

#Set fine tuning method
#Manual optimizer
# True: fine tuning all the layers 
# False: only fine tuning the classifier layers
# Since XLNet in 'pytorch_transformer' did not contian classifier layers
# FULL_FINETUNING = True need to set True

FULL_FINETUNING = True
if FULL_FINETUNING:
    # Fine tune model all layer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    # Only fine tune classifier parameters
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
#Fine-tuing model
# TRAIN loop
model.train()

print("***** Running training *****")
print("  Num examples = %d"%(len(tr_inputs)))
print("  Batch size = %d"%(batch_num))
print("  Num steps = %d"%(num_train_optimization_steps))

for _ in trange(epochs,desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_segs,b_labels = batch
        # forward pass
        outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
        loss, logits = outputs[:2]

        if n_gpu>1:
            # When multi gpu, average it
            loss = loss.mean()
       
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        optimizer.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
#Save model
xlnet_out_address = '/home/saul/xlnet/'

# Make dir if not exits
if not os.path.exists(xlnet_out_address):
        os.makedirs(xlnet_out_address)
 
# Save a trained model, configuration and tokenizer
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(xlnet_out_address, "pytorch_model.bin")
output_config_file = os.path.join(xlnet_out_address, "config.json")
# Save model into file
torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(xlnet_out_address)

#Load model
model = XLNetForSequenceClassification.from_pretrained(xlnet_out_address,num_labels=len(tag2idx))
# Set model to GPU
model.to(device)

if n_gpu >1:
    model = torch.nn.DataParallel(model)
#Eval model
# Evalue loop
model.eval()

# Set acc funtion
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
y_true = []
y_predict = []

print("***** Running evaluation *****")
print("  Num examples ={}".format(len(val_inputs)))
print("  Batch size = {}".format(batch_num))

for step, batch in enumerate(valid_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_segs,b_labels = batch
    with torch.no_grad():
        outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
        tmp_eval_loss, logits = outputs[:2]
    # Get textclassification predict result
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    tmp_eval_accuracy = accuracy(logits, label_ids)
#     print(tmp_eval_accuracy)
#     print(np.argmax(logits, axis=1))
#     print(label_ids)
#     Save predict and real label results for analyze

    for predict in np.argmax(logits, axis=1):
        y_predict.append(predict)
    for real_result in label_ids.tolist():
        y_true.append(real_result)
    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1
eval_loss = eval_loss / nb_eval_steps
eval_accuracy = eval_accuracy / len(val_inputs)
loss = tr_loss/nb_tr_steps 
result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'loss': loss}
report = classification_report(y_pred=np.array(y_predict),y_true=np.array(y_true))
# Save the report into file
output_eval_file = os.path.join(xlnet_out_address, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
        print("  %s = %s"%(key, str(result[key])))
        writer.write("%s = %s\n" % (key, str(result[key])))     
    print(report)
    writer.write("\n\n")  
    writer.write(report)
