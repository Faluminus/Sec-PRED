#!/usr/bin/env python3

#
# Arontier Inc.: Artificial Intelligence in Precision Medicine
# Copyright: 2018-present
#

import os
import torch
from torch import nn
import torch.nn.functional as F
import esm
import json
import numpy as np
import argparse
from einops import rearrange
import string


PROTEIN_PROPERTY = "secondary_structure"
LABEL_EXT = ".label.ss"

# SS_IND2CHAR = ['H', 'G', 'I', 'E', 'B', 'T', 'S', 'L']
# SS_CHAR2IND = {c: i for i, c in enumerate(SS_IND2CHAR)}

SS_IND2CHAR = ['H', 'G', 'I', 'E', 'B', 'T', 'S', 'C', 'X'] ### X-> Unknown, 8
SS_CHAR2IND = {c: i for i, c in enumerate(SS_IND2CHAR)}

SS_Q8_to_Q3 = {'H': 'H', 'G': 'H', 'I': 'H',
                 'E': 'E', 'B': 'E',
                 'T': 'C', 'S': 'C', 'C': 'C'}

MAX_MSA_ROW_NUM = 256  # 256
MAX_MSA_COL_NUM = 1023  # start token +1 1024
torch.set_grad_enabled(False)

class lstm_net(nn.Module):
    def __init__(self, input_feature_size=768, hidden_node=256, dropout=0.25, need_row_attention=False, class_num=8):
        super().__init__()
        self.need_row_attention = need_row_attention
        self.linear_proj = nn.Sequential(
            nn.Linear(input_feature_size, input_feature_size // 2),
            nn.InstanceNorm1d(input_feature_size // 2),
            nn.ReLU(),
            nn.Linear(input_feature_size // 2, input_feature_size // 4),
            nn.InstanceNorm1d(input_feature_size // 4),
            nn.ReLU(),
            nn.Linear(input_feature_size // 4, input_feature_size // 4),
        )

        if self.need_row_attention:
            lstm_input_feature_size = input_feature_size // 4 + 144*2
        else:
            lstm_input_feature_size = input_feature_size // 4

        self.lstm = nn.LSTM(
            input_size=lstm_input_feature_size,
            hidden_size=hidden_node,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )

        self.to_property = nn.Sequential(
            nn.Linear(hidden_node * 2, hidden_node * 2),
            nn.InstanceNorm1d(hidden_node * 2),
            nn.ReLU(),
            nn.Linear(hidden_node * 2, class_num),
        )

    def forward(self, msa_query_embeddings, msa_row_attentions):
        msa_query_embeddings = self.linear_proj(msa_query_embeddings)

        if self.need_row_attention:
            msa_row_attentions = rearrange(msa_row_attentions, 'b l h i j -> b (l h) i j')
            msa_attention_features = torch.cat((torch.mean(msa_row_attentions, dim=2), torch.mean(msa_row_attentions, dim=3)), dim=1)
            # msa_attention_features = (torch.mean(msa_row_attentions, dim=2) + torch.mean(msa_row_attentions, dim=3))/2
            msa_attention_features = msa_attention_features.permute((0, 2, 1))

            lstm_input = torch.cat([msa_query_embeddings, msa_attention_features], dim=2)

        else:
            lstm_input = msa_query_embeddings

        lstm_input = lstm_input.permute((1, 0, 2))
        lstm_output, lstm_hidden = self.lstm(lstm_input)
        lstm_output = lstm_output.permute((1, 0, 2))
        label_output = self.to_property(lstm_output)

        return label_output



def read_msa_json(msa_json_path, msa_method, msa_row_num):

    with open(msa_json_path) as json_file:
        msa_coord_json_dict = json.load(json_file)

    if not msa_method:
        msa_method = list(msa_coord_json_dict['MSA'].keys())[0]

    msa_seq = msa_coord_json_dict['MSA'][msa_method]['sequences']
    msa_seq = [seq[0:2] for seq in msa_seq]
    query_seq = msa_seq[0][1]

    if msa_row_num > MAX_MSA_ROW_NUM:
        msa_row_num = MAX_MSA_ROW_NUM
        print(f"The MSA row num is larger than {MAX_MSA_ROW_NUM}. This program force the msa row to under {MAX_MSA_ROW_NUM}")

    msa_seq = msa_seq[: msa_row_num]

    return msa_seq, query_seq

def read_msa_file(filepath, msa_row_num):

    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    with open(filepath,"r") as f:
        lines = f.readlines()
    # read file line by line
    for i in range(0,len(lines),2):

        seq = []
        seq.append(lines[i])
        seq.append(lines[i+1].rstrip().translate(table))
        seqs.append(seq)

    if msa_row_num > MAX_MSA_ROW_NUM:
        msa_row_num = MAX_MSA_ROW_NUM
        print(f"The MSA row num is larger than {MAX_MSA_ROW_NUM}. This program force the msa row to under {MAX_MSA_ROW_NUM}")

    seqs = seqs[: msa_row_num]
    return seqs, seqs[0]



def extract_msa_transformer_features(msa_seq, msa_transformer, msa_batch_converter, device=torch.device("cpu")):
    msa_seq_label, msa_seq_str, msa_seq_token = msa_batch_converter([msa_seq])
    msa_seq_token = msa_seq_token.to(device)
    msa_row, msa_col = msa_seq_token.shape[1], msa_seq_token.shape[2]
    print(f"{msa_seq_label[0][0]}, msa_row: {msa_row}, msa_col: {msa_col}")

    if msa_col > MAX_MSA_COL_NUM:
        print(f"msa col num should less than {MAX_MSA_COL_NUM}. This program force the msa col to under {MAX_MSA_COL_NUM}")
    msa_seq_token = msa_seq_token[:, :, :MAX_MSA_COL_NUM]

    ### keys: ['logits', 'representations', 'col_attentions', 'row_attentions', 'contacts']
    msa_transformer_outputs = msa_transformer(
        msa_seq_token, repr_layers=[12],
        need_head_weights=True, return_contacts=True)
    msa_row_attentions = msa_transformer_outputs['row_attentions']
    msa_representations = msa_transformer_outputs['representations'][12]
    msa_query_representation = msa_representations[:, 0, 1:, :]  # remove start token
    msa_row_attentions = msa_row_attentions[..., 1:, 1:]  # remove start token

    return msa_query_representation, msa_row_attentions

def str_find_ch(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]



def save_ss_to_json(out_ss_json, output_ss_np_argmax, query_seq):
    output_ss_np_argmax_list = output_ss_np_argmax.tolist()
    output_ss_char_list = [SS_IND2CHAR[ind] for ind in output_ss_np_argmax_list]
    output_ss_char_list_q3 = [SS_Q8_to_Q3[ind] for ind in output_ss_char_list]

    json_dict = {
        "secondary_structure_data_q8": output_ss_char_list,
        "secondary_structure_data_q3": output_ss_char_list_q3,

        "query_seq": query_seq,
        "metadata": {
            "precision": 0,
            "title": "secondary structure prediction",
            "data-min": 0,
            "data-max": 7,
            "status-8": SS_IND2CHAR,
            "status-8to3": SS_Q8_to_Q3,
        }
    }

    with open(out_ss_json, "w") as f:
        # json.dump(json_dict, f)
        json.dump(json_dict, f, indent=4)
    
def get_msa_transformer_prediction(aminoAcid):
    #input_path = 'examples/s_pred_ss.a3m'
    #output_path = 's_pred_ss.out'
    conv_model_path = 's_pred_ss_weights.pth'
    #msa_method = 'none'
    msa_row_num = 256
    device = 'gpu'

    #parser = argparse.ArgumentParser(description='S-Pred for secondary structrue prediction: input msa and output ss8 and ss3 (.json)')
    #parser.add_argument('-i', '--input_path', type=str, default='examples/s_pred_ss.a3m',
                        #help='input msa path (.json or .a3m)')
    #parser.add_argument('-o', '--output_path', type=str, default='s_pred_ss.out',
                        #help='output predicted ss8 ss3 secondary structure')
    #parser.add_argument('--conv_model_path', type=str,
                        #default='s_pred_ss_weights.pth',
                        #help='model weight path')
    #msa_args = parser.add_argument_group('MSA')


    #msa_args.add_argument('--msa_method', type=str, help='input msa method')
    #msa_args.add_argument('--msa_row_num', type=int, default=256,
                          #help='input msa row num to msa transformer')
    #parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                       # help='choose device: cpu or gpu')


    #args = parser.parse_args()


    print("===================================")
    print("Print Arguments:")
    print("===================================")

    #print(' '.join(f'{k} = {v}\n' for k, v in vars(args).items()))


    if device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("gpu is not available, run on cpu")
            device = torch.device("cpu")

    #if already have the msa_transformer_weight
    #msa_transformer, msa_alphabet = esm.pretrained.load_model_and_alphabet_local("s_pred_ss_weights.pth")

    msa_transformer, msa_alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
    msa_batch_converter = msa_alphabet.get_batch_converter()

    msa_transformer.to(device)
    msa_transformer.eval()

    conv_model = lstm_net(input_feature_size=768, hidden_node=256, dropout=0.25, need_row_attention=True, class_num=8)
    conv_model = conv_model.to(device)
    
    if device.type == 'cpu':
        ch = torch.load(conv_model_path, map_location=torch.device('cpu'))
    else:
        ch = torch.load(conv_model_path)

    conv_model.load_state_dict(ch['conv_model'])
    conv_model.to(device)
    conv_model.eval()
    
    for param in msa_transformer.parameters():
        param.requires_grad = False
    for param in conv_model.parameters():
        param.requires_grad = False

    print("===================================")
    print("Extract msa transformer features")
    print("===================================")

    #if input_path.endswith('.json'):
    #   msa_seq, query_seq = read_msa_json(input_path, msa_method, msa_row_num)
    #else:
    #msa_seq, query_seq = read_msa_file(input_path, msa_row_num)
    msa_seq = ''
    query_seq = aminoAcid
    
    msa_row_num = len(msa_seq)
    msa_col_num = len(query_seq)

    print(f"msa row number: {msa_row_num}")
    print(f"msa column number: {msa_col_num}")


    msa_query_representation, msa_row_attentions = extract_msa_transformer_features(msa_seq,
                                                                                    msa_transformer,
                                                                                    msa_batch_converter,
                                                                                    device=device)
    
    msa_query_representation.to(device)
    msa_row_attentions.to(device)

    output_property = conv_model(msa_query_representation, msa_row_attentions)
    output_property = output_property.permute((0, 2, 1))
    
    output_property_softmax = F.softmax(output_property, dim=1)
    output_property_softmax_np = output_property_softmax.data.cpu().numpy().squeeze()
    output_property_softmax_np_argmax = np.argmax(output_property_softmax_np, axis=0)
    
    return output_property_softmax_np_argmax
    
    #out_ss_json_path = output_path + '.ss.json'

    #save_ss_to_json(out_ss_json_path, output_property_softmax_np_argmax, query_seq)

    print("===================================")
    print("Done")
    print("===================================")