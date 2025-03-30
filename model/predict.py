import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from transformers import AutoTokenizer
from sklearn.metrics import matthews_corrcoef


input_data = "VMKANVTKKTLNEGLGLLERVIPSRSSNPLLTALKVETSEGGLTLSGTNLEIDLSCFVPAEVQQPENFVVPAHLFAQIVRNLGGELVELELSGQELSVRSGGSDFKLQTGDIEAYPPLSFPAQADVSLDGGELSRAFSSVRYAASNEAFQAVFRGIKLEHHGESARVVASDGYRVAIRDFPASGDGKNLIIPARSVDELIRVLKDGEARFTYGDGMLTVTTDRVKMNLKLLDGDFPDYERVIPKDIKLQVTLPATALKEAVNRVAVLADKNANNRVEFLVSEGTLRLAAEGDYGRAQDTLSVTQGGTEQAMSLAFNARHVLDALGPIDGDAELLFSGSTSPAIFRAVGGGGGYMAVMVTLR"
final_data = "CEEEEEEHHHHHHHHHHHHHHSCSCCSSTTTTEEEEEECSSEEEEEEECSSEEEEEEEECEEESCCCEEEEHHHHHHHHHHCCSSEEEEEEETTEEEEEETTEEEEEECCCGGGSPPPCCCCCCCEEEEHHHHHHHHHHHGGGCCTTCSSGGGGEEEEEEETTEEEEEEESSSSEEEEEEECBCCCCCEEEEHHHHHHHHHHCCSSEEEEEECSSEEEEECSSEEEEEECCCSPPPCGGGGSCCCCCEEEEEEHHHHHHHHHHHHTTSCTTTTTEEEEEEETTEEEEEEECSSEEEEEEEECEEEESCSEEEEEEEHHHHHHHHTTCCSEEEEEESCTTSCEEEEEGGGGGGEEEEEPPPC"
iidax_data = ""
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


src_data_tokenized = tokenizer(input_data, padding=True, truncation=True, max_length=1000, return_tensors="tf")
tgt_data_tokenized = tokenizer(final_data, padding=True, truncation=True, max_length=1000, return_tensors="tf")


src_input_ids = src_data_tokenized['input_ids']
tgt_input_ids = tgt_data_tokenized['input_ids']


src_input_ids = tf.where(src_input_ids == 1, 0, src_input_ids)
tgt_input_ids = tf.where(tgt_input_ids == 1, 0, tgt_input_ids)

src_input_ids_np = src_input_ids.numpy()
tgt_input_ids_np = tgt_input_ids.numpy()


model = tf.keras.models.load_model('./lstmSecPRED-PREDICTION.keras')

data = model.predict(src_input_ids_np)
arged = tf.argmax(data, axis=-1)
decoded = tokenizer.decode(arged[0])
final = list()
for e in decoded:
    if e != ' ':
        final.append(e)



matthews_corrcoef(final_data.split(), final)
