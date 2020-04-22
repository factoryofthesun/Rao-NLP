'''
Helper functions to prepare abstract data for inference
'''

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def preProcess(max_len, tokenizer, abstracts):
    abstracts = [ab for ab in abstracts]

    #Tokenize
    abs_tokens = [tokenizer.tokenize(ab) + ["<sep>", "<cls>"] for ab in abstracts]

    #Encode tokens
    encoded_ids = [tokenizer.convert_tokens_to_ids(x) for x in abs_tokens]
    MAX_LEN = max([len(x) for x in encoded_ids]) #Check max tokenized length
    print("Maximum length tokens vector found: {}".format(MAX_LEN))

    #If longer than MAX_LEN ids, take first and last MAX_LEN//2
    print("Middle-out truncating tokens to length {}".format(max_len))
    truncated_ids = [x[:max_len//2] + x[-max_len//2:] if len(x) > max_len else x for x in encoded_ids]

    for x in truncated_ids:
        if len(x) > max_len:
            print("Found inconsistent length {}".format(len(x)))
            print(x)

    #Need standardized input lengths - pad to maximum length encoded vector
    input_ids = pad_sequences(truncated_ids, maxlen = max_len, dtype = "long", truncating ="pre", padding = "post", value = 0)

    #Create attention masks
    attention_masks = []
    for seq in input_ids:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)

    input_tensor = torch.tensor(input_ids)
    mask_tensor = torch.tensor(attention_masks)

    return input_tensor, mask_tensor

# Inputs is a list of tensors to chunk
# Returns a dataloader with the batched inputs
def createDataLoader(batch_size, *inputs):
    ts_data = TensorDataset(*inputs)
    dataloader = DataLoader(ts_data, batch_size=batch_size)

    return dataloader
