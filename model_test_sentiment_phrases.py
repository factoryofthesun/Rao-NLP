import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import XLNetForSequenceClassification
from transformers import XLNetTokenizer
from nlp_data_prep import preProcess, createDataLoader
import time
import datetime

# Preprocessing parameters
max_len = 128

# Use the currently best performing model to tag abstract senetiment
models_dir = "Models/"
best_model_name = "RMSprop_dp0.3_sdp0.1_bsize128.pt"
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",
                                                      num_labels = 2,
                                                      dropout = 0.3,
                                                      summary_last_dropout = 0.1)
device = torch.device('cpu')
model.load_state_dict(torch.load(models_dir + best_model_name, map_location=device))
model.eval()

# Load tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

# Read in lists of positive words
data_dir = "opinion-lexicon-English/positive-words.txt"
with open(data_dir, 'r') as f:
    pos_words = f.read().splitlines()

# Preprocess
encodings_tensor, attention_masks_tensor = preProcess(max_len, tokenizer, pos_words)

# ==== Conduct Inference ====
# Time inference
t0 = time.time()

# Predict in batches
dataloader = createDataLoader(100, encodings_tensor, attention_masks_tensor)

probs_nonpos_full = []
probs_pos_full = []
preds_full = []
for step, batch in enumerate(dataloader):
    print("Predicting on batch {}...".format(step))
    batch = tuple(t.to(device) for t in batch)
    b_inputs, b_masks = batch

    with torch.no_grad():
        outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
        logits = outputs[0]
        probs = F.softmax(logits, dim=1)
        probs = probs.numpy()
        preds = np.argmax(probs, axis =1).flatten()
        try:
            probs_nonpos_full.extend(probs[:,0])
            probs_pos_full.extend(probs[:,1])
            preds_full.extend(preds)
        except:
            print("BROKE")
            print("Probs:", probs)
            print("Preds:",preds)

print("Total inference took " + str(datetime.timedelta(seconds=int(round(time.time()-t0)))))
# Output asbtracts, probabilities, and predictions as dataframe
# Save abstract code for easy merging with features later
preds_df = pd.DataFrame({"Word": pos_words, "Prob_NotPositive": probs_nonpos_full,
                        "Prob_Positive": probs_pos_full, "Prediction": preds_full})

preds_df.to_csv("pos_words_predictions.csv", index = False)
