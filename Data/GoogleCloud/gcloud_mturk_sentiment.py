import pandas as pd
import os
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

#Make sure to run the below line in the shell session
#export GOOGLE_APPLICATION_CREDENTIALS="/Users/guanzhi0/Documents/Anita_Rao_RP/NLP/Rao-SciBERT/Data/GoogleCloud/My First Project-25c38a10b149.json"

# Instantiates a client
client = language.LanguageServiceClient()

#Load data and apply polarity rule
from pathlib import Path

data_path = str(Path(__file__).parent / "../Train")
mturk_abstracts = pd.read_csv(data_path + "/mturk_train.csv")

#Polarity rule: If >=2 positive ratings, then label positive
mturk_abstracts['polarity'] = (mturk_abstracts['count_pos'] >= 2).astype(int)

abstracts = mturk_abstracts['inputtext'].tolist()
labels = mturk_abstracts['polarity'].tolist()

# Loop through each abstract and extract sentiment
sentiment_scores = []
sentiment_class = []
for a in abstracts:
    document = types.Document(
        content=a,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    sentiment_scores.append(sentiment.score)
    if sentiment.score > 0:
        sentiment_class.append(1)
    else:
        sentiment_class.append(0)


print("Outputting predictions...")
out = pd.DataFrame({"Abstracts":abstracts, "Targets":labels, "Preds":sentiment_class, "Scores":sentiment_scores})
out['Correct'] = out['Preds'] == out['Targets']

accuracy = out['Correct'].sum()/len(out['Correct'])
print("Naive GoogleCloud sentiment classifier accuracy: {}".format(accuracy))

out.to_csv("gcloud_sentiments.csv", index = False)
