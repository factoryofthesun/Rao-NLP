# Rao-NLP

Problem statement: Can we train an NLP model on a relatively small training set (~1400 abstracts) to accurately identify "positive" <sup>1</sup> results in food science paper abstracts? 

Models under evaluation: 
Fine-Tuning SciBERT using huggingface's BERTForSequenceClassification model https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification

 Top performance: 73% accuracy, 0.65 AUC, 0.55 Loss

Fine-Tuning XLNet using huggingface's XLNetForSequenceClassification model 
https://huggingface.co/transformers/model_doc/xlnet.html#xlnetforsequenceclassification

 Top performance: 80% accuracy, 0.88 AUC, 0.43 Loss 

Yoon Kim's CNN for Sentence Classification w/ SciBERT word vectors 

 Top performance: 73.4% accuracy, 0.575 Loss

Yoon Kim's CNN for Sentence Classification w/ Word2Vec word vectors 

 Top performance: 62% accuracy, 1.05 Loss
 
<sup>1</sup> Positive defined as: reporting a positive improvement to an existing process 
