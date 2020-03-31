# Rao-NLP

Problem statement: Can we train an NLP model on a relatively small training set (~1400 abstracts) to accurately identify "positive" <sup>1</sup> results in food science paper abstracts? 

Models under evaluation: 
Fine-Tuning SciBERT using huggingface's BERTForSequenceClassification model https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification

Top performance: 

Fine-Tuning XLNet using huggingface's XLNetForSequenceClassification model 
https://huggingface.co/transformers/model_doc/xlnet.html#xlnetforsequenceclassification

Top performance: 

Yoon Kim's CNN for Sentence Classification w/ SciBERT word vectors 

Top performance: 

Yoon Kim's CNN for Sentence Classification w/ Word2Vec word vectors 

Top performance: 
 
<sup>1</sup> Positive defined as: reporting a positive improvement to an existing process 
