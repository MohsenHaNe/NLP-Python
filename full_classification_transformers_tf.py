# -*- coding: utf-8 -*-
"""full classification-transformers-tf.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mYZszhqhy1ywd74qPFqDXrUj2TsnTpag
"""

from transformers import AutoModelForSequenceClassification , TFAutoModelForSequenceClassification
from transformers import AutoTokenizer , AutoConfig
import numpy as np
from scipy.special import softmax

MODEL = f'cardiffnlp/twitter-xml-roberta-base-sentment'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

tf_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
tf_model.save_pretrained(MODEL)

ex_text = 'have nice day❤'

encoded = tokenizer(ex_text,return_tensors=True)

output = tf_model(encoded)[0][0].numpy()

print(softmax(output))







