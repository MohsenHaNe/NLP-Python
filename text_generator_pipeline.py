# -*- coding: utf-8 -*-
"""Text generator pipeline.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10jFB8lYzyxaC7UJ6D3JKu-eE8vr3mNtA
"""

from transformers import pipeline

generator = pipeline('text-generation',model='distilgpt2')##or bigscience

texts = generator('The weather is very hot this season',max_length=20,num_return_sequences=3)

print(texts[2])

