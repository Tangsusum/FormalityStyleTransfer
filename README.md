# Formality Style Transfer with Pre-trained NLP Models

Formality style transfer refers to the translation of text between formal and informal tones. Current solutions suffer greatly from the lack of corpus pairs, long training times and high computational requirements. While most literature explores the application from informal to formal language, this thesis focuses on the translation from formal to informal text with the goal of being applied for chatbot applications.

This thesis aims to overcome the current challenges by leveraging the general language abilities of pre-trained Natural Language Processing (NLP) models, specifically, the
Text-To-Text Transfer Transformer (T5) model, to achieve high-quality translations. The pre-trained NLP model is based on <a href="https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-">Goutham’s</a> adaptation of the T5 model for paraphrasing. The solution extends the previous approach to formality style transfer by applying pre-processing and finetuning techniques on the <a href="https://github.com/raosudha89/GYAFC-corpus">Grammarly’s Yahoo Answers Formality Corpus (GYAFC)</a> dataset. After training, the pre-trained model displays a decent understanding of formality with the ability to demonstrate a substantial comprehension of language.

This repository contains the source code for the thesis. 

FINE-TUNING
------------

To fine tune the current model (from formal to informal), create CSV files with headings labelled *formal* and *informal*. Store the training and validaiton set under the ***Data*** folder. Simply run ***train.py*** to start training. 

***trainMulti.py***  requries an additional *prefix* column. This script is able to perform fine-tuning on bi-directional formality style transfer i.e. from formal to informal AND informal to formal 

SENTENCE TRANSLATION
------------

To obtain translation from the fine-tuned model, simply run the Jupyter notebook, ***demo.ipynb*** or the python3 script ***demo.py***. Insert the desired sentence in the ***sentence*** variable.

Make sure that the model is located at the same directory of ***demo.ipynb*** or ***demo.py***.  

RESOURCES
------------

Request access to the thesis and the fine-tuned models using the link below: 

https://drive.google.com/drive/folders/1oJv8BYMO4_s6-FzUXzmFP2hIBY2plvWw?usp=sharing

To load the model, rename the model folder to ***mymodel***. 

EXAMPLES
------------

```
Original: Your mother’s stupidity is so excessive she sold her car to acquire money for gasoline
Translated: Yo momma is so stupid she sold her car for gas money.

Original: I am very fond of superman.
Translated: I love superman a lot.
```

