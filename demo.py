import torch
import nltk
import ssl
from transformers import T5ForConditionalGeneration,T5Tokenizer
from Evaluation.evaluationMetric import ngramsScore
from Evaluation.postProcessing import postProcessing
from Evaluation.selection import set_seed, getMaxNgram, getSelected, preProcessing, getVector

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('./mymodel')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

"""
For bi-directional formality style transfer model, select the translation direction by feeding in the correct prefix 
"""

sentence = "designed to simplify our organization and increase our agility to better serve our clients at scale."

text =  "formal2informal: " + sentence
# text =  "informal2formal: " + sentence

max_len = 256

encoding = tokenizer.encode_plus(text, padding="max_length", return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

beam_outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=256,
    top_k=50,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=10
)


print ("\nOriginal Sentence ::")
print (sentence)
print ("\n")
print ("Translated Sentences :: ")

# Get ngrams for sentence selection
ngrams = []
sentences = []
final_outputs =[]

for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    if sent.lower() != sentence.lower() and sent not in final_outputs:
        final_outputs.append(sent)

# Get word vector distance 
distances = getVector(sentence, final_outputs)

for i, final_output in enumerate(final_outputs):
    ngrams.append(ngramsScore(sentence,final_output))
    print("{}: {} ngrams={} dist={}".format(i, final_output, ngramsScore(sentence, final_output), distances[i]))
    

index, maxNgram = getMaxNgram(sorted(ngrams, reverse=True), 1.1)
print("\n")
print('Selected Sentence (ngrams w/o threshold):')
print(final_outputs[ngrams.index(maxNgram)])

index, maxNgram = getMaxNgram(sorted(ngrams, reverse=True), 0.8)
print("\n")
print('Selected Sentence (ngrams w/ threshold):')
print(final_outputs[ngrams.index(maxNgram)])

print("\n")
dist, ngram, selected_sentence = getSelected(ngrams, distances, final_outputs, 0.8)
print('Selected Sentence (ngrams + wordVector):')
print(selected_sentence)

print("\n")
print("Post-processed sentence:")
print(postProcessing(selected_sentence))