from nltk.tokenize import sent_tokenize
import re
import stanfordnlp

# stanfordnlp.download('en')

constractions = ["tis","'twas", "ain't", "aren't", "can't", "could've", "couldn't", "didn't", "doesn't", "don't", "hasn't", "he'd", "he'll", "he's", "how'd", "how'll", "how's", "i'd", "i'll", "i'm", "i've", "isn't", "it's", "might've", "mightn't", "must've", "mustn't", "shan't", "she'd", "she'll", "she's", "should've", "shouldn't", "that'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "wasn't", "we'd", "we'll", "we're", "weren't", "what'd", "what's", "when", "when'd", "when'll", "when's", "where'd", "where'll", "where's", "who'd", "who'll", "who's", "why'd", "why'll", "why's", "won't", "would've", "wouldn't", "you'd", "you'll", "you're", "you've"]

def truecasing_by_sentence_segmentation(input_text):
    sentences = sent_tokenize(input_text, language='english')
    sentences_capitalized = [s.capitalize() for s in sentences]
    text_truecase = re.sub(" (?=[\.,'!?:;])", "", ' '.join(sentences_capitalized))
    return text_truecase

def truecasing_by_POS(input_text):
    # Capatalise first word in sentences
    new_input = truecasing_by_sentence_segmentation(input_text)

    # Capatalise name and places
    stf_nlp = stanfordnlp.Pipeline(processors='tokenize,pos')
    doc = stf_nlp(new_input)

    for sent in doc.sentences: 
        for w in sent.words:
            if w.upos in ["PROPN", "NNS"]:
                new_input = new_input.replace(w.text, w.text.capitalize())
    # text_truecase = re.sub(" (?=[\.,'!?:;])", "", ' '.join(sentences_capitalized))
    return new_input

def removeSpecial(sentences):
    if type(sentences) == str:
        sentences = re.sub(r'\W+', '', sentences)
        sentences = sentences.lower()
    else:
        sentences = [re.sub(r'\W+', '', s) for s in sentences]
        sentences = [s.lower() for s in sentences]
    return sentences



def postProcessing(input_text):
    preProcessed_text = truecasing_by_POS(input_text)
    for n in preProcessed_text.split(" "):
        if n in removeSpecial(constractions):
            preProcessed_text = preProcessed_text.replace(n, constractions[removeSpecial(constractions).index(n)])
        if n == "i":
            preProcessed_text = preProcessed_text.replace(" i ", " I ")
    return preProcessed_text




