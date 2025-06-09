from flask import Flask, render_template, request
import pickle
import re

# Load model
with open("tamil_ner_model.pkl", "rb") as f:
    crf = pickle.load(f)

# Flask app initialization
app = Flask(__name__)

# Tamil-aware tokenization
def tokenize_paragraph(paragraph):
    pattern = r'[\u0B80-\u0BFF]+|[^\s\u0B80-\u0BFF]'
    return re.findall(pattern, paragraph)

def split_into_sentences(tokens):
    sentences, sentence = [], []
    for token in tokens:
        sentence.append(token)
        if token in ['.', '।']:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences

def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length': len(word),
        'prefix-1': word[:1],
        'prefix-2': word[:2] if len(word) > 1 else word[:1],
        'prefix-3': word[:3] if len(word) > 2 else word[:1],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:] if len(word) > 1 else word[-1:],
        'suffix-3': word[-3:] if len(word) > 2 else word[-1:],
        'word.has_digit': any(char.isdigit() for char in word),
        'word.has_hyphen': '-' in word,
        'word.isalpha': word.isalpha(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def is_punctuation(word):
    return bool(re.match(r'^[.,!?;।]$', word))

def predict_ner_tags(sentence_words):
    feats = sent2features(sentence_words)
    preds = crf.predict_single(feats)
    results = []
    for word, tag in zip(sentence_words, preds):
        if tag == 'O' and is_punctuation(word):
            results.append((word, 'SpaceAfter=No'))
        else:
            results.append((word, tag))
    return results

def predict_paragraph(paragraph):
    tokens = tokenize_paragraph(paragraph)
    sentence_list = split_into_sentences(tokens)
    all_results = []
    for sentence_tokens in sentence_list:
        predicted = predict_ner_tags(sentence_tokens)
        all_results.extend(predicted)
    return all_results

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        paragraph = request.form['paragraph']
        results = predict_paragraph(paragraph)
    return render_template('index.html', results=results)

# ✅ PRODUCTION: Required for Render.com
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))  # Render uses dynamic ports
    app.run(host='0.0.0.0', port=port)
