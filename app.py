from flask import Flask, render_template, request, jsonify
import nltk
import numpy as np
from flask_cors import CORS

# Enable CORS for all routes
nltk.download('punkt')
nltk.download('punkt_tab')
app = Flask(__name__)
CORS(app)  # Enabling CORS for cross-origin requests

# Function to preprocess and extract features from the essay
def preprocess_essay(essay):
    words = nltk.word_tokenize(essay.lower())
    sentences = nltk.sent_tokenize(essay)
    words = [word for word in words if word.isalpha()]
    return words, sentences

# Function to calculate vocabulary richness (unique words / total words)
def vocabulary_richness(words):
    unique_words = set(words)
    return len(unique_words), len(words)

# Function to calculate average sentence length (in words)
def average_sentence_length(sentences):
    sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    return np.mean(sentence_lengths) if sentence_lengths else 0

# Function to extract useful insights (longest/shortest sentence, important words)
def extract_insights(essay, words, sentences):
    unique_words = set(words)
    longest_sentence = max(sentences, key=lambda x: len(nltk.word_tokenize(x)))
    shortest_sentence = min(sentences, key=lambda x: len(nltk.word_tokenize(x)))
    insights = {
        'keywords': list(unique_words),
        'longest_sentence': longest_sentence,
        'shortest_sentence': shortest_sentence
    }
    return insights

# Function to score the essay based on simple heuristics
def score_essay(essay):
    words, sentences = preprocess_essay(essay)
    vocab_unique, vocab_total = vocabulary_richness(words)
    avg_sentence_len = average_sentence_length(sentences)
    essay_len = len(essay)
    score = 0
    score += (vocab_unique / vocab_total) * 30
    score += avg_sentence_len * 2
    score += essay_len / 100
    score = min(max(score, 0), 100)
    return score, vocab_unique, vocab_total, avg_sentence_len

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit_essay', methods=['POST'])
def submit_essay():
    essay = request.form['essay']
    # Processing the essay
    score, vocab_unique, vocab_total, avg_sentence_len = score_essay(essay)
    words, sentences = preprocess_essay(essay)
    insights = extract_insights(essay, words, sentences)

    # Return results as a JSON response
    return jsonify({
        'score': round(score, 2),
        'vocab_unique': vocab_unique,
        'vocab_total': vocab_total,
        'avg_sentence_len': round(avg_sentence_len, 2),
        'keywords': insights['keywords'][:10],  # Show the first 10 unique words
        'longest_sentence': insights['longest_sentence'],
        'shortest_sentence': insights['shortest_sentence']
    })

if __name__ == '__main__':
    app.run(debug=True)
