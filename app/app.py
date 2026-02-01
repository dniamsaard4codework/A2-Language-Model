"""
TinyStories Language Model - Flask Web Application
===================================================
A simple web interface for the LSTM-based story generator.

Author: Dechathon Niamsa-ard [st126235]
"""

import os
import sys
import re
import math
import pickle
from collections import Counter
from flask import Flask, render_template, request, jsonify

import torch
import torch.nn as nn

# ============================================================================
# Configuration
# ============================================================================

# Get the parent directory to access model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_lm_complete.pt')
VOCAB_PATH = os.path.join(MODEL_DIR, 'vocab.pkl')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Torchtext Compatibility Layer (must match training notebook)
# ============================================================================
# The vocabulary was pickled with this custom class, so we need to define it
# here exactly as it was in the training notebook for unpickling to work.

class torchtext:
    """Compatibility layer for torchtext (deprecated library)"""
    
    class vocab:
        class Vocab:
            """Custom vocabulary class compatible with torchtext.vocab.Vocab."""
            def __init__(self, counter=None, min_freq=1):
                self.itos = []
                self.stoi = {}
                self.default_index = 0
                
                if counter is not None:
                    for token, freq in counter.most_common():
                        if freq >= min_freq:
                            self.stoi[token] = len(self.itos)
                            self.itos.append(token)
            
            def __len__(self):
                return len(self.itos)
            
            def __getitem__(self, token):
                return self.stoi.get(token, self.default_index)
            
            def insert_token(self, token, index):
                if token in self.stoi:
                    return
                for t, i in list(self.stoi.items()):
                    if i >= index:
                        self.stoi[t] = i + 1
                self.stoi[token] = index
                self.itos.insert(index, token)
            
            def set_default_index(self, index):
                self.default_index = index
            
            def get_itos(self):
                return self.itos

# Make the class available for unpickling
sys.modules['__main__'].torchtext = torchtext

# ============================================================================
# Tokenizer (same as training)
# ============================================================================

def basic_english_tokenizer(text):
    """Basic English tokenizer matching training preprocessing."""
    text = text.lower()
    text = re.sub(r"([.,!?;:'\"\(\)\[\]\{\}])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [t for t in text.strip().split() if t]
    return tokens

# ============================================================================
# Model Definition (same architecture as training)
# ============================================================================

class LSTMLanguageModel(nn.Module):
    """LSTM-based Language Model for text generation."""
    
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers,
                           dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
    
    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

# ============================================================================
# Load Model and Vocabulary
# ============================================================================

def load_model_and_vocab():
    """Load the trained model and vocabulary."""
    # Load vocabulary
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    
    # Load model checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Initialize model with saved hyperparameters
    model = LSTMLanguageModel(
        vocab_size=checkpoint['vocab_size'],
        emb_dim=checkpoint['emb_dim'],
        hid_dim=checkpoint['hid_dim'],
        num_layers=checkpoint['num_layers'],
        dropout_rate=checkpoint['dropout_rate']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab

# Load model at startup
print(f"Loading model from: {MODEL_PATH}")
print(f"Loading vocabulary from: {VOCAB_PATH}")
model, vocab = load_model_and_vocab()
print(f"Model loaded successfully! Device: {device}")

# ============================================================================
# Text Generation
# ============================================================================

def generate_text(prompt, max_length=50, temperature=0.7, seed=None):
    """
    Generate text continuation from a prompt.
    
    Args:
        prompt: Input text prompt
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        seed: Random seed for reproducibility
    
    Returns:
        Generated text string
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    tokens = basic_english_tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        for _ in range(max_length):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            # Temperature-scaled sampling
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Skip unknown tokens
            while next_token == vocab['<unk>']:
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop at end of sequence
            if next_token == vocab['<eos>']:
                break
            
            indices.append(next_token)
    
    # Convert back to text
    itos = vocab.get_itos()
    generated_tokens = [itos[i] for i in indices]
    return ' '.join(generated_tokens)

# ============================================================================
# Flask Application
# ============================================================================

app = Flask(__name__)

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint for text generation."""
    data = request.get_json()
    
    prompt = data.get('prompt', '').strip()
    temperature = float(data.get('temperature', 0.7))
    max_length = int(data.get('max_length', 50))
    
    if not prompt:
        return jsonify({'error': 'Please enter a prompt', 'generated_text': ''})
    
    try:
        generated_text = generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e), 'generated_text': ''})

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'device': str(device)})

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("TinyStories Language Model Web App")
    print("="*50)
    print(f"Open your browser: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False)
