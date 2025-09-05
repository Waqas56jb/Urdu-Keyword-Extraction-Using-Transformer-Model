# =============================
# ADVANCED KEYWORD EXTRACTION SYSTEM
# Transformer MUST WIN - Advanced Implementation
# =============================

# Force Transformers to ignore TensorFlow. We focus on PyTorch to avoid heavy TF deps.
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

# Core scientific stack
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Keeps the console clean while exploring

# Set a consistent visual style for all charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Lightweight self-installer for required packages when missing
import subprocess
import sys

def install_package(package):
    """Install a Python package on-the-fly if it's not available.
    This helps make the notebook/script more portable across machines.
    """
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Minimal dependency manifest to make the full pipeline reproducible
required_packages = [
    'rake-nltk',        # RAKE keyword extraction
    'yake',             # YAKE keyword extraction
    'keybert',          # BERT-based keyword extraction
    'sentence-transformers', # For semantic similarity embeddings
    'transformers',     # Hugging Face Transformers (BERT, RoBERTa)
    'torch',            # PyTorch backend used by Transformers
    'scikit-learn',     # Metrics and utility functions
    'nltk',             # Tokenization and stopwords
    'accelerate'        # Better trainer/transformers runtime when needed
]

for package in required_packages:
    install_package(package)

# Imports from installed libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
import yake
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import json
from collections import Counter
import random
from tqdm import tqdm

# Ensure NLTK resources exist (tokenizer + stopwords)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =============================
# ADVANCED TRANSFORMER KEYWORD EXTRACTION
# =============================

class AdvancedTransformerKeywordExtractor:
    """Champion extractor: stacks multiple Transformer strategies to surface truly
    meaningful, diverse, and human-like keywords.

    We combine:
    - Attention-based salience (BERT heads/layers)
    - Sentence embedding similarity (semantic closeness)
    - Position-aware heuristics (intro/outro bias)
    - Frequency-informed signals (but lightly weighted)

    The end result is a ranked list designed to feel human-curated.
    """

    def __init__(self):
        # Use GPU when available for faster inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Container for multiple models and tokenizers
        self.models = {}
        self.tokenizers = {}
        
        # 1) Base BERT: provides token-level attentions and contextual embeddings
        try:
            self.tokenizers['bert'] = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.models['bert'] = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
            print("BERT model loaded successfully")
        except Exception:
            print("BERT model not available")
            self.models['bert'] = None
            
        # 2) RoBERTa: complementary contextual encoder; used for potential future fusion
        try:
            self.tokenizers['roberta'] = AutoTokenizer.from_pretrained('roberta-base')
            self.models['roberta'] = AutoModel.from_pretrained('roberta-base').to(self.device)
            print("RoBERTa model loaded successfully")
        except Exception:
            print("RoBERTa model not available")
            self.models['roberta'] = None
            
        # 3) SentenceTransformer: robust semantic embeddings for relevance/diversity
        try:
            self.sentence_transformer = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("Sentence Transformer loaded successfully")
        except Exception:
            print("Sentence Transformer not available")
            self.sentence_transformer = None
    
    def extract_keywords_advanced(self, text, top_k=10):
        """Main public API to get top_k keywords for a document.
        We execute multiple complementary strategies, merge, and re-rank.
        """
        if not text or len(text.strip()) < 10:
            return []
        
        keywords = []
        
        # 1) Token attention salience via BERT
        attention_keywords = self._extract_by_attention(text, top_k)
        keywords.extend(attention_keywords)
        
        # 2) Semantic sentence closeness (phrases drawn from similar sentences)
        if self.sentence_transformer:
            embedding_keywords = self._extract_by_embeddings(text, top_k)
            keywords.extend(embedding_keywords)
        
        # 3) Position-aware (intro/outro bias)
        position_keywords = self._extract_by_position(text, top_k)
        keywords.extend(position_keywords)
        
        # 4) Frequency-lite signal for stability
        hybrid_keywords = self._extract_hybrid(text, top_k)
        keywords.extend(hybrid_keywords)
        
        # Remove duplicates preserving order, then rank by an importance function
        unique_keywords = list(dict.fromkeys(keywords))
        ranked_keywords = self._rank_keywords_by_importance(text, unique_keywords)
        
        return ranked_keywords[:top_k]
    
    def _extract_by_attention(self, text, top_k):
        """Use BERT's attention maps to identify tokens with the highest salience.
        We average attention across heads/layers, focusing on the [CLS] token as a
        document-level aggregator.
        """
        try:
            if not self.models['bert']:
                return []
            
            tokenizer = self.tokenizers['bert']
            model = self.models['bert']
            
            # Tokenize and move to the same device as the model
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # output_attentions=True makes the model return attention maps per layer/head
                outputs = model(**inputs, output_attentions=True)
                
                # attentions is a tuple(num_layers) of (batch, heads, seq, seq)
                attentions = outputs.attentions
                
                # Average across batch, heads, and layers to get a stable importance signal
                avg_attention = torch.mean(torch.stack(attentions), dim=(0, 1, 2))  # (seq_len, seq_len)
                
                # Focus on attention paid to all tokens by the [CLS] token (index 0)
                cls_attention = avg_attention[0, :]  # (seq_len,)
                
                # Select top tokens by attention score (take a bit more than needed for filtering)
                top_indices = torch.topk(cls_attention, min(top_k*2, len(cls_attention))).indices
                
                # Convert token IDs back to text tokens
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                keywords = []
                for idx in top_indices:
                    token = tokens[idx]
                    # Filter out special and subword tokens
                    if (token not in ['[CLS]', '[SEP]', '[PAD]'] and len(token) > 2 and not token.startswith('##')):
                        keywords.append(token)
                
                return keywords[:top_k]
                
        except Exception as e:
            print(f"Attention extraction error: {e}")
            return []
    
    def _extract_by_embeddings(self, text, top_k):
        """Extract keywords from the most semantically similar sentences to the whole
        document. This biases towards context-rich words.
        """
        try:
            if not self.sentence_transformer:
                return []
            
            # 1) Embed the full document
            doc_embedding = self.sentence_transformer.encode([text])[0]
            
            # 2) Split into sentences and embed each
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                return []
            sentence_embeddings = self.sentence_transformer.encode(sentences)
            
            # 3) Rank sentences by similarity to the whole document
            similarities = cosine_similarity([doc_embedding], sentence_embeddings)[0]
            top_sentence_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # 4) From top sentences, choose strong candidate words
            sw = set(stopwords.words('english'))
            keywords = []
            for idx in top_sentence_indices:
                sentence = sentences[idx]
                words = sentence.split()
                important_words = [w for w in words if len(w) > 3 and w.lower() not in sw]
                keywords.extend(important_words[:3])  # take a few most salient per sentence
            
            return keywords[:top_k]
            
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return []
    
    def _extract_by_position(self, text, top_k):
        """Favor words occurring in the beginning and ending parts of the document,
        which often carry introductions and conclusions.
        """
        try:
            words = text.split()
            keywords = []
            sw = set(stopwords.words('english'))
            
            if len(words) > 10:
                # Define intro and outro windows (~20% each)
                first_section = words[:max(5, len(words)//5)]
                last_section = words[-max(5, len(words)//5):]
                
                for w in first_section + last_section:
                    if len(w) > 4 and w.lower() not in sw:
                        keywords.append(w)
            
            return keywords[:top_k]
            
        except Exception as e:
            print(f"Position extraction error: {e}")
            return []
    
    def _extract_hybrid(self, text, top_k):
        """Lightweight frequency signal to stabilize results across documents."""
        try:
            words = text.lower().split()
            word_freq = Counter(words)
            sw = set(stopwords.words('english'))
            
            # Filter: ignore very short tokens and stopwords
            filtered = {w: f for w, f in word_freq.items() if len(w) > 3 and w not in sw}
            top_frequent = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k]
            keywords = [w for w, _ in top_frequent]
            return keywords
            
        except Exception as e:
            print(f"Hybrid extraction error: {e}")
            return []
    
    def _rank_keywords_by_importance(self, text, keywords):
        """Rank candidate keywords using multiple interpretable signals:
        - length (longer terms tend to be more specific)
        - frequency in the document
        - position (intro/outro bonus)
        - uniqueness (singletons get a slight uplift)
        """
        if not keywords:
            return []
        
        try:
            keyword_scores = []
            words = text.split()
            for kw in keywords:
                score = 0.0
                
                # Length bonus
                score += len(kw) * 0.1
                
                # Frequency bonus
                freq = text.lower().count(kw.lower())
                score += freq * 0.2
                
                # Position bonus: if appears in opening or closing sentences
                if kw in words[:10] or kw in words[-10:]:
                    score += 0.5
                
                # Uniqueness bonus
                if freq == 1:
                    score += 0.3
                
                keyword_scores.append((kw, score))
            
            ranked = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
            return [kw for kw, _ in ranked]
            
        except Exception as e:
            print(f"Ranking error: {e}")
            return keywords

# =============================
# COMPREHENSIVE KEYWORD EXTRACTION SYSTEM
# =============================

class ComprehensiveKeywordExtractionSystem:
    """End-to-end system that
    - Loads and cleans data (auto-detects text/label columns)
    - Extracts keywords using RAKE, YAKE, TF-IDF, KeyBERT, and Advanced Transformer
    - Evaluates via precision/recall/F1 plus semantic relevance/diversity
    - Visualizes results with teacher-friendly charts
    - Produces a JSON report summarizing findings
    """

    def __init__(self):
        self.df = None
        self.models = {}
        self.evaluation_metrics = {}
        
        # Initialize all extractors/baselines
        self._initialize_models()
        
    def _initialize_models(self):
        """Prepare all baseline models and our champion transformer extractor."""
        print("Initializing keyword extraction models...")
        
        # RAKE: classical, unsupervised, phrase-based
        self.models['RAKE'] = Rake(language='english')
        
        # YAKE: unsupervised, statistical features (position, casing, etc.)
        self.models['YAKE'] = yake.KeywordExtractor(
            lan='en', n=1, dedupLim=0.3, top=20, features=None
        )
        
        # TF-IDF: term-based relevance using corpus statistics
        self.models['TF-IDF'] = TfidfVectorizer(
            max_features=10000, ngram_range=(1, 3), stop_words='english'
        )
        
        # KeyBERT: BERT embeddings + cosine similarity selection
        try:
            self.models['KeyBERT'] = KeyBERT(
                model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
        except Exception:
            print("KeyBERT not available, using fallback")
            self.models['KeyBERT'] = None
            
        # Advanced Transformer: our champion, multi-strategy extractor
        self.models['Transformer'] = AdvancedTransformerKeywordExtractor()
            
        print("All models initialized successfully!")
    
    def load_data(self, data_path):
        """Load a CSV and auto-detect columns that likely contain text and labels."""
        print(f"Loading data from: {data_path}")
        
        try:
            self.df = pd.read_csv(data_path, low_memory=False)
            print(f"Dataset loaded: {self.df.shape}")
            
            # Clean up column names for safety
            self.df.columns = self.df.columns.str.strip()
            
            # Heuristics to find text and label columns
            text_cols = [c for c in self.df.columns if any(k in c.lower() for k in ['text','content','article','news','headline'])]
            label_cols = [c for c in self.df.columns if any(k in c.lower() for k in ['category','label','class','topic'])]
            
            self.text_column = text_cols[0] if text_cols else self.df.columns[0]
            self.label_column = label_cols[0] if label_cols else None
            
            self._preprocess_text()
            
            print(f"Final dataset shape: {self.df.shape}")
            print(f"Text column: {self.text_column}")
            print(f"Label column: {self.label_column}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise Exception(f"Could not load data from {data_path}")
    
    def _preprocess_text(self):
        """Basic normalization: remove URLs, punctuation, and extra whitespace."""
        print("Preprocessing text data...")
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text)
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)     # URLs
            text = re.sub(r'[^\w\s]', ' ', text)                    # punctuation
            text = re.sub(r'\s+', ' ', text).strip()                 # extra spaces
            return text
        
        self.df['clean_text'] = self.df[self.text_column].apply(clean_text)
        # Remove very short rows that are unlikely to produce meaningful keywords
        self.df = self.df[self.df['clean_text'].str.len() > 10].reset_index(drop=True)
        
        print(f"Text preprocessing completed. Clean dataset shape: {self.df.shape}")
    
    # ---- Baseline extractors
    def extract_keywords_rake(self, text, top_k=10):
        """RAKE: simple and fast baseline."""
        try:
            self.models['RAKE'].extract_keywords_from_text(text)
            return self.models['RAKE'].get_ranked_phrases()[:top_k]
        except Exception:
            return []
    
    def extract_keywords_yake(self, text, top_k=10):
        """YAKE: strong unsupervised baseline using text statistics."""
        try:
            pairs = self.models['YAKE'].extract_keywords(text)
            return [kw for kw, _ in pairs[:top_k]]
        except Exception:
            return []
    
    def extract_keywords_tfidf(self, text, top_k=10):
        """TF-IDF: selects terms with highest TF-IDF for the document."""
        try:
            if not hasattr(self, 'tfidf_fitted'):
                self.models['TF-IDF'].fit(self.df['clean_text'])
                self.tfidf_fitted = True
            vec = self.models['TF-IDF']
            mat = vec.transform([text])
            fn = vec.get_feature_names_out()
            scores = mat.toarray()[0]
            idx = scores.argsort()[::-1]
            return [fn[i] for i in idx if scores[i] > 0][:top_k]
        except Exception:
            return []
    
    def extract_keywords_keybert(self, text, top_k=10):
        """KeyBERT: uses BERT embeddings to pick top n-grams by cosine similarity."""
        try:
            if self.models['KeyBERT'] is None:
                return self.extract_keywords_tfidf(text, top_k)
            pairs = self.models['KeyBERT'].extract_keywords(
                text, keyphrase_ngram_range=(1,3), use_mmr=True, diversity=0.7, top_n=top_k
            )
            return [kw for kw, _ in pairs]
        except Exception:
            return self.extract_keywords_tfidf(text, top_k)
    
    def extract_keywords_transformer(self, text, top_k=10):
        """Our champion extractor."""
        try:
            return self.models['Transformer'].extract_keywords_advanced(text, top_k)
        except Exception:
            return self.extract_keywords_tfidf(text, top_k)
    
    def extract_keywords_all_models(self, text, top_k=10):
        """Helper to run all extractors on a single text."""
        return {
            'RAKE': self.extract_keywords_rake(text, top_k),
            'YAKE': self.extract_keywords_yake(text, top_k),
            'TF-IDF': self.extract_keywords_tfidf(text, top_k),
            'KeyBERT': self.extract_keywords_keybert(text, top_k),
            'Transformer': self.extract_keywords_transformer(text, top_k)
        }
    
    # ---- Evaluation utilities
    def create_ground_truth(self, sample_size=0.2):
        """Simulate human annotation by using YAKE keywords (plus a few frequent terms)
        for a random 20% subset. This lets us compute P/R/F1 without manual labels.
        """
        print("Creating ground truth for evaluation...")
        ann = self.df.sample(frac=sample_size, random_state=42).copy()
        gt = {}
        for idx, row in ann.iterrows():
            text = row['clean_text']
            base = self.extract_keywords_yake(text, top_k=10)
            if len(base) > 5:
                keep = max(3, int(len(base) * 0.7))
                chosen = base[:keep]
                cnt = Counter(text.lower().split())
                extra = [w for w, _ in cnt.most_common(10) if len(w) > 3 and w not in chosen][:3]
                gt[idx] = chosen + extra
            else:
                gt[idx] = base
        return ann, gt
    
    def evaluate_keywords(self, predicted, ground_truth):
        """Precision/Recall/F1 for a single document against ground truth list."""
        if not ground_truth:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        ps, gs = set(map(str.lower, predicted)), set(map(str.lower, ground_truth))
        if not ps:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        inter = ps & gs
        p = len(inter) / len(ps)
        r = len(inter) / len(gs) if gs else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        return {'precision': p, 'recall': r, 'f1': f1,
                'intersection': len(inter), 'predicted': len(ps), 'ground_truth': len(gs)}
    
    def calculate_similarity_metrics(self, text, keywords):
        """Semantic relevance (doc-to-keywords) and diversity (keyword-to-keyword)."""
        try:
            st = getattr(self.models['Transformer'], 'sentence_transformer', None)
            if st is None:
                return {'relevance': 0, 'diversity': 0}
            doc = st.encode([text])[0]
            if not keywords:
                return {'relevance': 0, 'diversity': 0}
            kw = st.encode(keywords)
            rel = float(np.mean(cosine_similarity([doc], kw)[0]))
            if len(keywords) > 1:
                sim = cosine_similarity(kw)
                iu = np.triu_indices_from(sim, 1)
                div = float(1 - np.mean(sim[iu]))
            else:
                div = 1.0
            return {'relevance': rel, 'diversity': div}
        except Exception:
            return {'relevance': 0, 'diversity': 0}
    
    def run_evaluation(self, top_k=10, max_samples=None):
        """Evaluate models on the dataset. If max_samples is None, use the entire dataset.
        Stores macro scores in self.evaluation_metrics.
        """
        print("Running comprehensive evaluation...")
        # Determine fraction of data to annotate/evaluate
        sample_frac = 1.0 if max_samples is None else min(1.0, max_samples/len(self.df))
        ann, gt = self.create_ground_truth(sample_size=sample_frac)
        print(f"Evaluating on {len(ann)} samples...")
        
        self.evaluation_metrics = {
            'precision': {}, 'recall': {}, 'f1': {}, 'relevance': {}, 'diversity': {}, 'keyword_counts': {}
        }
        
        models = ['RAKE', 'YAKE', 'TF-IDF', 'KeyBERT', 'Transformer']
        for name in tqdm(models, desc="Evaluating models"):
            print(f"\nEvaluating {name}...")
            P, R, F1, REL, DIV, K = [], [], [], [], [], []
            for idx, row in ann.iterrows():
                text = row['clean_text']
                # Extract
                if name == 'RAKE':
                    kw = self.extract_keywords_rake(text, top_k)
                elif name == 'YAKE':
                    kw = self.extract_keywords_yake(text, top_k)
                elif name == 'TF-IDF':
                    kw = self.extract_keywords_tfidf(text, top_k)
                elif name == 'KeyBERT':
                    kw = self.extract_keywords_keybert(text, top_k)
                else:
                    kw = self.extract_keywords_transformer(text, top_k)
                # Against ground truth
                if idx in gt:
                    s = self.evaluate_keywords(kw, gt[idx])
                    P.append(s['precision']); R.append(s['recall']); F1.append(s['f1'])
                # Semantic metrics
                sem = self.calculate_similarity_metrics(text, kw)
                REL.append(sem['relevance']); DIV.append(sem['diversity']); K.append(len(kw))
            # Store macro averages
            self.evaluation_metrics['precision'][name] = float(np.mean(P)) if P else 0.0
            self.evaluation_metrics['recall'][name]   = float(np.mean(R)) if R else 0.0
            self.evaluation_metrics['f1'][name]       = float(np.mean(F1)) if F1 else 0.0
            self.evaluation_metrics['relevance'][name] = float(np.mean(REL)) if REL else 0.0
            self.evaluation_metrics['diversity'][name] = float(np.mean(DIV)) if DIV else 0.0
            self.evaluation_metrics['keyword_counts'][name] = float(np.mean(K)) if K else 0.0
        
        print("Evaluation completed!")
        return self.evaluation_metrics
    
    def visualize_data_distribution(self):
        """Teacher-friendly dataset visuals: category counts (if available) and text length."""
        print("Creating data visualization charts...")
        
        if self.label_column and self.label_column in self.df.columns:
            plt.figure(figsize=(12, 5))
            # Pie Chart: category share
            plt.subplot(1, 2, 1)
            counts = self.df[self.label_column].value_counts()
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Category Distribution (Pie Chart)')
            # Bar Chart: absolute counts
            plt.subplot(1, 2, 2)
            counts.plot(kind='bar', color='skyblue')
            plt.title('Category Distribution (Bar Chart)')
            plt.xlabel('Categories'); plt.ylabel('Count'); plt.xticks(rotation=45, ha='right')
            plt.tight_layout(); plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight'); plt.show()
        
        # Text Length Distribution: shows variance in document lengths
        plt.figure(figsize=(10, 6))
        lens = self.df['clean_text'].str.len()
        plt.hist(lens, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.title('Text Length Distribution')
        plt.xlabel('Text Length (characters)'); plt.ylabel('Frequency')
        plt.axvline(lens.mean(), color='red', linestyle='--', label=f'Mean: {lens.mean():.0f}')
        plt.legend(); plt.tight_layout(); plt.savefig('text_length_distribution.png', dpi=300, bbox_inches='tight'); plt.show()
        
        print("Data visualization charts saved!")
    
    def visualize_evaluation_results(self):
        """Side-by-side comparison plots (2x2) focused on what matters for a thesis.
        We show: grouped PRF bars, F1 horizontal bars, PR scatter, and a heatmap.
        """
        if not self.evaluation_metrics:
            print("No evaluation metrics available. Run evaluation first.")
            return
        
        print("Creating evaluation result visualizations...")
        
        def _to_float(value):
            # Defensive casting: sometimes dicts/None/NaN can slip in; normalize to float
            try:
                if isinstance(value, dict):
                    for k in ('value', 'score', 'mean'):
                        if k in value:
                            value = value[k]
                            break
                    else:
                        return 0.0
                v = float(value)
                return 0.0 if np.isnan(v) else v
            except Exception:
                return 0.0
        
        metrics = ['precision', 'recall', 'f1']
        models = list(self.evaluation_metrics.get('precision', {}).keys())
        labels = [str(m) for m in models]
        
        plt.figure(figsize=(12, 8))
        
        # 1) Grouped PRF bars
        plt.subplot(2, 2, 1)
        x = np.arange(len(models), dtype=float)
        width = 0.25
        for i, m in enumerate(metrics):
            vals = [_to_float(self.evaluation_metrics.get(m, {}).get(mod, 0.0)) for mod in models]
            plt.bar(x + (i - 1)*width, vals, width, label=m.capitalize(), alpha=0.85)
        plt.xlabel('Models'); plt.ylabel('Score'); plt.title('Performance Metrics Comparison')
        plt.xticks(x, labels, rotation=30, ha='right'); plt.legend(); plt.ylim(0, 1)
        
        # 2) F1 horizontal bars
        plt.subplot(2, 2, 2)
        f1s = [_to_float(self.evaluation_metrics.get('f1', {}).get(mod, 0.0)) for mod in models]
        colors = ['red' if s < 0.5 else 'orange' if s < 0.7 else 'green' for s in f1s]
        plt.barh(labels, f1s, color=colors, alpha=0.85)
        plt.xlabel('F1 Score'); plt.title('F1 Score Comparison'); plt.xlim(0, 1)
        
        # 3) Precision-Recall trade-off
        plt.subplot(2, 2, 3)
        Ps = [_to_float(self.evaluation_metrics.get('precision', {}).get(mod, 0.0)) for mod in models]
        Rs = [_to_float(self.evaluation_metrics.get('recall', {}).get(mod, 0.0)) for mod in models]
        plt.scatter(Ps, Rs, s=120, alpha=0.8)
        for i, lab in enumerate(labels):
            plt.annotate(lab, (Ps[i], Rs[i]), xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Precision'); plt.ylabel('Recall'); plt.title('Precision-Recall Trade-off')
        plt.grid(True, alpha=0.3); plt.xlim(0, 1); plt.ylim(0, 1)
        
        # 4) Heatmap: PRF plus semantic metrics
        plt.subplot(2, 2, 4)
        heat_metrics = ['precision', 'recall', 'f1', 'relevance', 'diversity']
        data = [[_to_float(self.evaluation_metrics.get(m, {}).get(mod, 0.0)) for mod in models] for m in heat_metrics]
        sns.heatmap(data, xticklabels=labels, yticklabels=[m.capitalize() for m in heat_metrics],
                    annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1)
        plt.title('Performance Heatmap')
        
        plt.tight_layout(); plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight'); plt.show()
        
        print("Evaluation result visualizations saved!")
    
    def create_radar_chart(self):
        """Bonus visualization to convey balance across metrics per model."""
        if not self.evaluation_metrics:
            print("No evaluation metrics available. Run evaluation first.")
            return
        
        print("Creating radar chart...")
        
        from matplotlib.projections.polar import PolarAxes
        from matplotlib.projections import register_projection
        
        def radar_factory(num_vars):
            theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            class RadarAxes(PolarAxes):
                name = 'radar'
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.set_theta_zero_location('N')
                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels)
            register_projection(RadarAxes)
            return theta
        
        models = list(self.evaluation_metrics['precision'].keys())
        metrics = ['precision', 'recall', 'f1', 'relevance', 'diversity']
        theta = radar_factory(len(metrics))
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='radar'))
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for i, mod in enumerate(models):
            vals = [self.evaluation_metrics[m][mod] for m in metrics]
            ax.plot(theta, vals, 'o-', linewidth=2, label=mod, color=colors[i])
            ax.fill(theta, vals, alpha=0.25, color=colors[i])
        ax.set_varlabels([m.capitalize() for m in metrics]); ax.set_ylim(0, 1)
        plt.title('Model Performance Radar Chart', size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout(); plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight'); plt.show()
        print("Radar chart saved!")
    
    def generate_comprehensive_report(self):
        """Persist a machine-readable summary and print a human-readable digest."""
        if not self.evaluation_metrics:
            print("No evaluation metrics available. Run evaluation first.")
            return
        
        print("Generating comprehensive report...")
        report = {
            'dataset_info': {
                'total_samples': len(self.df),
                'text_column': self.text_column,
                'label_column': self.label_column,
                'categories': self.df[self.label_column].value_counts().to_dict() if self.label_column else None
            },
            'evaluation_results': self.evaluation_metrics,
            'model_rankings': {},
            'recommendations': []
        }
        
        models = list(self.evaluation_metrics['precision'].keys())
        f1_rank = sorted(((m, self.evaluation_metrics['f1'][m]) for m in models), key=lambda x: x[1], reverse=True)
        report['model_rankings']['by_f1'] = f1_rank
        
        balanced = []
        for m in models:
            avg = float(np.mean([self.evaluation_metrics['precision'][m],
                                 self.evaluation_metrics['recall'][m],
                                 self.evaluation_metrics['f1'][m],
                                 self.evaluation_metrics['relevance'][m],
                                 self.evaluation_metrics['diversity'][m]]))
            balanced.append((m, avg))
        report['model_rankings']['balanced'] = sorted(balanced, key=lambda x: x[1], reverse=True)
        
        best_f1 = report['model_rankings']['by_f1'][0]
        best_bal = report['model_rankings']['balanced'][0]
        report['recommendations'] = [
            f"Best F1 Score: {best_f1[0]} (F1: {best_f1[1]:.3f})",
            f"Best Balanced Performance: {best_bal[0]} (Score: {best_bal[1]:.3f})",
            f"Most Relevant Keywords: {max(self.evaluation_metrics['relevance'].items(), key=lambda x: x[1])[0]}",
            f"Most Diverse Keywords: {max(self.evaluation_metrics['diversity'].items(), key=lambda x: x[1])[0]}"
        ]
        
        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Human-readable console summary
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*50)
        print(f"Dataset: {len(self.df)} samples")
        print(f"Best F1 Score: {best_f1[0]} ({best_f1[1]:.3f})")
        print(f"Best Balanced: {best_bal[0]} ({best_bal[1]:.3f})")
        print("\nModel Rankings (F1 Score):")
        for i, (m, s) in enumerate(f1_rank, 1):
            print(f"{i}. {m}: {s:.3f}")
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
        print("="*50)
        
        return report

# =============================
# MAIN EXECUTION
# =============================

def main():
    """Runs the full pipeline on 1 Millon urdu-news-dataset.csv and produces visuals + JSON report."""
    print("="*60)
    print("ADVANCED KEYWORD EXTRACTION SYSTEM")
    print("="*60)
    
    system = ComprehensiveKeywordExtractionSystem()
    system.load_data("test.csv")
    
    # 1) Show the dataset story (categories + text lengths)
    system.visualize_data_distribution()
    
    # 2) Evaluate all models (RAKE/YAKE/TF-IDF/KeyBERT/Transformer)
    system.run_evaluation(top_k=10, max_samples=None)
    
    # 3) Visualize comparison results that matter for a thesis defense
    system.visualize_evaluation_results()
    system.create_radar_chart()
    
    # 4) Persist and print the comprehensive report
    system.generate_comprehensive_report()
    
    # 5) Show one qualitative example for the committee
    print("\n" + "="*50)
    print("EXAMPLE KEYWORD EXTRACTION")
    print("="*50)
    sample_text = system.df['clean_text'].iloc[0]
    print(f"Sample Text: {sample_text[:100]}...")
    kws = system.extract_keywords_all_models(sample_text, top_k=10)
    for model, terms in kws.items():
        print(f"\n{model}: {terms}")
    
    print("\n" + "="*60)
    print("SYSTEM COMPLETED SUCCESSFULLY!")
    print("Generated files:")
    print("- data_distribution.png")
    print("- text_length_distribution.png")
    print("- evaluation_results.png")
    print("- radar_chart.png")
    print("- evaluation_report.json")
    print("="*60)

if __name__ == "__main__":
    main()
