# -*- coding: utf-8 -*-
"""
English Emotion, Color & Person Analysis Model + Depression Risk Detection (CSV Version)
"""

import pandas as pd
import numpy as np
import os
import sys
import re
import csv
import colorsys
import random
import time
import math
import subprocess
import importlib.util
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class ImprovedEmotionAnalyzer:
    def __init__(self):
        
        self.text_model = None
        self.text_vectorizer = None
        self.color_model = None
        self.color_encoder = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.bert_device = 'cpu'
        self.bert_id2label = {}
        self._torch = None
        self.name_set = set()      # 이름 데이터 저장소
        self.risk_keywords = set() # 위험 어휘 데이터 저장소
        self._emotion_dataset_cache = None
        self.emoji_emotion_weights = {}
        self.source_weights = {
            'keyword': float(os.environ.get('CLIO_WEIGHT_KEYWORD', 1.0)),
            'bert': float(os.environ.get('CLIO_WEIGHT_BERT', 1.25)),
            'ml': float(os.environ.get('CLIO_WEIGHT_ML', 0.85)),
            'emoji': float(os.environ.get('CLIO_WEIGHT_EMOJI', 1.15))
        }

        self.emotion_colors = {
            'Happiness': {'color': '#FFD700', 'color_name': 'Gold', 'tone': 'Bright and Pastel'},
            'Sadness': {'color': '#4682B4', 'color_name': 'Steel Blue', 'tone': 'Calm and Dark'},
            'Anger': {'color': '#DC143C', 'color_name': 'Crimson Red', 'tone': 'Intense and Dark'},
            'Fear': {'color': '#808080', 'color_name': 'Grey', 'tone': 'Dark and Muted'},
            'Disgust': {'color': '#9ACD32', 'color_name': 'Yellow Green', 'tone': 'Muted and Dark'},
            'Surprise': {'color': '#FF69B4', 'color_name': 'Hot Pink', 'tone': 'Vivid and Bright'}
        }
        self.color_dataset = None
        self.emotion_colors_data = {}
        self._load_models()
    
    def _load_models(self):
        """Load models on server start"""
        print("🚀 Starting AI model loading...")
        
        # 1. Load Text Model
        self._load_text_model()

        # 1-b. Load BERT model for deep learning inference
        self._load_bert_model()
        
        # 2. Load Color Model
        self._load_color_model()
        
        # 3. Load Color Dataset
        self._load_color_dataset()

        # 4. Load Name Dataset
        self._load_name_dataset()

        # 5. [수정됨] Load Risk Dataset (CSV Version)
        self._load_risk_dataset()

        # 6. Emoji-aware emotion weight loader
        self._load_emoji_emotion_model()
        
        print("✅ All models loaded successfully!")

    def _print_progress(self, current, total, prefix=""):
        total = max(total, 1)
        ratio = max(0.0, min(1.0, current / total))
        bar_length = 30
        filled = int(ratio * bar_length)
        bar = '#' * filled + '-' * (bar_length - filled)
        message = f"\r{prefix}[{bar}] {ratio * 100:5.1f}%"
        sys.stdout.write(message)
        sys.stdout.flush()
        if current >= total:
            sys.stdout.write("\n")

    def _ensure_dependency(self, module_name, install_target=None):
        """Install Python dependency from within the script when missing."""
        try:
            if importlib.util.find_spec(module_name) is not None:
                return
        except ModuleNotFoundError:
            pass

        package = install_target or module_name
        print(f"📦 Installing missing dependency: {package}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception as exc:
            raise RuntimeError(f"Failed to install {package}: {exc}")

    def _load_bert_model(self):
        model_name = os.environ.get('EMOTION_BERT_MODEL', 'distilbert-base-uncased')
        cache_dir = os.environ.get(
            'EMOTION_BERT_CACHE',
            os.path.join(os.path.dirname(__file__), 'bert_finetuned')
        )
        dataset = self._get_emotion_dataset()
        if dataset is None or dataset.empty:
            print("⚠️ Skipping BERT: no dataset available")
            return

        label_list = sorted({str(label).lower() for label in dataset['label'].unique()})
        if len(label_list) < 2:
            print("⚠️ Skipping BERT: need at least two label classes")
            return
        label2id = {label: idx for idx, label in enumerate(label_list)}

        try:
            self._ensure_dependency('transformers')
            self._ensure_dependency('torch')
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            os.makedirs(cache_dir, exist_ok=True)
            model_loaded = False

            if os.path.exists(os.path.join(cache_dir, 'config.json')):
                try:
                    print(f"🧠 Loading fine-tuned BERT from {cache_dir}")
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(cache_dir)
                    self.bert_model = AutoModelForSequenceClassification.from_pretrained(cache_dir)
                    model_loaded = True
                except Exception as load_err:
                    print(f"⚠️ Failed to load cached BERT model: {load_err}. Re-training...")

            if model_loaded and self.bert_model is not None:
                cached_labels = sorted(
                    {str(label).lower() for label in self.bert_model.config.id2label.values()}
                )
                if cached_labels != label_list:
                    print("♻️ Cached BERT labels differ from dataset labels. Re-training...")
                    model_loaded = False
                    self.bert_model = None

            if not model_loaded:
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(label2id),
                    id2label={idx: label for label, idx in label2id.items()},
                    label2id=label2id
                )
                self._train_bert_model(base_model, self.bert_tokenizer, dataset, cache_dir, label2id, torch)
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(cache_dir)
                self.bert_tokenizer = AutoTokenizer.from_pretrained(cache_dir)

            self._torch = torch
            self.bert_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.bert_model.to(self.bert_device)
            self.bert_model.eval()
            self.bert_id2label = {idx: label.lower() for idx, label in self.bert_model.config.id2label.items()}
            print("✅ BERT model ready.")
        except Exception as e:
            print(f"⚠️ Unable to initialize BERT model: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
            self.bert_id2label = {}
            self._torch = None

    def _train_bert_model(self, model, tokenizer, dataset, cache_dir, label2id, torch_module):
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        class EmotionTextDataset(torch_module.utils.data.Dataset):
            def __init__(self, texts, labels, tokenizer_ref, max_length=256):
                self.texts = list(texts)
                self.labels = list(labels)
                self.tokenizer = tokenizer_ref
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx])
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                item = {key: val.squeeze(0) for key, val in encoding.items()}
                item['labels'] = torch_module.tensor(self.labels[idx], dtype=torch_module.long)
                return item

        labels_numeric = dataset['label'].map(label2id)
        texts = dataset['text'].tolist()
        labels = labels_numeric.tolist()

        if len(texts) < 4:
            raise RuntimeError("Not enough samples to fine-tune BERT")

        test_size = 0.1 if len(texts) >= 20 else 0.2
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        max_length = int(os.environ.get('EMOTION_BERT_MAXLEN', 200))
        batch_size = int(os.environ.get('EMOTION_BERT_BATCH', 8))
        epochs = int(os.environ.get('EMOTION_BERT_EPOCHS', 2))
        learning_rate = float(os.environ.get('EMOTION_BERT_LR', 2e-5))

        train_dataset = EmotionTextDataset(X_train, y_train, tokenizer, max_length=max_length)
        val_dataset = EmotionTextDataset(X_val, y_val, tokenizer, max_length=max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        device = 'cuda' if torch_module.cuda.is_available() else 'cpu'
        model.to(device)

        optimizer = torch_module.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = max(1, len(train_loader) * epochs)
        warmup_steps = max(1, int(total_steps * 0.1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print(f"🛠️ Fine-tuning BERT on {len(texts)} samples ({epochs} epochs)...")

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_batches = max(1, len(train_loader))
            for batch_idx, batch in enumerate(train_loader, start=1):
                optimizer.zero_grad()
                labels_batch = batch['labels'].to(device)
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                outputs = model(**inputs, labels=labels_batch)
                loss = outputs.loss
                loss.backward()
                torch_module.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                self._print_progress(batch_idx, total_batches, prefix=f"      Epoch {epoch+1}/{epochs} ")

            avg_loss = total_loss / max(1, len(train_loader))
            model.eval()
            correct = 0
            total = 0
            with torch_module.no_grad():
                for batch in val_loader:
                    labels_batch = batch['labels'].to(device)
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    logits = model(**inputs).logits
                    preds = torch_module.argmax(logits, dim=-1)
                    correct += (preds == labels_batch).sum().item()
                    total += labels_batch.size(0)
            val_acc = correct / max(1, total)
            print(f"   Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}, val_acc: {val_acc:.3f}")

        model.save_pretrained(cache_dir)
        tokenizer.save_pretrained(cache_dir)
        print(f"💾 Fine-tuned BERT saved to {cache_dir}")

    # --- [수정됨] 위험 어휘 데이터셋 로드 함수 (CSV 버전) ---
    def _load_risk_dataset(self):
        try:
            # 파일명을 .csv로 변경
            csv_path = os.path.join(os.path.dirname(__file__), 'depression_risk_words.csv')
            
            if not os.path.exists(csv_path):
                print(f"⚠️ Risk dataset not found: {csv_path}")
                # 파일이 없을 경우를 대비한 기본 키워드
                self.risk_keywords = {"suicide", "kill myself", "want to die", "hopeless"}
                return
            
            print("🛡️ Loading risk dataset (CSV)...")
            
            # CSV 파일 읽기 (쉼표 구분)
            with open(csv_path, 'r', encoding='utf-8') as f:
                # delimiter를 지정하지 않으면 기본값이 콤마(,) 입니다.
                reader = csv.DictReader(f) 
                count = 0
                for row in reader:
                    # CSV 파일의 헤더가 'word', 'risk_level' 이라고 가정합니다.
                    # 혹시 헤더 이름이 다르면 이 부분을 수정해야 합니다.
                    if 'risk_level' in row and row['risk_level'] in ['high', 'medium']:
                        if 'word' in row:
                            self.risk_keywords.add(row['word'].lower())
                            count += 1
            
            print(f"✅ Loaded {count} risk keywords.")
            
        except Exception as e:
            print(f"❌ Failed to load risk dataset: {e}")
            self.risk_keywords = {"suicide", "die"}

    def _load_emoji_emotion_model(self):
        """Load emoji-emotion mapping and associated weights."""
        csv_path = os.path.join(os.path.dirname(__file__), 'emoji_emotion_large.csv')
        if not os.path.exists(csv_path):
            print(f"⚠️ Emoji dataset not found: {csv_path}")
            self.emoji_emotion_weights = {}
            return

        emoji_map = {}
        try:
            print("😊 Loading emoji-emotion dataset...")
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                if reader.fieldnames is None:
                    raise ValueError("Emoji CSV has no header row.")
                normalized_fields = [field.lower() for field in reader.fieldnames]
                emoji_field = None
                emotion_field = None
                weight_field = None
                for idx, field in enumerate(normalized_fields):
                    if field in ('emoji', 'icon', 'symbol', 'character') and emoji_field is None:
                        emoji_field = reader.fieldnames[idx]
                    elif field in ('emotion', 'label', 'sentiment') and emotion_field is None:
                        emotion_field = reader.fieldnames[idx]
                    elif field in ('weight', 'score', 'count') and weight_field is None:
                        weight_field = reader.fieldnames[idx]

                if emoji_field is None or emotion_field is None:
                    raise ValueError("Emoji CSV must contain emoji and emotion columns.")

                entry_count = 0
                for row in reader:
                    emoji_value = (row.get(emoji_field) or '').strip()
                    raw_emotion = (row.get(emotion_field) or '').strip()
                    if not emoji_value or not raw_emotion:
                        continue
                    mapped_emotion = self._map_label_to_emotion(raw_emotion)
                    if not mapped_emotion:
                        continue
                    try:
                        raw_weight = row.get(weight_field) if weight_field else None
                        weight_value = float(raw_weight) if raw_weight not in (None, '') else 1.0
                    except ValueError:
                        weight_value = 1.0
                    weight_value = max(weight_value, 0.05)
                    if emoji_value not in emoji_map:
                        emoji_map[emoji_value] = Counter()
                    emoji_map[emoji_value][mapped_emotion] += weight_value
                    entry_count += 1

            self.emoji_emotion_weights = {
                emoji: dict(weight_counter)
                for emoji, weight_counter in emoji_map.items()
            }
            print(f"✅ Loaded {entry_count} emoji-emotion entries ({len(self.emoji_emotion_weights)} unique emoji).")
        except Exception as exc:
            print(f"❌ Failed to load emoji dataset: {exc}")
            self.emoji_emotion_weights = {}

    def _load_name_dataset(self):
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'name_gender_dataset.csv')
            
            if not os.path.exists(csv_path):
                print(f"⚠️ Name dataset not found: {csv_path}")
                return
            
            print("👥 Loading name dataset...")
            df = pd.read_csv(csv_path)
            self.name_set = set(df['Name'].str.lower().values)
            print(f"✅ Loaded {len(self.name_set)} names.")
            
        except Exception as e:
            print(f"❌ Failed to load name dataset: {e}")

    def _load_color_dataset(self):
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'your_file_name.csv')
            
            if not os.path.exists(csv_path):
                print(f"⚠️ Color dataset not found: {csv_path}")
                return
            
            print("🎨 Loading color dataset...")
            self.color_dataset = pd.read_csv(csv_path)
            self.color_dataset = self.color_dataset[self.color_dataset['is_error'] == False]
            
            for emotion in self.color_dataset['emotion'].unique():
                emotion_data = self.color_dataset[self.color_dataset['emotion'] == emotion]
                self.emotion_colors_data[emotion] = emotion_data[['h', 's', 'v']].values
            
            print(f"📊 Color dataset loaded: {len(self.color_dataset)} samples")
                
        except Exception as e:
            print(f"❌ Failed to load color dataset: {e}")
            self.color_dataset = None
            self.emotion_colors_data = {}
    
    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _map_label_to_emotion(self, label):
        if not label:
            return 'Happiness'
        label = label.lower()
        mapping = {
            'joy': 'Happiness',
            'love': 'Happiness',
            'happy': 'Happiness',
            'positive': 'Happiness',
            'sadness': 'Sadness',
            'sad': 'Sadness',
            'anger': 'Anger',
            'angry': 'Anger',
            'fear': 'Fear',
            'scared': 'Fear',
            'disgust': 'Disgust',
            'hatred': 'Disgust',
            'surprise': 'Surprise',
            'neutral': 'Happiness'
        }
        return mapping.get(label, 'Happiness')
    
    def _get_emotion_dataset(self):
        if self._emotion_dataset_cache is not None:
            return self._emotion_dataset_cache.copy()

        csv_path = os.path.join(os.path.dirname(__file__), 'emotion_sentimen_dataset.csv')

        if not os.path.exists(csv_path):
            print(f"⚠️ Dataset not found: {csv_path}")
            self._emotion_dataset_cache = None
            return None

        print("📊 Loading text dataset...")
        df = pd.read_csv(csv_path, encoding='latin1')
        df_renamed = df.rename(columns={'Emotion': 'label', 'text': 'text'})
        df_clean = df_renamed[['text', 'label']].copy()

        df_clean['text'] = df_clean['text'].apply(self._clean_text)
        df_clean.dropna(subset=['text', 'label'], inplace=True)
        df_final = df_clean[df_clean['text'] != ""]

        label_map = {
            'happiness': 'joy', 'happy': 'joy', 'fun': 'joy', 'enthusiasm': 'joy',
            'relief': 'joy', 'love': 'joy', 'joy': 'joy', 'pleasure': 'joy',
            'sadness': 'sadness', 'empty': 'sadness', 'boredom': 'sadness',
            'grief': 'sadness', 'sorrow': 'sadness',
            'anger': 'anger', 'annoyance': 'anger', 'rage': 'anger',
            'fear': 'fear', 'worry': 'fear', 'anxiety': 'fear', 'panic': 'fear',
            'disgust': 'disgust', 'hate': 'disgust', 'aversion': 'disgust',
            'surprise': 'surprise', 'shock': 'surprise'
        }

        df_final['label'] = df_final['label'].map(label_map)
        df_final = df_final.dropna(subset=['label'])

        dataset_size = len(df_final)
        if dataset_size == 0:
            print("⚠️ No labeled samples found in dataset.")
            self._emotion_dataset_cache = None
            return None

        print(f"📦 Dataset ready: {dataset_size} samples")
        self._emotion_dataset_cache = df_final
        return df_final.copy()

    def _load_text_model(self):
        try:
            df_final = self._get_emotion_dataset()
            if df_final is None or df_final.empty:
                return

            label_counts = Counter(df_final['label'])
            if len(label_counts) < 2:
                print("⚠️ Not enough label diversity for the text model.")
                return

            X = df_final['text']
            y = df_final['label']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_tfidf = self.text_vectorizer.fit_transform(X_train)
            X_test_tfidf = self.text_vectorizer.transform(X_test)

            self.text_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            self.text_model.fit(X_train_tfidf, y_train)
            
            # 간단한 검증 출력
            y_pred = self.text_model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"📈 Validation accuracy: {accuracy:.3f}")

        except Exception as e:
            print(f"❌ Failed to load text model: {e}")
            self.text_model = None
            self.text_vectorizer = None
    
    def _load_color_model(self):
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'your_file_name.csv')
            if not os.path.exists(csv_path):
                print(f"⚠️ Color dataset not found: {csv_path}")
                return
            
            print("🎨 Training color model...")
            data = pd.read_csv(csv_path)
            X = data[['h', 's', 'v']]
            y = data['emotion']
            self.color_encoder = LabelEncoder()
            y_encoded = self.color_encoder.fit_transform(y)
            self.color_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.color_model.fit(X, y_encoded)
        except Exception as e:
            print(f"❌ Failed to load color model: {e}")
            self.color_model = None
            self.color_encoder = None
    
    def analyze_emotion(self, text):
        emotion, _, _ = self.analyze_emotion_with_flow(text)
        return emotion

    def analyze_emotion_with_flow(self, text):
        if not isinstance(text, str) or not text.strip():
            return 'Happiness', [], None

        blocks = self._split_text_blocks(text)
        if not blocks:
            return 'Happiness', [], None

        emotion_scores = Counter()
        flow = []
        total_blocks = len(blocks)

        for idx, block in enumerate(blocks):
            block_emotion = self._analyze_single_block(block)
            weight = self._calculate_block_weight(block, idx, total_blocks)
            flow.append({
                'index': idx,
                'text': block,
                'emotion': block_emotion,
                'weight': weight
            })
            emotion_scores[block_emotion] += weight

        if not flow or not emotion_scores:
            return 'Happiness', flow, None

        total_weight = sum(item['weight'] for item in flow) or 1.0
        for item in flow:
            item['ratio'] = item['weight'] / total_weight
            color_meta = self.emotion_colors.get(item['emotion'], self.emotion_colors['Happiness'])
            item['color'] = color_meta['color']

        dominant = emotion_scores.most_common(1)[0][0]
        gradient = self._build_flow_gradient(flow)
        return dominant, flow, gradient

    def _split_text_blocks(self, text):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            sentences = [text.strip()]
        return sentences

    def _calculate_block_weight(self, block, index, total_blocks):
        word_count = max(len(block.split()), 1)
        length_factor = math.sqrt(word_count)
        if total_blocks > 1:
            recency_factor = 1.0 + (index / (total_blocks - 1)) * 0.3
        else:
            recency_factor = 1.0
        return length_factor * recency_factor

    def _build_flow_gradient(self, flow):
        if not flow:
            return None
        stops = []
        cumulative = 0.0
        for idx, item in enumerate(flow):
            color = item.get('color', self.emotion_colors['Happiness']['color'])
            if idx == 0:
                stops.append(f"{color} 0%")
            cumulative += item.get('ratio', 0)
            percent = max(0.0, min(100.0, cumulative * 100))
            stops.append(f"{color} {percent:.2f}%")
        return f"linear-gradient(90deg, {', '.join(stops)})"

    def _analyze_single_block(self, text):
        scores = Counter()

        def add_vote(emotion_label, source_tag):
            if not emotion_label:
                return
            weight = self.source_weights.get(source_tag, 1.0)
            if weight <= 0:
                return
            scores[emotion_label] += weight

        add_vote(self._analyze_english_emotion(text), 'keyword')
        add_vote(self._analyze_with_bert(text), 'bert')
        add_vote(self._analyze_with_ml(text), 'ml')

        emoji_scores = self._analyze_with_emoji(text)
        if emoji_scores:
            emoji_weight = self.source_weights.get('emoji', 1.0)
            for emotion_label, value in emoji_scores.items():
                scores[emotion_label] += value * emoji_weight

        if scores:
            return scores.most_common(1)[0][0]
        return 'Happiness'
    
    def _analyze_english_emotion(self, text):
        text_lower = text.lower()
        english_emotions = {
            'Fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', 'panic', 'fear', 'dread', 'horror', 'scary', 'frightened'],
            'Happiness': ['happy', 'joy', 'glad', 'excited', 'wonderful', 'amazing', 'great', 'good', 'love', 'smile', 'laugh', 'fun', 'best', 'perfect'],
            'Sadness': ['sad', 'cry', 'tears', 'lonely', 'depressed', 'down', 'blue', 'hurt', 'pain', 'sorrow', 'grief', 'miserable'],
            'Anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated', 'frustrated', 'outraged'],
            'Disgust': ['disgusted', 'gross', 'sick', 'nauseated', 'revolted', 'repulsed', 'awful', 'terrible', 'horrible'],
            'Surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'wow', 'incredible', 'unexpected', 'startled']
        }
        emotion_scores = {}
        for emotion, keywords in english_emotions.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        if emotion_scores and max(emotion_scores.values()) > 0:
            return max(emotion_scores, key=emotion_scores.get)
        return None

    def _analyze_with_bert(self, text):
        if not self.bert_model or not self.bert_tokenizer or not text:
            return None
        cleaned_text = text.strip()
        if not cleaned_text:
            return None
        try:
            tokenizer_kwargs = {
                'padding': True,
                'truncation': True,
                'max_length': 256,
                'return_tensors': 'pt'
            }
            encoded = self.bert_tokenizer(cleaned_text, **tokenizer_kwargs)
            if self._torch is None:
                return None
            encoded = {k: v.to(self.bert_device) for k, v in encoded.items()}
            with self._torch.no_grad():
                outputs = self.bert_model(**encoded)
                probs = self._torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_idx = probs.argmax(dim=-1).item()
            raw_label = self.bert_id2label.get(pred_idx)
            return self._map_label_to_emotion(raw_label)
        except Exception as e:
            print(f"BERT analysis failed: {e}", file=sys.stderr)
            return None
    
    def _analyze_with_ml(self, text):
        if self.text_model is not None and self.text_vectorizer is not None:
            try:
                cleaned_text = self._clean_text(text)
                if not cleaned_text.strip():
                    return None
                if len(cleaned_text.split()) >= 3:
                    text_vector = self.text_vectorizer.transform([cleaned_text])
                    prediction = self.text_model.predict(text_vector)[0]
                    return self._map_label_to_emotion(prediction)
            except Exception as e:
                print(f"ML analysis failed: {e}", file=sys.stderr)
        return None

    def _analyze_with_emoji(self, text):
        if not text or not self.emoji_emotion_weights:
            return {}
        emoji_scores = Counter()
        for emoji_symbol, emotion_weights in self.emoji_emotion_weights.items():
            occurrences = text.count(emoji_symbol)
            if occurrences <= 0:
                continue
            for emotion_label, base_weight in emotion_weights.items():
                emoji_scores[emotion_label] += base_weight * occurrences
        return dict(emoji_scores)

    # --- 위험 감지 함수 ---
    def check_mind_care_needed(self, text):
        if not self.risk_keywords:
            return False
        text_lower = text.lower()
        risk_count = 0
        for word in self.risk_keywords:
            if word in text_lower:
                risk_count += 1
        if risk_count >= 1:
            return True
        return False

    # --- 인물 분석 함수 ---
    def analyze_people(self, text):
        if not self.name_set: return []
        people_emotions = {}
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if not sentence.strip(): continue
            words = re.findall(r'\b[A-Z][a-z]*\b', sentence)
            found_names = []
            for word in words:
                if word.lower() in self.name_set:
                    found_names.append(word)
            if found_names:
                emotion = self.analyze_emotion(sentence)
                for name in found_names:
                    if name not in people_emotions:
                        people_emotions[name] = []
                    people_emotions[name].append(emotion)
        results = []
        for name, emotions in people_emotions.items():
            most_common_emotion = Counter(emotions).most_common(1)[0][0]
            color_info = self.get_color_recommendation(most_common_emotion)
            results.append({
                'name': name,
                'emotion': most_common_emotion,
                'color': color_info['color_hex'],
                'count': len(emotions)
            })
        return results
    
    def hsv_to_hex(self, h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    def get_color_from_dataset(self, emotion):
        if self.emotion_colors_data and emotion in self.emotion_colors_data:
            color_data = self.emotion_colors_data[emotion]
            if len(color_data) > 0:
                selected_hsv = random.choice(color_data)
                h, s, v = selected_hsv
                corrected_hsv = self._adjust_color_tone(h, s, v, emotion)
                hex_color = self.hsv_to_hex(*corrected_hsv)
                return {'hsv': corrected_hsv, 'hex': hex_color, 'from_dataset': True}
        if emotion in self.emotion_colors:
            default_color = self.emotion_colors[emotion]['color']
            return {'hsv': None, 'hex': default_color, 'from_dataset': False}
        return {'hsv': None, 'hex': '#FFD700', 'from_dataset': False}
    
    def _adjust_color_tone(self, h, s, v, emotion):
        negative_emotions = ['Anger', 'Disgust', 'Fear', 'Sadness']
        positive_emotions = ['Happiness', 'Surprise']
        if emotion in negative_emotions:
            adjusted_s = max(0.2, min(0.7, s * 0.7))
            adjusted_v = max(0.2, min(0.6, v * 0.6))
        elif emotion in positive_emotions:
            adjusted_s = max(0.1, min(0.4, s * 0.5))
            adjusted_v = max(0.8, min(1.0, v * 0.2 + 0.8))
        else:
            adjusted_s = max(0.1, min(0.8, s))
            adjusted_v = max(0.3, min(1.0, v))
        return (h, adjusted_s, adjusted_v)
    
    def get_color_name_from_hsv(self, h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        if r > 0.8 and g > 0.8 and b < 0.3: return "Yellow"
        elif r > 0.7 and g < 0.3 and b < 0.3: return "Red"
        elif r < 0.3 and g > 0.7 and b < 0.3: return "Green"
        elif r < 0.3 and g < 0.3 and b > 0.7: return "Blue"
        elif r > 0.7 and g < 0.5 and b > 0.7: return "Pink"
        elif r < 0.3 and g < 0.3 and b < 0.3: return "Grey"
        elif r > 0.5 and g > 0.5 and b > 0.5: return "Bright Color"
        else: return "Neutral Tone"
    
    def analyze_emotion_and_color(self, diary_entry, show_visualization=False):
        # 1. 감정 및 색상 분석
        emotion, emotion_flow, gradient = self.analyze_emotion_with_flow(diary_entry)
        result = self.get_color_recommendation(emotion)
        
        # 2. 인물 분석
        people_result = self.analyze_people(diary_entry)
        
        # 3. 우울증 위험 감지
        needs_care = self.check_mind_care_needed(diary_entry)
        
        print(f"🤖 AI Analysis: {emotion} / People: {len(people_result)} / Risk: {needs_care}")
        
        return {
            'emotion': emotion,
            'color_hex': result['color_hex'],
            'color_name': result['color_name'],
            'tone': result['tone'],
            'people': people_result,
            'emotion_flow': emotion_flow,
            'emotion_gradient': gradient,
            'source': result.get('source', 'default'),
            'needs_care': needs_care
        }
    
    def get_color_recommendation(self, emotion):
        color_info = self.get_color_from_dataset(emotion)
        if color_info['from_dataset'] and color_info['hsv']:
            h, s, v = color_info['hsv']
            color_name = self.get_color_name_from_hsv(h, s, v)
            negative_emotions = ['Anger', 'Disgust', 'Fear', 'Sadness']
            tone = "Calm and Dark" if emotion in negative_emotions else "Bright and Pastel"
            return {'emotion': emotion, 'color_hex': color_info['hex'], 'color_name': color_name, 'tone': tone, 'source': 'dataset'}
        else:
            if emotion in self.emotion_colors:
                color_data = self.emotion_colors[emotion]
                return {'emotion': emotion, 'color_hex': color_data['color'], 'color_name': color_data['color_name'], 'tone': color_data['tone'], 'source': 'default'}
            return {'emotion': 'Happiness', 'color_hex': self.emotion_colors['Happiness']['color'], 'color_name': self.emotion_colors['Happiness']['color_name'], 'tone': self.emotion_colors['Happiness']['tone'], 'source': 'fallback'}


# Global Instance
improved_analyzer = ImprovedEmotionAnalyzer()

def analyze_emotion_and_color(diary_entry, show_visualization=False):
    return improved_analyzer.analyze_emotion_and_color(diary_entry, show_visualization)

def check_mind_care_needed(text):
    return improved_analyzer.check_mind_care_needed(text)

if __name__ == "__main__":
    # Test Cases
    test_cases = [
        "I met James today and he was so funny. But Sarah made me very angry.",
        "The sun is shining, I aced the test. What a perfect day.",
        "I feel like I want to die. Everything is hopeless and I hate myself.",
    ]
    
    print("🧪 English Only Model Test:")
    for text in test_cases:
        result = analyze_emotion_and_color(text)
        print(f"Text: {text[:50]}...")
        print(f"Main Emotion: {result['emotion']}")
        print(f"People: {result['people']}")
        print(f"Needs Care: {result['needs_care']}")
        print("-" * 30)

    server_url = os.environ.get('CLIO_SERVER_URL', 'http://127.0.0.1:5000')
    print(f"🌐 Connect via: {server_url}")
