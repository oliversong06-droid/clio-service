# -*- coding: utf-8 -*-
"""
English Emotion, Color & Person Analysis Model + Depression Risk Detection
(LSA + fallback TF-IDF/LogisticRegression)
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
from collections import Counter

import joblib  # LSA 파이프라인 로딩용

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


class ImprovedEmotionAnalyzer:
    def __init__(self):

        self.text_model = None          # LSA 파이프라인 or 기존 ML 모델
        self.text_vectorizer = None     # 레거시 TF-IDF용 (LSA 파이프라인이면 None)
        self.color_model = None
        self.color_encoder = None
        self.name_set = set()      # 이름 데이터 저장소
        self.risk_keywords = set() # 위험 어휘 데이터 저장소

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

        # 1. Load Text Model (LSA 우선)
        self._load_text_model()

        # 2. Load Color Model
        self._load_color_model()

        # 3. Load Color Dataset
        self._load_color_dataset()

        # 4. Load Name Dataset
        self._load_name_dataset()

        # 5. Load Risk Dataset (CSV Version)
        self._load_risk_dataset()

        print("✅ All models loaded successfully!")

    # --- 위험 어휘 데이터셋 로드 함수 (CSV 버전) ---
    def _load_risk_dataset(self):
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'depression_risk_words.csv')

            if not os.path.exists(csv_path):
                print(f"⚠️ Risk dataset not found: {csv_path}")
                self.risk_keywords = {"suicide", "kill myself", "want to die", "hopeless"}
                return

            print("🛡️ Loading risk dataset (CSV)...")

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    if 'risk_level' in row and row['risk_level'] in ['high', 'medium']:
                        if 'word' in row:
                            self.risk_keywords.add(row['word'].lower())
                            count += 1

            print(f"✅ Loaded {count} risk keywords.")

        except Exception as e:
            print(f"❌ Failed to load risk dataset: {e}")
            self.risk_keywords = {"suicide", "die"}

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

    def _load_text_model(self):
        """
        1순위: 미리 학습된 LSA 파이프라인(lsa_emotion_model.pkl) 로드
        2순위: 없으면 기존 TF-IDF + LogisticRegression으로 학습 (fallback)
        """
        base_dir = os.path.dirname(__file__)
        lsa_model_path = os.path.join(base_dir, "lsa_emotion_model.pkl")

        # 1) LSA 파이프라인 로드 시도
        try:
            if os.path.exists(lsa_model_path):
                print(f"🧠 Loading LSA emotion model from: {lsa_model_path}")
                self.text_model = joblib.load(lsa_model_path)
                self.text_vectorizer = None  # 파이프라인 내부에 TF-IDF 포함
                print("✅ LSA emotion model loaded successfully.")
                return
            else:
                print(f"⚠️ LSA model file not found: {lsa_model_path}")
                print("   → Falling back to legacy TF-IDF + LogisticRegression training...")
        except Exception as e:
            print(f"❌ Failed to load LSA model, fallback to legacy training: {e}")

        # 2) (fallback) 기존 TF-IDF + LogisticRegression 학습
        try:
            csv_path = os.path.join(base_dir, 'emotion_sentimen_dataset.csv')

            if not os.path.exists(csv_path):
                print(f"⚠️ Dataset not found: {csv_path}")
                return

            print("📊 Loading text dataset for legacy model...")
            df = pd.read_csv(csv_path, encoding='latin1')

            df_renamed = df.rename(columns={'Emotion': 'label', 'text': 'text'})
            df_clean = df_renamed[['text', 'label']].copy()

            df_clean['text'] = df_clean['text'].apply(self._clean_text)
            df_clean.dropna(subset=['text', 'label'], inplace=True)
            df_final = df_clean[df_clean['text'] != ""]

            label_map = {
                'happiness': 'joy', 'fun': 'joy', 'enthusiasm': 'joy', 'relief': 'joy', 'love': 'joy',
                'sadness': 'sadness', 'empty': 'sadness', 'boredom': 'sadness',
                'anger': 'anger', 'worry': 'fear', 'hate': 'disgust', 'surprise': 'surprise'
            }
            df_final['label'] = df_final['label'].str.lower().map(label_map)
            df_final = df_final.dropna(subset=['label'])

            dataset_size = len(df_final)
            label_counts = Counter(df_final['label'])
            if dataset_size == 0 or len(label_counts) < 2:
                print("⚠️ Not enough labeled samples to train the text model.")
                return

            print(f"📦 Dataset ready: {dataset_size} samples")

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

            y_pred = self.text_model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"📈 Legacy TF-IDF model validation accuracy: {accuracy:.3f}")

        except Exception as e:
            print(f"❌ Failed to load or train text model: {e}")
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

    # -------------------- 감정 분석 메인 로직 --------------------

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
        blocks = []
        for i in range(0, len(sentences), 2):
            pair = ' '.join(sentences[i:i+2])
            if pair:
                blocks.append(pair)
        return blocks

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
        english_result = self._analyze_english_emotion(text)
        if english_result:
            return english_result
        ml_result = self._analyze_with_ml(text)
        if ml_result:
            return ml_result
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

    def _analyze_with_ml(self, text):
        """
        LSA 파이프라인(또는 기존 TF-IDF + LogisticRegression)을 이용해 감정 예측
        """
        if self.text_model is None:
            return None

        try:
            cleaned_text = self._clean_text(text)
            if not cleaned_text.strip():
                return None
            if len(cleaned_text.split()) < 3:
                return None

            prediction = None
            # 먼저 파이프라인처럼 raw text를 바로 넣어본다 (LSA 파이프라인 케이스)
            try:
                prediction = self.text_model.predict([cleaned_text])[0]
            except Exception:
                # 파이프라인이 아니면, 레거시 TF-IDF + LogisticRegression 구조일 수 있음
                if self.text_vectorizer is not None:
                    text_vector = self.text_vectorizer.transform([cleaned_text])
                    prediction = self.text_model.predict(text_vector)[0]
                else:
                    return None

            if prediction is None:
                return None

            pred_label = str(prediction).lower()

            emotion_map = {
                'happiness': 'Happiness',
                'fun': 'Happiness',
                'enthusiasm': 'Happiness',
                'relief': 'Happiness',
                'love': 'Happiness',
                'neutral': 'Happiness',

                'sadness': 'Sadness',
                'empty': 'Sadness',
                'boredom': 'Sadness',

                'anger': 'Anger',

                'worry': 'Fear',
                'anxiety': 'Fear',

                'hate': 'Disgust',

                'surprise': 'Surprise'
            }

            return emotion_map.get(pred_label, 'Happiness')

        except Exception as e:
            print(f"ML analysis failed: {e}", file=sys.stderr)
            return None

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
        if not self.name_set:
            return []
        people_emotions = {}
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if not sentence.strip():
                continue
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
        if r > 0.8 and g > 0.8 and b < 0.3:
            return "Yellow"
        elif r > 0.7 and g < 0.3 and b < 0.3:
            return "Red"
        elif r < 0.3 and g > 0.7 and b < 0.3:
            return "Green"
        elif r < 0.3 and g < 0.3 and b > 0.7:
            return "Blue"
        elif r > 0.7 and g < 0.5 and b > 0.7:
            return "Pink"
        elif r < 0.3 and g < 0.3 and b < 0.3:
            return "Grey"
        elif r > 0.5 and g > 0.5 and b > 0.5:
            return "Bright Color"
        else:
            return "Neutral Tone"

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
            return {
                'emotion': emotion,
                'color_hex': color_info['hex'],
                'color_name': color_name,
                'tone': tone,
                'source': 'dataset'
            }
        else:
            if emotion in self.emotion_colors:
                color_data = self.emotion_colors[emotion]
                return {
                    'emotion': emotion,
                    'color_hex': color_data['color'],
                    'color_name': color_data['color_name'],
                    'tone': color_data['tone'],
                    'source': 'default'
                }
            return {
                'emotion': 'Happiness',
                'color_hex': self.emotion_colors['Happiness']['color'],
                'color_name': self.emotion_colors['Happiness']['color_name'],
                'tone': self.emotion_colors['Happiness']['tone'],
                'source': 'fallback'
            }


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
