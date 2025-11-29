# -*- coding: utf-8 -*-
"""
English Emotion, Color & Person Analysis Model
"""

import pandas as pd
import numpy as np
import os
import sys
import re
import colorsys
import random
import time
import math
from collections import Counter # [추가] 빈도수 계산용
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class ImprovedEmotionAnalyzer:
    def __init__(self):
        
        self.text_model = None
        self.text_vectorizer = None
        self.color_model = None
        self.color_encoder = None
        self.name_set = set() # [추가] 이름 데이터 저장소
        
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
        
        # 2. Load Color Model
        self._load_color_model()
        
        # 3. Load Color Dataset
        self._load_color_dataset()

        # 4. [추가] Load Name Dataset
        self._load_name_dataset()
        
        print("✅ All models loaded successfully!")

    # --- [추가됨] 이름 데이터셋 로드 함수 ---
    def _load_name_dataset(self):
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'name_gender_dataset.csv')
            
            if not os.path.exists(csv_path):
                print(f"⚠️ Name dataset not found: {csv_path}")
                return
            
            print("👥 Loading name dataset...")
            df = pd.read_csv(csv_path)
            # 검색 속도를 위해 소문자로 변환하여 집합(Set)에 저장
            self.name_set = set(df['Name'].str.lower().values)
            print(f"✅ Loaded {len(self.name_set)} names.")
            
        except Exception as e:
            print(f"❌ Failed to load name dataset: {e}")

    def _load_color_dataset(self):
        """Load color dataset csv"""
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
        """Text cleaning (English only)"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Keep only alphabets and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _load_text_model(self):
        """Load text emotion analysis model"""
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'emotion_sentimen_dataset.csv')
            
            if not os.path.exists(csv_path):
                print(f"⚠️ Dataset not found: {csv_path}")
                return
            
            print("📊 Loading text dataset...")
            df = pd.read_csv(csv_path, encoding='latin1')
            
            df_renamed = df.rename(columns={'Emotion': 'label', 'text': 'text'})
            df_clean = df_renamed[['text', 'label']].copy()
            
            df_clean['text'] = df_clean['text'].apply(self._clean_text)
            df_clean.dropna(subset=['text', 'label'], inplace=True)
            df_final = df_clean[df_clean['text'] != ""]
            
            label_map = {
                'happiness': 'joy', 'fun': 'joy', 'enthusiasm': 'joy', 'relief': 'joy', 'love': 'joy',
                'sadness': 'sadness', 'empty': 'sadness', 'boredom': 'sadness',
                'anger': 'anger',
                'worry': 'fear',
                'hate': 'disgust',
                'surprise': 'surprise'
            }
            
            df_final['label'] = df_final['label'].map(label_map)
            df_final = df_final.dropna(subset=['label'])

            dataset_size = len(df_final)
            label_counts = Counter(df_final['label'])
            if dataset_size == 0 or len(label_counts) < 2:
                print("⚠️ Not enough labeled samples to train the text model.")
                return

            print(f"📦 Dataset ready: {dataset_size} samples")
            for label, count in label_counts.items():
                print(f"   • {label}: {count}")

            X = df_final['text']
            y = df_final['label']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_tfidf = self.text_vectorizer.fit_transform(X_train)
            X_test_tfidf = self.text_vectorizer.transform(X_test)

            self.text_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            start_time = time.perf_counter()
            self.text_model.fit(X_train_tfidf, y_train)
            train_time = time.perf_counter() - start_time
            print(f"⏱️ Text model training complete in {train_time:.2f}s")

            try:
                y_pred = self.text_model.predict(X_test_tfidf)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"📈 Validation accuracy: {accuracy:.3f}")
                print(classification_report(y_test, y_pred, zero_division=0))

                test_df = pd.DataFrame({'text': X_test.reset_index(drop=True),
                                        'label': y_test.reset_index(drop=True)})
                sample_size = min(100, len(test_df))
                if sample_size == 0:
                    print("⚠️ No samples available for random evaluation.")
                else:
                    sample_df = test_df.sample(n=sample_size)
                    sample_vectors = self.text_vectorizer.transform(sample_df['text'])
                    sample_pred = self.text_model.predict(sample_vectors)
                    sample_accuracy = accuracy_score(sample_df['label'], sample_pred)
                    print(f"🎯 Random {sample_size} sample accuracy: {sample_accuracy:.3f}")
            except Exception as eval_error:
                print(f"⚠️ Could not compute validation metrics: {eval_error}")
            
        except Exception as e:
            print(f"❌ Failed to load text model: {e}")
            self.text_model = None
            self.text_vectorizer = None
    
    def _load_color_model(self):
        """Load color model"""
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
        """Returns dominant emotion, flow metadata, and gradient string"""
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
        """English keyword analysis"""
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
        """Analysis using ML model"""
        if self.text_model is not None and self.text_vectorizer is not None:
            try:
                cleaned_text = self._clean_text(text)
                
                if not cleaned_text.strip():
                    return None

                if len(cleaned_text.split()) >= 3:
                    text_vector = self.text_vectorizer.transform([cleaned_text])
                    prediction = self.text_model.predict(text_vector)[0]
                    
                    emotion_map = {
                        'joy': 'Happiness',
                        'sadness': 'Sadness',
                        'anger': 'Anger',
                        'fear': 'Fear',
                        'disgust': 'Disgust',
                        'surprise': 'Surprise'
                    }
                    return emotion_map.get(prediction, 'Happiness')
            except Exception as e:
                print(f"ML analysis failed: {e}", file=sys.stderr)
        
        return None

    # --- [추가됨] 인물 분석 함수 ---
    def analyze_people(self, text):
        if not self.name_set: return []
        
        people_emotions = {} # { 'James': ['Happiness', 'Anger'], ... }
        
        # 1. 문장 단위로 나누기 (. ! ? 기준)
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if not sentence.strip(): continue
            
            # 2. 대문자로 시작하는 단어 찾기 (이름 후보)
            words = re.findall(r'\b[A-Z][a-z]*\b', sentence)
            
            found_names = []
            for word in words:
                # 3. 이름 데이터셋에 있는지 확인
                if word.lower() in self.name_set:
                    # 너무 짧은 단어나 일반적인 접속사 등은 제외할 수 있음 (여기선 단순 포함)
                    found_names.append(word)
            
            # 4. 이름이 발견되면 그 문장의 감정 분석
            if found_names:
                emotion = self.analyze_emotion(sentence)
                for name in found_names:
                    if name not in people_emotions:
                        people_emotions[name] = []
                    people_emotions[name].append(emotion)
        
        # 5. 결과 정리 (가장 많이 느낀 감정 선택)
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
        """Returns color names in English"""
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
        emotion, emotion_flow, gradient = self.analyze_emotion_with_flow(diary_entry)
        result = self.get_color_recommendation(emotion)
        
        # [중요] 인물 분석 실행
        people_result = self.analyze_people(diary_entry)
        
        print(f"🤖 AI Analysis: {emotion} / People found: {len(people_result)}")
        
        return {
            'emotion': emotion,
            'color_hex': result['color_hex'],
            'color_name': result['color_name'],
            'tone': result['tone'],
            'people': people_result, # 인물 분석 결과 추가
            'emotion_flow': emotion_flow,
            'emotion_gradient': gradient,
            'source': result.get('source', 'default')
        }
    
    def get_color_recommendation(self, emotion):
        """Returns tone descriptions in English"""
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

if __name__ == "__main__":
    # Test Cases with Names
    test_cases = [
        "I met James today and he was so funny. But Sarah made me very angry.",
        "The sun is shining, I aced the test, and my favorite song just came on. What a perfect day.",
    ]
    
    print("🧪 English Only Model Test:")
    for text in test_cases:
        result = analyze_emotion_and_color(text)
        print(f"Text: {text[:50]}...")
        print(f"Main Emotion: {result['emotion']}")
        print(f"People: {result['people']}")
        print()
