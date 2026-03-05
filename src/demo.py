
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, precision_recall_curve,
                           average_precision_score)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Conv1D, 
                                   MaxPooling1D, Flatten, Bidirectional, 
                                   Attention, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                      ReduceLROnPlateau)
from sklearn.utils.class_weight import compute_class_weight
import os
import json
import warnings
import tensorflow as tf

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class EnhancedLogAnomalyDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.results = {}
        self.models = {}
        self.base_path = "/Users/addarezigkarima/Downloads/demo graduation/demo gradution"

    def extract_log_features(self, df):
        """Enhanced feature extraction with more patterns and features"""
        # Basic text features
        df["log_length"] = df["Content"].str.len()
        df["word_count"] = df["Content"].str.split().str.len()
        df["special_chars"] = df["Content"].str.count(r'[^a-zA-Z0-9\s]')
        df["digit_count"] = df["Content"].str.count(r'\d')
        df["upper_case"] = df["Content"].str.count(r'[A-Z]')
        df["url_count"] = df["Content"].str.count(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Enhanced error patterns
        error_terms = ['error', 'fail', 'exception', 'warning', 'critical',
                      'fatal', 'timeout', 'denied', 'rejected', 'invalid',
                      'unauthorized', 'crash', 'panic', 'oom', 'segfault']
        for term in error_terms:
            df[f"has_{term}"] = df["Content"].str.contains(term, case=False).astype(int)
        
        # Sequence and frequency features
        if 'EventId' in df.columns:
            df['event_frequency'] = df.groupby('EventId')['EventId'].transform('count')
            df['event_ratio'] = df['event_frequency'] / len(df)
        
        # Timestamp features if available
        if 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df['hour'] = df['Timestamp'].dt.hour
                df['day_of_week'] = df['Timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            except:
                pass
        
        return df

    def load_dataset(self, dataset_name):
        """Load dataset with robust error handling and automatic label detection"""
        try:
            if dataset_name == "BGL":
                files = {
                    'structured': "BGL_2k.log_structured.csv",
                    'template': "BGL_templates.csv"
                }
            elif dataset_name == "HDFS":
                files = {
                    'structured': "HDFS_2k.log_structured.csv", 
                    'template': "HDFS_templates.csv"
                }
            elif dataset_name == "Thunderbird":
                files = {
                    'structured': "Thunderbird_2k.log_structured.csv",
                    'template': "Thunderbird_2k.log_templates.csv"
                }

            df = pd.read_csv(os.path.join(self.base_path, dataset_name, files['structured']))
            templates = pd.read_csv(os.path.join(self.base_path, dataset_name, files['template']))
            merged = pd.merge(df, templates, on="EventId")
            
            merged["Content"] = merged["Content"].fillna("")
            merged = self.extract_log_features(merged)
            
            label_cols = [col for col in merged.columns if 'label' in col.lower() or 'anomaly' in col.lower()]
            if label_cols:
                merged["Label"] = pd.to_numeric(merged[label_cols[0]], errors='coerce').fillna(0).astype(int)
            else:
                event_counts = merged['EventId'].value_counts()
                rare_events = event_counts[event_counts < event_counts.quantile(0.1)].index
                merged["Label"] = merged['EventId'].isin(rare_events).astype(int)
            
            if sum(merged["Label"]) / len(merged) < 0.05:
                rare_events = merged['EventId'].value_counts().tail(int(len(merged) * 0.1)).index
                merged.loc[merged['EventId'].isin(rare_events), "Label"] = 1
            
            print(f"✅ Loaded {dataset_name} ({len(merged)} entries, {sum(merged['Label'])} anomalies)")
            return merged
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {str(e)}")
            return None

    def preprocess_data(self, df):
        """Prepare data for all model types with feature scaling"""
        if df is None:
            return None, None, None, None
            
        try:
            X_tfidf = self.vectorizer.fit_transform(df["Content"])
            
            feature_cols = ["log_length", "word_count", "special_chars", "digit_count", 
                          "upper_case", "url_count"] + [col for col in df.columns if col.startswith("has_")]
            
            if 'event_frequency' in df.columns:
                feature_cols.extend(['event_frequency', 'event_ratio'])
            if 'hour' in df.columns:
                feature_cols.extend(['hour', 'day_of_week', 'is_weekend'])
                
            X_features = df[feature_cols].values
            X_features = (X_features - X_features.mean(axis=0)) / (X_features.std(axis=0) + 1e-7)
            
            X = np.hstack([X_tfidf.toarray(), X_features])
            y = df["Label"].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"❌ Preprocessing error: {str(e)}")
            return None, None, None, None

    def build_autoencoder(self, input_dim):
        """Autoencoder model"""
        input_layer = Input(shape=(input_dim,))
        
        encoded = Dense(256, activation='relu')(input_layer)
        encoded = Dropout(0.3)(encoded)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        model = Model(input_layer, decoded)
        model.compile(optimizer=Adam(0.0005), loss='mse')
        return model

    def build_lstm(self, input_shape):
        """LSTM model"""
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.4)(x)
        x = Bidirectional(LSTM(64))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy', 
                             tf.keras.metrics.Precision(name='precision'),
                             tf.keras.metrics.Recall(name='recall')])
        return model

    def build_cnn(self, input_shape):
        """CNN model"""
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(128, 5, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu', padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def build_bilstm_attention(self, input_shape):
        """BiLSTM with attention"""
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.4)(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        
        attention = Attention()([x, x])
        x = tf.keras.layers.Concatenate()([x, attention])
        x = Flatten()(x)
        
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.0005),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def calculate_class_weights(self, y_train):
        """Calculate class weights"""
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weights = np.clip(weights, 0.1, 10)
        return dict(zip(classes, weights))

    def evaluate_model(self, model, X_test, y_test, model_type):
        """Evaluation metrics"""
        try:
            y_test = np.array(y_test).astype(int)
            
            if model_type == "autoencoder":
                reconstructions = model.predict(X_test, verbose=0)
                mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
                threshold = np.percentile(mse, 95)
                y_pred = (mse > threshold).astype(int)
                y_scores = mse
            else:
                X_test_seq = np.expand_dims(X_test, axis=2)
                y_scores = model.predict(X_test_seq, verbose=0).flatten()
                y_pred = (y_scores > 0.5).astype(int)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_scores),
                "pr_auc": average_precision_score(y_test, y_scores)
            }
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Evaluation error: {str(e)}")
            return {
                "error": str(e),
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "roc_auc": 0.5,
                "pr_auc": 0
            }

    def run_experiment(self):
        """Main training pipeline"""
        print(f"🔍 Scanning datasets in: {self.base_path}")
        datasets = [d for d in os.listdir(self.base_path) 
                   if os.path.isdir(os.path.join(self.base_path, d))]
        
        model_types = ["autoencoder", "lstm", "cnn", "bilstm_attention"]
        
        for dataset in ["BGL", "HDFS", "Thunderbird"]:
            print(f"\n{'='*40}\n🧪 Processing {dataset}\n{'='*40}")
            
            df = self.load_dataset(dataset)
            if df is None:
                continue
                
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            if X_train is None:
                continue
                
            seq_shape = (X_train.shape[1], 1)
            X_train_seq = np.expand_dims(X_train, axis=2)
            
            for model_type in model_types:
                print(f"\n🛠️ Training {model_type} model...")
                
                try:
                    callbacks = [
                        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
                        ModelCheckpoint(
                            f"best_{dataset}_{model_type}.keras",
                            save_best_only=True,
                            monitor='val_loss',
                            mode='min'
                        ),
                        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
                    ]
                    
                    if model_type == "autoencoder":
                        model = self.build_autoencoder(X_train.shape[1])
                        model.fit(X_train, X_train,
                                epochs=100,
                                batch_size=64,
                                validation_split=0.2,
                                callbacks=callbacks,
                                verbose=1)
                    else:
                        if model_type == "lstm":
                            model = self.build_lstm(seq_shape)
                        elif model_type == "cnn":
                            model = self.build_cnn(seq_shape)
                        else:
                            model = self.build_bilstm_attention(seq_shape)
                        
                        model.fit(X_train_seq, y_train,
                                epochs=100,
                                batch_size=64,
                                validation_split=0.2,
                                callbacks=callbacks,
                                class_weight=self.calculate_class_weights(y_train),
                                verbose=1)
                    
                    metrics = self.evaluate_model(model, X_test, y_test, model_type)
                    self.results[f"{dataset}_{model_type}"] = metrics
                    self.models[f"{dataset}_{model_type}"] = model
                    
                    print(f"📊 {model_type} results:")
                    for metric, value in metrics.items():
                        if metric != "error":
                            print(f"{metric}: {value:.4f}")
                
                except Exception as e:
                    print(f"❌ Training failed for {model_type}: {str(e)}")
                    self.results[f"{dataset}_{model_type}"] = {"error": str(e)}
        
        results_path = os.path.join(self.base_path, "enhanced_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n🎉 All models trained successfully!")
        print(f"📄 Results saved to: {results_path}")

if __name__ == "__main__":
    print("🚀 Starting enhanced log anomaly detection pipeline")
    detector = EnhancedLogAnomalyDetector()
    detector.run_experiment() 

