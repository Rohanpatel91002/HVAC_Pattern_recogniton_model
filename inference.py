import os
import json
import pickle
import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame, scaler) -> pd.DataFrame:
   
    df = df.copy()
    
    # Parse datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    df = df.sort_index()
    
    # Select relevant columns
    cols = ['T1', 'T2', 'T3', 'RH_1', 'RH_2', 'RH_3', 'T_out', 'RH_out', 
            'Press_mm_hg', 'Windspeed', 'Visibility', 'Tdewpoint', 'HVAC_Energy']
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]
    
    
    df = df.ffill().bfill()
    
    
    df_normalized = pd.DataFrame(
        scaler.transform(df),
        index=df.index,
        columns=df.columns
    )
    
    return df_normalized


def create_sequences(df: pd.DataFrame, seq_length: int = 144, stride: int = 6):
    
    data = df.values
    timestamps = df.index
    
    sequences = []
    seq_times = []
    
    for i in range(0, len(data) - seq_length + 1, stride):
        seq = data[i:i + seq_length]
        sequences.append(seq)
        seq_times.append(timestamps[i + seq_length - 1])
    
    return np.array(sequences), np.array(seq_times)



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    # Time features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_office_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['weekday'] < 5)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # Season
    df['season'] = df['month'].apply(lambda m: 
        0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3)
    
    # Aggregates
    temp_cols = [c for c in ['T1', 'T2', 'T3'] if c in df.columns]
    rh_cols = [c for c in ['RH_1', 'RH_2', 'RH_3'] if c in df.columns]
    
    if temp_cols:
        df['T_indoor_mean'] = df[temp_cols].mean(axis=1)
    if rh_cols:
        df['RH_indoor_mean'] = df[rh_cols].mean(axis=1)
    
    return df


class BehavioralClusterModel:
   
    
    def __init__(self, n_clusters=6, method='kmeans', random_state=42):
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.scaler = None
        self.model = None
        self.pattern_names = []
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(n_clusters=data['n_clusters'], method=data['method'])
        model.model = data['model']
        model.scaler = data['scaler']
        model.pattern_names = data['pattern_names']
        return model


def label_patterns(cluster_labels: np.ndarray, 
                   timestamps: np.ndarray,
                   pattern_names: list[str]) -> pd.DataFrame:
   
    pattern_mapping = {i: name for i, name in enumerate(pattern_names)}
    
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    results = []
    for i, (ts, cluster_id) in enumerate(zip(timestamps, cluster_labels)):
        ts = pd.Timestamp(ts)
        hour = ts.hour
        weekday = ts.weekday()
        
        # Time period
        if 5 <= hour < 9:
            period = 'Early Morning'
        elif 9 <= hour < 12:
            period = 'Morning'
        elif 12 <= hour < 14:
            period = 'Midday'
        elif 14 <= hour < 18:
            period = 'Afternoon'
        elif 18 <= hour < 22:
            period = 'Evening'
        else:
            period = 'Night'
        
        day_type = 'Weekend' if weekday >= 5 else 'Weekday'
        context = f"{weekday_names[weekday]} {period} ({day_type})"
        
        results.append({
            'timestamp': ts,
            'cluster_id': int(cluster_id),
            'pattern_name': pattern_mapping.get(cluster_id, f'Pattern_{cluster_id}'),
            'hour': hour,
            'weekday': weekday,
            'month': ts.month,
            'context': context
        })
    
    return pd.DataFrame(results)



def model_fn(model_dir: str):
    
    import tensorflow as tf
    
    # Load the Keras model directly (.keras format)
    encoder = tf.keras.models.load_model(os.path.join(model_dir, 'temporal_encoder.keras'))
    cluster_model = BehavioralClusterModel.load(os.path.join(model_dir, 'clustering_model.pkl'))
    
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    return {
        'encoder': encoder,
        'cluster_model': cluster_model,
        'scaler': scaler
    }


def input_fn(request_body: str, request_content_type: str):

    if request_content_type == 'application/json':
        data = json.loads(request_body)
        df = pd.DataFrame(data)
    elif request_content_type == 'text/csv':
        from io import StringIO
        df = pd.read_csv(StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    return df



def predict_fn(input_data: pd.DataFrame, model: dict):
    
    encoder = model['encoder']
    cluster_model = model['cluster_model']
    scaler = model['scaler']
    
    # Preprocess
    df_normalized = preprocess_data(input_data, scaler)
    
    
    sequences, seq_times = create_sequences(df_normalized, seq_length=144, stride=6)
    
    if len(sequences) == 0:
        return pd.DataFrame({'error': ['Not enough data. Need at least 144 rows (24 hours).']})
    
    
    embeddings = encoder.predict(sequences, verbose=0)
    
    # Get features for clustering
    df_raw = input_data.copy()
    if 'date' in df_raw.columns:
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.set_index('date')
    
    df_features = engineer_features(df_raw)
    df_features = df_features.reindex(seq_times).ffill().bfill()
    
    cluster_features = df_features[['hour', 'weekday', 'is_weekend', 'is_office_hours',
                                     'is_night', 'season', 'T_indoor_mean', 'RH_indoor_mean',
                                     'HVAC_Energy']].values
    
    
    combined = np.hstack([embeddings, cluster_features])
    
   
    cluster_labels = cluster_model.predict(combined)
    
    # Create labeled output
    labels_df = label_patterns(cluster_labels, seq_times, cluster_model.pattern_names)
    
    return labels_df


def output_fn(prediction: pd.DataFrame, accept: str):

    if accept == 'application/json':
        return prediction.to_json(orient='records', date_format='iso')
    elif accept == 'text/csv':
        return prediction.to_csv(index=False)
    else:
        return prediction.to_json(orient='records', date_format='iso')



def run_inference(data_path: str, model_dir: str = 'models') -> pd.DataFrame:
    
    df = pd.read_csv(data_path)
    models = model_fn(model_dir)
  
    results = predict_fn(df, models)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <data_file.csv> [model_dir]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    
    print(f"Running inference on {data_file}...")
    results = run_inference(data_file, model_dir)
    
    print(f"\nGenerated {len(results)} predictions:")
    print(results.to_string())
