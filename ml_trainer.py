import numpy as np
from sklearn.ensemble import IsolationForest

def extract_features(tx, history_txs):
    """
    Extracts numerical features from a transaction and user history for ML.
    """
    # 1. Transaction Amount
    amount = tx["amount"]
    
    # 2. Recent Transaction Count (last 60s)
    # history_txs is expected to include the current tx
    # Using max(0...) just as a safeguard
    recent_txs = [t for t in history_txs if max(0, tx["timestamp"] - t["timestamp"]) < 60]
    recent_count = len(recent_txs)
    
    # 3. Unique Location Count (last 60s)
    locations = set(t["location"] for t in recent_txs)
    unique_locations = len(locations)
    
    return [amount, recent_count, unique_locations]

class MLTrainer:
    @staticmethod
    def train_isolation_forest(transactions):
        """
        Trains an Isolation Forest model on historical transactions.
        """
        if not transactions:
            return None
            
        print(f"Training Machine Learning model on {len(transactions)} records...")
            
        # Reconstruct histories to accurately pull features
        user_histories = {}
        features_matrix = []
        
        for tx in transactions:
            user_id = tx["user_id"]
            if user_id not in user_histories:
                user_histories[user_id] = []
            
            history = user_histories[user_id]
            history.append(tx)
            
            # Extract features for this tx state
            feats = extract_features(tx, history)
            features_matrix.append(feats)
            
            # Maintain same buffer size as analyzer
            if len(history) > 10:
                history.pop(0)
                
        X = np.array(features_matrix)
        
        # Fit the Isolation Forest
        # contamination represents the percentage of expected fraud
        model = IsolationForest(contamination=0.04, random_state=42)
        model.fit(X)
        
        print("Model training complete.")
        return model
