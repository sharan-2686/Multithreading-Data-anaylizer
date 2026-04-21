import threading
import math
import numpy as np
from ml_trainer import extract_features

class FraudAnalyzer:
    """
    Analyzes transactions for fraudulent behavior in a thread-safe manner.
    """
    def __init__(self, ml_model=None):
        self.lock = threading.Lock()
        self.ml_model = ml_model
        
        # Thread tracking
        self.thread_counts = {}
        self.thread_timeline = []
        
        # User history: Dict[user_id, List[transaction]]
        self.user_history = {}
        
        # Global stats for Welford's online algorithm (mean and variance)
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  
        
        # Aggregator Stats
        self.total_processed = 0
        self.total_flagged = 0
        self.flagged_transactions = []
        
    def _update_global_stats(self, amount):
        self.count += 1
        delta = amount - self.mean
        self.mean += delta / self.count
        delta2 = amount - self.mean
        self.M2 += delta * delta2

    def _get_std_dev(self):
        if self.count < 2:
            return 0.0
        return math.sqrt(self.M2 / (self.count - 1))

    def get_summary(self):
        with self.lock:
            return {
                "total_processed": self.total_processed,
                "total_flagged": self.total_flagged,
                "fraud_rate": f"{(self.total_flagged / self.total_processed * 100):.2f}%" if self.total_processed else "0.00%",
                "mean_amount": round(self.mean, 2),
                "std_dev_amount": round(self._get_std_dev(), 2)
            }

    def analyze(self, tx):
        """
        Analyze a single transaction and update state. Read/writes are locked for thread safety.
        Returns the analysis result.
        """
        import time
        import random
        start_time = time.time()
        
        # Simulate randomly variant I/O network conditions (e.g., 5ms to 25ms database latency).
        # This causes the threads to run asynchronously and creates realistic visible interleaving.
        time.sleep(random.uniform(0.005, 0.025)) 
        
        user_id = tx["user_id"]
        amount = tx["amount"]
        location = tx["location"]
        timestamp = tx["timestamp"]
        
        # 1. State modification and snapshotting inside minimal lock block
        with self.lock:
            # Update global stats safely
            self._update_global_stats(amount)
            
            # Track user history
            if user_id not in self.user_history:
                self.user_history[user_id] = []
            history = self.user_history[user_id]
            history.append(tx)
            
            if len(history) > 10:
                history.pop(0)  # Keep recent history
                
            # Snapshot for unlocked processing
            std_dev_copy = self._get_std_dev()
            mean_copy = self.mean
            count_copy = self.count
            history_copy = list(history)
            
        # 2. Heavy calculations running entirely OUTSIDE the lock concurrently
        risk_score = 0
        reasons = []
        
        # Rule 1: High-value transaction
        if amount > 10000.0:
            risk_score += 50
            reasons.append("High value transaction")
            
        # Rule 4: Statistical anomaly (using z-score)
        if count_copy >= 10 and std_dev_copy > 0:
            z_score = abs(amount - mean_copy) / std_dev_copy
            if z_score > 3.0:
                risk_score += 40
                reasons.append(f"Statistical anomaly (Z-score: {z_score:.2f})")
                
        # Rule 2 & 3: Rapid successive transactions & Location anomaly
        recent_txs = [t for t in history_copy if timestamp - t["timestamp"] < 60]
        
        if len(recent_txs) >= 4:
            risk_score += 30
            reasons.append("Rapid successive transactions")
            
        locations = set(t["location"] for t in recent_txs)
        if len(locations) > 1:
            risk_score += 60
            reasons.append("Location anomaly (impossible travel)")
            
        # ML Integration Rule (This was bottlenecking multi-threading before!)
        if self.ml_model is not None:
            feats = extract_features(tx, history_copy)
            prediction = self.ml_model.predict([feats])[0]
            if prediction == -1:
                risk_score += 50
                reasons.append("Machine Learning Anomaly")
                
        # Aggregate Final Score
        risk_score = min(100, risk_score)
        fraud_flag = risk_score >= 50
        reason_str = ", ".join(reasons) if reasons else "Normal"
        
        result = {
            "transaction_id": tx["transaction_id"],
            "risk_score": risk_score,
            "fraud_flag": fraud_flag,
            "reason": reason_str
        }
        
        
        end_time = time.time()
        
        # 3. Quickly update global metrics inside minimal lock
        with self.lock:
            thread_name = threading.current_thread().name
            self.thread_counts[thread_name] = self.thread_counts.get(thread_name, 0) + 1
            
            self.thread_timeline.append({
                "Thread": thread_name,
                "Start": start_time,
                "End": end_time
            })
            
            self.total_processed += 1
            if fraud_flag:
                self.total_flagged += 1
                self.flagged_transactions.append({
                    "tx": tx,
                    "analysis": result
                })
                
        return result
