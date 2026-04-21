import threading
import math

class FraudAnalyzer:
    """
    Analyzes transactions for fraudulent behavior in a thread-safe manner.
    """
    def __init__(self):
        self.lock = threading.Lock()
        
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
        # Simulate an external I/O bound operation like a database query or network request.
        # This is where multi-threading provides massive speed up in Python by releasing the GIL!
        time.sleep(0.001) 
        
        with self.lock:
            user_id = tx["user_id"]
            amount = tx["amount"]
            location = tx["location"]
            timestamp = tx["timestamp"]
            
            risk_score = 0
            reasons = []
            
            # Rule 1: High-value transaction
            if amount > 10000.0:
                risk_score += 50
                reasons.append("High value transaction")
                
            # Rule 4: Statistical anomaly (using z-score)
            std_dev = self._get_std_dev()
            if self.count >= 10 and std_dev > 0:
                z_score = abs(amount - self.mean) / std_dev
                if z_score > 3.0:
                    risk_score += 40
                    reasons.append(f"Statistical anomaly (Z-score: {z_score:.2f})")
            
            # Update global stats safely
            self._update_global_stats(amount)
            
            # Track user history
            if user_id not in self.user_history:
                self.user_history[user_id] = []
            history = self.user_history[user_id]
            history.append(tx)
            
            if len(history) > 10:
                history.pop(0)  # Keep recent history
                
            # Rule 2 & 3: Rapid successive transactions & Location anomaly
            # Check transactions within the last 60 seconds
            recent_txs = [t for t in history if timestamp - t["timestamp"] < 60]
            
            if len(recent_txs) >= 4:  # If user did 4+ transactions very recently
                risk_score += 30
                reasons.append("Rapid successive transactions")
                
            locations = set(t["location"] for t in recent_txs)
            if len(locations) > 1:
                risk_score += 60
                reasons.append("Location anomaly (impossible travel)")
                
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
            
            self.total_processed += 1
            if fraud_flag:
                self.total_flagged += 1
                self.flagged_transactions.append({
                    "tx": tx,
                    "analysis": result
                })
                
        return result
