import random
import time
import uuid

# List of sample cities
CITIES = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Mumbai", "Toronto", "Dubai", "Singapore"]

class DataGenerator:
    """
    Generates simulated financial transactions.
    """
    def __init__(self, num_users=1000):
        self.users = [f"USER_{i:04d}" for i in range(num_users)]
        self.user_cities = {user: random.choice(CITIES) for user in self.users}
    
    def generate_transaction(self):
        """
        Generate a single random transaction.
        """
        user_id = random.choice(self.users)
        
        # 95% chance of normal transaction, 5% chance of anomalous high value
        if random.random() < 0.95:
            amount = round(random.uniform(10.0, 1000.0), 2)
        else:
            amount = round(random.uniform(5000.0, 50000.0), 2)
            
        # 95% chance of normal location (home city), 5% chance of travel anomaly
        if random.random() < 0.95:
            location = self.user_cities[user_id]
        else:
            location = random.choice(CITIES)
        # Introduce sequential timestamps to simulate a real data stream
        if not hasattr(self, "current_time"):
            self.current_time = time.time() - (86400 * 7) # Start 7 days ago
            
        # Time moves forward by 0.1 to 5.0 seconds per transaction
        self.current_time += random.uniform(0.1, 5.0)
        
        return {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user_id,
            "amount": amount,
            "timestamp": self.current_time,
            "location": location
        }

    def generate_batch(self, count):
        """
        Generate a batch of transactions.
        """
        return [self.generate_transaction() for _ in range(count)]
