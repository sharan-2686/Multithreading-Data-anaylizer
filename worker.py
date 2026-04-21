import concurrent.futures
import queue
import threading

def worker_task(tx_queue, analyzer):
    """
    Worker function that continuously pulls from the queue until empty.
    """
    while True:
        try:
            # Non-blocking get. If empty, the worker will terminate.
            tx = tx_queue.get_nowait()
        except queue.Empty:
            break
            
        # Process the transaction
        analyzer.analyze(tx)
        
        # Mark as done
        tx_queue.task_done()

class WorkManager:
    """
    Manages the execution of workers (Consumers).
    """
    @staticmethod
    def process_single_threaded(transactions, analyzer):
        """
        Process transactions sequentially in the main thread.
        """
        for tx in transactions:
            analyzer.analyze(tx)

    @staticmethod
    def process_multi_threaded(transactions, analyzer, thread_count=4):
        """
        Process transactions using a ThreadPoolExecutor and a thread-safe Queue.
        """
        tx_queue = queue.Queue()
        
        # Producer: Populate the queue with all transactions.
        # In a fully streaming architecture this would happen concurrently with consumers,
        # but for benchmarking a fixed batch is loaded first.
        for tx in transactions:
            tx_queue.put(tx)
            
        # Consumers: Spin up workers to empty the queue.
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            # Submit enough worker tasks to match the desired thread count
            futures = [executor.submit(worker_task, tx_queue, analyzer) for _ in range(thread_count)]
            
            # Wait for all workers to finish
            concurrent.futures.wait(futures)
            
        # Optional: wait until queue reports all tasks are done.
        # Not strictly needed since workers drain the queue, but good practice.
        # tx_queue.join()
