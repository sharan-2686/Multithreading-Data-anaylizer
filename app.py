import streamlit as st
import time
import pandas as pd
from generator import DataGenerator
from analyzer import FraudAnalyzer
from worker import WorkManager
from ml_trainer import MLTrainer
import altair as alt

st.set_page_config(page_title="Fraud Detection Simulation", layout="wide")
st.title("🛡️ Real-Time Multi-Threaded Fraud Detection")

# Initialize session state for persistence
if "analyzer" not in st.session_state:
    st.session_state.analyzer = FraudAnalyzer()
    st.session_state.generator = DataGenerator(num_users=200)
    st.session_state.total_tx_data = []
    st.session_state.performance_log = []

st.sidebar.header("⚙️ Simulation Controls")
batch_size = st.sidebar.slider("Batch Size", 50, 2000, 500)
threads = st.sidebar.slider("Worker Threads", 1, 16, 4)
use_ml = st.sidebar.checkbox("🧠 Enable Machine Learning (Isolation Forest)", value=False)

if st.sidebar.button("🚀 Generate & Process Data"):
    with st.spinner(f"Generating and processing {batch_size} transactions..."):
        # Generate data
        tx_batch = st.session_state.generator.generate_batch(batch_size)
        st.session_state.total_tx_data.extend(tx_batch)
        
        # Handle ML Model
        if use_ml and getattr(st.session_state.analyzer, 'ml_model', None) is None:
            st.info("Initiating Machine Learning... Training Isolation Forest on warm-up data.")
            warmup_batch = st.session_state.generator.generate_batch(2000)
            ml_model = MLTrainer.train_isolation_forest(warmup_batch)
            st.session_state.analyzer.ml_model = ml_model
            st.toast("✅ ML Model Trained & Loaded!")
        elif not use_ml:
            st.session_state.analyzer.ml_model = None
        
        # 1. Benchmark Single-Threaded (using a dummy analyzer to not double-count stats)
        dummy_analyzer = FraudAnalyzer(ml_model=st.session_state.analyzer.ml_model)
        start = time.time()
        WorkManager.process_single_threaded(tx_batch, dummy_analyzer)
        single_time = time.time() - start
        
        # Save the single-thread timeline for visualization
        st.session_state.single_thread_timeline = dummy_analyzer.thread_timeline
        
        # 2. Process Multi-Threaded (updating the actual dashboard state)
        st.session_state.analyzer.thread_counts = {}
        st.session_state.analyzer.thread_timeline = []
        
        start = time.time()
        WorkManager.process_multi_threaded(tx_batch, st.session_state.analyzer, thread_count=threads)
        multi_time = time.time() - start
        
        speedup = single_time / multi_time if multi_time > 0 else 0
        
        st.session_state.performance_log.append({
            "Batch": batch_size,
            "Threads": threads,
            "Single Thread (s)": single_time,
            "Multi Thread (s)": multi_time,
            "Speedup": speedup
        })
        st.sidebar.success(f"Multi-thread processed in {multi_time:.4f}s ({speedup:.2f}x speedup)")

# Dashboard UI
summary = st.session_state.analyzer.get_summary()

st.markdown("### 📊 Live Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", summary["total_processed"])
col2.metric("Flagged as Fraud", summary["total_flagged"])
col3.metric("Fraud Rate", summary["fraud_rate"])
col4.metric("Avg Transaction", f"${summary['mean_amount']:.2f}")

st.markdown("---")

col_left, col_right = st.columns([2, 2])

with col_left:
    st.subheader("🚨 Recent Fraud Alerts")
    if st.session_state.analyzer.flagged_transactions:
        flagged_data = []
        # Show top 20 recent
        for item in reversed(st.session_state.analyzer.flagged_transactions[-20:]):
            tx = item['tx']
            an = item['analysis']
            flagged_data.append({
                "User": tx['user_id'],
                "Amount": tx['amount'],
                "Location": tx['location'],
                "Risk": an['risk_score'],
                "Reason": an['reason']
            })
        
        df = pd.DataFrame(flagged_data)
        st.dataframe(df.style.highlight_max(subset=['Risk'], color='red'), use_container_width=True)
    else:
        st.info("No fraudulent transactions detected yet.")

with col_right:
    st.subheader("📈 Performance Comparison")
    if st.session_state.performance_log:
        perf_df = pd.DataFrame(st.session_state.performance_log)
        
        # Show data table for details
        st.dataframe(
            perf_df[["Batch", "Threads", "Speedup"]].style.format({"Speedup": "{:.2f}x"}), 
            use_container_width=True
        )
        
        # Show bar chart comparing run times
        chart_data = perf_df[['Single Thread (s)', 'Multi Thread (s)']]
        st.bar_chart(chart_data)
    else:
        st.info("Run a simulation batch to see performance.")

st.markdown("---")
if st.session_state.total_tx_data:
    st.subheader("💸 Transaction Amount Distribution (Last 200)")
    # Show history of amounts
    recent_txs = st.session_state.total_tx_data[-200:]
    amounts_df = pd.DataFrame({
        "Amount": [t["amount"] for t in recent_txs]
    })
    st.line_chart(amounts_df)

st.markdown("---")
st.subheader("⏱️ Execution Timeline Comparison (Single vs Multi-Threaded)")

def generate_gantt_chart(timeline_data, title="Execution Timeline"):
    if not timeline_data:
        return None
    # Plotting first 30 entries so the spacing and idle times are explicitly clear
    plot_data = timeline_data[:30]
    df_gantt = pd.DataFrame(plot_data)
    min_time = df_gantt["Start"].min()
    
    # Calculate exactly how many milliseconds have passed since the batch began processing
    df_gantt["Start (ms)"] = (df_gantt["Start"] - min_time) * 1000
    df_gantt["End (ms)"] = (df_gantt["End"] - min_time) * 1000
    df_gantt["Worker"] = df_gantt["Thread"].apply(lambda x: x.replace("ThreadPoolExecutor-", "Worker-").replace("MainThread", "SingleThread"))
    
    chart = alt.Chart(df_gantt).mark_bar(cornerRadius=2, size=18).encode(
        x=alt.X('Start (ms):Q', title='Time since start (milliseconds)'),
        x2='End (ms):Q',
        y=alt.Y('Worker:N', title='Thread ID', sort='ascending'),
        color=alt.Color('Worker:N', legend=None),
        tooltip=['Worker', 'Start (ms)', 'End (ms)']
    ).properties(
        height=250,
        title=title
    ).interactive()
    return chart

col_gant_single, col_gant_multi = st.columns(2)

with col_gant_single:
    st.markdown("#### Single-Threaded Sequence")
    if getattr(st.session_state, 'single_thread_timeline', []):
        chart_single = generate_gantt_chart(st.session_state.single_thread_timeline, "Strictly Sequential")
        st.altair_chart(chart_single, use_container_width=True)
        st.caption("A single thread processes everything strictly successively. Notice how one task completely finishes before the next begins.")
    else:
        st.info("No single-threaded timeline data yet.")

with col_gant_multi:
    st.markdown("#### Multi-Threaded Concurrency")
    if getattr(st.session_state.analyzer, 'thread_timeline', []):
        chart_multi = generate_gantt_chart(st.session_state.analyzer.thread_timeline, "Concurrent Overlapping")
        st.altair_chart(chart_multi, use_container_width=True)
        st.caption("Multiple threads process transactions concurrently. Notice the overlapping colored execution blocks filling idle gaps!")
    else:
        st.info("No timeline data yet.")
