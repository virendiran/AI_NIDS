import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

st.title("AI-Powered Network Intrusion Detection System")
st.markdown("""
### Project Overview
This system uses Machine Learning (**Random Forest Algorithm**) to analyze network traffic in real-time.
It classifies traffic into two categories:
* **Benign:** Safe, normal traffic.
* **Malicious:** Potential cyberattacks (DDoS, Port Scan, etc.).
""")


def load_data():
    file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()  
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
   
    df['Label'] = df['Label'].map({'BENIGN': 0, 'DDoS': 1})
    df = df[df['Label'].isin([0, 1])]  
  
    df = df.select_dtypes(include=[np.number])
    return df

df = load_data()


st.sidebar.header("Control Panel")
st.sidebar.info("Adjust model parameters here.")
split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100)


X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - split_size) / 100, random_state=42
)


st.divider()
col_train, col_metrics = st.columns([1, 2])


with col_train:
    st.subheader("1. Model Training")
    if st.button("Train Model Now"):
        with st.spinner("Training Random Forest Classifier..."):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['model'] = model
            st.success("✅ Training Complete!")

    if 'model' in st.session_state:
        st.success("✅ Model is Ready for Inference")


with col_metrics:
    st.subheader("2. Performance Metrics")
    if 'model' in st.session_state:
        model = st.session_state['model']
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc * 100:.2f}%")
        m2.metric("Total Samples", len(df))
        m3.metric("Threats Detected", int(y_pred.sum()))

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please train the model first.")


st.divider()
st.subheader("3. Live Traffic Analyzer")

if 'model' not in st.session_state:
    st.info("👈 Train the model first to enable live analysis.")
else:
    st.markdown("""
    > 💡 **Note**: CIC-IDS2017 uses **78 features**. This demo simulates a full feature vector:
    > - **Benign**: Uses average benign traffic values.
    > - **DDoS-like**: Uses high packet rate + short duration (typical of attacks).
    """)

    
    feature_names = X.columns.tolist()  

    col1, col2 = st.columns(2)
    traffic_type = col1.radio("Select Traffic Type", ["Benign", "DDoS-like"])
    use_custom = col2.checkbox("Customize Key Features", value=False)

    if use_custom:
        st.subheader("🔧 Adjust Key Indicators")
        flow_dur = st.number_input("Flow Duration (ms)", 1, 100000, 500)
        total_pkts = st.number_input("Total Packets", 1, 10000, 1000)
        pkt_len = st.number_input("Avg Packet Length", 10, 1500, 500)
    else:
        flow_dur = 500 if traffic_type == "Benign" else 100
        total_pkts = 100 if traffic_type == "Benign" else 5000
        pkt_len = 500 if traffic_type == "Benign" else 54  

    if st.button("🔍 Analyze Simulated Traffic"):
        
        sample = np.zeros(len(feature_names))

        for i, col in enumerate(feature_names):
            if 'Flow_Duration' in col:
                sample[i] = flow_dur
            elif 'Total' in col and ('Packet' in col or 'Fwd Packet' in col):
                sample[i] = total_pkts
            elif 'Packet_Length' in col or 'Avg_Pkt_Size' in col:
                sample[i] = pkt_len
            elif 'Active' in col or 'Idle' in col:
                sample[i] = 10 if traffic_type == "DDoS-like" else 1000
            elif 'Bwd' in col:  
                sample[i] = 0 if traffic_type == "DDoS-like" else np.random.uniform(10, 100)
            else:
                
                sample[i] = np.random.uniform(10, 200)

        
        model = st.session_state['model']
        pred = model.predict(sample.reshape(1, -1))[0]
        prob = model.predict_proba(sample.reshape(1, -1))[0]

        if pred == 1:
            st.error(f"🚨 **MALICIOUS TRAFFIC DETECTED!** (Confidence: {prob[1]:.2%})")
            st.write("**Pattern**: High packet rate, short duration, small packet size — typical of DDoS.")
        else:
            st.success(f"✅ **BENIGN TRAFFIC** (Confidence: {prob[0]:.2%})")