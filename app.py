import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_excel('data.xlsx', sheet_name='data')
train_data = pd.read_excel('data.xlsx', sheet_name='oversample.train')
test_data = pd.read_excel('data.xlsx', sheet_name='test')

# Set page config
st.set_page_config(page_title="Dashboard Klasifikasi SVM dan KMeans SVM", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["Deskripsi Data", "Prediksi SVM", "Prediksi KMeans SVM", "Perbandingan Model", "Prediksi Baru"])

# Deskripsi Data
if page == "Deskripsi Data":
    st.title("Statistika Deskriptif")
    
    # Descriptive statistics
    st.subheader("Tabel Statistika Deskriptif")
    desc_stats = data.groupby('fraud').agg({
        'amount': ['mean', 'std', 'min', 'median', 'max'],
        'second': ['mean', 'std', 'min', 'median', 'max'],
        'days': ['mean', 'std', 'min', 'median', 'max']
    })
    st.table(desc_stats)
    
    # Pie chart for fraud variable
    st.subheader("Pie Chart Variabel Fraud")
    fraud_counts = data['fraud'].value_counts()
    plt.figure(figsize=(5,5))
    plt.pie(fraud_counts, labels=['Sah', 'Penipuan'], autopct='%1.1f%%', startangle=140)
    st.pyplot(plt)
    
    # Boxplot for variables based on fraud category
    st.subheader("Boxplot Berdasarkan Kategori Fraud")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(x='fraud', y='amount', data=data, ax=axes[0])
    axes[0].set_title('Amount')
    sns.boxplot(x='fraud', y='second', data=data, ax=axes[1])
    axes[1].set_title('Second')
    sns.boxplot(x='fraud', y='days', data=data, ax=axes[2])
    axes[2].set_title('Days')
    st.pyplot(fig)

# Prediksi SVM
elif page == "Prediksi SVM":
    st.title("Prediksi Menggunakan SVM")
    
    # Prepare data
    X_train = train_data[['amount', 'second', 'days']]
    y_train = train_data['fraud']
    X_test = test_data[['amount', 'second', 'days']]
    y_test = test_data['fraud']
    
    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    
    # Evaluation
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    recall_svm = recall_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm)
    
    st.subheader("Confusion Matrix")
    st.table(cm_svm)
    
    st.subheader("Evaluasi Model")
    st.write(f"Akurasi: {accuracy_svm:.2f}")
    st.write(f"Sensitivitas: {recall_svm:.2f}")
    st.write(f"Spesifisitas: {precision_svm:.2f}")

# Prediksi KMeans SVM
elif page == "Prediksi KMeans SVM":
    st.title("Prediksi Menggunakan KMeans SVM")
    
    # Prepare data
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_train)
    X_train['cluster'] = kmeans.labels_
    X_test['cluster'] = kmeans.predict(X_test)
    
    # Train SVM model on clusters
    cluster_svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    cluster_svm_model.fit(X_train, y_train)
    y_pred_cluster_svm = cluster_svm_model.predict(X_test)
    
    # Evaluation
    cm_cluster_svm = confusion_matrix(y_test, y_pred_cluster_svm)
    accuracy_cluster_svm = accuracy_score(y_test, y_pred_cluster_svm)
    recall_cluster_svm = recall_score(y_test, y_pred_cluster_svm)
    precision_cluster_svm = precision_score(y_test, y_pred_cluster_svm)
    
    st.subheader("Confusion Matrix")
    st.table(cm_cluster_svm)
    
    st.subheader("Evaluasi Model")
    st.write(f"Akurasi: {accuracy_cluster_svm:.2f}")
    st.write(f"Sensitivitas: {recall_cluster_svm:.2f}")
    st.write(f"Spesifisitas: {precision_cluster_svm:.2f}")

# Perbandingan Model
elif page == "Perbandingan Model":
    st.title("Perbandingan Model SVM dan KMeans SVM")
    
    st.subheader("Evaluasi Model SVM")
    st.write(f"Akurasi: {accuracy_svm:.2f}")
    st.write(f"Sensitivitas: {recall_svm:.2f}")
    st.write(f"Spesifisitas: {precision_svm:.2f}")
    
    st.subheader("Evaluasi Model KMeans SVM")
    st.write(f"Akurasi: {accuracy_cluster_svm:.2f}")
    st.write(f"Sensitivitas: {recall_cluster_svm:.2f}")
    st.write(f"Spesifisitas: {precision_cluster_svm:.2f}")

# Prediksi Baru
elif page == "Prediksi Baru":
    st.title("Prediksi Baru Menggunakan Model SVM")
    
    amount = st.number_input("Amount", min_value=0.0)
    second = st.number_input("Second", min_value=0.0)
    days = st.number_input("Days", min_value=0, value=int(second // 86400))
    
    # Update second based on days input
    second = st.number_input("Second", min_value=0.0, value=days * 86400)
    
    if st.button("Prediksi"):
        input_data = np.array([[amount, second, days]])
        prediction = svm_model.predict(input_data)
        st.write(f"Hasil Prediksi: {'Penipuan' if prediction[0] == 1 else 'Sah'}")

