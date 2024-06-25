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

# Set page config
st.set_page_config(page_title="Dashboard Klasifikasi SVM dan KMeans SVM")

# Cache the data loading function to avoid reloading the data on each rerun
@st.cache_resource
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name='data')
    train_data = pd.read_excel(file_path, sheet_name='oversample.train')
    test_data = pd.read_excel(file_path, sheet_name='test')
    return data, train_data, test_data

data, train_data, test_data = load_data('data.xlsx')

# Cache the model training functions to avoid retraining the models on each rerun
@st.cache_resource
def train_svm(X_train_svm, y_train_svm, kernel='linear', C=1.0, gamma='scale'):
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma)
    svm_model.fit(X_train_svm, y_train_svm)
    return svm_model

@st.cache_resource
def train_kmeans_svm(X_train_ksvm, y_train_ksvm, n_clusters=3, kernel='linear', C=1.0, gamma='scale'):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    X_train_ksvm['cluster'] = kmeans.labels_
    
    cluster_svm_model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma))
    cluster_svm_model.fit(X_train_ksvm, y_train_ksvm)
    return kmeans, cluster_svm_model



# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["Deskripsi Data", "Prediksi SVM", "Prediksi KMeans SVM", "Perbandingan Model", "Prediksi Baru"])

# Deskripsi Data
if page == "Deskripsi Data":
    st.title("Statistika Deskriptif")
    
    # Descriptive statistics for each variable
    def descriptive_stats(variable):
        stats = data.groupby('fraud')[variable].agg(['mean', 'std', 'min', 'median', 'max']).reset_index()
        stats['variable'] = variable
        return stats

    amount_stats = descriptive_stats('amount')
    second_stats = descriptive_stats('second')
    days_stats = descriptive_stats('days')

    # Concatenate all stats into a single DataFrame
    desc_stats = pd.concat([amount_stats, second_stats, days_stats], ignore_index=True)

    # Display the descriptive statistics
    st.subheader("Tabel Statistika Deskriptif")
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
    X_train_svm = train_data[['amount', 'second', 'days']]
    y_train_svm = train_data['fraud']
    X_test_svm = test_data[['amount', 'second', 'days']]
    y_test_svm = test_data['fraud']

    # Add evaluation to the comparison section
    accuracy_svm = 0.0
    recall_svm = 0.0
    precision_svm = 0.0

    
    # Train SVM model with specified parameters
    svm_model = train_svm(X_train_svm, y_train_svm, kernel='linear', C=1.0, gamma='scale')
    y_pred_svm = svm_model.predict(X_test_svm)
    
    # Evaluation
    cm_svm = confusion_matrix(y_test_svm, y_pred_svm)
    accuracy_svm = accuracy_score(y_test_svm, y_pred_svm)
    recall_svm = recall_score(y_test_svm, y_pred_svm)
    precision_svm = precision_score(y_test_svm, y_pred_svm)
    
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
    X_train_ksvm = train_data[['amount', 'second', 'days']]
    y_train_ksvm = train_data['fraud']
    X_test_ksvm = test_data[['amount', 'second', 'days']]
    y_test_ksvm = test_data['fraud']

    # Add evaluation to the comparison section
    accuracy_cluster_svm = 0.0
    recall_cluster_svm = 0.0
    precision_cluster_svm = 0.0
    
    # Train KMeans and SVM model with specified parameters
    kmeans, cluster_svm_model = train_kmeans_svm(X_train_ksvm, y_train_ksvm, n_clusters=3, kernel='linear', C=1.0, gamma='scale')
    X_test['cluster'] = kmeans.predict(X_test_ksvm)
    y_pred_cluster_svm = cluster_svm_model.predict(X_test_ksvm)
    
    # Evaluation
    cm_cluster_svm = confusion_matrix(y_test_ksvm, y_pred_cluster_svm)
    accuracy_cluster_svm = accuracy_score(y_test_ksvm, y_pred_cluster_svm)
    recall_cluster_svm = recall_score(y_test_ksvm, y_pred_cluster_svm)
    precision_cluster_svm = precision_score(y_test_ksvm, y_pred_cluster_svm)
    
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
    
    # Compare accuracy and display message based on comparison
    if accuracy_svm > accuracy_cluster_svm:
        st.write("Metode SVM lebih baik dalam memprediksi penipuan transaksi kartu kredit.")
    elif accuracy_svm < accuracy_cluster_svm:
        st.write("Metode KMeans SVM lebih baik dalam memprediksi penipuan transaksi kartu kredit.")
    else:
        st.write("Metode SVM dan KMeans SVM memiliki performa prediksi yang sama untuk penipuan transaksi kartu kredit.")

# Prediksi Baru
elif page == "Prediksi Baru":
    st.title("Prediksi Baru Menggunakan Model SVM")
    
    amount = st.number_input("Amount", min_value=0.0, max_value = 30000.0)
    second = st.number_input("Second", min_value=0.0, value=days * 86400.0)
    days = st.number_input("Days", min_value=0.0, value= second // 86400.0)

    
    if st.button("Prediksi"):
        input_data = np.array([[amount, second, days]])
        prediction = svm_model.predict(input_data)
        st.write(f"Hasil Prediksi: {'Penipuan' if prediction[0] == 1 else 'Sah'}")
