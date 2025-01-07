from flask import Flask, render_template, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from collections import OrderedDict
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt



app = Flask(__name__)

# 主頁面路由
@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/analysis')
def analysis():
    return render_template('index-2.html')  

def safe_mean(series):
    mean_value = series.mean()
    # 如果 mean_value 是 NaN，則返回 0
    return 0 if np.isnan(mean_value) else mean_value

# 圖表資料的 API 路由

@app.route('/calculationofEvents')
def generate_chart():
    # 讀取資料
    dataset_path = os.path.join('/Users/debbie/python/Flask-2/dataset', 'transcript.csv')
    df = pd.read_csv(dataset_path)

    # 數據處理
    event_counts = df['event'].value_counts()

    # 返回 JSON 資料
    data = {
        'labels': event_counts.index.tolist(),
        'values': event_counts.values.tolist()
    }

    return jsonify(data)



@app.route('/ProportionofEvents')
def generate_chart_2():
    # 讀取資料
    dataset_path = os.path.join('/Users/debbie/python/Flask-2/dataset', 'transcript.csv')
    df = pd.read_csv(dataset_path)

    # 數據處理
    event_counts = df['event'].value_counts()
    wedge_sizes = event_counts.tolist()
    wedge_labels = event_counts.index.tolist()
    wedge_percentages = [(count / sum(wedge_sizes)) * 100 for count in wedge_sizes]

    # 返回 JSON 資料
    data = {
        'labels': wedge_labels,
        'sizes': wedge_sizes,
        'percentages': [round(p, 1) for p in wedge_percentages]
    }

    return jsonify(data)

@app.route('/EventsOccurrence')
def generate_chart_3():
    # 讀取資料
    dataset_path = os.path.join('/Users/debbie/python/Flask-2/dataset', 'transcript_mod.csv')
    df = pd.read_csv(dataset_path)

    # 數據處理
    viewed_daily = df[df['event'] == 'offer viewed'].value_counts('days_since_start').sort_index()
    transaction_daily = df[df['event'] == 'transaction'].value_counts('days_since_start').sort_index()
    completed_daily = df[df['event'] == 'offer completed'].value_counts('days_since_start').sort_index()

    received_hourly = df[df['event'] == 'offer received'].value_counts('time').sort_index()
    viewed_hourly = df[df['event'] == 'offer viewed'].value_counts('time').sort_index()
    transaction_hourly = df[df['event'] == 'transaction'].value_counts('time').sort_index()
    completed_hourly = df[df['event'] == 'offer completed'].value_counts('time').sort_index()

    # 整理數據
    data = {
        'daily': {
            'days': viewed_daily.index.tolist(),
            'viewed': viewed_daily.values.tolist(),
            'transaction': transaction_daily.values.tolist(),
            'completed': completed_daily.values.tolist()
        },
        'hourly': {
            'hours': viewed_hourly.index.tolist(),
            'received': received_hourly.values.tolist(),
            'viewed': viewed_hourly.values.tolist(),
            'transaction': transaction_hourly.values.tolist(),
            'completed': completed_hourly.values.tolist()
        }
    }

    return jsonify(data)
@app.route('/datatable')
def datatable():
    # 讀取 CSV 檔案
    dataset_path = '/Users/debbie/python/Flask-2/dataset/out_transaction_event_portfolio.csv'
    df = pd.read_csv(dataset_path, encoding='utf-8-sig')

    # 確保欄位順序與 CSV 一致
    ordered_columns = ['person', 'event', 'offer_alias', 'reward', 'channels',
                       'difficulty', 'duration', 'offer_type', 'time', 'days_since_start', 'dict_key']
    df = df[ordered_columns]  # 強制按照指定欄位順序排列

    # 將資料轉換為有序 JSON
    data = [OrderedDict(row) for row in df.head(1000).to_dict(orient='records')]
    return jsonify(data)

@app.route('/analysis/ElbowMethod')
def generate_chart_4():
    # 讀取資料
    dataset_path = os.path.join('/Users/debbie/python/Flask-2/dataset', 'customer_behavior_scaled.csv')
    df = pd.read_csv(dataset_path)

    # 數據處理
    df_feature = df
    wcss = []  # 紀錄 WCSS 值
    sil_score = []  # 紀錄 Silhouette 分數
    max_clusters = 10

    # 計算 WCSS 和 Silhouette Score
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, random_state=10)
        kmeans.fit(df_feature)
        wcss.append(kmeans.inertia_)

        # 計算 Silhouette Score（k=1 時無法計算）
        if k >= 2:
            labels = kmeans.labels_
            sil_score.append(metrics.silhouette_score(df_feature, labels))

    # 返回 JSON 資料
    data = {
        "elbow_method": {
            "clusters": list(range(1, max_clusters)),  # 群數
            "wcss": wcss  # WCSS 值
        },
        "silhouette_method": {
            "clusters": list(range(2, max_clusters)),  # k=2 開始計算
            "sil_score": sil_score  # Silhouette Score
        }
    }
    
    return jsonify(data)


@app.route('/analysis/ElbowMethod_2')
def generate_chart_5():
  
    dataset_path = os.path.join('/Users/debbie/python/Flask-2/dataset', 'customer_behavior_scaled.csv')
    df = pd.read_csv(dataset_path)

    dataset_path_2 = os.path.join('/Users/debbie/python/Flask-2/dataset', 'customer_behavior.csv')
    df_2 = pd.read_csv(dataset_path_2)

    # K-means 分群分析
    k = 5
    df_feature = df
    kmeans = KMeans(n_clusters=k, random_state=10)
    kmeans.fit(df_feature)

    # 加入分群標籤
    customer_with_cluster = df_2.assign(cluster=kmeans.labels_)

    # 統計每個群組內的資料點數量
    cluster_counts = customer_with_cluster['cluster'].value_counts().sort_index().tolist()

    # 計算 Silhouette Score
    silhouette_score = metrics.silhouette_score(df_feature, labels=kmeans.labels_)

    # 分群的平均數據
    cluster_info = customer_with_cluster.groupby('cluster').agg([np.mean]).round(1)

    # 壓平多層級欄位名稱
    # 將多層級欄位名稱 (hierarchical columns) 轉換為單層名稱：
    # 例如：('money_spent', 'mean') → 'money_spent_mean'。
    cluster_info.columns = ['_'.join(col) for col in cluster_info.columns]

    # 將分群資訊整理為字典格式
    cluster_summary = cluster_info.to_dict(orient='index')

    # 繪圖資料準備
    color_list = ['red', 'blue', 'orange', 'green', 'yellow']
    legend_list = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    scatter_data = []

    for i in range(k):
        cluster_points = customer_with_cluster.loc[customer_with_cluster['cluster'] == i]
        scatter_data.append({
            'label': legend_list[i],
            'x': cluster_points['num_viewed'].tolist(),
            'y': cluster_points['money_spent'].tolist(),
            'color': color_list[i]
        })

    # 返回 JSON 資料
    data = {
        "cluster_counts": cluster_counts,  # 每個群組資料數量
        "silhouette_score": silhouette_score,  # Silhouette 分數
        "cluster_summary": cluster_summary,  # 各群組數據摘要
        "scatter_data": scatter_data  # 分群散點圖資料
    }

    return jsonify(data)

@app.route('/analysis/clusterResult')
def generate_chart_6():
    # 1. 讀取資料
    dataset_path = os.path.join('/Users/debbie/python/Flask-2/dataset', 'customer_with_cluster.csv')
    df = pd.read_csv(dataset_path)

    # 2. 定義群組數量與顏色清單
    k = 5
    color_list = ['red', 'blue', 'orange', 'green', 'purple']

    # 3. 數據分析結果整理
    cluster_summary = df.groupby('cluster').agg({
        'num_viewed': ['mean', 'std'],
        'money_spent': ['mean', 'std'],
        'num_completed': ['mean', 'std'],
        'num_transactions': ['mean', 'std']
    }).round(2)

    # 壓平多層級欄位名稱
    cluster_summary.columns = ['_'.join(col) for col in cluster_summary.columns]
    cluster_summary = cluster_summary.to_dict(orient='index')

    # 4. 散點圖資料準備
    scatter_data = []
    for i in range(k):
        cluster_points = df[df['cluster'] == i]
        scatter_data.append({
            'label': f'Cluster {i}',
            'x': cluster_points['num_viewed'].tolist(),
            'y': cluster_points['money_spent'].tolist(),
            'color': color_list[i]
        })

    # 5. 返回 JSON 資料
    data = {
        "cluster_summary": cluster_summary,
        "scatter_data": scatter_data
    }

    return jsonify(data)

   
if __name__ == '__main__':
    app.run(debug=True)
    

