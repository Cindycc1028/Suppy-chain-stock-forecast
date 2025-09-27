# Suppy-chain-stock-forecast
This project focuses on analyzing supply chain data to optimize operational efficiency by leveraging SQL for multi-table data integration, cleaning, and preparation, followed by Python-based time series modeling to uncover seasonal patterns and promotional trends.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置可视化风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. 数据加载与初步探索
def load_and_explore_data():
    # 从URL加载数据
    url = "https://sfile.chatglm.cn/chatglm4/file/43/436adf12de.csv"
    df = pd.read_csv(url)
    
    print("数据集形状:", df.shape)
    print("\n前5行数据:")
    print(df.head())
    
    print("\n数据集信息:")
    df.info()
    
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    print("\n描述性统计:")
    print(df.describe())
    
    return df

# 2. 数据清洗与预处理
def clean_and_preprocess(df):
    # 处理重复列名（数据集中有两个Lead time列）
    df.columns = ['Product_type', 'SKU', 'Price', 'Availability', 'Number_of_products_sold', 
                  'Revenue_generated', 'Customer_demographics', 'Stock_levels', 'Lead_times', 
                  'Order_quantities', 'Shipping_times', 'Shipping_carriers', 'Shipping_costs', 
                  'Supplier_name', 'Location', 'Lead_time', 'Production_volumes', 
                  'Manufacturing_lead_time', 'Manufacturing_costs', 'Inspection_results', 
                  'Defect_rates', 'Transportation_modes', 'Routes', 'Costs']
    
    # 处理缺失值
    # 数值列用中位数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # 分类列用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 处理异常值 - 使用IQR方法
    def remove_outliers(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df
    
    # 对关键数值列处理异常值
    key_numeric_cols = ['Price', 'Number_of_products_sold', 'Revenue_generated', 
                       'Stock_levels', 'Shipping_costs', 'Defect_rates']
    for col in key_numeric_cols:
        df = remove_outliers(df, col)
    
    # 创建时间索引（假设数据按时间顺序记录）
    df['Time_Index'] = range(len(df))
    
    # 创建促销标志（假设每10个时间单位有一次促销）
    df['Promotion'] = (df['Time_Index'] % 10 == 0).astype(int)
    
    # 计算衍生指标
    df['Inventory_Turnover'] = df['Number_of_products_sold'] / df['Stock_levels']
    df['Profit_Margin'] = (df['Revenue_generated'] - df['Costs']) / df['Revenue_generated']
    df['Order_Fulfillment_Rate'] = df['Number_of_products_sold'] / df['Order_quantities']
    
    print("\n清洗后数据集形状:", df.shape)
    print("\n清洗后缺失值统计:")
    print(df.isnull().sum())
    
    return df

# 3. 关键指标统计与可视化
def analyze_key_metrics(df):
    # 按产品类型分组统计
    product_metrics = df.groupby('Product_type').agg({
        'Number_of_products_sold': 'sum',
        'Revenue_generated': 'sum',
        'Stock_levels': 'mean',
        'Defect_rates': 'mean',
        'Shipping_costs': 'mean',
        'Inventory_Turnover': 'mean',
        'Profit_Margin': 'mean'
    }).reset_index()
    
    print("\n按产品类型的关键指标:")
    print(product_metrics)
    
    # 按供应商分组统计
    supplier_metrics = df.groupby('Supplier_name').agg({
        'Number_of_products_sold': 'sum',
        'Revenue_generated': 'sum',
        'Defect_rates': 'mean',
        'Manufacturing_costs': 'mean',
        'Lead_time': 'mean'
    }).reset_index()
    
    print("\n按供应商的关键指标:")
    print(supplier_metrics)
    
    # 可视化关键指标
    plt.figure(figsize=(15, 10))
    
    # 产品类型销售对比
    plt.subplot(2, 2, 1)
    sns.barplot(x='Product_type', y='Number_of_products_sold', data=product_metrics)
    plt.title('产品类型销售量对比')
    plt.xticks(rotation=45)
    
    # 产品类型收入对比
    plt.subplot(2, 2, 2)
    sns.barplot(x='Product_type', y='Revenue_generated', data=product_metrics)
    plt.title('产品类型收入对比')
    plt.xticks(rotation=45)
    
    # 供应商缺陷率对比
    plt.subplot(2, 2, 3)
    sns.barplot(x='Supplier_name', y='Defect_rates', data=supplier_metrics)
    plt.title('供应商缺陷率对比')
    plt.xticks(rotation=45)
    
    # 库存周转率分布
    plt.subplot(2, 2, 4)
    sns.histplot(df['Inventory_Turnover'], kde=True)
    plt.title('库存周转率分布')
    
    plt.tight_layout()
    plt.savefig('key_metrics.png')
    plt.show()
    
    return product_metrics, supplier_metrics

# 4. 时间序列分析
def time_series_analysis(df):
    # 按时间索引聚合销售数据
    ts_data = df.groupby('Time_Index')['Number_of_products_sold'].sum().reset_index()
    ts_data.set_index('Time_Index', inplace=True)
    
    # 时间序列分解
    decomposition = seasonal_decompose(ts_data, model='additive', period=10)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(ts_data, label='原始销售数据')
    plt.legend()
    plt.title('原始销售数据')
    
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label='趋势')
    plt.legend()
    plt.title('趋势成分')
    
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label='季节性')
    plt.legend()
    plt.title('季节性成分')
    
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label='残差')
    plt.legend()
    plt.title('残差成分')
    
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png')
    plt.show()
    
    # SARIMA模型预测
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data.iloc[:train_size], ts_data.iloc[train_size:]
    
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 10))
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(test))
    
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='训练数据')
    plt.plot(test, label='测试数据')
    plt.plot(forecast, label='预测数据')
    plt.title('SARIMA销售预测')
    plt.legend()
    plt.savefig('sales_forecast.png')
    plt.show()
    
    # 评估模型
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mse)
    print(f"\n预测均方根误差(RMSE): {rmse:.2f}")
    
    return decomposition, model_fit

# 5. 促销趋势建模
def promotion_trend_modeling(df):
    # 准备数据
    promo_data = df[['Time_Index', 'Number_of_products_sold', 'Promotion', 'Price', 'Stock_levels']]
    
    # 添加滞后特征
    for lag in [1, 2, 3]:
        promo_data[f'Sales_Lag_{lag}'] = promo_data['Number_of_products_sold'].shift(lag)
    
    # 去除缺失值
    promo_data.dropna(inplace=True)
    
    # 特征和目标变量
    X = promo_data.drop('Number_of_products_sold', axis=1)
    y = promo_data['Number_of_products_sold']
    
    # 数据标准化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 随机森林模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 预测
    y_pred = rf_model.predict(X_test)
    
    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n促销模型均方根误差(RMSE): {rmse:.2f}")
    print(f"促销模型R²分数: {r2:.2f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n特征重要性:")
    print(feature_importance)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('促销模型特征重要性')
    plt.savefig('feature_importance.png')
    plt.show()
    
    # 促销效果分析
    promo_effect = df.groupby('Promotion')['Number_of_products_sold'].mean()
    print("\n促销效果分析:")
    print(promo_effect)
    
    # 可视化促销效果
    plt.figure(figsize=(8, 5))
    sns.barplot(x=promo_effect.index, y=promo_effect.values)
    plt.title('促销对销售量的影响')
    plt.xticks([0, 1], ['非促销期', '促销期'])
    plt.ylabel('平均销售量')
    plt.savefig('promotion_effect.png')
    plt.show()
    
    return rf_model, feature_importance

# 6. 库存管理优化建议
def inventory_optimization(df):
    # 计算安全库存
    df['Safety_Stock'] = df['Number_of_products_sold'].std() * 1.65  # 95%服务水平
    
    # 计算再订货点
    df['Reorder_Point'] = df['Safety_Stock'] + (df['Number_of_products_sold'].mean() * df['Lead_time'])
    
    # 计算经济订货量(EOQ)
    holding_cost = 0.2  # 持有成本率
    ordering_cost = 50  # 每次订货成本
    df['EOQ'] = np.sqrt((2 * ordering_cost * df['Number_of_products_sold'].sum()) / 
                        (holding_cost * df['Price']))
    
    # 按产品类型汇总库存指标
    inventory_metrics = df.groupby('Product_type').agg({
        'Stock_levels': 'mean',
        'Safety_Stock': 'mean',
        'Reorder_Point': 'mean',
        'EOQ': 'mean',
        'Inventory_Turnover': 'mean'
    }).reset_index()
    
    print("\n库存优化指标:")
    print(inventory_metrics)
    
    # 可视化库存指标
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='Product_type', y='Stock_levels', data=inventory_metrics)
    plt.title('平均库存水平')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(x='Product_type', y='Safety_Stock', data=inventory_metrics)
    plt.title('安全库存水平')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.barplot(x='Product_type', y='Reorder_Point', data=inventory_metrics)
    plt.title('再订货点')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.barplot(x='Product_type', y='EOQ', data=inventory_metrics)
    plt.title('经济订货量')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('inventory_optimization.png')
    plt.show()
    
    return inventory_metrics

# 主函数
def main():
    # 1. 加载数据
    df = load_and_explore_data()
    
    # 2. 数据清洗与预处理
    df_clean = clean_and_preprocess(df)
    
    # 3. 关键指标分析
    product_metrics, supplier_metrics = analyze_key_metrics(df_clean)
    
    # 4. 时间序列分析
    decomposition, sarima_model = time_series_analysis(df_clean)
    
    # 5. 促销趋势建模
    rf_model, feature_importance = promotion_trend_modeling(df_clean)
    
    # 6. 库存优化建议
    inventory_metrics = inventory_optimization(df_clean)
    
    print("\n供应链数据分析完成！")
    print("生成的可视化图表已保存为PNG文件。")

if __name__ == "__main__":
    main()
