from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Point, Daily
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os
from matplotlib.font_manager import FontProperties

# 设置中文字体
font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SimHei.ttf')
custom_font = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

def get_weather_data():
    """获取北京气温数据"""
    file_path = 'beijing_weather.csv'
    
    if os.path.exists(file_path):
        print("从本地加载天气数据...")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
    else:
        print("从Meteostat获取天气数据...")
        # 北京气象站点
        beijing = Point(39.9042, 116.4074, 45)
        
        # 获取2010年到现在的数据
        start = datetime(2010, 1, 1)
        end = datetime.now()
        
        # 获取每日数据
        data = Daily(beijing, start, end)
        df = data.fetch()
        
        # 重置索引，将日期变为列
        df = df.reset_index()
        df = df.rename(columns={'time': 'date'})
        
        # 保存数据
        df.to_csv(file_path, index=False)
    
    return df[['date', 'tavg', 'tmin', 'tmax']]

def create_features(df):
    """创建特征"""
    df = df.copy()
    
    # 提取日期特征
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    
    # 滞后特征
    for i in range(1, 8):  # 使用前7天的数据
        df[f'tavg_lag_{i}'] = df['tavg'].shift(i)
        df[f'tmin_lag_{i}'] = df['tmin'].shift(i)
        df[f'tmax_lag_{i}'] = df['tmax'].shift(i)
    
    # 移动平均
    for window in [7, 14, 30]:
        df[f'tavg_ma_{window}'] = df['tavg'].rolling(window=window).mean()
        df[f'tmin_ma_{window}'] = df['tmin'].rolling(window=window).mean()
        df[f'tmax_ma_{window}'] = df['tmax'].rolling(window=window).mean()
    
    # 温度范围
    df['temp_range'] = df['tmax'] - df['tmin']
    
    return df.dropna()

def prepare_data(df, target_col='tavg'):
    """准备训练和测试数据"""
    df = create_features(df)
    
    # 分离2024年7-10月的数据作为测试集
    train_mask = (df['date'] < '2024-07-01')
    test_mask = (df['date'] >= '2024-07-01') & (df['date'] <= '2024-10-31')
    
    # 分离特征
    feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
    
    # 准备训练集和测试集
    X_train = df[train_mask][feature_cols]
    X_test = df[test_mask][feature_cols]
    y_train = df[train_mask][target_col]
    y_test = df[test_mask][target_col]
    test_dates = df[test_mask]['date']
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, test_dates, feature_cols

def plot_predictions(dates, y_test, y_pred):
    """绘制预测结果"""
    plt.figure(figsize=(15, 8))
    plt.plot(dates, y_test, label='实际气温', color='black', linewidth=2)
    plt.plot(dates, y_pred, label='预测气温', color='blue', alpha=0.7, linewidth=2)
    
    plt.title('北京2024年7-10月气温预测对比', fontproperties=custom_font, fontsize=14)
    plt.xlabel('日期', fontproperties=custom_font, fontsize=12)
    plt.ylabel('气温 (°C)', fontproperties=custom_font, fontsize=12)
    plt.legend(prop=custom_font)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('temperature_prediction.png', dpi=300, bbox_inches='tight')
    return plt.gcf()

def plot_feature_importance(feature_names, importance, top_n=10):
    """绘制特征重要性"""
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_imp = feature_imp.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n), feature_imp['importance'])
    plt.xticks(range(top_n), feature_imp['feature'], rotation=45, ha='right')
    
    plt.title('特征重要性排序', fontproperties=custom_font, fontsize=14)
    plt.xlabel('特征', fontproperties=custom_font)
    plt.ylabel('重要性', fontproperties=custom_font)
    plt.tight_layout()
    
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    return plt.gcf()

def main():
    """主函数"""
    try:
        # 设置随机种子
        np.random.seed(42)
        
        # 1. 获取数据
        df = get_weather_data()
        print(f"数据获取完成，共 {len(df)} 条记录")
        
        # 2. 准备数据
        X_train, X_test, y_train, y_test, test_dates, feature_cols = prepare_data(df)
        print(f"数据预处理完成，训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        # 3. 训练随机森林模型
        print("\n开始训练随机森林模型...")
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. 预测和评估
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n模型评估结果:")
        print(f"R2分数: {r2:.4f}")
        print(f"均方误差(MSE): {mse:.4f}")
        print(f"均方根误差(RMSE): {rmse:.4f}")
        print(f"平均绝对误差(MAE): {mae:.4f}")
        
        # 5. 绘制预测图
        plot_predictions(test_dates, y_test, y_pred)
        
        # 6. 绘制特征重要性
        plot_feature_importance(feature_cols, model.feature_importances_)
        
        # 7. 输出预测结果
        results_df = pd.DataFrame({
            '日期': test_dates,
            '实际气温': y_test,
            '预测气温': y_pred
        })
        results_df.to_csv('temperature_predictions.csv', index=False, encoding='utf-8-sig')
        
        plt.show()
        
        return model, results_df
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    model, results_df = main()