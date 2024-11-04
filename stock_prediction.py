import efinance as ef
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os
from matplotlib.font_manager import FontProperties
import time

# 设置中文字体
font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SimHei.ttf')
custom_font = FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

def get_stock_data(stock_code='GOOGL', start_date='20230101', end_date='20241130'):
    """获取股票数据"""
    file_path = f'results/{stock_code}_history.csv'
    
    if os.path.exists(file_path):
        print(f"从本地加载{stock_code}的历史数据...")
        df = pd.read_csv(file_path)
    else:
        print(f"从网络获取{stock_code}的历史数据...")
        df = ef.stock.get_quote_history(stock_code, beg=start_date, end=end_date)
        df.to_csv(file_path, encoding='utf-8', index=False)
    
    df['日期'] = pd.to_datetime(df['日期'])
    cols_to_keep = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率']
    df = df[cols_to_keep]
    
    return df

def create_features(df):
    """创建特征"""
    df = df.copy()
    
    # 确保数值类型
    numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 成交量相关特征
    df['volume_price_ratio'] = df['成交量'] / df['收盘']
    
    return df.dropna()

def prepare_data(df, target_col='收盘', test_size=0.2):
    """准备训练和测试数据"""
    df = create_features(df)
    
    feature_columns = [col for col in df.columns if col != target_col and col != '日期']
    X = df[feature_columns].astype(float)
    y = df[target_col].astype(float)
    
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:-1]
    y_train, y_test = y[1:split_idx+1], y[split_idx+1:]
    # y_test[-1:] = 300
    #import ipdb; ipdb.set_trace()    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, df['日期'][split_idx:]
def plot_predictions_dynamic(dates, y_test, y_pred, stock_code='GOOGL', start_from=100):
    """动态绘制预测结果，预测点和实际点交替显示"""
    plt.ion()  # 打开交互模式
    
    # 转换数据为numpy数组，确保可以通过索引访问
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # 从指定天数开始的数据
    dates = dates[start_from:]
    y_test = y_test[start_from:]
    y_pred = y_pred[start_from:]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('股价预测结果和实际股价对比图', fontproperties=custom_font, fontsize=14, pad=20)
    plt.xlabel('日期', fontproperties=custom_font, fontsize=12)
    plt.ylabel('股价', fontproperties=custom_font, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(-1, len(dates))
    ax.set_ylim(min(min(y_test), min(y_pred)) * 0.95,
                max(max(y_test), max(y_pred)) * 1.05)
    
    # 处理日期刻度
    date_list = dates.tolist()
    tick_indices = range(0, len(date_list), max(1, len(date_list)//10))
    date_labels = [date_list[i].strftime('%Y-%m-%d') if isinstance(date_list[i], pd.Timestamp) 
                  else pd.Timestamp(date_list[i]).strftime('%Y-%m-%d') 
                  for i in tick_indices]
    plt.xticks(list(tick_indices), date_labels, rotation=45)
    
    # 初始化线条和散点
    actual_line, = ax.plot([], [], 'black', label='实际值', linewidth=2)
    pred_line, = ax.plot([], [], 'blue', label='预测值', alpha=0.7, linewidth=2)
    actual_scatter = ax.scatter([], [], color='black', s=50, alpha=0.6)
    pred_scatter = ax.scatter([], [], color='blue', s=50, alpha=0.6)
    
    plt.legend(prop=custom_font)
    plt.tight_layout()
    
    # 动态更新数据
    actual_x_data = []
    actual_y_data = []
    pred_x_data = []
    pred_y_data = []
    
    try:
        # 交替显示预测点和实际点
        for i in range(len(y_test)):
            # 先显示预测点
            pred_x_data.append(i)
            pred_y_data.append(y_pred[i])
            pred_line.set_data(pred_x_data, pred_y_data)
            pred_scatter.set_offsets(np.c_[pred_x_data, pred_y_data])
            
            # 更新图形
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.3)  # 暂停一小段时间
            
            # 再显示实际点
            actual_x_data.append(i)
            actual_y_data.append(y_test[i])
            actual_line.set_data(actual_x_data, actual_y_data)
            actual_scatter.set_offsets(np.c_[actual_x_data, actual_y_data])
            
            # 更新图形
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)  # 暂停一小段时间
    except Exception as e:
        print(f"绘图过程中发生错误: {str(e)}")
        print(f"当前索引: {i}")
        print(f"数据形状: y_test: {y_test.shape}, y_pred: {y_pred.shape}")
    
    plt.ioff()  # 关闭交互模式
    plt.show()
    
    return fig

def plot_feature_importance(feature_columns, importance, top_n=10):
    """绘制特征重要性"""
    importance_dict = dict(zip(feature_columns, importance))
    importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    plt.figure(figsize=(12, 6))
    plt.bar(importance_sorted.keys(), importance_sorted.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('特征重要性排序', fontproperties=custom_font, fontsize=14)
    plt.xlabel('特征', fontproperties=custom_font)
    plt.ylabel('重要性', fontproperties=custom_font)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def main():
    """主函数"""
    try:
        # 设置随机种子
        np.random.seed(42)
        
        # 1. 获取数据
        # 提示用户输入股票代码
        stock_code = input("请输入股票代码（例如：MSFT）：")
        # 提示用户输入开始日期
        start_date = input("请输入开始日期（YYYYMMDD）：")
        # 提示用户输入结束日期
        end_date = input("请输入结束日期（YYYYMMDD）：")
        # 获取股票数据
        df = get_stock_data(stock_code, start_date=start_date, end_date=end_date)

        print(f"数据获取完成，共 {len(df)} 条记录")
        
        # 2. 准备数据
        X_train, X_test, y_train, y_test, scaler, feature_columns, test_dates = prepare_data(df)
        print(f"数据预处理完成，训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        # 3. 训练随机森林模型
        print("\n开始训练随机森林模型...")
        model = RandomForestRegressor(n_estimators=200, 
                                    max_depth=10,
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    random_state=42)
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
        
        # 5. 绘制动态预测图
        fig = plot_predictions_dynamic(test_dates, y_test, y_pred, stock_code, start_from=20)
        plt.show()
        
        # 6. 绘制特征重要性
        plot_feature_importance(feature_columns, model.feature_importances_)
        
        return model, scaler, feature_columns
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    model, scaler, feature_columns = main()