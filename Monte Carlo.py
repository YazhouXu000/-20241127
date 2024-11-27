import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载训练好的模型
loaded_model = joblib.load('final_model.pkl')

# 定义参数范围
num_simulations = 10000
random_parameters = {
    'Heating rate (℃/min)': np.random.uniform(1, 30, num_simulations),
    'Steam': np.random.uniform(0.0, 0.3, num_simulations),
    'WHSV': np.random.uniform(10000, 152788, num_simulations),
    'Temperature': np.random.uniform(500, 900, num_simulations)
}

# 生成随机参数组合
random_combinations = pd.DataFrame(random_parameters)

# 使用已加载的模型预测每个组合的效率
predicted_efficiencies = loaded_model.predict(random_combinations)

# 找出最高效的参数组合
optimal_index = np.argmax(predicted_efficiencies)
optimal_parameters = random_combinations.iloc[optimal_index]
optimal_efficiency = predicted_efficiencies[optimal_index]

# 输出最优参数组合及其预测效率
print("最优参数组合：")
print(optimal_parameters)
print("预测效率：", optimal_efficiency)

# 创建三维散点图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# # 三个主要参数作为坐标轴
# x = random_combinations['Heating rate (℃/min)']
# y = random_combinations['Steam']
# z = random_combinations['Temperature']
# c = predicted_efficiencies
# 三个主要参数作为坐标轴
x = random_combinations['WHSV']
y = random_combinations['Steam']
z = random_combinations['Temperature']
c = predicted_efficiencies

# 绘制散点图
sc = ax.scatter(x, y, z, c=c, cmap='viridis', alpha=0.6)
plt.colorbar(sc, label='Predicted Efficiency')

# 标记最优参数组合
ax.scatter(optimal_parameters['WHSV'],
           optimal_parameters['Steam'],
           optimal_parameters['Temperature'],
           color='red', s=100, label='Optimal Combination')

ax.set_xlabel('WHSV')
ax.set_ylabel('Steam')
ax.set_zlabel('Temperature')
plt.title('Monte Carlo Simulation for Parameter Optimization (3D)')

plt.legend()
plt.savefig("Monte Carlo_1")
