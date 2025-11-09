import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(8, 2.7), layout='constrained')
t=np.arange(0,5,0.01)
s=np.sin(2*np.pi*t)
l1, = ax1.plot(t, s)

ax2 = ax1.twinx() # 创建共享x轴的第二个y轴
l2, = ax2.plot(t, range(len(t)), color='orange')
ax12=ax1.twinx() # 创建共享x轴的第三个y轴
l3, =ax12.plot(t,t**2,color='#87CEFA')
print(range(len(t)))
ax2.legend([l1, l2, l3], ['Sine (left)', 'Straight (right)','HHHHH'])


ax3.plot(t, s)
ax3.set_xlabel('Angle [°]')
ax4 = ax3.secondary_xaxis('top', functions=(np.rad2deg, np.deg2rad))
ax4.set_xlabel('Angle [rad]')
plt.show()
'''

# --- 定义转换函数  ---
def celsius_to_kelvin(celsius):
    """将摄氏度转换为开尔文"""
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    """将开尔文转换为摄氏度"""
    return kelvin - 273.15

# --- 创建画布和子图 ---
# 调整 figsize 让图表更清晰
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 4), layout='constrained') # 创建两个子图ax1, ax3,1行2列,layout='constrained'自动调整布局
time = np.arange(0, 5, 0.01) # X轴：时间 (Time)


#左侧子图：三重Y轴演示


# --- Y1 (ax1): 振荡信号 (电流/电压) ---
signal = np.sin(2 * np.pi * time) * 10 
l1, = ax1.plot(time, signal, color='blue', label='Voltage (V)')
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Voltage (V)', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue') # 设置第一个轴的刻度颜色为蓝色

# --- Y2 (ax2): 线性背景噪音 ---
ax2 = ax1.twinx() # 创建共享x轴的第二个y轴
noise = time * 2 + 5
l2, = ax2.plot(time, noise, color='red', linestyle=':', label='Temperature (K)')
ax2.set_ylabel('Temperature (K)', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red') # 设置第二个轴的刻度颜色为红色

# --- Y3 (ax12): 能量/功率 (平方关系) ---
# 注意：第三个轴需要手动调整位置，否则会叠在第二个轴上
# 使用 make_artist_v2 创建一个独立的 Y 轴
ax12 = ax1.twinx()
ax12.spines['right'].set_position(('outward', 60)) # 将第三个轴向右平移 60 points,spines['right']表示右侧的轴线
power = signal**2 / 10
l3, = ax12.plot(time, power, color='green', linestyle='--', label='Power (W)')
ax12.set_ylabel('Power (W)', color='green', fontsize=12)
ax12.tick_params(axis='y', labelcolor='green') # 设置第三个轴的刻度颜色为绿色

# 统一图例 (放在 Y1 的轴上)
lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left') # 图例放在左上角
ax1.set_title('Three Quantities on Shared X-Axis')



#右侧子图：辅助轴演示（温度转换）

# --- X1 (ax3): 主轴 (摄氏度) ---
temp_c = np.arange(-50, 50, 1) # 摄氏度数据
data_y = np.exp(temp_c / 20) # 模拟一个随温度指数增长的物理量
ax3.plot(temp_c, data_y, color='purple')
ax3.set_xlabel('Temperature ($^{\circ}C$)', fontsize=12)
ax3.set_ylabel('Reaction Rate', fontsize=12)
ax3.set_title('Secondary Axis for Unit Conversion')

# --- X2 (ax4): 辅助轴 (开尔文) ---
ax4 = ax3.secondary_xaxis('top', 
                          functions=(celsius_to_kelvin, kelvin_to_celsius)) # 温度转化,top表示辅助轴在上方

ax4.set_xlabel('Temperature (K)', fontsize=12, color='darkorange')
ax4.tick_params(axis='x', labelcolor='darkorange') # 辅助轴使用橙色标记

# 绘制 y 轴的 0 刻度线作为参考
ax3.axhline(0, color='gray', linestyle='--', linewidth=0.5)

# --- 最终显示 ---
plt.tight_layout()
plt.show()