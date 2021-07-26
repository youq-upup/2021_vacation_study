import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# 定义随机生成信号序列函数
def information(n):
    signal = np.array([])
    for j in range(n):
        x = random.random()
        if x >= 0.5:
            x = 1
        else:
            x = 0
        signal = np.insert(signal, len(signal), x)
    return signal


# 定义加性高斯白噪声
def awgn(y, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(y ** 2) / len(y)
    npower = xpower / snr
    return np.random.randn(len(y)) * np.sqrt(npower) + y


# 经过串并转换得到I、Q两路信号-----------------------------------------------------------------------------------------------
N = 20  # 码元个数
T = 1  # 码元周期
fc = 2  # 载波频率
Fs = 100  # 采样频率
bitstream = information(N)
bitstream = 2*bitstream-1  # 映射编码
x_I = np.array([])
x_Q = np.array([])
for i in range(1, N+1):
    if np.mod(i, 2) != 0:
        x_I = np.insert(x_I, len(x_I), bitstream[i-1])  # I路信号为bitstream[]的偶数位置序列
    else:
        x_Q = np.insert(x_Q, len(x_Q), bitstream[i-1])  # Q路信号为bitstream[]的奇数位置序列
bit_data = np.array([])
# 将码元序列扩展
for i in range(1, N+1):
    bit_data = np.insert(bit_data, len(bit_data), bitstream[i-1]*np.ones(T*Fs))
I_data = np.array([])
Q_data = np.array([])
for i in range(1, int(N/2)+1):
    I_data = np.insert(I_data, len(I_data), x_I[i-1]*np.ones(T*Fs*2))  # 注意这里为了绘图方便，将I、Q两路信号的码元长度放大了两倍
    Q_data = np.insert(Q_data, len(Q_data), x_Q[i-1]*np.ones(T*Fs*2))
t = np.array([])
for i in np.arange(0, N*T, 1/Fs):
    t = np.insert(t, len(t), i)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置图片中的字体
plt.rcParams['axes.unicode_minus'] = False  # 识别负号
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(t, bit_data)
plt.legend(["Bitstream"], loc='upper right')
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
plt.subplot(3, 1, 2)
plt.plot(t, I_data)
plt.legend(["I_Bitstream"], loc='upper right')
x_major_locator = MultipleLocator(2)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.subplot(3, 1, 3)
plt.plot(t, Q_data)
plt.legend(["Q_Bitstream"], loc='upper right')
x_major_locator = MultipleLocator(2)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
# QPSK信号调制-----------------------------------------------------------------------------------------------------------
bit_t = np.array([])
for i in np.arange(0, 2*T, 1/Fs):
    bit_t = np.insert(bit_t, len(bit_t), i)
I_carrier = np.array([])
Q_carrier = np.array([])
for i in range(1, int(N/2)+1):
    I_carrier = np.insert(I_carrier, len(I_carrier), x_I[i-1]*np.cos(2*np.pi*fc*bit_t))
    Q_carrier = np.insert(Q_carrier, len(Q_carrier), x_Q[i-1]*np.cos(2*np.pi*fc*bit_t+np.pi/2))
QPSK_signal = I_carrier+Q_carrier
plt.figure(2)
plt.subplot(4, 1, 1)
plt.plot(t, I_carrier)
plt.legend(["I_signal"], loc='upper right')
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.subplot(4, 1, 2)
plt.plot(t, Q_carrier)
plt.legend(["Q_signal"], loc='upper right')
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.subplot(4, 1, 3)
plt.plot(t, QPSK_signal)
plt.legend(["QPSK_signal"], loc='upper right')
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
# QPSK调制信号解调--------------------------------------------------------------------------------------------------------
SNR = 1
QPSK_receive = awgn(QPSK_signal, SNR)
I_recover = np.array([])
Q_recover = np.array([])
plt.subplot(4, 1, 4)
plt.plot(t, QPSK_receive)
plt.legend(["QPSK_receive"], loc='upper right')
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
# 接收信号与恢复载波相乘
for i in range(1, int(N/2)+1):
    I_output = QPSK_receive[(i-1)*len(bit_t):i*len(bit_t)]*np.cos(2*np.pi*fc*bit_t)
    if np.sum(I_output) > 0:
        I_recover = np.insert(I_recover, len(I_recover), 1)
    else:
        I_recover = np.insert(I_recover, len(I_recover), -1)
    Q_output = QPSK_receive[(i-1)*len(bit_t):i*len(bit_t)]*np.cos(2*np.pi*fc*bit_t+np.pi/2)
    if np.sum(Q_output) > 0:
        Q_recover = np.insert(Q_recover, len(Q_recover), 1)
    else:
        Q_recover = np.insert(Q_recover, len(Q_recover), -1)
# 并串转换
bit_recover = np.array([])
for i in range(1, N+1):
    if np.mod(i, 2) != 0:
        bit_recover = np.insert(bit_recover, len(bit_recover), I_recover[int((i-1)/2)])
    else:
        bit_recover = np.insert(bit_recover, len(bit_recover), Q_recover[int(i/2)-1])
recover_data = np.array([])
for i in range(1, N+1):
    recover_data = np.insert(recover_data, len(recover_data), bit_recover[i-1]*np.ones(T*Fs))
I_recover_data = np.array([])
Q_recover_data = np.array([])
for i in range(1, int(N/2)+1):
    I_recover_data = np.insert(I_recover_data, len(I_recover_data), I_recover[i-1]*np.ones(T*Fs*2))
    Q_recover_data = np.insert(Q_recover_data, len(Q_recover_data), Q_recover[i-1]*np.ones(T*Fs*2))
plt.figure(3)
plt.subplot(3, 1, 1)
plt.plot(t, recover_data)
plt.legend(["Bitstream"], loc='upper right')
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.subplot(3, 1, 2)
plt.plot(t, I_recover_data)
plt.legend(["I_Bitstream"], loc='upper right')
x_major_locator = MultipleLocator(2)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.subplot(3, 1, 3)
plt.plot(t, Q_recover_data)
plt.legend(["Q_Bitstream"], loc='upper right')
x_major_locator = MultipleLocator(2)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()
