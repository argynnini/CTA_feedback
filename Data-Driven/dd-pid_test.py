import numpy as np
import scipy.signal as signal

import os
import atexit
import matplotlib as mpl
import matplotlib.pyplot as plt


# SSH接続中かどうかを判定する関数
def is_under_ssh_connection():
    # The environment variable `SSH_CONNECTION` exists only in the SSH session.
    return "SSH_CONNECTION" in os.environ.keys()


# WebAggを利用する関数
def use_WebAgg(port=8000, port_retries=50):
    """use WebAgg for matplotlib backend."""
    current_backend = mpl.get_backend()
    current_webagg_configs = {
        "port": mpl.rcParams["webagg.port"],
        "port_retries": mpl.rcParams["webagg.port_retries"],
        "open_in_browser": mpl.rcParams["webagg.open_in_browser"],
    }

    def reset():
        mpl.use(current_backend)
        mpl.rc("webagg", **current_webagg_configs)

    mpl.use("WebAgg")
    mpl.rc(
        "webagg",
        **{"port": port, "port_retries": port_retries, "open_in_browser": False}
    )
    atexit.register(reset)


# SSH接続中の場合はWebAggを利用
if is_under_ssh_connection():
    use_WebAgg(port=8000, port_retries=50)


# 非線形ブロック（多項式関数）
def nonlinear_block(u, t):
    if t < 70:
        return 1.5 * u[t] - 1.5 * u[t] ** 2 + 0.5 * u[t] ** 3 # システム1
    else:
        return 1.0 * u[t] - 1.0 * u[t] ** 2 + 1.0 * u[t] ** 3 # システム2


# 線形ブロック（FIRフィルタ）
# 線形ブロック
def linear_block(x, y, t, noise):
    if t == 0:
        return 0
    elif t == 1:
        return 0.6 * y[t - 1] + 1.2 * x[t - 1] + noise[t]
    else:
        return (
            0.6 * y[t - 1] - 0.1 * y[t - 2] + 1.2 * x[t - 1] - 0.1 * x[t - 2] + noise[t]
        )


# ガウス白色雑音を生成
def generate_noise(length, mean=0, variance=0.001):
    return np.random.normal(mean, np.sqrt(variance), length)


# Hammersteinモデルの出力を計算する関数
def hammerstein_model(u, noise):
    length = len(u)
    x = np.zeros(length)
    y = np.zeros(length)

    for t in range(length):
        x[t] = nonlinear_block(u, t)
        y[t] = linear_block(x, y, t, noise)

    return y


# 入力信号を生成
length = 100
u = np.linspace(0, 2.5, length)
# ガウス白色雑音を生成
noise = generate_noise(length)

# Hammersteinモデルの出力を計算
output_signal = hammerstein_model(u, noise)

# hammerstein_model関数の静特性をプロットu入力に対するシステム1,2の出力
plt.figure()
plt.plot(u, label="Input Signal")
plt.plot([nonlinear_block(u, t) for t in range(length)], label="Nonlinear Block Output")
plt.plot([linear_block(u, output_signal, t, noise) for t in range(length)], label="Linear Block Output")
plt.legend()


# 結果を表示
plt.figure()
plt.plot(u, label="Input Signal")
plt.plot(output_signal, label="Output Signal")
plt.legend()
plt.show()
