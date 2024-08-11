# -*- coding: utf-8 -*-

# 固定PID制御器によって得られた操業データを用いて、初期データベースを作成する

import numpy as np
from enum import IntEnum
from dataclasses import dataclass


INIT_DATA = './Data-Driven/pid_sikoku24.csv' # 初期データのファイル名
INIT_GAIN = [40, 10, 0.5] # 初期のPID制御器のゲイン
n_u = 2 # 入力変数の数
n_y = 2 # 出力変数の数

# ------------------- クラス定義 -------------------
# ヘッダーの列挙型
class Header(IntEnum):
    TIME = 0
    VOLTAGE = 1
    MVAVG   = 2
    FSTLAG = 3
    RTGT = 4
    RACT = 5
    PWM = 6

# データセットのクラス
class DataSet:
    def __init__(self, time, required_point, pid_gain):
        self.time = time
        self.required_point = required_point
        self.pid_gain = pid_gain

# 操業データのクラス
class InitData:
    def __init__(self, file_path, n_u, n_y, init_gain):
        self.file_path = file_path
        self.n_u = n_u
        self.n_y = n_y
        self.data = None
        self.dataset = []
        self.init_gain = init_gain
        # 初期化時にデータを読み込む
        self.load()

    def load(self):
        self.data = np.loadtxt(self.file_path, delimiter=',', skiprows=1, encoding='utf-8')

    def create(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        # 操業データを用いた時刻tにおける要求点の生成 (t: 時間, r_0:目標値, y_0:出力値, u_0:入力値)
        # [[t, [r_0(t+1), r_0(t), y_0(t), ..., y_0(t-n_y+1), u_0(t), ..., u_0(t-n_u+1)], [Kp, Ki, Kd]], ...]
        for i in range(self.n_y, len(self.data) - 1):
            t_0 = self.data[i, Header.TIME]
            r_0 = [self.data[i+1, Header.RTGT], self.data[i, Header.RTGT]]
            y_0 = [self.data[i-j, Header.RACT] for j in range(self.n_y)]
            u_0 = [self.data[i-j, Header.PWM] for j in range(self.n_u)]
            self.dataset.append(DataSet(t_0, r_0 + y_0 + u_0, self.init_gain))

# ------------------- メイン処理 -------------------
# 初期データベースの作成
initdata = InitData(INIT_DATA, n_u, n_y, INIT_GAIN)
initdata.create()
print(initdata.dataset[0].time)
