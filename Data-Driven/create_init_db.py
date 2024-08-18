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
    TIME    = 0
    VOLTAGE = 1
    MVAVG   = 2
    FSTLAG  = 3
    RTGT    = 4
    RACT    = 5
    PWM     = 6

# データセットのクラス
class DataSet:
    def __init__(self, time, required_point, pid_gain):
        self.time = time
        self.required_point = required_point
        self.pid_gain = pid_gain

# 操業データのクラス
class InitData:
    # 初期化
    def __init__(self, file_path, n_u, n_y, init_gain):
        self.file_path = file_path
        self.n_u = n_u
        self.n_y = n_y
        self.data = None
        self.dataset = []
        self.init_gain = init_gain
        # 初期化時にデータを読み込む
        self.load()
    
    # ファイル読み込み
    def load(self):
        self.data = np.loadtxt(self.file_path, delimiter=',', skiprows=1, encoding='utf-8')

    # データベース作成
    def create(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        # 操業データを用いた時刻tにおける要求点の生成 (t: 時間, r_0:目標値, y_0:出力値, u_0:入力値)
        # [[t, [r_0(t+1), r_0(t), y_0(t), ..., y_0(t-n_y+1), u_0(t), ..., u_0(t-n_u+1)], [Kp, Ki, Kd]], ...]
        for i in range(self.n_y, len(self.data) - 1):
            t_0 = self.data[i, Header.TIME] # 時間
            r_0 = [self.data[i+1, Header.RTGT], self.data[i, Header.RTGT]] # 目標値
            y_0 = [self.data[i-j, Header.RACT] for j in range(self.n_y)] # 出力値
            u_0 = [self.data[i-j, Header.PWM] for j in range(self.n_u)] # 入力値
            self.dataset.append(DataSet(t_0, r_0 + y_0 + u_0, self.init_gain)) # データセットに追加

# ------------------- メイン処理 -------------------
# 初期データベースの作成
initdata = InitData(INIT_DATA, n_u, n_y, INIT_GAIN) # 操業データ読み込み
initdata.create() # 操業データベース作成

# データベースにあるすべての情報ベクトルのi番目の要素の中で，最も大きな要素と最も小さな要素
max_m = []
min_m = []
for i in range(0, n_u + n_y + 2):
    max_m.append(max([data.required_point[i] for data in initdata.dataset]))
    min_m.append(min([data.required_point[i] for data in initdata.dataset]))

max_min_diff = [x - y for x, y in zip(max_m, min_m)]
# print(max_m, '\n', min_m, max_min_diff) # 最大値，最小値，最大値-最小値

# 要求点とデータベース内の情報ベクトルの距離を計算して配列に格納する．ここでは，重み付きL1ノルムを用いる.
# 距離 = Σ(|(要求点i - データベース内の情報ベクトルij)| / (最大値i - 最小値i))   

entire_distance = []
distances = []
# len(initdata.dataset)
for j in range(0, 3):
    for data in initdata.dataset:
        distance = 0
        for i in range(0, n_u + n_y + 2):
            distance += abs(data.required_point[i] - initdata.dataset[j].required_point[i]) / max_min_diff[i]
        distances.append(distance)
    entire_distance.append(distances)
    print(j)

# smallest_n_elements = sorted(array)[:n]

# for index, data in enumerate(initdata.dataset):
#     distance = 0
#     for i in range(0, n_u + n_y + 2):
#         distance += abs(data.required_point[i] - initdata.dataset[index].required_point[i]) / max_min_diff[i]
#     distances.append(distance)

print(distances) # 距離