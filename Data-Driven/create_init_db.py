# -*- coding: utf-8 -*-

# 固定PID制御器によって得られた操業データを用いて、初期データベースを作成する

import numpy as np
from enum import IntEnum


# INIT_DATA = './Data-Driven/pid_sikoku24.csv' # 初期データのファイル名
INIT_DATA = './Data-Driven/pid_test.csv' # 初期データのファイル名
INIT_GAIN = [40.0, 10.0, 0.50] # 初期のPID制御器のゲイン
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
        # [[t, [r_0(t+1), r_0(t), y_0(t), ..., y_0(t-n_y+1), u_0(t-1), ..., u_0(t-n_u+1)], [Kp, Ki, Kd]], ...]
        for i in range(0, len(self.data)):
            t_0 = self.data[i, Header.TIME] # 時間
            r_0 = [self.data[i+1 if i < len(self.data)-1 else i, Header.RTGT],
                   self.data[i, Header.RTGT]] # 目標値(最後の行は最後の行の値を使う)
            y_0 = [self.data[i-j if i >= j else 0, Header.RACT] for j in range(0, self.n_y)] # 出力値(最初の行は最初の行の値を使う)
            u_0 = [self.data[i-j if i >= j else 0, Header.PWM] for j in range(1, self.n_u)] # 入力値(最初の行は最初の行の値を使う)
            self.dataset.append(DataSet(t_0, r_0 + y_0 + u_0, self.init_gain)) # データセットに追加

# ------------------- メイン処理 -------------------
# 初期データベースの作成
initdata = InitData(INIT_DATA, n_u, n_y, INIT_GAIN) # 操業データ読み込み
initdata.create() # 操業データベース作成

# データセットの抽出
dataset = np.array([data.required_point for data in initdata.dataset])
pid_gain = np.array([data.pid_gain for data in initdata.dataset])

# for文の範囲(デバッグ用)
for_min = 0
for_max = len(initdata.dataset)

# 要求点とデータベース内の情報ベクトルの距離を計算して配列に格納する．ここでは，重み付きL1ノルムを用いる.
# 距離 = Σ(|(要求点i - データベース内の情報ベクトルij)| / (最大値i - 最小値i))   

# データベースにあるすべての情報ベクトルのi番目の要素の中で，最も大きな要素と最も小さな要素(分母)
max_m = np.max(dataset, axis=0)
min_m = np.min(dataset, axis=0)
max_min_diff = max_m - min_m # 最大値-最小値(分母)

# 重み付きL1ノルムの計算
distances = np.empty((for_max - for_min, dataset.shape[0]))
print('\n\r\r')
for j in range(for_min, for_max):
    print('\r距離計算中 ', j+1,' / ', for_max, end='')
    distance = np.abs(dataset - dataset[j]) / max_min_diff
    distance = np.sum(distance, axis=1)
    distances[j - for_min] = distance  # NumPy 配列に直接格納
del distance  # 不要な変数を削除
print('\n完了')

# 距離djが小さいものからn個の情報ベクトルを近傍データとして取り出す
n = 3  # 取り出す要素の数
nearest_data = np.empty((for_max - for_min, n, dataset.shape[1]))

for i in range(for_min, for_max):
    print('\r近傍計算中 ', i+1,' / ', for_max, end='')
    # 配列をソートしてインデックスを取得し、小さい順にn個のインデックスを取り出す
    nearest_indices = np.argsort(distances[i])[:n]
    # 近傍データを取り出す
    nearest_data[i - for_min] = dataset[nearest_indices]
del nearest_indices  # 不要な変数を削除
print('\n完了')
# print(nearest_data)  # 近傍データ

# PIDゲインの算出
# 近傍に対し，重み付き線形平均法(linearly weighted average: LWA)により，局所モデルを構成する．
# [Kp_old, Ki_old, Kd_old] = Σ(i=0~n) 重みw_i * [Kp, Ki, Kd]
# [Kp, Ki, Kd] = [40, 10, 0.5]  # 初期のPID制御器のゲイン
# 重みw_i = exp(-d_i) / Σ(i=0~n) exp(-d_i)
# 重みw_iはΣ(i=0~n) w_i = 1を満たすように正規化する．
# d_i = Σ(j=0~m) |(要求点j - データベース内の情報ベクトルij)| / (最大値j - 最小値j) = distances

# 重みの計算
weights = np.empty((for_max - for_min, distances.shape[1]))

for i in range(for_min, for_max):
    print('\r重み計算中 ', i+1,' / ', for_max, end='')
    # 重みの計算
    exp_distances = np.exp(-distances[i])
    weight = exp_distances / np.sum(exp_distances)
    weights[i - for_min] = weight  # NumPy 配列に直接格納
del weight, exp_distances  # 不要な変数を削除
print('\n完了')

# 重み付き線形平均法による局所モデルの構成
local_model = np.empty((for_max - for_min, 3))

for i in range(for_min, for_max):
    print('\r局所モデル計算中 ', i+1,' / ', for_max, end='')
    # 重み付き線形平均法による局所モデルの構成
    Kp_old, Ki_old, Kd_old = 0, 0, 0
    for j in range(0, n):
        Kp_old += weights[i][j] * pid_gain[j][0]
        Ki_old += weights[i][j] * pid_gain[j][1]
        Kd_old += weights[i][j] * pid_gain[j][2]
    local_model[i - for_min] = [Kp_old, Ki_old, Kd_old]
del Kp_old, Ki_old, Kd_old  # 不要な変数を削除
print('\n完了')

# それぞれのゲインの最大値
print(np.max(local_model, axis=0))
print('\n')

# print(local_model)  # 局所モデル

