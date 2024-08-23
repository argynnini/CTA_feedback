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
n = 4  # 取り出す要素の数
nearest_distance = np.empty((for_max - for_min, n))
nearest_data = np.empty((for_max - for_min, n, dataset.shape[1]))
for i in range(for_min, for_max):
    print('\r近傍計算中 ', i+1,' / ', for_max, end='')
    # 配列をソートしてインデックスを取得し、小さい順にn個のインデックスを取り出す
    nearest_indices = np.argsort(distances[i])[:n]
    # 近傍データを取り出す
    nearest_distance[i - for_min] = distances[i][nearest_indices]
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

# 近傍に対する重みwの計算(自分自身は除外)
nearest_distance_without_self = nearest_distance[:, 1:]
weights = np.empty((for_max - for_min, nearest_distance_without_self.shape[1]))
for i in range(for_min, for_max):
    print('\r重み計算中 ', i+1,' / ', for_max, end='')
    # 重みの計算
    exp_distances = np.exp(-nearest_distance_without_self[i])
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
    for j in range(0, n - 1): # 近傍データの数(自分自身は除外なので-1)
        Kp_old += weights[i][j] * pid_gain[i][0]
        Ki_old += weights[i][j] * pid_gain[i][1]
        Kd_old += weights[i][j] * pid_gain[i][2]
    local_model[i - for_min] = [Kp_old, Ki_old, Kd_old]
del Kp_old, Ki_old, Kd_old  # 不要な変数を削除
print('\n完了')

# それぞれのゲインの最大値
print(np.max(local_model, axis=0))
print('\n')

# local model から重複を削除
local_model_unique = np.unique(local_model, axis=0)
print('\n')
# print(local_model)  # 局所モデル

# 次に，Kp, Ki, Kdの値に対して，次の最急降下法を用いて，学習を行う．
# θnew = θold(t) - η * ∂J(t+1) / ∂old(t)
# θ = [Kp, Ki, Kd], η = 学習係数ベクトル[ηp, ηi, ηd]
# J(t) = 1/2{(y0(t) - yr(t))^2 + λ * fsΔu(t-1)^2}
# yr(t)は疑似参照入力r(t)と参照モデルGm(z^-1)を用いて計算する．
# yr(t) = Gm(z^-1) * r(t)
# r(t) = {Δu0(t) + (Kp(t)+Ki(t)+Kd(t))y0(t) - (Kp(t)+2Kd(t))y0(t-1) + Kd(t)y0(t-2)} / Ki(t)}
# 参照モデルGm(z^-1)は2次遅れ系として，次のように表される．
# Gm(z^-1) = z^-1 P(1) / P(z^-1)
# ただし，P(z^-1) はGm(z^-1)の特性多項式であり，次のように表される．
# P(z^-1) = 1 + p1 z^-1 + p2 z^-2
# p1 = -2exp(-ρ/2u)cos(ρ*√(4u-1)/2u)
# p2 = exp(-ρ/u)
# ρ := Ts / σ
# u := 0.25(1-σ) + 0.51δ
# σ, δ はそれぞれ制御系の立ち上がり特性と減衰特性を表し，設計者が任意に設定する．
# Ts はサンプリング時間である．
# また，Δu(t)を次式で表す．
# Δu(t) = Ki(t){r(t) - yr(t)} - Kp(t)yr(t) - Kd(t)Δyr(t)^2
# 以上の関係から，θnewの右辺第2項の微分は次のようになる．
# ∂J(t+1) / ∂Kp(t) = ∂J(t+1) / ∂yr(t+1) * ∂yr(t+1) / ∂r(t) * ∂r(t) / ∂Kp(t) + λ * fs * ∂Δu(t)^2 / ∂Δu(t) * ∂Δu(t) / ∂Kp(t)
# ∂J(t+1) / ∂Ki(t) = ∂J(t+1) / ∂yr(t+1) * ∂yr(t+1) / ∂r(t) * ∂r(t) / ∂Ki(t) + λ * fs * ∂Δu(t)^2 / ∂Δu(t) * ∂Δu(t) / ∂Ki(t)
# ∂J(t+1) / ∂Kd(t) = ∂J(t+1) / ∂yr(t+1) * ∂yr(t+1) / ∂r(t) * ∂r(t) / ∂Kd(t) + λ * fs * ∂Δu(t)^2 / ∂Δu(t) * ∂Δu(t) / ∂Kd(t)
# これらの微分を用いて，θnewを求める．

# 参照モデルの特性多項式の係数
def calc_p1(rho):
    return -2 * np.exp(-rho / 2) * np.cos(rho * np.sqrt(4 - 1) / 2)
def calc_p2(rho):
    return np.exp(-rho)
def calc_p(rho):
    return np.array([1, calc_p1(rho), calc_p2(rho)])
# 参照モデルの特性
def calc_Gm(rho):
    return calc_p(1) / calc_p(rho)
# Δu(t)の計算
def calc_delta_u(Kp, Ki, Kd, r, yr, y, y_1, y_2, delta_yr):
    return Ki * (r - yr) - Kp * yr - Kd * delta_yr**2
# r(t)の計算
def calc_r(Kp, Ki, Kd, y, y_1, y_2):
    return (y + (Kp + Ki + Kd) * y - (Kp + 2 * Kd) * y_1 + Kd * y_2) / Ki
# yr(t)の計算
def calc_yr(Kp, Ki, Kd, r, y, y_1, y_2):
    return calc_Gm(1) * r
# J(t)の計算
def calc_J(y, yr, lambda_fs, delta_u):
    return 0.5 * (y - yr)**2 + lambda_fs * delta_u**2
# ∂J(t+1) / ∂Kp(t)の計算
def calc_dJ_dKp(y, yr, r, Kp, Ki, Kd, lambda_fs, delta_u, delta_yr):
    return (y - yr) * (r - Kp * yr - Kd * delta_yr**2) + lambda_fs * delta_u * delta_yr**2
# ∂J(t+1) / ∂Ki(t)の計算
def calc_dJ_dKi(y, yr, r, Kp, Ki, Kd, lambda_fs, delta_u, delta_yr):
    return (y - yr) * (r - Kp * yr - Kd * delta_yr**2)
# ∂J(t+1) / ∂Kd(t)の計算
def calc_dJ_dKd(y, yr, r, Kp, Ki, Kd, lambda_fs, delta_u, delta_yr):
    return (y - yr) * (r - Kp * yr - Kd * delta_yr**2) + lambda_fs * delta_u * 2 * Kd * delta_yr
# θnewの計算
def calc_theta_new(Kp, Ki, Kd, dJ_dKp, dJ_dKi, dJ_dKd, eta_p, eta_i, eta_d):
    return [Kp - eta_p * dJ_dKp, Ki - eta_i * dJ_dKi, Kd - eta_d * dJ_dKd]

# パラメータ
lambda_fs = 1  # λ * fs
eta_p = 0.1  # ηp
eta_i = 0.1  # ηi
eta_d = 0.1  # ηd
sigma = 0.5  # σ
delta = 0.5  # δ
Ts = 1  # サンプリング時間
rho = Ts / sigma  # ρ
u = 0.25 * (1 - sigma) + 0.51 * delta  # u
Gm = calc_Gm(rho)  # 参照モデル
p = calc_p(rho)  # 参照モデルの特性多項式
print(Gm, p)
