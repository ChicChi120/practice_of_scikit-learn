#scikit-learnを学ぶ
import sklearn
print("Version : ",sklearn.__version__)

#Chainer Tutorials

'''
#Step1 の改善 : データセットの準備 + 前処理
'''
from sklearn.datasets import load_boston
dataset = load_boston()

x = dataset.data
t = dataset.target

#xとtを訓練用データセットとテスト用データセットに分割
#ホールドアウト法で分割する.

# データセットを分割する関数の読み込み
from sklearn.model_selection import train_test_split

# 訓練用データセットとテスト用データセットへの分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

from sklearn.preprocessing import PowerTransformer

scaler = PowerTransformer()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


'''
Step2~4 : モデル・目的関数・最適化手法を決める
'''
#LinearRegressionは最小二乗法を行うクラス

from sklearn.linear_model import LinearRegression

# モデルの定義
reg_model = LinearRegression()

'''
Step5 : モデルの訓練
'''
# モデルの訓練
reg_model.fit(x_train_scaled, t_train)

# 訓練後のパラメータ 重みw
print("w = ",reg_model.coef_)

# 訓練後のバイアス バイアスb
print("b = ",reg_model.intercept_)

# 精度の検証
print("score : ",reg_model.score(x_train_scaled, t_train))

#テスト用データセットで評価
print("Test : ",reg_model.score(x_test_scaled, t_test))

'''
結果が変わらなかったので標準化とは別のべき変換をする前処理をした
'''