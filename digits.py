#scikit-learnを学ぶ
#Chainer Tutorials のやり方を参考にする

from matplotlib import pyplot as plt

'''
Step1 : データセットの準備
'''

from sklearn.datasets import load_digits
dataset = load_digits()

x = dataset.data
t = dataset.target

print("x_shape : ",x.shape)
print("t_shape : ",t.shape)

# データセットを分割する関数の読み込み
from sklearn.model_selection import train_test_split

# 訓練用データセットとテスト用データセットへの分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

'''
前処理1 : 標準化


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(x_train)

# 平均
scaler.mean_
# 分散
scaler.var_

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#結果はやはり変わらなかった
'''

'''
前処理2 : べき変換


from sklearn.preprocessing import PowerTransformer

scaler = PowerTransformer()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#結果は悪くなった
'''

'''
Step2~4 : モデル・目的関数・最適手法を決める
'''

from sklearn.linear_model import LinearRegression

# モデルの定義
reg_model = LinearRegression()


'''
Step5 : モデルの訓練
'''

# モデルの訓練
reg_model.fit(x_train, t_train)

# 訓練後のパラメータ w
reg_model.coef_

# 訓練後のバイアス b
reg_model.intercept_

# 精度の検証
print("train score : ",reg_model.score(x_train, t_train))

#テスト用データセットで評価
print("test score : ",reg_model.score(x_test, t_test))