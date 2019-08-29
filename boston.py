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

#前処理(べき変換)
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

#LinearRegressionは最小二乗法を行うクラス

from sklearn.linear_model import LinearRegression

# モデルの定義
reg_model = LinearRegression()

'''
パイプライン化
前処理のscaler と 重回帰分析を行う reg_model が両方同じ fit()メソッドを持っていたことから
パイプラインと呼ばれる、これらの処理を統合する機能を使ってみる.
'''

# パイプラインの作成 (scaler -> svr)
pipeline = Pipeline([
    ('scaler', PowerTransformer()),
    ('reg', LinearRegression())
])
'''
scaler = PowerTransformer()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
'''

'''
Step5 : モデルの訓練
'''
# モデルの訓練
pipeline.fit(x_train, t_train)

# 訓練後のパラメータ 重みw
#print("w = ",reg_model.coef_)

# 訓練後のバイアス バイアスb
#print("b = ",reg_model.intercept_)

# 精度の検証
print("Train Score : ",pipeline.score(x_train, t_train))

#テスト用データセットで評価
print("Test Score : ",pipeline.score(x_test, t_test))

'''
結果が変わらなかったので標準化とは別のべき変換をする前処理をした
'''