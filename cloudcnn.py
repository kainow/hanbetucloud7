from keras.models import Sequential#ニューラルネットワークのモデルの定義に使う
from keras.layers import Conv2D, MaxPooling2D#畳み込みやプーリングの処理に必要な関数を読み込む。二次元。
from keras.layers import Activation, Dropout, Flatten, Dense#kerasのlayers処理から活性化関数、ドロップアウト関数、データを１次元に変換する関数、
from keras.utils import np_utils#データを扱うためにkerasのutilsからナンパイのutilsをインポート
import keras
import numpy as np 

classes = ["altocumulus","stratus","nimbostratus","altostratus","cirrocumulus","cirrostratus","cirrus","cumulonimbus","cumulus","stratocumulus"]
num_classes = len(classes)
image_size = 50

def main():#データを読み込んでトレーニングを実行するメイン関数を提議
    X_train, X_test, Y_train, Y_test = np.load("./cloud.npy")#変数をそれぞれ用意してファイルからデータを読み込む
    X_train = X_train.astype("float") /256#データの正規化を行うために最大値でわる。
    X_test = X_test.astype("float") /256#同上
    Y_train = np_utils.to_categorical(Y_train, num_classes)#one-hot-vectorという正解だけ１で他は０に配列を変換
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    model = model_train(X_train, Y_train)#モデルの学習と評価。

    model_eval(model, X_test, Y_test)#モデルの評価
#ケラスのドキュメントを参照に典型的なＮＮを用いるゾ。cifar10をまねます
def model_train(X,Y):
    model = Sequential()#モデルの作成
    model.add(Conv2D(32,(3,3),padding = 'same',input_shape = X.shape[1:]))#ニューラルネットワークの層を足していく。32このフィルターを同じサイズで畳み込む。
    model.add(Activation('relu'))#正だけ通して不は排除するrelu関数
    model.add(Conv2D(32,(3,3)))#繰り返す
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))#プールサイズを指定。一番大きい値を取り出して特徴づける
    model.add(Dropout(0.25))#２５パーセントを捨ててデータの偏りをなくすということ。

    model.add(Conv2D(64,(3,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))#最後の出力層を決める。データの数が違う時はここを指定
    model.add(Activation('softmax'))#足すと１になる

    opt = keras.optimizers.rmsprop(lr=0.0001, decay = 1e-6) #最適化の手法の宣言。ラーニングレートを指定して学習率を下げていく

    model.compile(loss='categorical_crossentropy',#モデルの最適化を宣言
                    optimizer = opt, metrics=['accuracy'])#再開と推定地がどれくらい離れているかと言う数値を使う。それが小さくなるように最適化を行う。

    model.fit(X,Y,batch_size = 32, epochs = 50)#引数で渡された値と１回のトレーニングの際のデータの数、回数を指定。

    model.save('./cloudcnn.h5')#結果をファイルに保存する。
    json_string = model.to_json()
    open('cloudmodel.json', 'w').write(json_string)
    return model


def model_eval(model,X,Y):
    scores = model.evaluate(X,Y, verbose = 1)#結果を変数で受け取って、引数で受け取ったモデルに値を与える。途中の経過表示を有効化。
    print('Test Loss:' , scores[0])
    print('Test Accuracy: ' , scores[1])


if __name__ == "__main__":#このプログラムを呼ばれたときにだけメインを実行するようにする。
    main()

