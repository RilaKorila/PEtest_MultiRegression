import pandas as pd
import numpy as np

score = pd.read_csv('./data/score.csv')

names = ['学年','性別','身長','体重','座高','握力',
'上体起こし','長座体前屈','反復横跳び','シャトルラン','50ｍ走','立ち幅跳び','ハンドボール投げ',
'握力得点','上体起こし得点','長座体前屈得点','反復横跳び得点','シャトルラン得点','50ｍ走得点',
'立ち幅跳び得点','ハンドボール投げ得点']

def get_num_data():
    tmp = score

    # 列順を変える
    # tmp = tmp.drop(['順位'], axis=1)
    # tmp.insert(1, '順位', soccer['順位'])

    # # 任意の行をとる
    # # delete = teams - rows
    rows = ['学年', '性別']
    tmp = tmp.drop(rows, axis=1)
    
    # # データの処理はあとでここに書く

    return tmp

def get_full_data():
    return score



def get_corrcoef(data, x_label, y_label):
    cor = np.corrcoef(data[x_label].values, data[y_label].values)
    print(data[y_label].values)
    return np.round(cor[0,1], 3)

