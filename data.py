import pandas as pd
import numpy as np

DATA_SOURCE = './data/score_0nan_small.csv'
score = pd.read_csv(DATA_SOURCE)
names = score.drop(['学年', '性別'], axis=1).columns

def get_num_data():
    tmp = score
    # 任意の行をとる
    # delete = teams - rows
    rows = ['学年', '性別']
    tmp = tmp.drop(rows, axis=1)
    return tmp


def get_full_data():
    return score


def get_corrcoef(data, x_label, y_label):
    cor = np.corrcoef(data[x_label], data[y_label])
    return cor[0,1].round(4)


def pick_up_df(df, genre):
    ans = pd.DataFrame()

    for elem in genre:
        grade = elem[0:2]
        gender = elem[2]
        ans = ans.append(df[(df['学年'] == grade) & (df['性別'] == gender)])


# scoreでの高1女子のデータ：409~589
# テストデータは、ここから20件とる
def split_train_test(df):
    # 加工を加えない生データ score
    test = df.iloc[500: 521,:]
    train  = pd.concat([df.iloc[:500, :], df.iloc[521:, :] ])
    return train, test


# ジャンルに応じてデータをフィルタリングして返す
def load_filtered_data(data, genre_filter):
    # 数値でフィルター(何点以上)
    # filtered_data = data[data['num_rooms'].between(rooms_filter[0], rooms_filter[1])]
    grade_filter = []
    gender_filter = []
    for elem in genre_filter:
        grade_filter.append(str(elem[0:2]))
        gender_filter.append(str(elem[2]))

    filtered_data = data[data['学年'].isin(grade_filter)]
    filtered_data = filtered_data[filtered_data['性別'].isin(gender_filter)]

    return filtered_data