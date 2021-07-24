import streamlit as st
import data
import plotly.express as px


st.set_page_config(
    # page_title="Ex-stream-ly Cool App",
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    )

names = ['学年','性別','身長','体重','座高','握力',
'上体起こし','長座体前屈','反復横跳び','シャトルラン','50ｍ走','立ち幅跳び','ハンドボール投げ',
'握力得点','上体起こし得点','長座体前屈得点','反復横跳び得点','シャトルラン得点','50ｍ走得点',
'立ち幅跳び得点','ハンドボール投げ得点']

st.title("体力測定 データ")

score = data.get_num_data()
full_data = data.get_full_data()

# 色分けオプション
coloring = st.radio(
    "グラフの色分け",
    ('なし', '学年', '性別')
)



left, right = st.beta_columns(2)

with left: # 散布図の表示 
    label = score.columns
    x_label = st.selectbox('横軸を選択',label)
    y_label = st.selectbox('縦軸を選択',label)


    if coloring == '学年':
        fig = px.scatter(
        full_data,
        x=x_label,
        y=y_label,
        color="学年"
        )   
      
    elif coloring == "性別":
        fig = px.scatter(
            full_data,
            x=x_label,
            y=y_label,
            color="性別",
            )
        
    else:
        fig = px.scatter(
            full_data,
            x=x_label,
            y=y_label,
            )
    st.plotly_chart(fig, use_container_width=True)

    cor = data.get_corrcoef(score, x_label, y_label)
    st.write('相関係数：' + str(cor))

    

with right: # ヒストグラムの表示
    hist_val = st.selectbox('変数を選択',label)
    fig = px.histogram(score, x=hist_val)
    st.plotly_chart(fig, use_container_width=True)


menu = st.sidebar.selectbox(
    '何をする？',
    ['ここから選ぼう','散布図を表示']
)

# 風船とぶ、
# st.balloons()

# 待たせられる
# with st.spinner('Wait for it...'):
#     time.sleep(5)
# st.success('Done!')