import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prophet import Prophet, plot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Ridge



ticker_name = st.sidebar.text_input("Enter Company Ticker: ").upper()
df = pd.DataFrame()
hasError = False
try:
    df = yf.download(ticker_name)
    hasError = False
except Exception as e:
    hasError = True
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def homepage():

    st.header("Stock Forecast Analytics")
    st.text("")
    st.text("")
    st.text("")
    st.subheader('\n\n\n Program Overview')
    st.write('\n\nIn this program you are asked to enter a publicly listed companies ticker symbol. \n\n'
             'After you input the ticker you can view the historical stock price data of the firm since it was listed on the stock exchange and filter a specific date range.\n\n'
             'The program also shows the company\'s cumulative and returns distribution for the filtered date\n\n'
             'On the second page of this program you can view the forecasted change in closing stock price for the company selected for a period of time and query the exact stock price at a certain date\n\n'
             'On the third page this program graphs out stock price predictions for each stock price type such as opening, high, low, and adjusted close for the following year and also shows the historical data.\n\n'
             'On the last page you can calculate the predicted closing stock price of a company on a certain day of the week by giving entering the opening, high, low price of the stock and the volume traded')
    st.text("")
    st.write('For more information please find my LinkedIn and Resume attached below')
    st.write('\n\n\n[LinkedIn](https://www.linkedin.com/in/vivek-khetan/)')

    with open('Resume.pdf', 'rb') as file:
        st.download_button(label='Resume',
                           data= file,
                           file_name='Resume.pdf'
        )
    st.text("")
    st.text("")
    st.text("")
    from PIL import Image
    image = Image.open('My project.jpg')
    st.image(image)
def subheader(df):
    col1, col2, col3 = st.columns(3)
    col1.subheader(f"${df['Close'].tail(1).iloc[0]:.2f}")
    last = df['Close'].tail(2)
    col2.subheader(f"${(last.iloc[1] - last.iloc[0]):.2f}")
    col3.subheader(f"{((last.iloc[1] - last.iloc[0]) / last.iloc[0] * 100):.2f}%\n\n")
def sel():
    select_dic = {"1W": 7, "1M": 30, "3M":90, "6M": 182, "1YR":365}
    x = st.selectbox('Data From',
                     select_dic.keys())
    return select_dic, x
def min_max(x, _30):
    col1, col2 = st.columns(2)
    col1.write(f"Max (Last {x}): ${_30['Close'].max():.2f}")
    col2.write(f"Min (Last {x}): ${_30['Close'].min():.2f}")
def graph1(_30):
    fig = go.Figure(data=[go.Candlestick(
        x = _30.index, high=_30['High'], low=_30['Low'], open=_30['Open'], close=_30['Close']
    )])
    fig.update_layout(title = {'text': 'Stock Data', "x":0.5}, xaxis_title = 'Date', yaxis_title = 'Price')
    st.plotly_chart(fig)
def graph2(_30):
    fig = go.Figure(data=go.Scatter(
        x = _30.index, y=_30['Cumulative Returns']
    ))
    fig.update_layout(title = {'text': 'Cumulative Returns', "x":0.5}, xaxis_title = 'Date', yaxis_title = 'Returns')
    st.plotly_chart(fig)
def graph3(_30):
    fig = go.Figure(data=go.Histogram(
        x=_30['Returns']
    ))
    fig.update_layout(title = {'text': 'Returns Distribution', "x":0.5}, xaxis_title = 'Return', yaxis_title = 'Count')
    st.plotly_chart(fig)
def calc(_30):
    _30['Returns'] = (_30['Close']/_30['Close'].shift(1) - 1)
    _30['Cumulative Returns'] = (1-_30['Returns']).cumprod()
    return _30

def overview():
    if df.shape[0] > 0:
        subheader(df)
        select_dic, x = sel()
        _30 = df.tail(select_dic[x])
        min_max(x, _30)
        _30 = calc(_30)
        graph1(_30)
        graph2(_30)
        graph3(_30)
        st.write(_30.sort_index(ascending=False))

def prediction():
    select_dic = {'1 YR':365, '2 YR': 2*365, '3 YR': 3*365}
    selection = st.selectbox('Select number of years for forecast',
                             select_dic.keys())
    model = Prophet()
    d = df['Close'].reset_index().rename(columns={'Date':'ds', 'Close':'y'})
    model.fit(d)
    d_future = model.make_future_dataframe(periods=select_dic[selection])
    forecast = model.predict(d_future)
    fig = plot.plot_plotly(model, forecast)
    fig1 = plot.plot_components_plotly(model, forecast)
    st.plotly_chart(fig)
    st.plotly_chart(fig1)
    date = st.date_input('Enter a date')
    st.write(f"${(forecast.query('ds == @date')['yhat'].iloc[0]):.2f}")

def forecast():
    _df = calc(df)
    selection = st.selectbox('Select a Stock Price Type',
                             _df.columns)
    model = Prophet()
    _df1 = _df[selection].reset_index().rename(columns={'Date':'ds', selection:'y'})
    model.fit(_df1)
    future = model.make_future_dataframe(periods=365)
    f = model.predict(future)
    fig = plot.plot_plotly(model, f)
    st.plotly_chart(fig)

def sk():
    _df = df.reset_index()
    _df['Day'] = _df['Date'].dt.dayofweek

    x = _df[['Open', 'High', 'Low', 'Volume', 'Day']]
    y = _df['Close']
    week = {'Monday':0, 'Tueday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    model = Ridge()
    model.fit(x, y)
    col1, col2, col3, col4 = st.columns(4)
    _open = col1.number_input('Enter Open: ')
    high = col2.number_input('Enter High: ')
    low = col3.number_input('Enter Low: ')
    vol = col4.number_input('Enter Volume: ')
    day = st.selectbox("Choose Day",
                       week.keys())
    pred = model.predict([[_open, high, low, vol, week[day]]])
    st.write(pred)

def mainHeader():
    if(hasError):
        st.write("Invalid Ticker")
    else:
        st.header(ticker_name)

def main():
    select = st.sidebar.radio('Select Dashboard',
                              ['Homepage', 'Historical Data', 'Forecast', 'Stock Type Forecast', 'Day-to-Day Prediction'])
    if select == 'Historical Data':
        mainHeader()
        overview()
        add_bg_from_local('stock chart.jpeg')
    elif select == 'Homepage':
        homepage()
    elif select == 'Forecast':
        mainHeader()
        add_bg_from_local('stock chart.jpeg')
        prediction()
    elif select == 'Stock Type Forecast':
        mainHeader()
        add_bg_from_local('stock chart.jpeg')
        forecast()
    else:
        mainHeader()
        add_bg_from_local('stock chart.jpeg')
        sk()

main()
