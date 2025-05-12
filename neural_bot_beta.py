from matplotlib.pylab import f
import streamlit as st
import pandas as pd
# from yahooquery import Ticker
import yfinance as yf 
from requests_html import HTMLSession
# import requests
# from bs4 import BeautifulSoup
import talib
# import investpy
from collections import OrderedDict
import plotly.graph_objects as go
# from streamlit.components.v1 import html
from curl_cffi import requests
yf_session = requests.Session(impersonate="chrome")

st.set_page_config(page_title="Neural Bot Analysis", layout="wide")
st.title("📈 Neural Bot Analysis")


st.sidebar.header("Configure")

input_str = st.sidebar.text_input("Enter stock symbols (comma separated):", "AAPL, MSFT, GOOGL")
# YF : [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]
timeframe = st.sidebar.selectbox("Timeframe for Technicals", ["5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])
stock_list = [s.strip().upper() for s in input_str.split(",")] if input_str else []


# events = investpy.economic_calendar(
#     countries=['united states'], 
#     from_date='12/04/2025', 
#     to_date='15/04/2025'
# )


class DailyInfo:
    def __init__(self, site, count = 100):
        site = site.split("count=")[0] + f"count={count}"
        self.site = site
        self.session = HTMLSession()
    def get_info(self, site = None, drop_col = True):
        if site is None:
            site = self.site
        resp = self.session.get(site)
        
        tables = pd.read_html(resp.html.raw_html)  
        
        df = tables[0].copy()
        
        df.columns = tables[0].columns

        try:
            if drop_col:
                column_names = df.columns 
                df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
                desired_columns = ['Symbol', 'Name', 'Price', 'Change', 'Change %', 'Volume', 'Avg Vol (3M)', 'Market Cap', 'P/E Ratio (TTM)', '52 Wk Change %']
                temp_df = df.copy()
                for col in column_names:
                    if col in desired_columns:
                        continue
                    else:
                        del temp_df[col]
                if not temp_df.empty:
                    df = temp_df

        except:
            pass
        # session.close()
        # self.session.close()
        return df


def render_52w_visual(row):
    try:
        range_str = row['52 Week Range']
        cmp = row['CMP']
        if not range_str or cmp is None:
            return ''
        low, high = map(float, range_str.replace(" ", "").split("-"))
        if high <= low:
            return ''

        # Normalize CMP
        ratio = (cmp - low) / (high - low)
        ratio = max(0, min(ratio, 1))  # clip between 0-1

        total_blocks = 12
        pos = int(ratio * total_blocks)
        bar = ''.join(['█' if i == pos else '▁' for i in range(total_blocks)])
        # bar = ''.join([f'<span style="color:blue">█</span>' if i == pos else '▁' for i in range(total_blocks)])
        return bar
    except:
        return ''
    

def render_price_target_visual(row):
    try:
        high, low = row['Highest Target'], row['Lowest Target']
        range_str = f"{low} - {high}"
        cmp = row['CMP']
        mean = row['Average Target']
        if not range_str or cmp is None:
            return ''
        low, high = map(float, range_str.replace(" ", "").split("-"))
        if high <= low:
            return ''

        # Normalize CMP
        ratio = (cmp - low) / (high - low)
        ratio = max(0, min(ratio, 1))  # clip between 0-1
        # Normalize Mean
        mean_ratio = (mean - low) / (high - low)
        mean_ratio = max(0, min(mean_ratio, 1))  # clip between 0-1

        total_blocks = 20
        pos = int(ratio * total_blocks)
        mean_pos = int(mean_ratio * total_blocks)
        
        # bar = ''.join(['█' if i == pos else '▁' for i in range(total_blocks)])
        bar = ''
        for i in range(total_blocks):
            if i == pos:
                bar += ' 🚀 ' if cmp < mean else " 🚴🏻 "
            elif i == mean_pos:
                bar += ' 📍 '
            else:
                bar += '='


        # bar = ''.join([f'<span style="color:blue">█</span>' if i == pos else '▁' for i in range(total_blocks)])
        return bar
    except:
        return ''



class EarningsInfo:
    def __init__(self, symbol):
        self.symbol = symbol

    def get_earnings_info(self, symbol = None):
        if symbol is None:
            symbol = self.symbol
        ticker = yf.Ticker(symbol, session=yf_session)
        info = ticker.earnings_dates
        info.reset_index(inplace=True)
        info['Earnings Date'] = info['Earnings Date'].dt.date
        stock_data = yf.download(symbol, period='5y', interval='1d', session=yf_session)
        stock_data.columns = stock_data.columns.get_level_values(0)
        stock_data['Change %'] = round(stock_data['Close'].pct_change() * 100, 3)
        stock_data['Close'] = round(stock_data['Close'], 3)
        stock_data.reset_index(inplace=True)
        
        for idx, row in info.dropna().iterrows():
            # break
            price_df = stock_data.loc[(stock_data.Date.dt.date >= row['Earnings Date'])]
            first_row = price_df.iloc[0]
            second_row = price_df.iloc[1]

            if float(abs(first_row['Change %'])) > float(abs(second_row['Change %'])):
                price_row = first_row
            else:
                price_row = second_row

            info.loc[idx, 'Close'] = price_row['Close']
            info.loc[idx, 'Change %'] = price_row['Change %']

        
        estimates = ticker.earnings_estimate
        estimates.reset_index(inplace=True)
        estimate_row = estimates[estimates['period']=='0q']
        if not estimate_row.empty:
            estimate_row = estimate_row.iloc[0]
            current_date = pd.Timestamp('now', tz='America/New_York')
            info.sort_values(by=['Earnings Date'], inplace=True, ascending=False)
            est_date = info.loc[(info['Earnings Date'] >= current_date.date())].iloc[-1]['Earnings Date']
            info.loc[(info['Earnings Date'] == est_date), 'EPS Estimate'] = estimate_row['avg']
        # info_data = stock_data.loc[(stock_data.Date.dt.date.isin(info['Earnings Date']))]

        # info_data = info_data[['Date', 'Close', 'Change %']].reset_index(drop=True)
        # # info.reset_index(inplace=True)
        # merge_data = pd.merge(info_data, info, how='right', left_on=info_data.Date.dt.date, right_on=info['Earnings Date'])
        # merge_data = merge_data[['Earnings Date', 'EPS Estimate', 'Reported EPS', 'Surprise(%)', 'Close', 'Change %']].reset_index(drop=True)

        return info


class TechnicalInfo:
    def __init__(self, symbol):
        self.symbol = symbol
    
    def calculate_technical_indicators(self, symbol = None, timeframe = '1d'):
        # Fetch historical data
        if symbol is None:
            symbol =self.symbol
        if timeframe in ["1d", "5d", "1wk", "1mo", "3mo"]:
            period = '5y'
        else:
            period = '60d'
        df = yf.download(symbol, period=period, interval=timeframe, session=yf_session) 
        if df.empty or len(df) < 200:
            return None, None  # Not enough data
        
        df.columns = df.columns.get_level_values(0)

        # df.dropna(inplace=True)

        # Price arrays
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        thresholds = {
        'RSI': (30, 70),
        'Stoch %K': (20, 80),
        'CCI': (-100, 100),
        'ADX': (20, 25),
        'Stoch RSI': (20, 80),
        'W%R': (-80, -20),
        'Bull/Bear Power': (-0.05, 0.05),
        'Ultimate Oscillator': (30, 70)}

        tech_data = {}


        tech_data['RSI']  = talib.RSI(close, timeperiod=14).iloc[-1]
        tech_data['Stoch %K'] = talib.STOCH(high, low, close)[0].iloc[-1]
        tech_data['CCI'] = talib.CCI(high, low, close, timeperiod=20).iloc[-1]
        tech_data['ADX']  = talib.ADX(high, low, close, timeperiod=14).iloc[-1]
        # tech_data['Awesome Oscillator']  = ((high + low)/2 - (high + low).rolling(5).mean()).iloc[-1]
        tech_data['Momentum'] = talib.MOM(close, timeperiod=10).iloc[-1]
        macd, macdsignal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        tech_data['MACD'] = macd.iloc[-1] - macdsignal.iloc[-1]
        tech_data['Stoch RSI'] =  talib.STOCHRSI(close, timeperiod=14)[1].iloc[-1]
        tech_data['W%R'] =  talib.WILLR(high, low, close, timeperiod=14).iloc[-1]
        tech_data['Bull/Bear Power'] = close.iloc[-1] - talib.EMA(close, timeperiod=13).iloc[-1]
        tech_data['Ultimate Oscillator'] = talib.ULTOSC(high, low, close).iloc[-1]


        for ma in [10, 20, 30, 50, 100, 200]:
            tech_data[f'SMA_{ma}'] = talib.SMA(close, timeperiod=ma).iloc[-1]
            tech_data[f'EMA_{ma}'] = talib.EMA(close, timeperiod=ma).iloc[-1]


        wma_9 = talib.WMA(close, timeperiod=9)
        tech_data['Hull MA (9)'] = 2 * talib.WMA(close, timeperiod=9//2).iloc[-1] - wma_9.iloc[-1]


        vwma = (close * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        tech_data['VWMA_20'] = vwma.iloc[-1]

        nine_high = high.rolling(window=9).max()
        nine_low = low.rolling(window=9).min()
        tech_data['Ichimoku Base Line'] = (nine_high + nine_low) / 2.0
        tech_data['Ichimoku Base Line'] = tech_data['Ichimoku Base Line'].iloc[-1]

        # Ratings
        ratings = {}

        current_price = float(close.iloc[-1])

        for key, value in tech_data.items():
            if 'MA' in key or 'VWMA' in key or 'Ichimoku' in key:
                if current_price > value:
                    ratings[key] = 'Buy'
                elif current_price < value:
                    ratings[key] = 'Sell'
                else:
                    ratings[key] = 'Neutral'
            else:
                if pd.isna(value):
                    ratings[key] = 'Neutral'
                elif key in thresholds.keys():
                    thers_tupple = thresholds[key]
                    if value < thers_tupple[0]:
                        ratings[key] = 'Buy'
                    elif value > thers_tupple[1]:
                        ratings[key] = 'Sell'
                    else:
                        ratings[key] = 'Neutral'
                elif key == 'Momentum':
                    ratings[key] = 'Buy' if value > 0 else 'Sell'
                # elif key == 'Awesome Oscillator':
                #     ratings[key] = 'Buy' if value > 0 else 'Sell'
                elif key == 'MACD':
                    ratings[key] = 'Buy' if value > 0 else 'Sell'
                elif key == 'ADX':
                    ratings[key] = 'Neutral'
                else:
                    ratings[key] = 'Neutral'

        # final_rating = max(set(ratings.values()), key=pd.Series(ratings.values()).value_counts())
        final_rating = pd.Series(ratings.values()).value_counts()
        max_final_rating = final_rating.idxmax()
        tech_data['Symbol'] = symbol
        tech_data = OrderedDict(tech_data)
        tech_data.move_to_end('Symbol', last=False)
        tech_data = dict(tech_data)
        ratings['Symbol'] = symbol
        ratings = OrderedDict(ratings)
        ratings.move_to_end('Symbol', last=False)
        ratings = dict(ratings)
        # ratings['Buy_Count'] = int(final_rating['Buy'])
        # ratings['Sell_Count'] = int(final_rating['Sell'])
        # ratings['Neutral_Count'] = int(final_rating['Neutral'])
        for k, v in final_rating.items():
            # break
            ratings[f'{k}_Count'] = int(v)
        ratings['Final Rating'] = max_final_rating
        return tech_data, ratings


if st.sidebar.button("Run Bot"):
    if stock_list:
        try:
            data = []
            technical = []
            technical_rating = []
            earnings_date_perfomance = {}
            recom_dict = {}
            analyst_target_list = []
            for symbol in stock_list:
                # break
                tickers = yf.Ticker(symbol, session=yf_session)
                base_data = tickers.info

                # stock_data = financial_data.get(symbol, {})
                # row = {label: stock_data.get(key, None) for key, label in desired_metrics.items()}
                row = {}
                row['symbol'] = symbol
                row['Sector'] = base_data.get('sector', None)
                row['Industry'] = base_data.get('industry', None)
                row['CMP'] = base_data.get('currentPrice', None)
                row['Pct Change %'] = base_data.get('regularMarketChangePercent', None)
                row['Volume'] = base_data.get('regularMarketVolume', None)
                row['Market Cap'] =  f"{round(base_data.get('marketCap', 0) / 1e9, 4)} B"
                row['Beta'] = base_data.get('beta', None)
                row['Total Cash'] = f"{round(base_data.get('totalCash', 0) / 1e9, 4)} B"
                row['Total Debt'] = f"{round(base_data.get('totalDebt', 0) / 1e9, 4)} B"
                row['Total Revenue'] = f"{round(base_data.get('totalRevenue', 0) / 1e9, 4)} B"
                row['Enterprise Value'] = f"{round(base_data.get('enterpriseValue', 0) / 1e9, 4)} B"
                row['Debt to (Cash + Mcap)'] = round(base_data.get('totalDebt', 0) / (base_data.get('totalCash', 0) + base_data.get('marketCap', 0)) * 100, 2) if base_data.get('totalCash', 0) and base_data.get('marketCap', 0) else None
                row['Debt to (Cash + Revenue)'] = round(base_data.get('totalDebt', 0) / (base_data.get('totalCash', 0) + base_data.get('totalRevenue', 0)) * 100, 2) if base_data.get('totalCash', 0) and base_data.get('totalRevenue', 0) else None
                row['Debt to Revenue'] = round((base_data.get('totalDebt', 0) / base_data.get('totalRevenue', 0)) * 100, 2) if base_data.get('totalRevenue', 0) else None
                row['Debt to Equity'] = round(base_data.get('totalDebt', 0) / base_data.get('marketCap', 0) * 100, 2) if base_data.get('marketCap', 0) else None
                row['Debt to Cash'] = round(base_data.get('totalDebt', 0) / base_data.get('totalCash', 0) * 100, 2) if base_data.get('totalCash', 0) else None
                row['Revenue to Mcap'] = round(base_data.get('totalRevenue', 0) / base_data.get('marketCap', 0) * 100, 2) if base_data.get('marketCap', 0) else None
                row['Revenue Multiples'] = round(base_data.get('marketCap', 0) / base_data.get('totalRevenue', 0), 2) if base_data.get('totalRevenue', 0) else None
                row['PE Ratio'] = base_data.get('trailingPE', None) if base_data.get('trailingPE', None) is not None else base_data.get('forwardPE', None) 
                row['PB Ratio'] = base_data.get('priceToBook', None)
                row['Price to Sales'] = base_data.get('priceToSalesTrailing12Months', None)
                row['Revenue per Share'] = base_data.get('revenuePerShare', None)
                row['Cash Per Share'] = base_data.get('totalCashPerShare', None)
                row['Free Cash Flow'] = f"{round(base_data.get('freeCashflow', 0) / 1e9, 4)} B"
                row['Operating Cash Flow'] = f"{round(base_data.get('operatingCashflow', 0) / 1e9, 4)} B"
                row['EBITDA'] =  f"{round(base_data.get('ebitda', 0) / 1e9, 4)} B"
                row['EPS Multiples'] = round(base_data.get('priceEpsCurrentYear', None), 4)
                row['EPS (TTM)'] = base_data.get('trailingEps', None)
                row['EPS (Yr)'] = base_data.get('epsCurrentYear', None)
                row['EPS (Pred)'] = base_data.get('forwardEps', None)

                row['Share Outstanding'] = base_data.get('sharesOutstanding', None) 
                row['52 Week Range'] = base_data.get('fiftyTwoWeekRange', None)

                data.append(row)

                # Technical 
                tech_info = TechnicalInfo(symbol = symbol)
                tech_data, tech_rating = tech_info.calculate_technical_indicators(symbol = symbol, timeframe=timeframe)
                if tech_data is not None and tech_rating is not None:
                    technical.append(tech_data)
                    technical_rating.append(tech_rating)

                # Earnings
                earnings = EarningsInfo(symbol = symbol)
                earnings_data = earnings.get_earnings_info(symbol = symbol)
                earnings_date_perfomance[symbol] = earnings_data

                recom_dict[symbol] = tickers.recommendations_summary
                target_price = tickers.analyst_price_targets
                target_price['Symbol'] = symbol
                target_price = OrderedDict(target_price)
                target_price.move_to_end('Symbol', last=False)
                target_price = dict(target_price)

                analyst_target_list.append(target_price)

            df_selected = pd.DataFrame(data)
            df_selected['52W Visual'] = df_selected.apply(render_52w_visual, axis=1)
            technical_df = pd.DataFrame(technical)
            technical_rating_df = pd.DataFrame(technical_rating)
            cols = list(technical_rating_df.columns)
            new_order = [cols[0], cols[-1]] + cols[1:-1]
            technical_rating_df = technical_rating_df[new_order]
            
            recom_count_dict = {}
            for k, v in recom_dict.items():
                recom_count_dict[k] = int((v[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].sum(axis=1)).median())
                # break

            analyst_target_df = pd.DataFrame(analyst_target_list)
            analyst_target_df['Analyst Count'] = analyst_target_df['Symbol'].map(recom_count_dict) 
            analyst_target_df['Expected Change %'] = (round(((analyst_target_df['mean'] / analyst_target_df['current']) - 1) * 100, 3)).astype(str) + ' %'
            analyst_target_df.rename(columns={'current' : 'CMP', 'high' : 'Highest Target', 'low' : 'Lowest Target', 'mean' : 'Average Target', 'median' : 'Median Target'}, inplace=True)
            analyst_target_df['Price Target Visual'] = analyst_target_df.apply(lambda row: render_price_target_visual(row), axis=1)

            if df_selected.empty:
                st.warning(f"No data Financial found for the given {symbol}.")
            if technical_df.empty:
                st.warning(f"No data Technical found for the given {symbol}.")
            
            else:
                st.subheader("Fundamental Analysis")
                st.dataframe(df_selected)
                # st.write(df_selected.to_html(escape=False), unsafe_allow_html=True)

                st.subheader("Technical Indicators")
                st.dataframe(technical_df)
                st.subheader('Technical Rating')
                # st.dataframe(technical_rating_df)
                # st.dataframe(technical_rating_df.style.applymap(
                #     lambda val: (
                #         "background-color: green; color: white" if val == "Buy"
                #         else "background-color: red; color: white" if val == "Sell"
                #         else "background-color: gray; color: black" if val == "Neutral"
                #         else ""
                #     ),
                #     subset=technical_rating_df.columns
                # ), use_container_width=True)
                st.dataframe(technical_rating_df.style.applymap(
                    lambda val: (
                        "background-color: #b6e6b6; color: black" if val == "Buy"
                        else "background-color: #f5b6b6; color: black" if val == "Sell"
                        else "background-color: #dcdcdc; color: black" if val == "Neutral"
                        else ""
                    ),
                    subset=technical_rating_df.columns
                ), use_container_width=True)

                st.subheader('Analyst Target Price')
                st.dataframe(analyst_target_df)

            st.subheader("Earnings")

            for symbol, data in earnings_date_perfomance.items():
                st.write(f"**{symbol}**")
                st.dataframe(data)

            st.subheader('Analyst Recomendations')
            num_cols = 3
            chunks = [stock_list[i:i+num_cols] for i in range(0, len(stock_list), num_cols)]
            for chunk in chunks:
                cols = st.columns(len(chunk))

                for i, symbol in enumerate(chunk):
                    df = recom_dict[symbol]
                    colors = {
                        'strongBuy': '#006400',
                        'buy': '#66c266',
                        'hold': '#ffd700',
                        'sell': '#e67300',
                        'strongSell': '#b22222',
                    }

                    fig = go.Figure()
                    for col in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']:
                        fig.add_trace(go.Bar(
                            name=col.capitalize(),
                            x=df['period'],
                            y=df[col],
                            marker_color=colors[col],
                            text=df[col],
                            textposition='inside',
                            insidetextanchor='middle',
                            textfont=dict(size=10, color='white'),
                        ))

                    fig.update_layout(
                        barmode='stack',
                        height=300,
                        margin=dict(l=10, r=10, t=30, b=30),
                        plot_bgcolor='white',
                        xaxis=dict(title='', tickangle=0),
                        yaxis=dict(title=''),
                        showlegend=False,
                        font=dict(size=12)
                    )

                    with cols[i]:
                        st.markdown(f"**{symbol}**", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
            
            
            
            # num_cols = 2
            # chunks = [stock_list[i:i+num_cols] for i in range(0, len(stock_list), num_cols)]

            # for chunk in chunks:
            #     cols = st.columns(len(chunk))

            #     for i, symbol in enumerate(chunk):
            #         data = analyst_target_dict[symbol]
            #         current = data['current']
            #         low = data['low']
            #         high = data['high']
            #         mean = data['mean']

            #         # Render HTML inside Streamlit
            #         html_code = render_price_target_html(symbol, current, low, high, mean)
            #         with cols[i]:
            #             html(html_code, height=130)

            # st.write(df.to_html(escape=False), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")
    else:
        st.warning("No Stock Symbol has entered.")
    count = 100
    dailyobj = DailyInfo(site = f"https://finance.yahoo.com/gainers?offset=0&count=", count = count)
    try:
        st.subheader('Today Gainer')
        df = dailyobj.get_info(site = f"https://finance.yahoo.com/gainers?offset=0&count={count}")
        st.dataframe(df)
    except:
        st.error("Today's winner haven't fetched")

    try:
        st.subheader('Today Loser')
        count = 100
        # dailyobj = DailyInfo(site = f"https://finance.yahoo.com/losers?offset=0&count=", count = count)
        df = dailyobj.get_info(site = f"https://finance.yahoo.com/losers?offset=0&count={count}")
        st.dataframe(df)
    except:
        st.error("Today's Loser haven't fetched")

    try:
        st.subheader('Today Most Active')
        count = 100
        # dailyobj = DailyInfo(site = f"https://finance.yahoo.com/most-active?offset=0&count=", count = count)
        df = dailyobj.get_info(site = f"https://finance.yahoo.com/most-active?offset=0&count={count}")
        st.dataframe(df)
    except:
        st.error("Today's Most Active haven't fetched")

    try:
        st.subheader('Undervalued Large Cap')
        count = 100
        site = f"https://finance.yahoo.com/screener/predefined/undervalued_large_caps?offset=0&count={count}"
        df = dailyobj.get_info(site = site, drop_col=True)
        st.dataframe(df)
    except:
        st.error("Undervalued Large Cap haven't fetched")

    dailyobj.session.close()
    st.markdown("---")
    st.subheader("Upcoming Macroeconomic & Political Events")

    economic_events = [
        {"event": "CPI Release", "date": "2025-04-16", "impact": "High"},
        {"event": "PPI Release", "date": "2025-04-11", "impact": "High"},
        {"event": "FOMC Minutes", "date": "2025-05-01", "impact": "Critical"},
        {"event": "Powell Speech", "date": "2025-04-22", "impact": "High"},
        {"event": "Trump Tariff Threat", "date": "2025-04-04", "impact": "Market Shock"},
        {"event": "Fed Governor Waller Speech", "date": "2025-04-18", "impact": "Medium"},
        {"event": "Debt Ceiling Talks", "date": "2025-04-25", "impact": "High"},
    ]

    df_events = pd.DataFrame(economic_events)
    df_events['date'] = pd.to_datetime(df_events['date'])

    today = pd.to_datetime("today").normalize()
    upcoming = df_events[df_events['date'] >= today].sort_values(by="date")

    st.dataframe(upcoming.style.applymap(
        lambda val: (
            "background-color: #990000; color: white" if val == "Critical"
            else "background-color: #cc3300; color: white" if val == "High"
            else "background-color: #e6b800; color: black" if val == "Medium"
            else ""
        ),
        subset=["impact"]
    ), use_container_width=True)

    # # -------------------------------
    # # FOOTNOTE
    # # -------------------------------
    # st.markdown("---")
    # st.markdown("**Note:** This bot combines technical charting with static macroeconomic triggers.")
    # st.markdown("Future enhancement will fetch live calendar feeds and analyze real-time event impact.")

    with st.expander("📘 Help Guide"):
        st.markdown(""" 
        # Neural Bot Analysis – Documentation

        ## Overview

        **Neural Bot Analysis** is a Streamlit-based dashboard for analyzing stocks using both fundamental and technical indicators. It fetches data from Yahoo Finance and other sources, processes it, and presents key metrics, visualizations, and analyst recommendations to help users make informed investment decisions.

        ---

        ## Features

        - **User Input:** Enter stock symbols and select technical analysis timeframes.
        - **Fundamental Analysis:** Displays key financial ratios and metrics for each stock.
        - **Technical Analysis:** Calculates and rates technical indicators using TA-Lib.
        - **Visualizations:** Shows price position in 52-week range and analyst price targets.
        - **Earnings Analysis:** Presents historical earnings data and performance.
        - **Analyst Recommendations:** Visualizes buy/hold/sell recommendations.
        - **Market Movers:** Lists top gainers, losers, most active, and undervalued large caps.
        - **Upcoming Events:** Displays major macroeconomic and political events.

        ---

        ## Code Structure

        - **DailyInfo:** Fetches tabular data from Yahoo Finance screeners.
        - **EarningsInfo:** Retrieves earnings dates and post-earnings price performance.
        - **TechnicalInfo:** Calculates technical indicators and ratings.
        - **render_52w_visual:** Visualizes current price within 52-week range.
        - **render_price_target_visual:** Visualizes price and analyst targets.
        - **Main App:** Handles user input, data fetching, processing, and display.

        ---

        ## Fundamental Parameters Explained

        Below are the key financial metrics calculated and displayed, with detailed explanations and interpretations:

        ### 1. **Pct Change % (Percent Change)**
        - **Meaning:** The percentage change in the stock price over the last trading session.
        - **Interpretation:**  
        - **Higher is better** if positive (price is rising).
        - **Lower is better** if negative (price is falling, which may be good for short sellers).

        ---

        ### 2. **Volume**
        - **Meaning:** The total number of shares traded during the last session.
        - **Interpretation:**  
        - **Higher is better** for liquidity and market interest.
        - **Lower** may indicate weak interest or illiquidity.

        ---

        ### 3. **Market Cap (Market Capitalization)**
        - **Meaning:** The total value of a company’s outstanding shares (`Price × Shares Outstanding`).
        - **Interpretation:**  
        - **Higher market cap** means a larger, often more stable company.
        - **Lower market cap** companies may offer higher growth but are riskier.

        ---

        ### 4. **Beta**
        - **Meaning:** Measures a stock’s volatility relative to the market.
        - **Interpretation:**  
        - **Beta = 1:** Moves with the market.
        - **Beta > 1:** More volatile than the market.
        - **Beta < 1:** Less volatile.
        - **Higher or lower is better** depends on your risk preference.

        ---

        ### 5. **Total Cash**
        - **Meaning:** The total cash and cash equivalents held by the company.
        - **Interpretation:**  
        - **Higher is better**; indicates financial flexibility and ability to weather downturns.

        ---

        ### 6. **Total Debt**
        - **Meaning:** The sum of all short-term and long-term debt.
        - **Interpretation:**  
        - **Lower is better**; high debt increases financial risk.

        ---

        ### 7. **Total Revenue**
        - **Meaning:** The total income from sales of goods/services.
        - **Interpretation:**  
        - **Higher is better**; indicates business scale and growth.

        ---

        ### 8. **Enterprise Value**
        - **Meaning:** Market cap plus debt, minus cash. Represents the total value of a business.
        - **Interpretation:**  
        - Used for valuation comparisons; **lower EV** can mean undervaluation.

        ---

        ### 9. **Debt to (Cash + Mcap)**
        - **Meaning:** Debt divided by the sum of cash and market cap, expressed as a percentage.
        - **Interpretation:**  
        - **Lower is better**; shows how much debt is covered by cash and equity value.

        ---

        ### 10. **Debt to (Cash + Revenue)**
        - **Meaning:** Debt divided by the sum of cash and revenue, as a percentage.
        - **Interpretation:**  
        - **Lower is better**; indicates ability to cover debt with cash and income.

        ---

        ### 11. **Debt to Revenue**
        - **Meaning:** Debt divided by revenue, as a percentage.
        - **Interpretation:**  
        - **Lower is better**; high values may indicate over-leverage.

        ---

        ### 12. **Debt to Equity**
        - **Meaning:** Debt divided by market cap (proxy for equity), as a percentage.
        - **Interpretation:**  
        - **Lower is better**; high values mean more risk.

        ---

        ### 13. **Debt to Cash**
        - **Meaning:** Debt divided by cash, as a percentage.
        - **Interpretation:**  
        - **Lower is better**; high values mean less liquidity to cover debt.

        ---

        ### 14. **Revenue to Mcap**
        - **Meaning:** Revenue divided by market cap, as a percentage.
        - **Interpretation:**  
        - **Higher is better**; shows how much revenue is generated per unit of market value.

        ---

        ### 15. **Revenue Multiples**
        - **Meaning:** Market cap divided by revenue.
        - **Interpretation:**  
        - **Lower is better**; lower multiples may indicate undervaluation.

        ---

        ### 16. **PE Ratio (Price-to-Earnings)**
        - **Meaning:** Price per share divided by earnings per share.
        - **Interpretation:**  
        - **Lower is better** for value investing (cheaper relative to earnings).
        - **Higher PE** may indicate growth expectations or overvaluation.

        ---

        ### 17. **PB Ratio (Price-to-Book)**
        - **Meaning:** Price per share divided by book value per share.
        - **Interpretation:**  
        - **Lower is better** for value; high PB may mean overvaluation.

        ---

        ### 18. **Price to Sales**
        - **Meaning:** Price per share divided by sales per share.
        - **Interpretation:**  
        - **Lower is better**; high values may indicate overvaluation.

        ---

        ### 19. **Revenue per Share**
        - **Meaning:** Total revenue divided by shares outstanding.
        - **Interpretation:**  
        - **Higher is better**; shows how much revenue each share generates.

        ---

        ### 20. **Cash Per Share**
        - **Meaning:** Total cash divided by shares outstanding.
        - **Interpretation:**  
        - **Higher is better**; indicates liquidity per share.

        ---

        ### 21. **Free Cash Flow**
        - **Meaning:** Cash generated after capital expenditures.
        - **Interpretation:**  
        - **Higher is better**; positive free cash flow means the company can reinvest, pay dividends, or reduce debt.

        ---

        ### 22. **Operating Cash Flow**
        - **Meaning:** Cash generated from core business operations.
        - **Interpretation:**  
        - **Higher is better**; indicates healthy business operations.

        ---

        ### 23. **EBITDA**
        - **Meaning:** Earnings before interest, taxes, depreciation, and amortization.
        - **Interpretation:**  
        - **Higher is better**; used to compare profitability between companies.

        ---

        ### 24. **EPS Multiples**
        - **Meaning:** Price per share divided by earnings per share for the current year.
        - **Interpretation:**  
        - **Lower is better** for value; higher may indicate growth expectations.

        ---

        ### 25. **EPS (TTM), EPS (Yr), EPS (Pred)**
        - **Meaning:** Earnings per share for trailing twelve months, current year, and predicted (forward).
        - **Interpretation:**  
        - **Higher is better**; shows profitability per share.

        ---

        ### 26. **Share Outstanding**
        - **Meaning:** Number of shares currently issued and held by investors.
        - **Interpretation:**  
        - Used in calculating per-share metrics.

        ---

        ### 27. **52 Week Range**
        - **Meaning:** The lowest and highest price at which a stock has traded in the last 52 weeks.
        - **Interpretation:**  
        - Used to gauge volatility and current price position.

        ---

        ## Visualizations

        - **52 Week Range Bar:**  
        Shows where the current price sits within its 52-week range using block characters.

        - **Price Target Bar:**  
        Shows current price, mean analyst target, and range using emojis and block characters.

        ---

        ## Usage Notes

        - **Higher or lower is better** depends on the metric and investment style (value vs. growth).
        - Always interpret ratios in the context of industry averages and company history.

        ---

        ## Future Enhancements

        - Live macroeconomic calendar feeds.
        - Real-time event impact analysis.
        - More advanced technical and fundamental screening.

        ---
                    
        """)