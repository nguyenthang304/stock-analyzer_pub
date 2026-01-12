import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

# ==========================================
# 1. Configuration & Setup
# ==========================================
st.set_page_config(
    page_title="QuantAlgo Pro: Tech Analysis System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Professional" Look 
st.markdown("""
<style>
   .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        color: white;
    }
   .signal-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Ingestion Module 
# ==========================================
@st.cache_data(ttl=300)
def get_data(ticker, period, interval):
    """
    Fetches historical OHLCV data using yfinance.
    Handles MultiIndex columns and data cleaning.
    """
    try:
        # Fetch data
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            return None
            
        # Handle MultiIndex columns (yfinance update fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Rename standard columns
        df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                           'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        
        # Drop NaN values created by gaps
        df.dropna(subset=['Close'], inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ==========================================
# 3. Technical Analysis Engine (The 7 Steps)
# ==========================================
def calculate_indicators(df):
    """
    Step 1-6: Adds Technical Indicators to the DataFrame.
    Uses pandas-ta for vectorized calculation.
    """
    data = df.copy()

    # --- Step 1: Trend Identification [2, 4] ---
    data.ta.ema(length=50, append=True)
    data.ta.ema(length=200, append=True)
    # ADX for Trend Strength 
    data.ta.adx(length=14, append=True) # Adds ADX_14, DMP_14, DMN_14

    # --- Step 3: Volume Analysis  ---
    data.ta.obv(append=True)
    # Relative Volume (Current Vol / 20-day Avg Vol)
    vol_sma = ta.sma(data['Volume'], length=20)
    data = data['Volume'] / vol_sma

    # --- Step 4: Momentum [14, 15] ---
    data.ta.rsi(length=14, append=True)
    data.ta.stoch(append=True) # STOCHk, STOCHd
    data.ta.macd(append=True)  # MACD, MACDh (hist), MACDs

    # --- Step 6: Volatility  ---
    # Bollinger Bands
    data.ta.bbands(length=20, std=2, append=True) 
    # BB Bandwidth for Squeeze detection
    data = (data - data) / data
    
    # ATR for Risk Management
    data.ta.atr(length=14, append=True)

    return data

def identify_levels(df, window=15):
    """
    Step 2: Key Levels (Support/Resistance) using Fractal Logic.
    Uses scipy.signal.argrelextrema to find local peaks/valleys.
    """
    data = df.copy()
    
    # Find local minima (Support) and maxima (Resistance)
    # The 'order' parameter defines the window size for comparison
    min_idxs = argrelextrema(data['Close'].values, np.less_equal, order=window)
    max_idxs = argrelextrema(data['Close'].values, np.greater_equal, order=window)
    
    levels_support = data.iloc[min_idxs]['Close'].tolist()
    levels_resistance = data.iloc[max_idxs]['Close'].tolist()
    
    # Clustering Logic: Group nearby levels to avoid noise
    def cluster_levels(levels, threshold=0.015):
        if not levels: return
        levels.sort()
        clustered =
        current_cluster = [levels]
        
        for i in range(1, len(levels)):
            # If next level is within threshold % of the cluster average
            if levels[i] <= np.mean(current_cluster) * (1 + threshold):
                current_cluster.append(levels[i])
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [levels[i]]
        clustered.append(np.mean(current_cluster))
        return clustered

    clean_supports = cluster_levels(levels_support)
    clean_resistances = cluster_levels(levels_resistance)
    
    return clean_supports, clean_resistances

def calculate_fibonacci(df):
    """
    Calculates Fibonacci Retracement Levels based on the period High/Low.[8, 10]
    """
    max_price = df['High'].max()
    min_price = df['Low'].min()
    diff = max_price - min_price
    
    levels = {
        '0.0 (Low)': min_price,
        '0.236': min_price + 0.236 * diff,
        '0.382': min_price + 0.382 * diff,
        '0.5': min_price + 0.5 * diff,
        '0.618': min_price + 0.618 * diff,
        '0.786': min_price + 0.786 * diff,
        '1.0 (High)': max_price
    }
    return levels

def detect_patterns(df):
    """
    Step 5: Price Action (Candlestick Patterns).
    Custom Vectorized Implementation for portability.[17, 18, 19]
    """
    data = df.copy()
    
    # Pre-calculate body and wicks
    data = np.abs(data['Close'] - data['Open'])
    data = data['High'] - np.maximum(data['Close'], data['Open'])
    data = np.minimum(data['Close'], data['Open']) - data['Low']
    data = data['High'] - data['Low']
    
    # 1. Bullish Engulfing 
    data = (
        (data['Close'].shift(1) < data['Open'].shift(1)) & # Prev Red
        (data['Close'] > data['Open']) &                   # Curr Green
        (data['Open'] < data['Close'].shift(1)) &          # Engulf Body
        (data['Close'] > data['Open'].shift(1))
    )

    # 2. Bearish Engulfing
    data = (
        (data['Close'].shift(1) > data['Open'].shift(1)) & # Prev Green
        (data['Close'] < data['Open']) &                   # Curr Red
        (data['Open'] > data['Close'].shift(1)) &          # Engulf Body
        (data['Close'] < data['Open'].shift(1))
    )

    # 3. Hammer (Bullish Rejection) 
    data['Hammer'] = (
        (data > 2 * data) & 
        (data < 0.5 * data)
    )

    # 4. Shooting Star (Bearish Rejection)
    data = (
        (data > 2 * data) & 
        (data < 0.5 * data)
    )
    
    # 5. Doji (Indecision) 
    data = data <= (0.1 * data)
    
    # 6. Double Bottom Detection (Vectorized approximation) [21]
    # Check if current low is close to a previous low (within 20 bars)
    rolling_min = data['Low'].rolling(window=20).min()
    # If current low is within 1% of the rolling min (which isn't this candle)
    data = (
        (np.abs(data['Low'] - rolling_min.shift(5)) < (data['Close'] * 0.01)) &
        (data['Low'] > rolling_min.shift(5)) & # Ideally slightly higher low
        (data > 30) # Momentum confirming
    )

    return data

# ==========================================
# 4. The Recommendation Engine (Signal Logic)
# ==========================================
def generate_signal(df, supports, resistances):
    """
    Step 7: Risk Management & Final Recommendation.
    """
    last = df.iloc[-1]
    
    # 1. Determine Trend Regime [2]
    trend = "Neutral"
    ema_50 = last['EMA_50']
    ema_200 = last['EMA_200']
    price = last['Close']
    
    if price > ema_200:
        trend = "Bullish" if ema_50 > ema_200 else "Weak Bullish"
    else:
        trend = "Bearish" if ema_50 < ema_200 else "Weak Bearish"
        
    # 2. Key Level Proximity (within 1.5%)
    dist_sup = [abs(price - s)/price for s in supports]
    dist_res = [abs(price - r)/price for r in resistances]
    
    at_support = min(dist_sup) < 0.015 if dist_sup else False
    at_resistance = min(dist_res) < 0.015 if dist_res else False
    
    # 3. Momentum State
    rsi = last
    overbought = rsi > 70
    oversold = rsi < 30
    
    # 4. Decision Tree
    signal = "WAIT"
    reasons =
    
    # --- BUY LOGIC ---
    if "Bullish" in trend:
        # Pullback Setup
        if at_support and not overbought:
            signal = "BUY"
            reasons.append("Trend Pullback to Support.")
        # Breakout Setup
        elif price > last and last > 1.2:
            signal = "BUY"
            reasons.append("Volatility Breakout (Price > Upper BB) with Volume.")
        # Pattern Confirmation
        if last['Hammer'] or last:
            if signal == "WAIT": signal = "BUY (Aggressive)"
            reasons.append("Bullish Candlestick Pattern detected.")
            
    # --- SELL LOGIC ---
    elif "Bearish" in trend:
        # Rally rejection
        if at_resistance and not oversold:
            signal = "SELL"
            reasons.append("Trend Rally to Resistance.")
        # Breakdown
        elif price < last and last > 1.2:
            signal = "SELL"
            reasons.append("Volatility Breakdown (Price < Lower BB) with Volume.")
        # Pattern Confirmation
        if last or last:
            if signal == "WAIT": signal = "SELL (Aggressive)"
            reasons.append("Bearish Candlestick Pattern detected.")

    # --- WAIT LOGIC (Overrides) ---
    if overbought and signal == "BUY":
        signal = "WAIT"
        reasons.append("Market is Overbought (RSI > 70). Risk of pullback.")
    if oversold and signal == "SELL":
        signal = "WAIT"
        reasons.append("Market is Oversold (RSI < 30). Risk of bounce.")
        
    if not reasons:
        reasons.append("No clear edge detected. Market may be choppy (Wait for ADX > 25).")

    # Risk Management 
    atr = last
    stop_loss = price - (2 * atr) if "BUY" in signal else price + (2 * atr)
    target = price + (4 * atr) if "BUY" in signal else price - (4 * atr)

    return {
        "Signal": signal,
        "Trend": trend,
        "Reasons": reasons,
        "Stop": stop_loss,
        "Target": target,
        "ATR": atr
    }

# ==========================================
# 5. UI Layout & Execution
# ==========================================
def main():
    st.title("🤖 QuantAlgo Pro: Automated Technical Analysis")
    st.markdown("Automated 7-Step Strategy: Trend > Levels > Volume > Momentum > Patterns > Volatility > Risk.")
    
    # Sidebar
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
    timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1wk", "1h", "15m"], index=0)
    period = st.sidebar.selectbox("Data Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    
    if st.sidebar.button("Run Analysis"):
        with st.spinner('Analyzing Market Structure...'):
            # 1. Fetch
            df = get_data(ticker, period, timeframe)
            if df is None: return
            
            # 2. Process
            df = calculate_indicators(df)
            df = detect_patterns(df)
            sup, res = identify_levels(df)
            fibs = calculate_fibonacci(df)
            
            # 3. Analyze
            result = generate_signal(df, sup, res)
            
            # 4. Dashboard
            # Top Metrics
            last = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"{last['Close']:.2f}")
            col2.metric("Trend", result)
            col3.metric("RSI", f"{last:.1f}")
            col4.metric("Rel Vol", f"{last:.1f}x")
            
            # Signal Card
            color = "#00ff00" if "BUY" in result else "#ff0000" if "SELL" in result else "#ffa500"
            st.markdown(f"""
            <div class="signal-box" style="border: 2px solid {color}; background-color: rgba(0,0,0,0.2);">
                <h1 style="color: {color};">{result}</h1>
                <h3>{', '.join(result)}</h3>
                <p><strong>Stop Loss:</strong> {result:.2f} | <strong>Target:</strong> {result:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Charting
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_width=[0.2, 0.7])
            
            # Main Price Chart
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='orange'), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='blue'), name='EMA 200'), row=1, col=1)
            
            # Plot Levels (Limit to closest 3 to current price to avoid clutter)
            for level in res[-3:]:
                fig.add_hline(y=level, line_dash="dash", line_color="red", row=1, col=1, annotation_text="Res")
            for level in sup[-3:]:
                fig.add_hline(y=level, line_dash="dash", line_color="green", row=1, col=1, annotation_text="Sup")
                
            # Plot Fibonacci [32]
            for name, level in fibs.items():
                 fig.add_hline(y=level, line_color="gray", opacity=0.3, row=1, col=1, annotation_text=f"Fib {name}")

            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("### Data View")
            st.dataframe(df.tail(10))

if __name__ == "__main__":
    main()