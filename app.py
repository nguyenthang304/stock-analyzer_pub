"""
Professional 7-Step Technical Analysis Strategy App
Senior Quant Developer Implementation
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="7-Step TA Strategy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Mobile First, Dark Professional Theme
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --accent-green: #00ff88;
        --accent-red: #ff4757;
        --accent-yellow: #ffd93d;
        --accent-blue: #6c5ce7;
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --border-color: #2a2a3a;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    .signal-card {
        font-family: 'Space Grotesk', sans-serif;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid var(--border-color);
    }
    
    .signal-bullish {
        background: linear-gradient(135deg, rgba(0,255,136,0.15) 0%, rgba(0,255,136,0.05) 100%);
        border-color: var(--accent-green);
    }
    
    .signal-bearish {
        background: linear-gradient(135deg, rgba(255,71,87,0.15) 0%, rgba(255,71,87,0.05) 100%);
        border-color: var(--accent-red);
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, rgba(255,217,61,0.15) 0%, rgba(255,217,61,0.05) 100%);
        border-color: var(--accent-yellow);
    }
    
    .signal-text {
        font-size: clamp(2rem, 8vw, 4rem);
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .signal-bullish .signal-text { color: var(--accent-green); }
    .signal-bearish .signal-text { color: var(--accent-red); }
    .signal-neutral .signal-text { color: var(--accent-yellow); }
    
    .score-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
        margin-top: 1rem;
    }
    
    .step-card {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--border-color);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .step-pass { border-left-color: var(--accent-green); }
    .step-fail { border-left-color: var(--accent-red); }
    .step-warn { border-left-color: var(--accent-yellow); }
    
    .metric-box {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    .trade-setup {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(108,92,231,0.1) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--accent-blue);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stTextInput label {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-secondary);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        background: var(--bg-card);
        border-radius: 8px;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .signal-card { padding: 1.5rem; }
        .metric-box { padding: 1rem; }
        .metric-value { font-size: 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch stock data from Yahoo Finance with caching."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [col if col != 'Date' else 'Date' for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMAs for trend identification."""
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    return df


def analyze_trend(df: pd.DataFrame) -> Dict:
    """Step 1: Trend Identification using EMA alignment."""
    latest = df.iloc[-1]
    price = latest['Close']
    ema20 = latest['EMA20']
    ema50 = latest['EMA50']
    ema200 = latest['EMA200']
    
    # Check for uptrend
    if price > ema20 > ema50 > ema200:
        trend = "UPTREND"
        signal = "bullish"
        score = 1
    # Check for downtrend
    elif price < ema20 < ema50 < ema200:
        trend = "DOWNTREND"
        signal = "bearish"
        score = -1
    else:
        trend = "SIDEWAYS"
        signal = "neutral"
        score = 0
    
    # Calculate EMA slopes for additional context
    ema20_slope = (df['EMA20'].iloc[-1] - df['EMA20'].iloc[-5]) / df['EMA20'].iloc[-5] * 100
    ema50_slope = (df['EMA50'].iloc[-1] - df['EMA50'].iloc[-5]) / df['EMA50'].iloc[-5] * 100
    
    return {
        "trend": trend,
        "signal": signal,
        "score": score,
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "ema20_slope": ema20_slope,
        "ema50_slope": ema50_slope
    }


def find_support_resistance(df: pd.DataFrame, order: int = 10) -> Dict:
    """Step 2: Find Support and Resistance levels using local extrema."""
    # Find local maxima and minima
    close_prices = df['Close'].values
    
    # Find swing highs (resistance)
    high_idx = argrelextrema(close_prices, np.greater, order=order)[0]
    resistance_levels = close_prices[high_idx][-5:] if len(high_idx) >= 5 else close_prices[high_idx]
    
    # Find swing lows (support)
    low_idx = argrelextrema(close_prices, np.less, order=order)[0]
    support_levels = close_prices[low_idx][-5:] if len(low_idx) >= 5 else close_prices[low_idx]
    
    # Calculate Fibonacci levels
    recent_high = df['High'].tail(50).max()
    recent_low = df['Low'].tail(50).min()
    diff = recent_high - recent_low
    
    fib_levels = {
        "0.0": recent_high,
        "0.236": recent_high - diff * 0.236,
        "0.382": recent_high - diff * 0.382,
        "0.5": recent_high - diff * 0.5,
        "0.618": recent_high - diff * 0.618,
        "0.786": recent_high - diff * 0.786,
        "1.0": recent_low
    }
    
    current_price = df['Close'].iloc[-1]
    nearest_support = support_levels[support_levels < current_price].max() if len(support_levels[support_levels < current_price]) > 0 else recent_low
    nearest_resistance = resistance_levels[resistance_levels > current_price].min() if len(resistance_levels[resistance_levels > current_price]) > 0 else recent_high
    
    return {
        "support_levels": support_levels.tolist(),
        "resistance_levels": resistance_levels.tolist(),
        "fib_levels": fib_levels,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "recent_high": recent_high,
        "recent_low": recent_low
    }


def analyze_volume(df: pd.DataFrame) -> Dict:
    """Step 3: Volume Analysis."""
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    
    latest_volume = df['Volume'].iloc[-1]
    vol_sma = df['Vol_SMA20'].iloc[-1]
    
    # Volume spike detection
    volume_ratio = latest_volume / vol_sma if vol_sma > 0 else 0
    strong_volume = volume_ratio > 1.5
    
    # OBV slope (last 5 periods)
    obv_slope = (df['OBV'].iloc[-1] - df['OBV'].iloc[-5]) / abs(df['OBV'].iloc[-5]) * 100 if df['OBV'].iloc[-5] != 0 else 0
    obv_bullish = obv_slope > 0
    
    # Score based on volume confirmation
    if strong_volume and obv_bullish:
        score = 1
        signal = "bullish"
    elif strong_volume and not obv_bullish:
        score = -1
        signal = "bearish"
    else:
        score = 0
        signal = "neutral"
    
    return {
        "current_volume": latest_volume,
        "vol_sma20": vol_sma,
        "volume_ratio": volume_ratio,
        "strong_volume": strong_volume,
        "obv_slope": obv_slope,
        "obv_bullish": obv_bullish,
        "score": score,
        "signal": signal
    }


def analyze_momentum(df: pd.DataFrame) -> Dict:
    """Step 4: Momentum & Oscillators Analysis."""
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    rsi = df['RSI'].iloc[-1]
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    
    macd_val = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    macd_hist_prev = df['MACD_Hist'].iloc[-2]
    
    macd_crossover = macd_val > macd_signal and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]
    macd_crossunder = macd_val < macd_signal and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]
    
    # Stochastic
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['Stoch_K'] = stoch['STOCHk_14_3_3']
    df['Stoch_D'] = stoch['STOCHd_14_3_3']
    
    stoch_k = df['Stoch_K'].iloc[-1]
    stoch_d = df['Stoch_D'].iloc[-1]
    
    stoch_oversold_cross = stoch_k > stoch_d and stoch_k < 30 and df['Stoch_K'].iloc[-2] <= df['Stoch_D'].iloc[-2]
    stoch_overbought_cross = stoch_k < stoch_d and stoch_k > 70 and df['Stoch_K'].iloc[-2] >= df['Stoch_D'].iloc[-2]
    
    # Scoring
    score = 0
    signals = []
    
    if rsi < 30:
        score += 1
        signals.append("RSI Oversold")
    elif rsi > 70:
        score -= 1
        signals.append("RSI Overbought")
    
    if macd_crossover:
        score += 1
        signals.append("MACD Bullish Cross")
    elif macd_crossunder:
        score -= 1
        signals.append("MACD Bearish Cross")
    
    if stoch_oversold_cross:
        score += 1
        signals.append("Stoch Oversold Cross")
    elif stoch_overbought_cross:
        score -= 1
        signals.append("Stoch Overbought Cross")
    
    return {
        "rsi": rsi,
        "rsi_status": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
        "macd": macd_val,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "macd_crossover": macd_crossover,
        "macd_crossunder": macd_crossunder,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "stoch_oversold_cross": stoch_oversold_cross,
        "stoch_overbought_cross": stoch_overbought_cross,
        "score": max(min(score, 1), -1),  # Clamp between -1 and 1
        "signals": signals
    }


def detect_price_patterns(df: pd.DataFrame) -> Dict:
    """Step 5: Price Action Patterns Detection."""
    patterns = []
    score = 0
    
    # Get recent lows for double bottom detection
    recent_lows = df['Low'].tail(20).values
    low_indices = argrelextrema(recent_lows, np.less, order=3)[0]
    
    # Double Bottom Detection
    if len(low_indices) >= 2:
        last_two_lows = recent_lows[low_indices[-2:]]
        pct_diff = abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] * 100
        if pct_diff <= 2:
            patterns.append("Double Bottom (Bullish)")
            score += 1
    
    # Double Top Detection
    recent_highs = df['High'].tail(20).values
    high_indices = argrelextrema(recent_highs, np.greater, order=3)[0]
    
    if len(high_indices) >= 2:
        last_two_highs = recent_highs[high_indices[-2:]]
        pct_diff = abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] * 100
        if pct_diff <= 2:
            patterns.append("Double Top (Bearish)")
            score -= 1
    
    # Bollinger Band Width for Consolidation
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close'] * 100
    
    current_bb_width = df['BB_Width'].iloc[-1]
    avg_bb_width = df['BB_Width'].tail(50).mean()
    
    if current_bb_width < avg_bb_width * 0.7:
        patterns.append("Consolidation/Squeeze")
    
    # Flag Pattern (after strong move)
    price_change_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
    recent_range = (df['High'].tail(5).max() - df['Low'].tail(5).min()) / df['Close'].iloc[-1] * 100
    
    if abs(price_change_5d) > 5 and recent_range < 3:
        if price_change_5d > 0:
            patterns.append("Bull Flag")
            score += 0.5
        else:
            patterns.append("Bear Flag")
            score -= 0.5
    
    return {
        "patterns": patterns if patterns else ["No significant patterns detected"],
        "bb_width": current_bb_width,
        "avg_bb_width": avg_bb_width,
        "score": max(min(score, 1), -1)
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
    """Step 6: Candlestick Pattern Detection."""
    patterns = []
    score = 0
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    open_price = latest['Open']
    close_price = latest['Close']
    high_price = latest['High']
    low_price = latest['Low']
    
    body = abs(close_price - open_price)
    full_range = high_price - low_price
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    
    # Pin Bar / Hammer / Shooting Star
    if full_range > 0:
        body_ratio = body / full_range
        
        # Hammer (bullish) - long lower wick, small body at top
        if body_ratio < 0.3 and lower_wick > body * 2 and upper_wick < body:
            patterns.append("Hammer (Bullish)")
            score += 1
        
        # Shooting Star (bearish) - long upper wick, small body at bottom
        elif body_ratio < 0.3 and upper_wick > body * 2 and lower_wick < body:
            patterns.append("Shooting Star (Bearish)")
            score -= 1
        
        # Doji - open approximately equals close
        if body_ratio < 0.1:
            patterns.append("Doji (Indecision)")
    
    # Engulfing patterns
    prev_body = abs(prev['Close'] - prev['Open'])
    
    # Bullish Engulfing
    if (prev['Close'] < prev['Open'] and  # Previous bearish
        close_price > open_price and  # Current bullish
        open_price < prev['Close'] and  # Current opens below prev close
        close_price > prev['Open']):  # Current closes above prev open
        patterns.append("Bullish Engulfing")
        score += 1
    
    # Bearish Engulfing
    elif (prev['Close'] > prev['Open'] and  # Previous bullish
          close_price < open_price and  # Current bearish
          open_price > prev['Close'] and  # Current opens above prev close
          close_price < prev['Open']):  # Current closes below prev open
        patterns.append("Bearish Engulfing")
        score -= 1
    
    return {
        "patterns": patterns if patterns else ["No candlestick patterns detected"],
        "body_ratio": body / full_range if full_range > 0 else 0,
        "score": max(min(score, 1), -1)
    }


def calculate_risk_management(
    entry_price: float,
    support_level: float,
    account_size: float,
    risk_percent: float
) -> Dict:
    """Step 7: Risk Management Calculator."""
    # Stop Loss slightly below support (0.5% buffer)
    stop_loss = support_level * 0.995
    
    # Calculate risk per share
    risk_per_share = entry_price - stop_loss
    
    if risk_per_share <= 0:
        return {
            "error": "Invalid stop loss calculation",
            "stop_loss": stop_loss,
            "take_profit": entry_price,
            "position_size": 0,
            "risk_amount": 0,
            "reward_amount": 0
        }
    
    # Calculate position size
    risk_amount = account_size * (risk_percent / 100)
    position_size = int(risk_amount / risk_per_share)
    
    # Take Profit at 1:2 Risk:Reward
    take_profit = entry_price + (risk_per_share * 2)
    
    # Potential reward
    reward_amount = position_size * (take_profit - entry_price)
    
    return {
        "entry_price": entry_price,
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "position_size": position_size,
        "risk_amount": round(risk_amount, 2),
        "reward_amount": round(reward_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "risk_reward_ratio": "1:2"
    }


def calculate_overall_signal(
    trend: Dict,
    volume: Dict,
    momentum: Dict,
    patterns: Dict,
    candles: Dict
) -> Dict:
    """Calculate overall signal based on all analysis steps."""
    # Weighted scoring
    total_score = 0
    max_score = 5
    
    # Step 1: Trend (most important)
    total_score += trend['score'] * 1.5
    
    # Step 3: Volume
    total_score += volume['score'] * 1.0
    
    # Step 4: Momentum
    total_score += momentum['score'] * 1.0
    
    # Step 5: Price Patterns
    total_score += patterns['score'] * 0.75
    
    # Step 6: Candlestick Patterns
    total_score += candles['score'] * 0.75
    
    # Normalize
    normalized_score = total_score / max_score
    
    if normalized_score > 0.3 and trend['trend'] == "UPTREND":
        signal = "BULLISH"
        recommendation = "BUY"
    elif normalized_score < -0.3:
        signal = "BEARISH"
        recommendation = "SELL"
    else:
        signal = "NEUTRAL"
        recommendation = "WAIT"
    
    return {
        "signal": signal,
        "recommendation": recommendation,
        "score": round(normalized_score * 100, 1),
        "raw_score": round(total_score, 2)
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_main_chart(df: pd.DataFrame, sr_data: Dict, ticker: str) -> go.Figure:
    """Create the main candlestick chart with indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker} Price Action', 'Volume', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4757'
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['EMA20'], name='EMA 20',
                   line=dict(color='#ffd93d', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['EMA50'], name='EMA 50',
                   line=dict(color='#6c5ce7', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['EMA200'], name='EMA 200',
                   line=dict(color='#fd79a8', width=2)),
        row=1, col=1
    )
    
    # Support/Resistance lines
    for level in sr_data['support_levels'][-3:]:
        fig.add_hline(y=level, line_dash="dash", line_color="rgba(0,255,136,0.5)",
                      annotation_text=f"S: ${level:.2f}", row=1, col=1)
    
    for level in sr_data['resistance_levels'][-3:]:
        fig.add_hline(y=level, line_dash="dash", line_color="rgba(255,71,87,0.5)",
                      annotation_text=f"R: ${level:.2f}", row=1, col=1)
    
    # Fibonacci levels
    for label, level in list(sr_data['fib_levels'].items())[1:-1]:
        fig.add_hline(y=level, line_dash="dot", line_color="rgba(108,92,231,0.4)",
                      annotation_text=f"Fib {label}", row=1, col=1)
    
    # Volume
    colors = ['#00ff88' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff4757' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Vol_SMA20'], name='Vol SMA20',
                   line=dict(color='#ffd93d', width=1)),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI'], name='RSI',
                   line=dict(color='#6c5ce7', width=2)),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.7)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.7)", row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(108,92,231,0.1)", 
                  line_width=0, row=3, col=1)
    
    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(10,10,15,0)',
        plot_bgcolor='rgba(26,26,36,0.8)',
        font=dict(family='JetBrains Mono', color='#a0a0b0'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(26,26,36,0.8)'
        ),
        height=800,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(gridcolor='rgba(42,42,58,0.5)')
    fig.update_yaxes(gridcolor='rgba(42,42,58,0.5)')
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 7-Step Technical Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Analysis Settings")
        
        ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol").upper()
        
        period_options = {
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y"
        }
        period_label = st.selectbox("Date Range", options=list(period_options.keys()), index=2)
        period = period_options[period_label]
        
        st.markdown("---")
        st.markdown("### 💰 Risk Management")
        
        account_size = st.number_input("Account Size ($)", value=10000, min_value=100, step=1000)
        risk_percent = st.slider("Max Risk (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        
        st.markdown("---")
        analyze_btn = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if analyze_btn or ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            # Fetch data
            df = fetch_stock_data(ticker, period)
            
            if df is None or len(df) < 200:
                st.error(f"❌ Unable to fetch sufficient data for {ticker}. Please check the ticker symbol.")
                return
            
            # Run analysis
            df = calculate_emas(df)
            
            trend_data = analyze_trend(df)
            sr_data = find_support_resistance(df)
            volume_data = analyze_volume(df)
            momentum_data = analyze_momentum(df)
            pattern_data = detect_price_patterns(df)
            candle_data = detect_candlestick_patterns(df)
            
            overall = calculate_overall_signal(
                trend_data, volume_data, momentum_data, pattern_data, candle_data
            )
            
            risk_data = calculate_risk_management(
                entry_price=df['Close'].iloc[-1],
                support_level=sr_data['nearest_support'],
                account_size=account_size,
                risk_percent=risk_percent
            )
            
            # Display Results
            # Executive Summary Card
            signal_class = f"signal-{overall['signal'].lower()}"
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div style="font-size: 1rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 2px;">
                    Signal for {ticker}
                </div>
                <div class="signal-text">{overall['signal']}</div>
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                    Recommendation: <strong>{overall['recommendation']}</strong>
                </div>
                <div class="score-badge" style="background: rgba(255,255,255,0.1);">
                    Confidence Score: {overall['score']}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Price Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">${df['Close'].iloc[-1]:.2f}</div>
                    <div class="metric-label">Current Price</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                color = "#00ff88" if change >= 0 else "#ff4757"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: {color}">{change:+.2f}%</div>
                    <div class="metric-label">Daily Change</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{trend_data['trend']}</div>
                    <div class="metric-label">Trend</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{momentum_data['rsi']:.1f}</div>
                    <div class="metric-label">RSI (14)</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Main Chart
            st.markdown("### 📈 Price Chart with Indicators")
            chart = create_main_chart(df, sr_data, ticker)
            st.plotly_chart(chart, use_container_width=True)
            
            # Step-by-Step Analysis Report
            st.markdown("### 📋 Step-by-Step Analysis Report")
            
            with st.expander("**Step 1: Trend Identification (EMA Filter)**", expanded=True):
                status = "✅" if trend_data['trend'] == "UPTREND" else ("❌" if trend_data['trend'] == "DOWNTREND" else "⚠️")
                step_class = "step-pass" if trend_data['trend'] == "UPTREND" else ("step-fail" if trend_data['trend'] == "DOWNTREND" else "step-warn")
                
                st.markdown(f"""
                <div class="step-card {step_class}">
                    <strong>{status} Trend: {trend_data['trend']}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("EMA 20", f"${trend_data['ema20']:.2f}")
                col2.metric("EMA 50", f"${trend_data['ema50']:.2f}")
                col3.metric("EMA 200", f"${trend_data['ema200']:.2f}")
                
                st.caption(f"EMA20 Slope: {trend_data['ema20_slope']:.2f}% | EMA50 Slope: {trend_data['ema50_slope']:.2f}%")
            
            with st.expander("**Step 2: Key Levels (Support & Resistance)**"):
                st.markdown(f"""
                <div class="step-card step-pass">
                    <strong>📍 Key Levels Identified</strong>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Support Levels:**")
                    for level in sr_data['support_levels'][-3:]:
                        st.write(f"• ${level:.2f}")
                
                with col2:
                    st.markdown("**Resistance Levels:**")
                    for level in sr_data['resistance_levels'][-3:]:
                        st.write(f"• ${level:.2f}")
                
                st.markdown("**Fibonacci Retracement:**")
                fib_cols = st.columns(4)
                for i, (label, level) in enumerate(list(sr_data['fib_levels'].items())[1:5]):
                    fib_cols[i].metric(f"Fib {label}", f"${level:.2f}")
            
            with st.expander("**Step 3: Volume Analysis**"):
                status = "✅" if volume_data['strong_volume'] and volume_data['obv_bullish'] else ("❌" if volume_data['strong_volume'] and not volume_data['obv_bullish'] else "⚠️")
                step_class = "step-pass" if volume_data['signal'] == "bullish" else ("step-fail" if volume_data['signal'] == "bearish" else "step-warn")
                
                vol_status = "Strong Volume Spike!" if volume_data['strong_volume'] else "Normal Volume"
                obv_status = "Bullish" if volume_data['obv_bullish'] else "Bearish"
                
                st.markdown(f"""
                <div class="step-card {step_class}">
                    <strong>{status} {vol_status} | OBV: {obv_status}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Volume", f"{volume_data['current_volume']:,.0f}")
                col2.metric("Vol SMA20", f"{volume_data['vol_sma20']:,.0f}")
                col3.metric("Volume Ratio", f"{volume_data['volume_ratio']:.2f}x")
            
            with st.expander("**Step 4: Momentum & Oscillators**"):
                status = "✅" if momentum_data['score'] > 0 else ("❌" if momentum_data['score'] < 0 else "⚠️")
                step_class = "step-pass" if momentum_data['score'] > 0 else ("step-fail" if momentum_data['score'] < 0 else "step-warn")
                
                signals_text = ", ".join(momentum_data['signals']) if momentum_data['signals'] else "No significant signals"
                
                st.markdown(f"""
                <div class="step-card {step_class}">
                    <strong>{status} {signals_text}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                rsi_color = "#ff4757" if momentum_data['rsi'] > 70 else ("#00ff88" if momentum_data['rsi'] < 30 else "#ffd93d")
                col1.metric("RSI (14)", f"{momentum_data['rsi']:.1f}", momentum_data['rsi_status'])
                col2.metric("MACD", f"{momentum_data['macd']:.3f}", 
                           "Bullish Cross" if momentum_data['macd_crossover'] else ("Bearish Cross" if momentum_data['macd_crossunder'] else "No Cross"))
                col3.metric("Stochastic K/D", f"{momentum_data['stoch_k']:.1f}/{momentum_data['stoch_d']:.1f}")
            
            with st.expander("**Step 5: Price Action Patterns**"):
                status = "✅" if pattern_data['score'] > 0 else ("❌" if pattern_data['score'] < 0 else "⚠️")
                step_class = "step-pass" if pattern_data['score'] > 0 else ("step-fail" if pattern_data['score'] < 0 else "step-warn")
                
                st.markdown(f"""
                <div class="step-card {step_class}">
                    <strong>{status} Patterns: {', '.join(pattern_data['patterns'])}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                col1.metric("BB Width", f"{pattern_data['bb_width']:.2f}%")
                col2.metric("Avg BB Width", f"{pattern_data['avg_bb_width']:.2f}%")
            
            with st.expander("**Step 6: Candlestick Patterns (Trigger)**"):
                status = "✅" if candle_data['score'] > 0 else ("❌" if candle_data['score'] < 0 else "⚠️")
                step_class = "step-pass" if candle_data['score'] > 0 else ("step-fail" if candle_data['score'] < 0 else "step-warn")
                
                st.markdown(f"""
                <div class="step-card {step_class}">
                    <strong>{status} Latest Candle: {', '.join(candle_data['patterns'])}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Body Ratio", f"{candle_data['body_ratio']:.2%}")
            
            with st.expander("**Step 7: Trade Setup & Risk Management**", expanded=True):
                if "error" in risk_data:
                    st.warning(risk_data['error'])
                else:
                    st.markdown(f"""
                    <div class="trade-setup">
                        <h4 style="color: #6c5ce7; margin-bottom: 1rem;">💼 Trade Setup Calculator</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📥 Entry Price", f"${risk_data['entry_price']:.2f}")
                    col2.metric("🛑 Stop Loss", f"${risk_data['stop_loss']:.2f}", f"-${risk_data['risk_per_share']:.2f}/share")
                    col3.metric("🎯 Take Profit", f"${risk_data['take_profit']:.2f}", f"R:R {risk_data['risk_reward_ratio']}")
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📊 Position Size", f"{risk_data['position_size']} shares")
                    col2.metric("💵 Risk Amount", f"${risk_data['risk_amount']:.2f}")
                    col3.metric("💰 Potential Reward", f"${risk_data['reward_amount']:.2f}")
                    
                    total_investment = risk_data['position_size'] * risk_data['entry_price']
                    st.info(f"💡 Total Investment Required: **${total_investment:,.2f}** ({(total_investment/account_size)*100:.1f}% of account)")
            
            # Disclaimer
            st.markdown("---")
            st.caption("""
            ⚠️ **Disclaimer:** This tool is for educational and informational purposes only. 
            It does not constitute financial advice. Always do your own research and consult 
            with a qualified financial advisor before making investment decisions. 
            Past performance is not indicative of future results.
            """)


if __name__ == "__main__":
    main()
