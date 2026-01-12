"""
Công cụ Phân tích Kỹ thuật 7 Bước Chuyên nghiệp
Hỗ trợ Cổ phiếu Việt Nam (HOSE, HNX)
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

# Danh sách cổ phiếu Việt Nam phổ biến
VN_STOCKS = {
    "HOSE (Sàn HCM)": {
        "FPT": "FPT Corporation - Công nghệ",
        "VNM": "Vinamilk - Sữa",
        "VIC": "Vingroup - Bất động sản",
        "VHM": "Vinhomes - Bất động sản",
        "VCB": "Vietcombank - Ngân hàng",
        "BID": "BIDV - Ngân hàng",
        "CTG": "VietinBank - Ngân hàng",
        "TCB": "Techcombank - Ngân hàng",
        "MBB": "MB Bank - Ngân hàng",
        "VPB": "VPBank - Ngân hàng",
        "HPG": "Hòa Phát - Thép",
        "MSN": "Masan - Tiêu dùng",
        "MWG": "Thế Giới Di Động - Bán lẻ",
        "VRE": "Vincom Retail - Bán lẻ",
        "SAB": "Sabeco - Bia",
        "GAS": "PV Gas - Dầu khí",
        "PLX": "Petrolimex - Xăng dầu",
        "VJC": "Vietjet Air - Hàng không",
        "SSI": "SSI - Chứng khoán",
        "VND": "VNDirect - Chứng khoán",
        "HCM": "HCMC Securities - Chứng khoán",
        "REE": "REE - Cơ điện lạnh",
        "PNJ": "PNJ - Vàng bạc",
        "ACB": "ACB - Ngân hàng",
        "STB": "Sacombank - Ngân hàng",
        "EIB": "Eximbank - Ngân hàng",
        "SHB": "SHB - Ngân hàng",
        "TPB": "TPBank - Ngân hàng",
        "HDB": "HDBank - Ngân hàng",
        "LPB": "LienVietPostBank - Ngân hàng",
        "NVL": "Novaland - Bất động sản",
        "PDR": "Phát Đạt - Bất động sản",
        "DXG": "Đất Xanh - Bất động sản",
        "KDH": "Khang Điền - Bất động sản",
        "DIG": "DIC Corp - Bất động sản",
        "BCM": "Becamex - Bất động sản",
        "KBC": "Kinh Bắc - KCN",
        "GVR": "Cao su Việt Nam - Cao su",
        "BVH": "Bảo Việt - Bảo hiểm",
        "POW": "PV Power - Điện",
        "GMD": "Gemadept - Cảng biển",
        "DCM": "Đạm Cà Mau - Phân bón",
        "DPM": "Đạm Phú Mỹ - Phân bón",
        "PVD": "PV Drilling - Dầu khí",
    },
}

st.set_page_config(
    page_title="Phân tích Kỹ thuật 7 Bước - Cổ phiếu Việt Nam",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
    :root {
        --bg-primary: #0a0a0f; --bg-secondary: #12121a; --bg-card: #1a1a24;
        --accent-green: #00ff88; --accent-red: #ff4757; --accent-yellow: #ffd93d;
        --accent-blue: #6c5ce7; --text-primary: #ffffff; --text-secondary: #a0a0b0;
        --border-color: #2a2a3a;
    }
    .stApp { background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%); }
    .main-header {
        font-family: 'Be Vietnam Pro', sans-serif; font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 700; background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0; margin-bottom: 1rem;
    }
    .signal-card {
        font-family: 'Be Vietnam Pro', sans-serif; padding: 2rem; border-radius: 20px;
        text-align: center; margin: 1rem 0; box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid var(--border-color);
    }
    .signal-bullish { background: linear-gradient(135deg, rgba(0,255,136,0.15) 0%, rgba(0,255,136,0.05) 100%); border-color: var(--accent-green); }
    .signal-bearish { background: linear-gradient(135deg, rgba(255,71,87,0.15) 0%, rgba(255,71,87,0.05) 100%); border-color: var(--accent-red); }
    .signal-neutral { background: linear-gradient(135deg, rgba(255,217,61,0.15) 0%, rgba(255,217,61,0.05) 100%); border-color: var(--accent-yellow); }
    .signal-text { font-size: clamp(2rem, 8vw, 4rem); font-weight: 700; margin: 0.5rem 0; }
    .signal-bullish .signal-text { color: var(--accent-green); }
    .signal-bearish .signal-text { color: var(--accent-red); }
    .signal-neutral .signal-text { color: var(--accent-yellow); }
    .score-badge { font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; padding: 0.5rem 1.5rem; border-radius: 50px; display: inline-block; margin-top: 1rem; }
    .step-card { background: var(--bg-card); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid var(--border-color); font-family: 'Be Vietnam Pro', sans-serif; }
    .step-pass { border-left-color: var(--accent-green); }
    .step-fail { border-left-color: var(--accent-red); }
    .step-warn { border-left-color: var(--accent-yellow); }
    .metric-box { background: var(--bg-card); border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid var(--border-color); }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--text-primary); }
    .metric-label { font-family: 'Be Vietnam Pro', sans-serif; font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem; }
    .trade-setup { background: linear-gradient(135deg, var(--bg-card) 0%, rgba(108,92,231,0.1) 100%); border-radius: 16px; padding: 1.5rem; border: 1px solid var(--accent-blue); }
    .stock-info { background: var(--bg-card); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border: 1px solid var(--accent-blue); }
    [data-testid="stSidebar"] { background: var(--bg-secondary); }
    @media (max-width: 768px) { .signal-card { padding: 1.5rem; } .metric-box { padding: 1rem; } .metric-value { font-size: 1.2rem; } }
</style>
""", unsafe_allow_html=True)


def safe_value(val, default=0):
    """Safely get value, return default if NaN or None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def format_price(val, currency, is_vn):
    """Format price with currency, handle NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if is_vn:
        return f"{currency}{val:,.0f}"
    return f"{currency}{val:.2f}"


def get_vn_ticker(ticker: str) -> str:
    """Chuyển đổi mã cổ phiếu VN sang format Yahoo Finance."""
    ticker = ticker.upper().strip()
    if '.VN' in ticker or '.HN' in ticker:
        return ticker
    return f"{ticker}.VN"


def get_stock_info(ticker: str) -> str:
    """Lấy thông tin cổ phiếu từ danh sách."""
    clean_ticker = ticker.replace('.VN', '').replace('.HN', '').upper()
    for exchange, stocks in VN_STOCKS.items():
        if clean_ticker in stocks:
            return f"{clean_ticker} - {stocks[clean_ticker]}"
    return ticker


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        df = df.reset_index()
        return df
    except Exception as e:
        return None


def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    return df


def analyze_trend(df: pd.DataFrame) -> Dict:
    latest = df.iloc[-1]
    price = safe_value(latest['Close'])
    ema20 = safe_value(latest['EMA20'])
    ema50 = safe_value(latest['EMA50'])
    ema200 = safe_value(latest['EMA200'])
    
    # Kiểm tra có đủ EMA không
    has_ema200 = not (pd.isna(latest['EMA200']) if 'EMA200' in latest else True)
    
    if has_ema200 and price > ema20 > ema50 > ema200:
        trend, trend_en, signal, score = "XU HƯỚNG TĂNG", "UPTREND", "bullish", 1
    elif has_ema200 and price < ema20 < ema50 < ema200:
        trend, trend_en, signal, score = "XU HƯỚNG GIẢM", "DOWNTREND", "bearish", -1
    elif price > ema20 > ema50:
        trend, trend_en, signal, score = "XU HƯỚNG TĂNG (ngắn hạn)", "UPTREND", "bullish", 0.5
    elif price < ema20 < ema50:
        trend, trend_en, signal, score = "XU HƯỚNG GIẢM (ngắn hạn)", "DOWNTREND", "bearish", -0.5
    else:
        trend, trend_en, signal, score = "ĐI NGANG", "SIDEWAYS", "neutral", 0
    
    # Tính độ dốc an toàn
    try:
        ema20_slope = (df['EMA20'].iloc[-1] - df['EMA20'].iloc[-5]) / df['EMA20'].iloc[-5] * 100
        if np.isnan(ema20_slope):
            ema20_slope = 0
    except:
        ema20_slope = 0
    
    try:
        ema50_slope = (df['EMA50'].iloc[-1] - df['EMA50'].iloc[-5]) / df['EMA50'].iloc[-5] * 100
        if np.isnan(ema50_slope):
            ema50_slope = 0
    except:
        ema50_slope = 0
    
    return {"trend": trend, "trend_en": trend_en, "signal": signal, "score": score,
            "price": price, "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "ema20_slope": ema20_slope, "ema50_slope": ema50_slope, "has_ema200": has_ema200}


def find_support_resistance(df: pd.DataFrame, order: int = 5) -> Dict:
    close_prices = df['Close'].values
    
    # Điều chỉnh order dựa trên độ dài data
    order = min(order, len(df) // 10) if len(df) > 20 else 2
    order = max(order, 2)
    
    high_idx = argrelextrema(close_prices, np.greater, order=order)[0]
    resistance_levels = close_prices[high_idx][-5:] if len(high_idx) >= 5 else (close_prices[high_idx] if len(high_idx) > 0 else np.array([df['High'].max()]))
    
    low_idx = argrelextrema(close_prices, np.less, order=order)[0]
    support_levels = close_prices[low_idx][-5:] if len(low_idx) >= 5 else (close_prices[low_idx] if len(low_idx) > 0 else np.array([df['Low'].min()]))
    
    recent_high, recent_low = df['High'].tail(50).max(), df['Low'].tail(50).min()
    diff = recent_high - recent_low
    fib_levels = {"0.0": recent_high, "0.236": recent_high - diff * 0.236,
                  "0.382": recent_high - diff * 0.382, "0.5": recent_high - diff * 0.5,
                  "0.618": recent_high - diff * 0.618, "0.786": recent_high - diff * 0.786, "1.0": recent_low}
    
    current_price = df['Close'].iloc[-1]
    support_below = support_levels[support_levels < current_price]
    nearest_support = support_below.max() if len(support_below) > 0 else recent_low
    
    resistance_above = resistance_levels[resistance_levels > current_price]
    nearest_resistance = resistance_above.min() if len(resistance_above) > 0 else recent_high
    
    return {"support_levels": support_levels.tolist(), "resistance_levels": resistance_levels.tolist(),
            "fib_levels": fib_levels, "nearest_support": nearest_support, "nearest_resistance": nearest_resistance,
            "recent_high": recent_high, "recent_low": recent_low}


def analyze_volume(df: pd.DataFrame) -> Dict:
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    
    latest_volume = safe_value(df['Volume'].iloc[-1])
    vol_sma = safe_value(df['Vol_SMA20'].iloc[-1], 1)
    
    volume_ratio = latest_volume / vol_sma if vol_sma > 0 else 0
    strong_volume = volume_ratio > 1.5
    
    try:
        obv_now = df['OBV'].iloc[-1]
        obv_prev = df['OBV'].iloc[-5]
        obv_slope = (obv_now - obv_prev) / abs(obv_prev) * 100 if obv_prev != 0 else 0
        if np.isnan(obv_slope):
            obv_slope = 0
    except:
        obv_slope = 0
    
    obv_bullish = obv_slope > 0
    
    if strong_volume and obv_bullish:
        score, signal = 1, "bullish"
    elif strong_volume and not obv_bullish:
        score, signal = -1, "bearish"
    else:
        score, signal = 0, "neutral"
    
    return {"current_volume": latest_volume, "vol_sma20": vol_sma, "volume_ratio": volume_ratio,
            "strong_volume": strong_volume, "obv_slope": obv_slope, "obv_bullish": obv_bullish,
            "score": score, "signal": signal}


def analyze_momentum(df: pd.DataFrame) -> Dict:
    df['RSI'] = ta.rsi(df['Close'], length=14)
    rsi = safe_value(df['RSI'].iloc[-1], 50)
    
    try:
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        macd_val = safe_value(df['MACD'].iloc[-1])
        macd_signal_val = safe_value(df['MACD_Signal'].iloc[-1])
        macd_crossover = macd_val > macd_signal_val and safe_value(df['MACD'].iloc[-2]) <= safe_value(df['MACD_Signal'].iloc[-2])
        macd_crossunder = macd_val < macd_signal_val and safe_value(df['MACD'].iloc[-2]) >= safe_value(df['MACD_Signal'].iloc[-2])
    except:
        macd_val, macd_signal_val = 0, 0
        macd_crossover, macd_crossunder = False, False
    
    try:
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
        df['Stoch_K'] = stoch['STOCHk_14_3_3']
        df['Stoch_D'] = stoch['STOCHd_14_3_3']
        stoch_k = safe_value(df['Stoch_K'].iloc[-1], 50)
        stoch_d = safe_value(df['Stoch_D'].iloc[-1], 50)
        stoch_oversold_cross = stoch_k > stoch_d and stoch_k < 30
        stoch_overbought_cross = stoch_k < stoch_d and stoch_k > 70
    except:
        stoch_k, stoch_d = 50, 50
        stoch_oversold_cross, stoch_overbought_cross = False, False
    
    score, signals = 0, []
    if rsi < 30: score += 1; signals.append("RSI Quá bán")
    elif rsi > 70: score -= 1; signals.append("RSI Quá mua")
    if macd_crossover: score += 1; signals.append("MACD Cắt lên")
    elif macd_crossunder: score -= 1; signals.append("MACD Cắt xuống")
    if stoch_oversold_cross: score += 1; signals.append("Stoch Quá bán")
    elif stoch_overbought_cross: score -= 1; signals.append("Stoch Quá mua")
    
    rsi_status = "Quá mua" if rsi > 70 else ("Quá bán" if rsi < 30 else "Trung tính")
    
    return {"rsi": rsi, "rsi_status": rsi_status, "macd": macd_val, "macd_signal": macd_signal_val,
            "macd_crossover": macd_crossover, "macd_crossunder": macd_crossunder,
            "stoch_k": stoch_k, "stoch_d": stoch_d, "score": max(min(score, 1), -1), "signals": signals}


def detect_price_patterns(df: pd.DataFrame) -> Dict:
    patterns, score = [], 0
    current_bb_width, avg_bb_width = 0, 0
    
    try:
        order = min(3, len(df) // 10) if len(df) > 10 else 2
        recent_lows = df['Low'].tail(20).values
        low_indices = argrelextrema(recent_lows, np.less, order=order)[0]
        if len(low_indices) >= 2:
            last_two_lows = recent_lows[low_indices[-2:]]
            if abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] * 100 <= 2:
                patterns.append("Đáy đôi (Tăng giá)"); score += 1
        
        recent_highs = df['High'].tail(20).values
        high_indices = argrelextrema(recent_highs, np.greater, order=order)[0]
        if len(high_indices) >= 2:
            last_two_highs = recent_highs[high_indices[-2:]]
            if abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] * 100 <= 2:
                patterns.append("Đỉnh đôi (Giảm giá)"); score -= 1
    except:
        pass
    
    try:
        bb = ta.bbands(df['Close'], length=20, std=2)
        bb_cols = list(bb.columns)
        bb_upper_col = next((c for c in bb_cols if 'BBU' in c), None)
        bb_lower_col = next((c for c in bb_cols if 'BBL' in c), None)
        if bb_upper_col and bb_lower_col:
            bb_upper = safe_value(bb[bb_upper_col].iloc[-1])
            bb_lower = safe_value(bb[bb_lower_col].iloc[-1])
            close_price = safe_value(df['Close'].iloc[-1], 1)
            if close_price > 0:
                current_bb_width = (bb_upper - bb_lower) / close_price * 100
                avg_bb_width = current_bb_width  # Simplified
                if current_bb_width < 5:
                    patterns.append("Tích lũy/Nén giá")
    except:
        pass
    
    try:
        price_change_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
        recent_range = (df['High'].tail(5).max() - df['Low'].tail(5).min()) / df['Close'].iloc[-1] * 100
        if abs(price_change_5d) > 5 and recent_range < 3:
            if price_change_5d > 0: patterns.append("Cờ tăng"); score += 0.5
            else: patterns.append("Cờ giảm"); score -= 0.5
    except:
        pass
    
    return {"patterns": patterns if patterns else ["Không phát hiện mô hình đáng kể"],
            "bb_width": current_bb_width, "avg_bb_width": avg_bb_width, "score": max(min(score, 1), -1)}


def detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
    patterns, score = [], 0
    body_ratio = 0
    
    try:
        latest, prev = df.iloc[-1], df.iloc[-2]
        o, c, h, l = latest['Open'], latest['Close'], latest['High'], latest['Low']
        body, full_range = abs(c - o), h - l
        upper_wick, lower_wick = h - max(o, c), min(o, c) - l
        body_ratio = body / full_range if full_range > 0 else 0
        
        if full_range > 0:
            if body_ratio < 0.3 and lower_wick > body * 2 and upper_wick < body:
                patterns.append("Nến Búa (Tăng giá)"); score += 1
            elif body_ratio < 0.3 and upper_wick > body * 2 and lower_wick < body:
                patterns.append("Sao Băng (Giảm giá)"); score -= 1
            if body_ratio < 0.1: patterns.append("Doji (Phân vân)")
        
        if (prev['Close'] < prev['Open'] and c > o and o < prev['Close'] and c > prev['Open']):
            patterns.append("Nhấn chìm tăng"); score += 1
        elif (prev['Close'] > prev['Open'] and c < o and o > prev['Close'] and c < prev['Open']):
            patterns.append("Nhấn chìm giảm"); score -= 1
    except:
        pass
    
    return {"patterns": patterns if patterns else ["Không phát hiện mô hình nến"],
            "body_ratio": body_ratio, "score": max(min(score, 1), -1)}


def calculate_risk_management(entry_price: float, support_level: float, account_size: float, risk_percent: float, is_vnd: bool = True) -> Dict:
    stop_loss = support_level * 0.995
    risk_per_share = entry_price - stop_loss
    
    if risk_per_share <= 0:
        return {"error": "Tính toán dừng lỗ không hợp lệ", "stop_loss": stop_loss, "take_profit": entry_price,
                "position_size": 0, "risk_amount": 0, "reward_amount": 0, "risk_per_share": 0, "risk_reward_ratio": "N/A", "is_vnd": is_vnd}
    
    risk_amount = account_size * (risk_percent / 100)
    position_size = int(risk_amount / risk_per_share)
    if is_vnd:
        position_size = (position_size // 100) * 100
    take_profit = entry_price + (risk_per_share * 2)
    reward_amount = position_size * (take_profit - entry_price)
    
    return {"entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
            "position_size": position_size, "risk_amount": risk_amount, "reward_amount": reward_amount,
            "risk_per_share": risk_per_share, "risk_reward_ratio": "1:2", "is_vnd": is_vnd}


def calculate_overall_signal(trend: Dict, volume: Dict, momentum: Dict, patterns: Dict, candles: Dict) -> Dict:
    total_score = trend['score']*1.5 + volume['score']*1.0 + momentum['score']*1.0 + patterns['score']*0.75 + candles['score']*0.75
    normalized_score = total_score / 5
    
    if normalized_score > 0.3 and "UPTREND" in trend['trend_en']:
        signal, signal_class, recommendation = "TĂNG GIÁ", "bullish", "MUA"
    elif normalized_score < -0.3:
        signal, signal_class, recommendation = "GIẢM GIÁ", "bearish", "BÁN"
    else:
        signal, signal_class, recommendation = "TRUNG TÍNH", "neutral", "CHỜ"
    
    return {"signal": signal, "signal_class": signal_class, "recommendation": recommendation,
            "score": round(normalized_score * 100, 1), "raw_score": round(total_score, 2)}


def create_main_chart(df: pd.DataFrame, sr_data: Dict, ticker: str, is_vnd: bool = True) -> go.Figure:
    currency = "₫" if is_vnd else "$"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2], subplot_titles=(f'{ticker} - Biến động Giá', 'Khối lượng', 'RSI'))
    
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                  name='Giá', increasing_line_color='#00ff88', decreasing_line_color='#ff4757'), row=1, col=1)
    
    # Chỉ vẽ EMA nếu có dữ liệu
    if 'EMA20' in df.columns and not df['EMA20'].isna().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name='EMA 20', line=dict(color='#ffd93d', width=1.5)), row=1, col=1)
    if 'EMA50' in df.columns and not df['EMA50'].isna().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], name='EMA 50', line=dict(color='#6c5ce7', width=1.5)), row=1, col=1)
    if 'EMA200' in df.columns and not df['EMA200'].isna().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA200'], name='EMA 200', line=dict(color='#fd79a8', width=2)), row=1, col=1)
    
    # Support/Resistance
    for level in sr_data['support_levels'][-3:]:
        label = f"HT: {currency}{level:,.0f}" if is_vnd else f"HT: {currency}{level:.2f}"
        fig.add_hline(y=level, line_dash="dash", line_color="rgba(0,255,136,0.5)", annotation_text=label, row=1, col=1)
    for level in sr_data['resistance_levels'][-3:]:
        label = f"KC: {currency}{level:,.0f}" if is_vnd else f"KC: {currency}{level:.2f}"
        fig.add_hline(y=level, line_dash="dash", line_color="rgba(255,71,87,0.5)", annotation_text=label, row=1, col=1)
    
    # Volume
    colors = ['#00ff88' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff4757' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Khối lượng', marker_color=colors), row=2, col=1)
    if 'Vol_SMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Vol_SMA20'], name='Vol SMA20', line=dict(color='#ffd93d', width=1)), row=2, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#6c5ce7', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.7)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.7)", row=3, col=1)
    
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(10,10,15,0)', plot_bgcolor='rgba(26,26,36,0.8)',
                      font=dict(family='JetBrains Mono', color='#a0a0b0'), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(26,26,36,0.8)'),
                      height=700, margin=dict(l=10, r=10, t=50, b=10), xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor='rgba(42,42,58,0.5)')
    fig.update_yaxes(gridcolor='rgba(42,42,58,0.5)')
    return fig


def main():
    st.markdown('<h1 class="main-header">📊 Phân tích Kỹ thuật 7 Bước</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #a0a0b0; margin-top: -1rem;">🇻🇳 Hỗ trợ Cổ phiếu Việt Nam (HOSE, HNX)</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎯 Cài đặt Phân tích")
        
        market = st.radio("Chọn thị trường", ["🇻🇳 Việt Nam", "🌍 Quốc tế"], horizontal=True)
        is_vn_market = market == "🇻🇳 Việt Nam"
        
        if is_vn_market:
            st.markdown("##### Cổ phiếu phổ biến:")
            all_stocks = {}
            for exchange, stocks in VN_STOCKS.items():
                for code, name in stocks.items():
                    all_stocks[f"{code} - {name}"] = code
            
            selected_stock = st.selectbox("Chọn từ danh sách", options=["-- Nhập mã khác --"] + list(all_stocks.keys()), index=0)
            
            if selected_stock == "-- Nhập mã khác --":
                ticker_input = st.text_input("Nhập mã cổ phiếu", value="FPT", help="VD: FPT, VNM, VIC, VCB...").upper()
            else:
                ticker_input = all_stocks[selected_stock]
            
            ticker = get_vn_ticker(ticker_input)
            currency = "₫"
            default_account = 100000000
        else:
            ticker = st.text_input("Mã cổ phiếu", value="AAPL", help="VD: AAPL, GOOGL, MSFT...").upper()
            currency = "$"
            default_account = 10000
        
        period_options = {"1 Tháng": "1mo", "3 Tháng": "3mo", "6 Tháng": "6mo", "1 Năm": "1y", "2 Năm": "2y"}
        period_label = st.selectbox("Khoảng thời gian", options=list(period_options.keys()), index=3)
        period = period_options[period_label]
        
        st.markdown("---")
        st.markdown("### 💰 Quản lý Rủi ro")
        
        if is_vn_market:
            account_size = st.number_input("Vốn tài khoản (VNĐ)", value=default_account, min_value=1000000, step=10000000, format="%d")
            st.caption(f"💵 {account_size:,.0f} VNĐ")
        else:
            account_size = st.number_input("Vốn tài khoản ($)", value=default_account, min_value=100, step=1000)
        
        risk_percent = st.slider("Rủi ro tối đa (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        
        st.markdown("---")
        analyze_btn = st.button("🔍 Chạy Phân tích", type="primary", use_container_width=True)
    
    if analyze_btn or ticker:
        display_ticker = ticker.replace('.VN', '').replace('.HN', '')
        with st.spinner(f"Đang phân tích {display_ticker}..."):
            df = fetch_stock_data(ticker, period)
            
            if df is None or len(df) < 20:
                st.error(f"❌ Không thể lấy dữ liệu cho **{display_ticker}**.")
                st.info("💡 Thử các mã: FPT, VNM, VIC, VCB, HPG, MBB, TCB")
                return
            
            # Cảnh báo nếu không đủ dữ liệu cho EMA200
            if len(df) < 200:
                st.warning(f"⚠️ Chỉ có {len(df)} phiên. EMA200 cần ít nhất 200 phiên để chính xác. Nên chọn khoảng thời gian dài hơn (1-2 năm).")
            
            df = calculate_emas(df)
            trend_data = analyze_trend(df)
            sr_data = find_support_resistance(df)
            volume_data = analyze_volume(df)
            momentum_data = analyze_momentum(df)
            pattern_data = detect_price_patterns(df)
            candle_data = detect_candlestick_patterns(df)
            overall = calculate_overall_signal(trend_data, volume_data, momentum_data, pattern_data, candle_data)
            risk_data = calculate_risk_management(df['Close'].iloc[-1], sr_data['nearest_support'], account_size, risk_percent, is_vn_market)
            
            # Header info
            stock_info = get_stock_info(ticker)
            st.markdown(f'<div class="stock-info"><strong>📈 {stock_info}</strong></div>', unsafe_allow_html=True)
            
            # Signal card
            signal_class = f"signal-{overall['signal_class']}"
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div style="font-size: 1rem; color: #a0a0b0; text-transform: uppercase; letter-spacing: 2px;">Tín hiệu cho {display_ticker}</div>
                <div class="signal-text">{overall['signal']}</div>
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Khuyến nghị: <strong>{overall['recommendation']}</strong></div>
                <div class="score-badge" style="background: rgba(255,255,255,0.1);">Độ tin cậy: {overall['score']}%</div>
            </div>""", unsafe_allow_html=True)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{format_price(df["Close"].iloc[-1], currency, is_vn_market)}</div><div class="metric-label">Giá hiện tại</div></div>', unsafe_allow_html=True)
            with col2:
                change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                color = "#00ff88" if change >= 0 else "#ff4757"
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="color: {color}">{change:+.2f}%</div><div class="metric-label">Thay đổi</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="font-size: 1rem;">{trend_data["trend"]}</div><div class="metric-label">Xu hướng</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{momentum_data["rsi"]:.1f}</div><div class="metric-label">RSI (14)</div></div>', unsafe_allow_html=True)
            
            # Chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Biểu đồ Giá với Chỉ báo")
            st.plotly_chart(create_main_chart(df, sr_data, display_ticker, is_vn_market), use_container_width=True)
            
            # Analysis report
            st.markdown("### 📋 Báo cáo Phân tích Chi tiết")
            
            with st.expander("**Bước 1: Xác định Xu hướng (Bộ lọc EMA)**", expanded=True):
                status = "✅" if "UPTREND" in trend_data['trend_en'] else ("❌" if "DOWNTREND" in trend_data['trend_en'] else "⚠️")
                step_class = "step-pass" if "UPTREND" in trend_data['trend_en'] else ("step-fail" if "DOWNTREND" in trend_data['trend_en'] else "step-warn")
                st.markdown(f'<div class="step-card {step_class}"><strong>{status} Xu hướng: {trend_data["trend"]}</strong></div>', unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("EMA 20", format_price(trend_data['ema20'], currency, is_vn_market))
                c2.metric("EMA 50", format_price(trend_data['ema50'], currency, is_vn_market))
                c3.metric("EMA 200", format_price(trend_data['ema200'], currency, is_vn_market) if trend_data['has_ema200'] else "N/A (cần thêm dữ liệu)")
                st.caption(f"Độ dốc EMA20: {trend_data['ema20_slope']:.2f}% | Độ dốc EMA50: {trend_data['ema50_slope']:.2f}%")
            
            with st.expander("**Bước 2: Mức then chốt (Hỗ trợ & Kháng cự)**"):
                st.markdown('<div class="step-card step-pass"><strong>📍 Đã xác định các mức then chốt</strong></div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Mức Hỗ trợ:**")
                    for level in sr_data['support_levels'][-3:]:
                        st.write(f"• {format_price(level, currency, is_vn_market)}")
                with c2:
                    st.markdown("**Mức Kháng cự:**")
                    for level in sr_data['resistance_levels'][-3:]:
                        st.write(f"• {format_price(level, currency, is_vn_market)}")
            
            with st.expander("**Bước 3: Phân tích Khối lượng**"):
                status = "✅" if volume_data['signal'] == "bullish" else ("❌" if volume_data['signal'] == "bearish" else "⚠️")
                step_class = "step-pass" if volume_data['signal'] == "bullish" else ("step-fail" if volume_data['signal'] == "bearish" else "step-warn")
                vol_status = "Đột biến khối lượng!" if volume_data['strong_volume'] else "Khối lượng bình thường"
                st.markdown(f'<div class="step-card {step_class}"><strong>{status} {vol_status}</strong></div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("KL hiện tại", f"{volume_data['current_volume']:,.0f}")
                c2.metric("KL TB 20 phiên", f"{volume_data['vol_sma20']:,.0f}")
                c3.metric("Tỷ lệ KL", f"{volume_data['volume_ratio']:.2f}x")
            
            with st.expander("**Bước 4: Động lượng & Chỉ báo**"):
                status = "✅" if momentum_data['score'] > 0 else ("❌" if momentum_data['score'] < 0 else "⚠️")
                step_class = "step-pass" if momentum_data['score'] > 0 else ("step-fail" if momentum_data['score'] < 0 else "step-warn")
                signals_text = ", ".join(momentum_data['signals']) if momentum_data['signals'] else "Không có tín hiệu đặc biệt"
                st.markdown(f'<div class="step-card {step_class}"><strong>{status} {signals_text}</strong></div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("RSI (14)", f"{momentum_data['rsi']:.1f}", momentum_data['rsi_status'])
                c2.metric("MACD", f"{momentum_data['macd']:.2f}")
                c3.metric("Stochastic K/D", f"{momentum_data['stoch_k']:.1f}/{momentum_data['stoch_d']:.1f}")
            
            with st.expander("**Bước 5: Mô hình Giá**"):
                status = "✅" if pattern_data['score'] > 0 else ("❌" if pattern_data['score'] < 0 else "⚠️")
                step_class = "step-pass" if pattern_data['score'] > 0 else ("step-fail" if pattern_data['score'] < 0 else "step-warn")
                st.markdown(f'<div class="step-card {step_class}"><strong>{status} {", ".join(pattern_data["patterns"])}</strong></div>', unsafe_allow_html=True)
            
            with st.expander("**Bước 6: Mô hình Nến**"):
                status = "✅" if candle_data['score'] > 0 else ("❌" if candle_data['score'] < 0 else "⚠️")
                step_class = "step-pass" if candle_data['score'] > 0 else ("step-fail" if candle_data['score'] < 0 else "step-warn")
                st.markdown(f'<div class="step-card {step_class}"><strong>{status} {", ".join(candle_data["patterns"])}</strong></div>', unsafe_allow_html=True)
            
            with st.expander("**Bước 7: Quản lý Rủi ro**", expanded=True):
                if "error" in risk_data and risk_data.get("position_size", 0) == 0:
                    st.warning(risk_data.get('error', 'Lỗi tính toán'))
                else:
                    st.markdown('<div class="trade-setup"><h4 style="color: #6c5ce7;">💼 Thiết lập Giao dịch</h4></div>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("📥 Giá vào", format_price(risk_data['entry_price'], currency, is_vn_market))
                    c2.metric("🛑 Dừng lỗ", format_price(risk_data['stop_loss'], currency, is_vn_market))
                    c3.metric("🎯 Chốt lời", format_price(risk_data['take_profit'], currency, is_vn_market))
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("📊 Khối lượng", f"{risk_data['position_size']:,} CP")
                    c2.metric("💵 Rủi ro", format_price(risk_data['risk_amount'], currency, is_vn_market))
                    c3.metric("💰 Lợi nhuận", format_price(risk_data['reward_amount'], currency, is_vn_market))
                    total = risk_data['position_size'] * risk_data['entry_price']
                    st.info(f"💡 Tổng vốn: **{format_price(total, currency, is_vn_market)}** ({(total/account_size)*100:.1f}% tài khoản)")
            
            st.markdown("---")
            st.caption("⚠️ **Miễn trừ:** Công cụ chỉ dành cho giáo dục. Không phải tư vấn tài chính.")


if __name__ == "__main__":
    main()
