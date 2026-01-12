"""
Công cụ Phân tích Kỹ thuật 7 Bước Chuyên nghiệp
Hỗ trợ Cổ phiếu Việt Nam (HOSE, HNX)
+ Tính năng Top 10 Cổ phiếu Tín hiệu Tốt Nhất
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
from typing import Dict, List, Optional
import warnings
import time
warnings.filterwarnings('ignore')

# Danh sách cổ phiếu Việt Nam phổ biến
VN_STOCKS = {

    # === NGÂN HÀNG (25) ===
    "VCB": "Vietcombank",
    "BID": "BIDV",
    "CTG": "VietinBank",
    "TCB": "Techcombank",
    "MBB": "MB Bank",
    "ACB": "ACB",
    "VPB": "VPBank",
    "HDB": "HDBank",
    "TPB": "TPBank",
    "STB": "Sacombank",
    "SHB": "SHB",
    "LPB": "LPBank",
    "VIB": "VIB",
    "OCB": "OCB",
    "MSB": "MSB",
    "NAB": "NamABank",
    "ABB": "ABBank",
    "BAB": "BacABank",
    "EIB": "Eximbank",
    "PGB": "PGBank",
    "SGB": "SaigonBank",
    "VBB": "VietBank",
    "KLB": "KienLongBank",
    "BVB": "BaoVietBank",
    "SSB": "SeABank",

    # === BẤT ĐỘNG SẢN (15) ===
    "VHM": "Vinhomes",
    "VIC": "Vingroup",
    "VRE": "Vincom Retail",
    "NVL": "Novaland",
    "PDR": "Phat Dat",
    "DIG": "DIC Corp",
    "DXG": "Dat Xanh",
    "KDH": "Khang Dien",
    "NLG": "Nam Long",
    "HDG": "Ha Do",
    "SCR": "TTC Land",
    "CEO": "CEO Group",
    "BCM": "Becamex IDC",
    "SZC": "Sonadezi Chau Duc",
    "IDC": "IDICO",

    # === TIÊU DÙNG – BÁN LẺ (10) ===
    "VNM": "Vinamilk",
    "MSN": "Masan Group",
    "MWG": "The Gioi Di Dong",
    "PNJ": "PNJ",
    "SAB": "Sabeco",
    "KDC": "KIDO Group",
    "ANV": "Nam Viet",
    "VHC": "Vinh Hoan",
    "FRT": "FPT Retail",
    "DGW": "Digiworld",

    # === CÔNG NGHỆ (5) ===
    "FPT": "FPT Corp",
    "CMG": "CMC Group",
    "ELC": "Elcom",
    "FOX": "FPT Telecom",
    "VGI": "Viettel Global",

    # === CHỨNG KHOÁN (10) ===
    "SSI": "SSI",
    "VND": "VNDirect",
    "HCM": "HSC",
    "VCI": "Vietcap",
    "BSI": "BIDV Securities",
    "FTS": "FPT Securities",
    "CTS": "VietinBank Securities",
    "AGR": "Agriseco",
    "TVS": "Thien Viet",
    "ORS": "Tien Phong Sec",

    # === DẦU KHÍ – ĐIỆN – NĂNG LƯỢNG (10) ===
    "GAS": "PV Gas",
    "PLX": "Petrolimex",
    "POW": "PV Power",
    "PVD": "PV Drilling",
    "PVS": "PTSC",
    "BSR": "Binh Son Refinery",
    "NT2": "Nhiet Dien Nhon Trach 2",
    "PC1": "PC1 Group",
    "GEG": "Gia Lai Electric",
    "REE": "REE Corp",

    # === NGUYÊN VẬT LIỆU – CÔNG NGHIỆP (10) ===
    "HPG": "Hoa Phat",
    "HSG": "Hoa Sen",
    "NKG": "Nam Kim",
    "DGC": "Duc Giang",
    "DCM": "Dam Ca Mau",
    "DPM": "Dam Phu My",
    "VGC": "Viglacera",
    "BMP": "Binh Minh Plastic",
    "BFC": "Binh Dien Fertilizer",
    "DHC": "Dong Hai",

    # === LOGISTICS – CẢNG – KCN (5) ===
    "GMD": "Gemadept",
    "VSC": "Viconship",
    "HAH": "Hai An Transport",
    "SCS": "Saigon Cargo",
    "KBC": "Kinh Bac City"
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
    .top-stock-card {
        background: var(--bg-card); border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
        border: 1px solid var(--border-color); transition: all 0.3s ease;
    }
    .top-stock-card:hover { border-color: var(--accent-green); transform: translateY(-2px); }
    .rank-badge {
        display: inline-block; width: 32px; height: 32px; border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
        color: white; font-weight: 700; text-align: center; line-height: 32px;
        font-family: 'JetBrains Mono', monospace;
    }
    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A0A0A0); }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #8B4513); }
    [data-testid="stSidebar"] { background: var(--bg-secondary); }
    @media (max-width: 768px) { .signal-card { padding: 1.5rem; } .metric-box { padding: 1rem; } .metric-value { font-size: 1.2rem; } }
</style>
""", unsafe_allow_html=True)


def safe_value(val, default=0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def format_price(val, currency="₫", is_vn=True):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if is_vn:
        return f"{currency}{val:,.0f}"
    return f"{currency}{val:.2f}"


def get_vn_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if '.VN' in ticker or '.HN' in ticker:
        return ticker
    return f"{ticker}.VN"


@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        df = df.reset_index()
        return df
    except:
        return None


def quick_analyze(df: pd.DataFrame) -> Dict:
    """Phân tích nhanh để xếp hạng cổ phiếu."""
    if df is None or len(df) < 20:
        return None
    
    try:
        # EMAs
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        
        latest = df.iloc[-1]
        price = safe_value(latest['Close'])
        ema20 = safe_value(latest['EMA20'])
        ema50 = safe_value(latest['EMA50'])
        
        # Trend score
        trend_score = 0
        if price > ema20 > ema50:
            trend_score = 1.5
        elif price > ema20:
            trend_score = 0.5
        elif price < ema20 < ema50:
            trend_score = -1.5
        elif price < ema20:
            trend_score = -0.5
        
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        rsi = safe_value(df['RSI'].iloc[-1], 50)
        
        rsi_score = 0
        if 30 < rsi < 50:  # Oversold bouncing
            rsi_score = 1
        elif 50 <= rsi < 70:  # Healthy uptrend
            rsi_score = 0.5
        elif rsi >= 70:
            rsi_score = -0.5
        elif rsi <= 30:
            rsi_score = 0.5  # Potential bounce
        
        # Volume
        df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
        vol_ratio = df['Volume'].iloc[-1] / df['Vol_SMA20'].iloc[-1] if df['Vol_SMA20'].iloc[-1] > 0 else 1
        
        vol_score = 0
        if vol_ratio > 1.5 and trend_score > 0:
            vol_score = 1
        elif vol_ratio > 1.2 and trend_score > 0:
            vol_score = 0.5
        
        # MACD
        try:
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            macd_val = safe_value(macd['MACD_12_26_9'].iloc[-1])
            macd_signal = safe_value(macd['MACDs_12_26_9'].iloc[-1])
            macd_hist = safe_value(macd['MACDh_12_26_9'].iloc[-1])
            macd_hist_prev = safe_value(macd['MACDh_12_26_9'].iloc[-2])
            
            macd_score = 0
            if macd_val > macd_signal and macd_hist > macd_hist_prev:
                macd_score = 1
            elif macd_val > macd_signal:
                macd_score = 0.5
            elif macd_val < macd_signal:
                macd_score = -0.5
        except:
            macd_score = 0
            macd_hist = 0
        
        # Price change
        price_change_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) > 6 else 0
        price_change_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100 if len(df) > 21 else 0
        
        # Total score
        total_score = trend_score + rsi_score + vol_score + macd_score
        
        # Signal
        if total_score >= 2:
            signal = "MUA MẠNH"
            signal_class = "bullish"
        elif total_score >= 1:
            signal = "MUA"
            signal_class = "bullish"
        elif total_score <= -2:
            signal = "BÁN MẠNH"
            signal_class = "bearish"
        elif total_score <= -1:
            signal = "BÁN"
            signal_class = "bearish"
        else:
            signal = "CHỜ"
            signal_class = "neutral"
        
        return {
            "price": price,
            "rsi": rsi,
            "trend_score": trend_score,
            "rsi_score": rsi_score,
            "vol_score": vol_score,
            "macd_score": macd_score,
            "total_score": total_score,
            "signal": signal,
            "signal_class": signal_class,
            "vol_ratio": vol_ratio,
            "price_change_5d": price_change_5d,
            "price_change_20d": price_change_20d,
            "macd_hist": macd_hist
        }
    except Exception as e:
        return None


def scan_top_stocks(stock_list: Dict, period: str = "6mo", progress_callback=None) -> List[Dict]:
    """Quét và xếp hạng tất cả cổ phiếu."""
    results = []
    total = len(stock_list)
    
    for i, (code, name) in enumerate(stock_list.items()):
        if progress_callback:
            progress_callback((i + 1) / total, f"Đang quét {code}... ({i+1}/{total})")
        
        ticker = get_vn_ticker(code)
        df = fetch_stock_data(ticker, period)
        
        if df is not None and len(df) >= 20:
            analysis = quick_analyze(df)
            if analysis:
                results.append({
                    "code": code,
                    "name": name,
                    "ticker": ticker,
                    **analysis
                })
        
        time.sleep(0.1)  # Rate limiting
    
    # Sort by total score
    results.sort(key=lambda x: x['total_score'], reverse=True)
    return results


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
    
    has_ema200 = not (pd.isna(latest['EMA200']) if 'EMA200' in latest else True)
    
    if has_ema200 and price > ema20 > ema50 > ema200:
        trend, trend_en, signal, score = "XU HƯỚNG TĂNG", "UPTREND", "bullish", 1
    elif has_ema200 and price < ema20 < ema50 < ema200:
        trend, trend_en, signal, score = "XU HƯỚNG GIẢM", "DOWNTREND", "bearish", -1
    elif price > ema20 > ema50:
        trend, trend_en, signal, score = "TĂNG (ngắn hạn)", "UPTREND", "bullish", 0.5
    elif price < ema20 < ema50:
        trend, trend_en, signal, score = "GIẢM (ngắn hạn)", "DOWNTREND", "bearish", -0.5
    else:
        trend, trend_en, signal, score = "ĐI NGANG", "SIDEWAYS", "neutral", 0
    
    try:
        ema20_slope = (df['EMA20'].iloc[-1] - df['EMA20'].iloc[-5]) / df['EMA20'].iloc[-5] * 100
        if np.isnan(ema20_slope): ema20_slope = 0
    except: ema20_slope = 0
    
    try:
        ema50_slope = (df['EMA50'].iloc[-1] - df['EMA50'].iloc[-5]) / df['EMA50'].iloc[-5] * 100
        if np.isnan(ema50_slope): ema50_slope = 0
    except: ema50_slope = 0
    
    return {"trend": trend, "trend_en": trend_en, "signal": signal, "score": score,
            "price": price, "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "ema20_slope": ema20_slope, "ema50_slope": ema50_slope, "has_ema200": has_ema200}


def find_support_resistance(df: pd.DataFrame, order: int = 5) -> Dict:
    close_prices = df['Close'].values
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
                  "0.618": recent_high - diff * 0.618, "1.0": recent_low}
    
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
        obv_slope = (df['OBV'].iloc[-1] - df['OBV'].iloc[-5]) / abs(df['OBV'].iloc[-5]) * 100 if df['OBV'].iloc[-5] != 0 else 0
        if np.isnan(obv_slope): obv_slope = 0
    except: obv_slope = 0
    
    obv_bullish = obv_slope > 0
    
    if strong_volume and obv_bullish: score, signal = 1, "bullish"
    elif strong_volume and not obv_bullish: score, signal = -1, "bearish"
    else: score, signal = 0, "neutral"
    
    return {"current_volume": latest_volume, "vol_sma20": vol_sma, "volume_ratio": volume_ratio,
            "strong_volume": strong_volume, "obv_slope": obv_slope, "obv_bullish": obv_bullish,
            "score": score, "signal": signal}


def analyze_momentum(df: pd.DataFrame) -> Dict:
    df['RSI'] = ta.rsi(df['Close'], length=14)
    rsi = safe_value(df['RSI'].iloc[-1], 50)
    
    try:
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'], df['MACD_Signal'] = macd['MACD_12_26_9'], macd['MACDs_12_26_9']
        macd_val, macd_signal_val = safe_value(df['MACD'].iloc[-1]), safe_value(df['MACD_Signal'].iloc[-1])
        macd_crossover = macd_val > macd_signal_val and safe_value(df['MACD'].iloc[-2]) <= safe_value(df['MACD_Signal'].iloc[-2])
        macd_crossunder = macd_val < macd_signal_val and safe_value(df['MACD'].iloc[-2]) >= safe_value(df['MACD_Signal'].iloc[-2])
    except:
        macd_val, macd_signal_val = 0, 0
        macd_crossover, macd_crossunder = False, False
    
    try:
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
        stoch_k, stoch_d = safe_value(stoch['STOCHk_14_3_3'].iloc[-1], 50), safe_value(stoch['STOCHd_14_3_3'].iloc[-1], 50)
    except:
        stoch_k, stoch_d = 50, 50
    
    score, signals = 0, []
    if rsi < 30: score += 1; signals.append("RSI Quá bán")
    elif rsi > 70: score -= 1; signals.append("RSI Quá mua")
    if macd_crossover: score += 1; signals.append("MACD Cắt lên")
    elif macd_crossunder: score -= 1; signals.append("MACD Cắt xuống")
    
    rsi_status = "Quá mua" if rsi > 70 else ("Quá bán" if rsi < 30 else "Trung tính")
    
    return {"rsi": rsi, "rsi_status": rsi_status, "macd": macd_val, "macd_signal": macd_signal_val,
            "macd_crossover": macd_crossover, "macd_crossunder": macd_crossunder,
            "stoch_k": stoch_k, "stoch_d": stoch_d, "score": max(min(score, 1), -1), "signals": signals}


def detect_patterns(df: pd.DataFrame) -> Dict:
    patterns, score = [], 0
    try:
        order = min(3, len(df) // 10) if len(df) > 10 else 2
        recent_lows = df['Low'].tail(20).values
        low_indices = argrelextrema(recent_lows, np.less, order=order)[0]
        if len(low_indices) >= 2:
            last_two = recent_lows[low_indices[-2:]]
            if abs(last_two[0] - last_two[1]) / last_two[0] * 100 <= 2:
                patterns.append("Đáy đôi"); score += 1
    except: pass
    return {"patterns": patterns if patterns else ["Không có"], "score": max(min(score, 1), -1)}


def detect_candles(df: pd.DataFrame) -> Dict:
    patterns, score = [], 0
    try:
        latest, prev = df.iloc[-1], df.iloc[-2]
        o, c, h, l = latest['Open'], latest['Close'], latest['High'], latest['Low']
        body, full_range = abs(c - o), h - l
        body_ratio = body / full_range if full_range > 0 else 0
        
        if full_range > 0:
            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)
            if body_ratio < 0.3 and lower_wick > body * 2:
                patterns.append("Nến Búa"); score += 1
            if body_ratio < 0.1:
                patterns.append("Doji")
        
        if prev['Close'] < prev['Open'] and c > o and o < prev['Close'] and c > prev['Open']:
            patterns.append("Nhấn chìm tăng"); score += 1
    except: pass
    return {"patterns": patterns if patterns else ["Không có"], "body_ratio": body_ratio if 'body_ratio' in dir() else 0, "score": max(min(score, 1), -1)}


def calculate_risk(entry: float, support: float, account: float, risk_pct: float, is_vn: bool = True) -> Dict:
    stop_loss = support * 0.995
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        return {"error": True, "position_size": 0}
    
    risk_amount = account * (risk_pct / 100)
    position_size = int(risk_amount / risk_per_share)
    if is_vn: position_size = (position_size // 100) * 100
    take_profit = entry + (risk_per_share * 2)
    
    return {"entry_price": entry, "stop_loss": stop_loss, "take_profit": take_profit,
            "position_size": position_size, "risk_amount": risk_amount,
            "reward_amount": position_size * (take_profit - entry), "error": False}


def calculate_signal(trend, volume, momentum, patterns, candles) -> Dict:
    total = trend['score']*1.5 + volume['score'] + momentum['score'] + patterns['score']*0.75 + candles['score']*0.75
    norm = total / 5
    
    if norm > 0.3 and "UPTREND" in trend['trend_en']:
        return {"signal": "TĂNG GIÁ", "signal_class": "bullish", "recommendation": "MUA", "score": round(norm*100, 1)}
    elif norm < -0.3:
        return {"signal": "GIẢM GIÁ", "signal_class": "bearish", "recommendation": "BÁN", "score": round(norm*100, 1)}
    return {"signal": "TRUNG TÍNH", "signal_class": "neutral", "recommendation": "CHỜ", "score": round(norm*100, 1)}


def create_chart(df: pd.DataFrame, sr_data: Dict, ticker: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2], subplot_titles=(f'{ticker}', 'Khối lượng', 'RSI'))
    
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                  name='Giá', increasing_line_color='#00ff88', decreasing_line_color='#ff4757'), row=1, col=1)
    
    if 'EMA20' in df.columns and not df['EMA20'].isna().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name='EMA20', line=dict(color='#ffd93d', width=1.5)), row=1, col=1)
    if 'EMA50' in df.columns and not df['EMA50'].isna().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], name='EMA50', line=dict(color='#6c5ce7', width=1.5)), row=1, col=1)
    if 'EMA200' in df.columns and not df['EMA200'].isna().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA200'], name='EMA200', line=dict(color='#fd79a8', width=2)), row=1, col=1)
    
    for level in sr_data['support_levels'][-2:]:
        fig.add_hline(y=level, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=1, col=1)
    for level in sr_data['resistance_levels'][-2:]:
        fig.add_hline(y=level, line_dash="dash", line_color="rgba(255,71,87,0.5)", row=1, col=1)
    
    colors = ['#00ff88' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff4757' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='KL', marker_color=colors), row=2, col=1)
    
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#6c5ce7', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.5)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=3, col=1)
    
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(10,10,15,0)', plot_bgcolor='rgba(26,26,36,0.8)',
                      font=dict(family='JetBrains Mono', color='#a0a0b0'), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      height=600, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    return fig


def main():
    st.markdown('<h1 class="main-header">📊 Phân tích Kỹ thuật 7 Bước</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #a0a0b0;">🇻🇳 Cổ phiếu Việt Nam | Top 10 Tín hiệu Tốt Nhất</p>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["🔍 Phân tích Chi tiết", "🏆 Top 10 Cổ phiếu Tốt Nhất"])
    
    with tab1:
        with st.sidebar:
            st.markdown("### 🎯 Cài đặt")
            
            market = st.radio("Thị trường", ["🇻🇳 Việt Nam", "🌍 Quốc tế"], horizontal=True)
            is_vn = market == "🇻🇳 Việt Nam"
            
            if is_vn:
                all_stocks = {f"{k} - {v}": k for k, v in VN_STOCKS.items()}
                selected = st.selectbox("Cổ phiếu", ["-- Nhập mã --"] + list(all_stocks.keys()))
                ticker_input = st.text_input("Mã CP", value="FPT").upper() if selected == "-- Nhập mã --" else all_stocks[selected]
                ticker = get_vn_ticker(ticker_input)
                currency, default_acc = "₫", 100000000
            else:
                ticker = st.text_input("Mã CP", value="AAPL").upper()
                currency, default_acc = "$", 10000
            
            period_opts = {"1 Tháng": "1mo", "3 Tháng": "3mo", "6 Tháng": "6mo", "1 Năm": "1y", "2 Năm": "2y"}
            period = period_opts[st.selectbox("Thời gian", list(period_opts.keys()), index=3)]
            
            st.markdown("---")
            st.markdown("### 💰 Quản lý Vốn")
            account = st.number_input("Vốn" + (" (VNĐ)" if is_vn else " ($)"), value=default_acc, min_value=1000 if not is_vn else 1000000)
            risk_pct = st.slider("Rủi ro (%)", 0.5, 5.0, 2.0, 0.5)
            
            st.markdown("---")
            analyze_btn = st.button("🔍 Phân tích", type="primary", use_container_width=True)
        
        if analyze_btn or ticker:
            display = ticker.replace('.VN', '')
            with st.spinner(f"Đang phân tích {display}..."):
                df = fetch_stock_data(ticker, period)
                
                if df is None or len(df) < 20:
                    st.error(f"❌ Không có dữ liệu cho {display}")
                    return
                
                if len(df) < 200:
                    st.warning(f"⚠️ Chỉ có {len(df)} phiên. Nên chọn 1-2 năm để có EMA200.")
                
                df = calculate_emas(df)
                trend = analyze_trend(df)
                sr = find_support_resistance(df)
                vol = analyze_volume(df)
                mom = analyze_momentum(df)
                patt = detect_patterns(df)
                cand = detect_candles(df)
                sig = calculate_signal(trend, vol, mom, patt, cand)
                risk = calculate_risk(df['Close'].iloc[-1], sr['nearest_support'], account, risk_pct, is_vn)
                
                # Stock info
                name = VN_STOCKS.get(display, display)
                st.markdown(f'<div class="stock-info"><strong>📈 {display} - {name}</strong></div>', unsafe_allow_html=True)
                
                # Signal card
                st.markdown(f"""
                <div class="signal-card signal-{sig['signal_class']}">
                    <div style="font-size: 1rem; color: #a0a0b0;">Tín hiệu {display}</div>
                    <div class="signal-text">{sig['signal']}</div>
                    <div>Khuyến nghị: <strong>{sig['recommendation']}</strong></div>
                    <div class="score-badge" style="background: rgba(255,255,255,0.1);">Điểm: {sig['score']}%</div>
                </div>""", unsafe_allow_html=True)
                
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f'<div class="metric-box"><div class="metric-value">{format_price(df["Close"].iloc[-1], currency, is_vn)}</div><div class="metric-label">Giá</div></div>', unsafe_allow_html=True)
                chg = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                c2.markdown(f'<div class="metric-box"><div class="metric-value" style="color: {"#00ff88" if chg >= 0 else "#ff4757"}">{chg:+.2f}%</div><div class="metric-label">Thay đổi</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-box"><div class="metric-value" style="font-size:1rem">{trend["trend"]}</div><div class="metric-label">Xu hướng</div></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-box"><div class="metric-value">{mom["rsi"]:.1f}</div><div class="metric-label">RSI</div></div>', unsafe_allow_html=True)
                
                # Chart
                st.plotly_chart(create_chart(df, sr, display), use_container_width=True)
                
                # Analysis details
                st.markdown("### 📋 Chi tiết Phân tích")
                
                with st.expander("**Bước 1-2: Xu hướng & Hỗ trợ/Kháng cự**", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Xu hướng:** {trend['trend']}")
                        st.metric("EMA20", format_price(trend['ema20'], currency, is_vn))
                        st.metric("EMA50", format_price(trend['ema50'], currency, is_vn))
                    with c2:
                        st.markdown(f"**Hỗ trợ gần nhất:** {format_price(sr['nearest_support'], currency, is_vn)}")
                        st.markdown(f"**Kháng cự gần nhất:** {format_price(sr['nearest_resistance'], currency, is_vn)}")
                
                with st.expander("**Bước 3-4: Khối lượng & Động lượng**"):
                    c1, c2 = st.columns(2)
                    c1.metric("Khối lượng", f"{vol['current_volume']:,.0f}", f"{vol['volume_ratio']:.1f}x TB")
                    c1.metric("RSI", f"{mom['rsi']:.1f}", mom['rsi_status'])
                    c2.metric("MACD", f"{mom['macd']:.2f}")
                    c2.metric("Stoch K/D", f"{mom['stoch_k']:.0f}/{mom['stoch_d']:.0f}")
                
                with st.expander("**Bước 7: Quản lý Rủi ro**", expanded=True):
                    if risk.get('error'):
                        st.warning("Không thể tính toán vị thế")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("📥 Vào lệnh", format_price(risk['entry_price'], currency, is_vn))
                        c2.metric("🛑 Dừng lỗ", format_price(risk['stop_loss'], currency, is_vn))
                        c3.metric("🎯 Chốt lời", format_price(risk['take_profit'], currency, is_vn))
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("📊 Khối lượng", f"{risk['position_size']:,} CP")
                        c2.metric("💵 Rủi ro", format_price(risk['risk_amount'], currency, is_vn))
                        c3.metric("💰 Lợi nhuận", format_price(risk['reward_amount'], currency, is_vn))
    
    with tab2:
        st.markdown("### 🏆 Top 10 Cổ phiếu Có Tín hiệu Kỹ thuật Tốt Nhất")
        st.markdown("*Quét và xếp hạng dựa trên: Xu hướng EMA, RSI, MACD, Khối lượng*")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            scan_btn = st.button("🔄 Quét thị trường", type="primary", use_container_width=True)
        
        if scan_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(pct, text):
                progress_bar.progress(pct)
                status_text.text(text)
            
            results = scan_top_stocks(VN_STOCKS, "6mo", update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if results:
                # Store in session
                st.session_state['scan_results'] = results
                st.session_state['scan_time'] = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        
        # Display results
        if 'scan_results' in st.session_state:
            results = st.session_state['scan_results']
            st.caption(f"🕐 Cập nhật lúc: {st.session_state.get('scan_time', 'N/A')}")
            
            # Top 10 cards
            top_10 = results[:10]
            
            for i, stock in enumerate(top_10):
                rank = i + 1
                rank_class = f"rank-{rank}" if rank <= 3 else ""
                
                signal_color = "#00ff88" if stock['signal_class'] == 'bullish' else ("#ff4757" if stock['signal_class'] == 'bearish' else "#ffd93d")
                
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 1.5, 1.5, 1.5])
                    
                    with col1:
                        st.markdown(f'<div class="rank-badge {rank_class}">{rank}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**{stock['code']}**")
                        st.caption(stock['name'])
                    
                    with col3:
                        st.markdown(f"<span style='color:{signal_color};font-weight:700'>{stock['signal']}</span>", unsafe_allow_html=True)
                        st.caption(f"Điểm: {stock['total_score']:.1f}")
                    
                    with col4:
                        st.metric("Giá", f"₫{stock['price']:,.0f}", f"{stock['price_change_5d']:+.1f}% (5D)")
                    
                    with col5:
                        st.metric("RSI", f"{stock['rsi']:.0f}", f"Vol: {stock['vol_ratio']:.1f}x")
                    
                    st.markdown("---")
            
            # Summary table
            with st.expander("📊 Bảng tổng hợp đầy đủ"):
                df_results = pd.DataFrame(results)
                df_display = df_results[['code', 'name', 'price', 'signal', 'total_score', 'rsi', 'vol_ratio', 'price_change_5d', 'price_change_20d']].copy()
                df_display.columns = ['Mã', 'Tên', 'Giá', 'Tín hiệu', 'Điểm', 'RSI', 'Vol Ratio', '% 5D', '% 20D']
                df_display['Giá'] = df_display['Giá'].apply(lambda x: f"₫{x:,.0f}")
                df_display['Điểm'] = df_display['Điểm'].round(2)
                df_display['RSI'] = df_display['RSI'].round(1)
                df_display['Vol Ratio'] = df_display['Vol Ratio'].round(2)
                df_display['% 5D'] = df_display['% 5D'].round(2)
                df_display['% 20D'] = df_display['% 20D'].round(2)
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Bottom 5 (worst signals)
            with st.expander("📉 5 Cổ phiếu Tín hiệu Yếu Nhất"):
                bottom_5 = results[-5:][::-1]
                for stock in bottom_5:
                    signal_color = "#ff4757" if stock['signal_class'] == 'bearish' else "#ffd93d"
                    st.markdown(f"**{stock['code']}** - {stock['name']} | <span style='color:{signal_color}'>{stock['signal']}</span> | Điểm: {stock['total_score']:.1f}", unsafe_allow_html=True)
        else:
            st.info("👆 Nhấn **Quét thị trường** để tìm Top 10 cổ phiếu có tín hiệu tốt nhất")
            
            st.markdown("""
            #### 📖 Tiêu chí xếp hạng:
            - **Xu hướng EMA**: Giá > EMA20 > EMA50 (+1.5 điểm)
            - **RSI**: Vùng 30-50 tăng (+1 điểm), 50-70 (+0.5 điểm)
            - **Khối lượng**: > 1.5x trung bình (+1 điểm)
            - **MACD**: Histogram tăng (+1 điểm)
            
            *Quét ~50 cổ phiếu phổ biến trên HOSE*
            """)
    
    st.markdown("---")
    st.caption("⚠️ **Miễn trừ:** Công cụ chỉ dành cho giáo dục. Không phải tư vấn tài chính.")


if __name__ == "__main__":
    main()
