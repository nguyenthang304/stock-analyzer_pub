"""
Công cụ Phân tích Đầu tư Toàn diện
Kết hợp: Phân tích Cơ bản + Kỹ thuật + AI (Claude)
Top 10 Cổ phiếu Đáng Đầu Tư Nhất
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
import json
import requests
warnings.filterwarnings('ignore')

# ======================= STOCK DATA =======================
VN_STOCKS = {
    # === NGÂN HÀNG (25) ===
    "VCB": {"name": "Vietcombank", "sector": "Ngân hàng"},
    "BID": {"name": "BIDV", "sector": "Ngân hàng"},
    "CTG": {"name": "VietinBank", "sector": "Ngân hàng"},
    "TCB": {"name": "Techcombank", "sector": "Ngân hàng"},
    "MBB": {"name": "MB Bank", "sector": "Ngân hàng"},
    "ACB": {"name": "ACB", "sector": "Ngân hàng"},
    "VPB": {"name": "VPBank", "sector": "Ngân hàng"},
    "HDB": {"name": "HDBank", "sector": "Ngân hàng"},
    "TPB": {"name": "TPBank", "sector": "Ngân hàng"},
    "STB": {"name": "Sacombank", "sector": "Ngân hàng"},
    "SHB": {"name": "SHB", "sector": "Ngân hàng"},
    "LPB": {"name": "LPBank", "sector": "Ngân hàng"},
    "VIB": {"name": "VIB", "sector": "Ngân hàng"},
    "OCB": {"name": "OCB", "sector": "Ngân hàng"},
    "MSB": {"name": "MSB", "sector": "Ngân hàng"},
    "NAB": {"name": "NamABank", "sector": "Ngân hàng"},
    "ABB": {"name": "ABBank", "sector": "Ngân hàng"},
    "BAB": {"name": "BacABank", "sector": "Ngân hàng"},
    "EIB": {"name": "Eximbank", "sector": "Ngân hàng"},
    "PGB": {"name": "PGBank", "sector": "Ngân hàng"},
    "SGB": {"name": "SaigonBank", "sector": "Ngân hàng"},
    "VBB": {"name": "VietBank", "sector": "Ngân hàng"},
    "KLB": {"name": "KienLongBank", "sector": "Ngân hàng"},
    "BVB": {"name": "BaoVietBank", "sector": "Ngân hàng"},
    "SSB": {"name": "SeABank", "sector": "Ngân hàng"},
    # === BẤT ĐỘNG SẢN (15) ===
    "VHM": {"name": "Vinhomes", "sector": "Bất động sản"},
    "VIC": {"name": "Vingroup", "sector": "Bất động sản"},
    "VRE": {"name": "Vincom Retail", "sector": "Bất động sản"},
    "NVL": {"name": "Novaland", "sector": "Bất động sản"},
    "PDR": {"name": "Phat Dat", "sector": "Bất động sản"},
    "DIG": {"name": "DIC Corp", "sector": "Bất động sản"},
    "DXG": {"name": "Dat Xanh", "sector": "Bất động sản"},
    "KDH": {"name": "Khang Dien", "sector": "Bất động sản"},
    "NLG": {"name": "Nam Long", "sector": "Bất động sản"},
    "HDG": {"name": "Ha Do", "sector": "Bất động sản"},
    "SCR": {"name": "TTC Land", "sector": "Bất động sản"},
    "CEO": {"name": "CEO Group", "sector": "Bất động sản"},
    "BCM": {"name": "Becamex IDC", "sector": "Bất động sản"},
    "SZC": {"name": "Sonadezi Chau Duc", "sector": "Bất động sản"},
    "IDC": {"name": "IDICO", "sector": "Bất động sản"},
    # === TIÊU DÙNG – BÁN LẺ (10) ===
    "VNM": {"name": "Vinamilk", "sector": "Tiêu dùng"},
    "MSN": {"name": "Masan Group", "sector": "Tiêu dùng"},
    "MWG": {"name": "The Gioi Di Dong", "sector": "Bán lẻ"},
    "PNJ": {"name": "PNJ", "sector": "Bán lẻ"},
    "SAB": {"name": "Sabeco", "sector": "Tiêu dùng"},
    "KDC": {"name": "KIDO Group", "sector": "Tiêu dùng"},
    "ANV": {"name": "Nam Viet", "sector": "Thủy sản"},
    "VHC": {"name": "Vinh Hoan", "sector": "Thủy sản"},
    "FRT": {"name": "FPT Retail", "sector": "Bán lẻ"},
    "DGW": {"name": "Digiworld", "sector": "Công nghệ"},
    # === CÔNG NGHỆ – VIỄN THÔNG (7) ===
    "FPT": {"name": "FPT Corp", "sector": "Công nghệ"},
    "CMG": {"name": "CMC Group", "sector": "Công nghệ"},
    "ELC": {"name": "Elcom", "sector": "Công nghệ"},
    "FOX": {"name": "FPT Telecom", "sector": "Viễn thông"},
    "VGI": {"name": "Viettel Global", "sector": "Viễn thông"},
    "CTR": {"name": "Viettel Construction", "sector": "Viễn thông"},
    "ITD": {"name": "ITD", "sector": "Công nghệ"},
    # === CHỨNG KHOÁN (10) ===
    "SSI": {"name": "SSI", "sector": "Chứng khoán"},
    "VND": {"name": "VNDirect", "sector": "Chứng khoán"},
    "HCM": {"name": "HSC", "sector": "Chứng khoán"},
    "VCI": {"name": "Vietcap", "sector": "Chứng khoán"},
    "BSI": {"name": "BIDV Securities", "sector": "Chứng khoán"},
    "FTS": {"name": "FPT Securities", "sector": "Chứng khoán"},
    "CTS": {"name": "VietinBank Securities", "sector": "Chứng khoán"},
    "AGR": {"name": "Agriseco", "sector": "Chứng khoán"},
    "TVS": {"name": "Thien Viet", "sector": "Chứng khoán"},
    "ORS": {"name": "Tien Phong Sec", "sector": "Chứng khoán"},
    # === DẦU KHÍ – ĐIỆN – NĂNG LƯỢNG (10) ===
    "GAS": {"name": "PV Gas", "sector": "Dầu khí"},
    "PLX": {"name": "Petrolimex", "sector": "Dầu khí"},
    "POW": {"name": "PV Power", "sector": "Điện"},
    "PVD": {"name": "PV Drilling", "sector": "Dầu khí"},
    "PVS": {"name": "PTSC", "sector": "Dầu khí"},
    "BSR": {"name": "Binh Son Refinery", "sector": "Dầu khí"},
    "NT2": {"name": "NT2 Power", "sector": "Điện"},
    "PC1": {"name": "PC1 Group", "sector": "Điện"},
    "GEG": {"name": "Gia Lai Electric", "sector": "Điện"},
    "REE": {"name": "REE Corp", "sector": "Điện"},
    # === NGUYÊN VẬT LIỆU – CÔNG NGHIỆP (12) ===
    "HPG": {"name": "Hoa Phat", "sector": "Thép"},
    "HSG": {"name": "Hoa Sen", "sector": "Thép"},
    "NKG": {"name": "Nam Kim", "sector": "Thép"},
    "DGC": {"name": "Duc Giang", "sector": "Hóa chất"},
    "DCM": {"name": "Dam Ca Mau", "sector": "Phân bón"},
    "DPM": {"name": "Dam Phu My", "sector": "Phân bón"},
    "VGC": {"name": "Viglacera", "sector": "VLXD"},
    "BMP": {"name": "Binh Minh Plastic", "sector": "Nhựa"},
    "BFC": {"name": "Binh Dien Fertilizer", "sector": "Phân bón"},
    "DHC": {"name": "Dong Hai", "sector": "Giấy"},
    "CSV": {"name": "Hoa Chat Co Ban", "sector": "Hóa chất"},
    "GVR": {"name": "Vietnam Rubber Group", "sector": "Cao su"},
    # === LOGISTICS – CẢNG – KCN (7) ===
    "GMD": {"name": "Gemadept", "sector": "Logistics"},
    "VSC": {"name": "Viconship", "sector": "Cảng biển"},
    "HAH": {"name": "Hai An Transport", "sector": "Logistics"},
    "SCS": {"name": "Saigon Cargo", "sector": "Logistics"},
    "KBC": {"name": "Kinh Bac City", "sector": "KCN"},
    "LHG": {"name": "Long Hau", "sector": "KCN"},
    "TIP": {"name": "Tan Tao IP", "sector": "KCN"},
    # === XÂY DỰNG – HẠ TẦNG (6) ===
    "CTD": {"name": "Coteccons", "sector": "Xây dựng"},
    "HBC": {"name": "Hoa Binh", "sector": "Xây dựng"},
    "FCN": {"name": "FECON", "sector": "Xây dựng"},
    "HHV": {"name": "Deo Ca", "sector": "Hạ tầng"},
    "CII": {"name": "CII", "sector": "Hạ tầng"},
    "VCG": {"name": "Vinaconex", "sector": "Xây dựng"},
    # === BẢO HIỂM (5) ===
    "BVH": {"name": "Bao Viet Holdings", "sector": "Bảo hiểm"},
    "PVI": {"name": "PVI Insurance", "sector": "Bảo hiểm"},
    "BMI": {"name": "Bao Minh", "sector": "Bảo hiểm"},
    "BIC": {"name": "BIDV Insurance", "sector": "Bảo hiểm"},
    "MIG": {"name": "Military Insurance", "sector": "Bảo hiểm"},
    # === HÀNG KHÔNG – DU LỊCH (4) ===
    "HVN": {"name": "Vietnam Airlines", "sector": "Hàng không"},
    "VJC": {"name": "VietJet Air", "sector": "Hàng không"},
    "ACV": {"name": "Airports Corporation", "sector": "Hàng không"},
    "SKG": {"name": "Superdong", "sector": "Du lịch"},
    # === Y TẾ – DƯỢC (5) ===
    "DHG": {"name": "Hau Giang Pharma", "sector": "Dược phẩm"},
    "TRA": {"name": "Traphaco", "sector": "Dược phẩm"},
    "IMP": {"name": "Imexpharm", "sector": "Dược phẩm"},
    "DMC": {"name": "Domesco", "sector": "Dược phẩm"},
    "AME": {"name": "AME Pharma", "sector": "Dược phẩm"},
}

SECTOR_PE = {
    "Ngân hàng": 10, "Bất động sản": 12, "Công nghệ": 18, "Tiêu dùng": 16,
    "Thép": 8, "Dầu khí": 10, "Chứng khoán": 12, "Bán lẻ": 14,
    "Điện": 12, "Hàng không": 15, "Logistics": 14, "Phân bón": 8,
    "Hóa chất": 10, "Xây dựng": 10, "VLXD": 10, "Thủy sản": 10,
    "KCN": 15, "Cao su": 10, "Bảo hiểm": 12, "Viễn thông": 14,
    "Cảng biển": 12, "Hạ tầng": 12, "Du lịch": 15, "Dược phẩm": 16,
    "Nhựa": 10, "Giấy": 10,
}

# ======================= PAGE CONFIG =======================
st.set_page_config(
    page_title="Phân tích Đầu tư AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    :root {
        --bg-primary: #0a0a0f; --bg-secondary: #12121a; --bg-card: #1a1a24;
        --green: #00ff88; --red: #ff4757; --yellow: #ffd93d;
        --blue: #6c5ce7; --purple: #a855f7; --cyan: #00d4ff;
        --text-primary: #ffffff; --text-secondary: #a0a0b0;
    }
    .stApp { background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%); }
    .main-title {
        font-family: 'Be Vietnam Pro', sans-serif;
        font-size: clamp(1.8rem, 5vw, 3rem);
        font-weight: 700;
        background: linear-gradient(135deg, var(--green), var(--cyan), var(--purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-title { text-align: center; color: var(--text-secondary); font-size: 1rem; margin-top: -0.5rem; }
    .card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #2a2a3a;
        margin: 0.5rem 0;
    }
    .signal-card {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
    }
    .signal-buy { background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,255,136,0.05)); border: 2px solid var(--green); }
    .signal-sell { background: linear-gradient(135deg, rgba(255,71,87,0.15), rgba(255,71,87,0.05)); border: 2px solid var(--red); }
    .signal-hold { background: linear-gradient(135deg, rgba(255,217,61,0.15), rgba(255,217,61,0.05)); border: 2px solid var(--yellow); }
    .signal-text { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
    .buy-text { color: var(--green); }
    .sell-text { color: var(--red); }
    .hold-text { color: var(--yellow); }
    .metric-box {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a2a3a;
    }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; }
    .metric-label { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.3rem; }
    .ai-box {
        background: linear-gradient(135deg, rgba(168,85,247,0.1), rgba(0,212,255,0.1));
        border: 1px solid var(--purple);
        border-radius: 16px;
        padding: 1.5rem;
    }
    .ai-title { color: var(--purple); font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; }
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; }
    .rank-2 { background: linear-gradient(135deg, #E8E8E8, #B0B0B0); color: #000; }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #8B4513); color: #fff; }
    .rank-other { background: var(--blue); color: #fff; }
    .stock-row {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #2a2a3a;
        transition: all 0.3s;
    }
    .stock-row:hover { border-color: var(--green); transform: translateX(5px); }
    .good { color: var(--green); }
    .bad { color: var(--red); }
    .neutral { color: var(--yellow); }
    .score-bar {
        height: 8px;
        background: #2a2a3a;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .score-fill { height: 100%; border-radius: 4px; }
    [data-testid="stSidebar"] { background: var(--bg-secondary); }
</style>
""", unsafe_allow_html=True)


# ======================= UTILITY FUNCTIONS =======================

def safe_val(val, default=0):
    """Safe value extraction."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        if np.isnan(val) or np.isinf(val):
            return default
    return val

def fmt_price(val, prefix="₫"):
    """Format price."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{prefix}{val:,.0f}"

def fmt_pct(val):
    """Format percentage."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:+.1f}%"

def fmt_num(val):
    """Format large numbers."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1e12:
        return f"{val/1e12:.1f}T"
    if abs(val) >= 1e9:
        return f"{val/1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"{val/1e6:.1f}M"
    return f"{val:,.0f}"

def get_ticker(code):
    """Get Yahoo Finance ticker."""
    return f"{code.upper()}.VN"


# ======================= DATA FETCHING =======================

@st.cache_data(ttl=600)
def get_fundamentals(ticker: str) -> Optional[Dict]:
    """Fetch fundamental data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "price": safe_val(info.get('currentPrice') or info.get('regularMarketPrice')),
            "market_cap": safe_val(info.get('marketCap')),
            "pe": safe_val(info.get('trailingPE')),
            "forward_pe": safe_val(info.get('forwardPE')),
            "pb": safe_val(info.get('priceToBook')),
            "ps": safe_val(info.get('priceToSalesTrailing12Months')),
            "peg": safe_val(info.get('pegRatio')),
            "ev_ebitda": safe_val(info.get('enterpriseToEbitda')),
            "revenue": safe_val(info.get('totalRevenue')),
            "revenue_growth": safe_val(info.get('revenueGrowth'), 0) * 100,
            "earnings": safe_val(info.get('netIncomeToCommon')),
            "earnings_growth": safe_val(info.get('earningsGrowth'), 0) * 100,
            "gross_margin": safe_val(info.get('grossMargins'), 0) * 100,
            "operating_margin": safe_val(info.get('operatingMargins'), 0) * 100,
            "profit_margin": safe_val(info.get('profitMargins'), 0) * 100,
            "roe": safe_val(info.get('returnOnEquity'), 0) * 100,
            "roa": safe_val(info.get('returnOnAssets'), 0) * 100,
            "debt_equity": safe_val(info.get('debtToEquity')),
            "current_ratio": safe_val(info.get('currentRatio')),
            "quick_ratio": safe_val(info.get('quickRatio')),
            "fcf": safe_val(info.get('freeCashflow')),
            "operating_cf": safe_val(info.get('operatingCashflow')),
            "dividend_yield": safe_val(info.get('dividendYield'), 0) * 100,
            "payout_ratio": safe_val(info.get('payoutRatio'), 0) * 100,
            "beta": safe_val(info.get('beta'), 1),
            "52w_high": safe_val(info.get('fiftyTwoWeekHigh')),
            "52w_low": safe_val(info.get('fiftyTwoWeekLow')),
            "eps": safe_val(info.get('trailingEps')),
            "forward_eps": safe_val(info.get('forwardEps')),
            "book_value": safe_val(info.get('bookValue')),
            "shares": safe_val(info.get('sharesOutstanding')),
        }
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def get_price_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch price data."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df.reset_index()
    except:
        return None


# ======================= FUNDAMENTAL ANALYSIS =======================

def calc_intrinsic_value(data: Dict, sector: str) -> Dict:
    """Calculate intrinsic value using multiple methods."""
    price = data['price']
    eps = data['eps']
    book = data['book_value']
    roe = data['roe']
    growth = data['earnings_growth']
    pe_benchmark = SECTOR_PE.get(sector, 12)
    
    values = {}
    
    # 1. Graham Number
    if eps > 0 and book > 0:
        values['graham'] = np.sqrt(22.5 * eps * book)
    
    # 2. P/E Fair Value
    if eps > 0:
        values['pe_fair'] = eps * pe_benchmark
    
    # 3. PEG Fair Value
    if eps > 0 and growth > 0:
        values['peg_fair'] = eps * max(growth, 8)
    
    # 4. Simple DCF
    if eps > 0:
        g = min(max(growth / 100, 0.03), 0.20)
        r = 0.12  # discount rate
        dcf = 0
        fut_eps = eps
        for yr in range(1, 6):
            fut_eps *= (1 + g)
            dcf += fut_eps / ((1 + r) ** yr)
        # Terminal
        terminal = fut_eps * 1.03 / (r - 0.03)
        dcf += terminal / ((1 + r) ** 5)
        values['dcf'] = dcf
    
    # 5. Book Value Based
    if book > 0 and roe > 0:
        fair_pb = max(1, roe / 10)
        values['bv_fair'] = book * fair_pb
    
    # Weighted Average
    weights = {'dcf': 0.30, 'graham': 0.25, 'pe_fair': 0.20, 'peg_fair': 0.15, 'bv_fair': 0.10}
    total_weight = 0
    weighted_sum = 0
    for k, v in values.items():
        if v and v > 0:
            weighted_sum += v * weights.get(k, 0.1)
            total_weight += weights.get(k, 0.1)
    
    avg_value = weighted_sum / total_weight if total_weight > 0 else 0
    upside = ((avg_value - price) / price * 100) if avg_value > 0 and price > 0 else 0
    
    return {
        "values": values,
        "avg_value": avg_value,
        "upside": upside,
        "pe_benchmark": pe_benchmark
    }


def score_fundamentals(data: Dict, sector: str) -> Dict:
    """Score fundamental analysis (0-100)."""
    score = 0
    details = []
    
    # 1. Profitability (25 pts)
    roe = data['roe']
    if roe >= 20:
        score += 12
        details.append(("ROE", f"{roe:.1f}%", "Xuất sắc", "good"))
    elif roe >= 15:
        score += 9
        details.append(("ROE", f"{roe:.1f}%", "Tốt", "good"))
    elif roe >= 10:
        score += 5
        details.append(("ROE", f"{roe:.1f}%", "TB", "neutral"))
    else:
        details.append(("ROE", f"{roe:.1f}%", "Yếu", "bad"))
    
    margin = data['profit_margin']
    if margin >= 15:
        score += 8
        details.append(("Biên LN", f"{margin:.1f}%", "Cao", "good"))
    elif margin >= 8:
        score += 5
        details.append(("Biên LN", f"{margin:.1f}%", "Khá", "good"))
    elif margin >= 3:
        score += 2
        details.append(("Biên LN", f"{margin:.1f}%", "Thấp", "neutral"))
    else:
        details.append(("Biên LN", f"{margin:.1f}%", "Rất thấp", "bad"))
    
    roa = data['roa']
    if roa >= 8:
        score += 5
    elif roa >= 4:
        score += 3
    
    # 2. Growth (25 pts)
    eg = data['earnings_growth']
    if eg >= 20:
        score += 15
        details.append(("Tăng trưởng LN", f"{eg:.1f}%", "Cao", "good"))
    elif eg >= 10:
        score += 10
        details.append(("Tăng trưởng LN", f"{eg:.1f}%", "Khá", "good"))
    elif eg >= 0:
        score += 5
        details.append(("Tăng trưởng LN", f"{eg:.1f}%", "Ổn định", "neutral"))
    else:
        details.append(("Tăng trưởng LN", f"{eg:.1f}%", "Âm", "bad"))
    
    rg = data['revenue_growth']
    if rg >= 15:
        score += 10
    elif rg >= 8:
        score += 7
    elif rg >= 0:
        score += 3
    
    # 3. Financial Health (25 pts)
    de = data['debt_equity']
    if de == 0 or de < 50:
        score += 12
        details.append(("Nợ/Vốn", f"{de:.0f}%", "An toàn", "good"))
    elif de < 100:
        score += 8
        details.append(("Nợ/Vốn", f"{de:.0f}%", "Chấp nhận", "good"))
    elif de < 150:
        score += 4
        details.append(("Nợ/Vốn", f"{de:.0f}%", "Cao", "neutral"))
    else:
        details.append(("Nợ/Vốn", f"{de:.0f}%", "Rủi ro", "bad"))
    
    cr = data['current_ratio']
    if cr >= 2:
        score += 8
    elif cr >= 1.5:
        score += 6
    elif cr >= 1:
        score += 3
    
    fcf = data['fcf']
    if fcf > 0:
        score += 5
        details.append(("Dòng tiền", "Dương", "Tốt", "good"))
    else:
        details.append(("Dòng tiền", "Âm", "Cẩn thận", "bad"))
    
    # 4. Valuation (25 pts)
    pe_benchmark = SECTOR_PE.get(sector, 12)
    pe = data['pe']
    if 0 < pe < pe_benchmark * 0.7:
        score += 12
        details.append(("P/E", f"{pe:.1f}", "Rẻ", "good"))
    elif 0 < pe < pe_benchmark:
        score += 8
        details.append(("P/E", f"{pe:.1f}", "Hợp lý", "good"))
    elif 0 < pe < pe_benchmark * 1.3:
        score += 4
        details.append(("P/E", f"{pe:.1f}", "Khá cao", "neutral"))
    else:
        details.append(("P/E", f"{pe:.1f}" if pe > 0 else "N/A", "Đắt/N/A", "bad"))
    
    pb = data['pb']
    if 0 < pb < 1.5:
        score += 8
    elif 0 < pb < 2.5:
        score += 5
    elif 0 < pb < 4:
        score += 2
    
    peg = data['peg']
    if 0 < peg < 1:
        score += 5
    elif 0 < peg < 1.5:
        score += 3
    
    # Rating
    if score >= 80:
        rating, rating_class = "XUẤT SẮC", "good"
    elif score >= 65:
        rating, rating_class = "TỐT", "good"
    elif score >= 50:
        rating, rating_class = "TRUNG BÌNH", "neutral"
    elif score >= 35:
        rating, rating_class = "YẾU", "bad"
    else:
        rating, rating_class = "RỦI RO", "bad"
    
    return {
        "score": min(score, 100),
        "rating": rating,
        "rating_class": rating_class,
        "details": details
    }


# ======================= TECHNICAL ANALYSIS =======================

def analyze_technical(df: pd.DataFrame) -> Optional[Dict]:
    """Comprehensive technical analysis."""
    if df is None or len(df) < 20:
        return None
    
    try:
        # EMAs
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        
        latest = df.iloc[-1]
        price = safe_val(latest['Close'])
        ema20 = safe_val(latest['EMA20'])
        ema50 = safe_val(latest['EMA50'])
        ema200 = safe_val(latest['EMA200'])
        
        # Trend Score
        trend_score = 0
        if price > ema20 > ema50:
            trend_score = 2 if ema200 > 0 and ema50 > ema200 else 1.5
            trend = "TĂNG"
        elif price < ema20 < ema50:
            trend_score = -2 if ema200 > 0 and ema50 < ema200 else -1.5
            trend = "GIẢM"
        else:
            trend = "ĐI NGANG"
        
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        rsi = safe_val(df['RSI'].iloc[-1], 50)
        
        rsi_score = 0
        if 30 < rsi < 45:
            rsi_score = 1.5
        elif 45 <= rsi < 55:
            rsi_score = 1
        elif 55 <= rsi < 70:
            rsi_score = 0.5
        elif rsi >= 70:
            rsi_score = -1
        elif rsi <= 30:
            rsi_score = 0.5
        
        # Volume
        df['Vol_SMA'] = df['Volume'].rolling(20).mean()
        vol_ratio = df['Volume'].iloc[-1] / df['Vol_SMA'].iloc[-1] if safe_val(df['Vol_SMA'].iloc[-1]) > 0 else 1
        
        vol_score = 0
        if vol_ratio > 1.5 and trend_score > 0:
            vol_score = 1.5
        elif vol_ratio > 1.2 and trend_score > 0:
            vol_score = 1
        elif vol_ratio > 1.5 and trend_score < 0:
            vol_score = -1
        
        # MACD
        try:
            macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            macd_val = safe_val(macd_df['MACD_12_26_9'].iloc[-1])
            macd_sig = safe_val(macd_df['MACDs_12_26_9'].iloc[-1])
            macd_hist = safe_val(macd_df['MACDh_12_26_9'].iloc[-1])
            macd_hist_prev = safe_val(macd_df['MACDh_12_26_9'].iloc[-2])
            
            macd_score = 0
            if macd_val > macd_sig and macd_hist > macd_hist_prev:
                macd_score = 1.5
            elif macd_val > macd_sig:
                macd_score = 1
            elif macd_val < macd_sig and macd_hist < macd_hist_prev:
                macd_score = -1.5
            elif macd_val < macd_sig:
                macd_score = -1
        except:
            macd_score = 0
            macd_hist = 0
        
        # Price momentum
        price_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) > 6 else 0
        price_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100 if len(df) > 21 else 0
        
        # Support/Resistance
        high_52w = df['High'].max()
        low_52w = df['Low'].min()
        price_pos = (price - low_52w) / (high_52w - low_52w) * 100 if (high_52w - low_52w) > 0 else 50
        
        # Total Score
        total_score = trend_score + rsi_score + vol_score + macd_score
        
        # Signal
        if total_score >= 3:
            signal, signal_class = "MUA MẠNH", "buy"
        elif total_score >= 1.5:
            signal, signal_class = "MUA", "buy"
        elif total_score <= -3:
            signal, signal_class = "BÁN MẠNH", "sell"
        elif total_score <= -1.5:
            signal, signal_class = "BÁN", "sell"
        else:
            signal, signal_class = "TRUNG LẬP", "hold"
        
        # Normalize to 0-100
        tech_score = ((total_score + 6) / 12) * 100
        tech_score = max(0, min(100, tech_score))
        
        return {
            "price": price,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "trend": trend,
            "trend_score": trend_score,
            "rsi": rsi,
            "rsi_score": rsi_score,
            "vol_ratio": vol_ratio,
            "vol_score": vol_score,
            "macd_score": macd_score,
            "macd_hist": macd_hist if 'macd_hist' in dir() else 0,
            "total_score": total_score,
            "tech_score": tech_score,
            "signal": signal,
            "signal_class": signal_class,
            "price_5d": price_5d,
            "price_20d": price_20d,
            "price_pos": price_pos,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "df": df
        }
    except Exception as e:
        return None


# ======================= AI ANALYSIS =======================

def get_ai_analysis(code: str, name: str, sector: str, fund_data: Dict, fund_score: Dict, 
                    intrinsic: Dict, tech_data: Dict) -> str:
    """Get AI analysis using Claude API."""
    
    # Prepare data summary
    data_summary = f"""
## Dữ liệu Cổ phiếu {code} - {name}
**Ngành:** {sector}

### Phân tích Cơ bản:
- Giá hiện tại: {fmt_price(fund_data['price'])}
- P/E: {fund_data['pe']:.1f} (Ngành TB: {intrinsic['pe_benchmark']})
- P/B: {fund_data['pb']:.1f}
- ROE: {fund_data['roe']:.1f}%
- ROA: {fund_data['roa']:.1f}%
- Biên LN ròng: {fund_data['profit_margin']:.1f}%
- Tăng trưởng LN: {fund_data['earnings_growth']:.1f}%
- Tăng trưởng DT: {fund_data['revenue_growth']:.1f}%
- Nợ/Vốn CSH: {fund_data['debt_equity']:.0f}%
- Current Ratio: {fund_data['current_ratio']:.2f}
- Dòng tiền tự do: {fmt_num(fund_data['fcf'])}
- EPS: {fund_data['eps']:.0f}
- Book Value: {fund_data['book_value']:.0f}

**Điểm Cơ bản: {fund_score['score']}/100 - {fund_score['rating']}**

### Định giá:
- Giá trị nội tại ước tính: {fmt_price(intrinsic['avg_value'])}
- Tiềm năng tăng giá: {intrinsic['upside']:+.1f}%

### Phân tích Kỹ thuật:
- Xu hướng: {tech_data['trend'] if tech_data else 'N/A'}
- RSI(14): {tech_data['rsi']:.1f if tech_data else 'N/A'}
- Khối lượng/TB: {tech_data['vol_ratio']:.2f if tech_data else 'N/A'}x
- Thay đổi 5 ngày: {tech_data['price_5d']:.1f if tech_data else 0}%
- Thay đổi 20 ngày: {tech_data['price_20d']:.1f if tech_data else 0}%
- Vị trí giá (so với 52 tuần): {tech_data['price_pos']:.0f if tech_data else 50}%

**Tín hiệu Kỹ thuật: {tech_data['signal'] if tech_data else 'N/A'}**
**Điểm Kỹ thuật: {tech_data['tech_score']:.0f if tech_data else 50}/100**
"""

    prompt = f"""Bạn là chuyên gia phân tích đầu tư chứng khoán Việt Nam với hơn 20 năm kinh nghiệm. 
Hãy phân tích cổ phiếu dựa trên dữ liệu sau và đưa ra nhận định đầu tư:

{data_summary}

Hãy viết một bản phân tích ngắn gọn (300-400 từ) bao gồm:

1. **TỔNG QUAN**: Đánh giá tổng thể về doanh nghiệp và vị thế trong ngành

2. **ĐIỂM MẠNH**: 2-3 điểm mạnh nổi bật (dựa trên dữ liệu)

3. **ĐIỂM YẾU/RỦI RO**: 2-3 rủi ro cần lưu ý

4. **NHẬN ĐỊNH KỸ THUẬT**: Xu hướng ngắn hạn và điểm vào/ra hợp lý

5. **KHUYẾN NGHỊ ĐẦU TƯ**: 
   - MUA MẠNH / MUA / GIỮ / BÁN / BÁN MẠNH
   - Mục tiêu giá ngắn hạn (1-3 tháng)
   - Mức cắt lỗ đề xuất

Trả lời bằng tiếng Việt, súc tích và thực tế. Không cần lặp lại số liệu đã có."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['content'][0]['text']
        else:
            return f"⚠️ Không thể kết nối AI. Mã lỗi: {response.status_code}"
    except Exception as e:
        return f"⚠️ Lỗi kết nối AI: {str(e)}"


# ======================= COMBINED SCORING =======================

def calc_combined_score(fund_score: int, tech_score: float, upside: float) -> Dict:
    """Calculate combined investment score."""
    
    # Valuation score from upside
    if upside >= 50:
        val_score = 100
    elif upside >= 30:
        val_score = 85
    elif upside >= 15:
        val_score = 70
    elif upside >= 0:
        val_score = 55
    elif upside >= -15:
        val_score = 40
    else:
        val_score = 20
    
    # Weighted: 45% Fundamental, 35% Technical, 20% Valuation
    combined = fund_score * 0.45 + tech_score * 0.35 + val_score * 0.20
    
    if combined >= 75:
        rating, rating_class = "MUA MẠNH", "buy"
    elif combined >= 60:
        rating, rating_class = "MUA", "buy"
    elif combined >= 45:
        rating, rating_class = "THEO DÕI", "hold"
    elif combined >= 35:
        rating, rating_class = "CẨN TRỌNG", "hold"
    else:
        rating, rating_class = "TRÁNH", "sell"
    
    return {
        "combined_score": round(combined, 1),
        "fund_score": fund_score,
        "tech_score": round(tech_score, 1),
        "val_score": val_score,
        "rating": rating,
        "rating_class": rating_class
    }


# ======================= SCANNING =======================

def scan_all_stocks(progress_cb=None) -> List[Dict]:
    """Scan all stocks and rank them."""
    results = []
    total = len(VN_STOCKS)
    
    for i, (code, info) in enumerate(VN_STOCKS.items()):
        if progress_cb:
            progress_cb((i + 1) / total, f"Đang phân tích {code}... ({i+1}/{total})")
        
        ticker = get_ticker(code)
        sector = info['sector']
        
        # Fundamentals
        fund_data = get_fundamentals(ticker)
        if not fund_data or fund_data['price'] == 0:
            continue
        
        fund_score = score_fundamentals(fund_data, sector)
        intrinsic = calc_intrinsic_value(fund_data, sector)
        
        # Technical
        df = get_price_data(ticker, "6mo")
        tech_data = analyze_technical(df)
        tech_score = tech_data['tech_score'] if tech_data else 50
        
        # Combined
        combined = calc_combined_score(fund_score['score'], tech_score, intrinsic['upside'])
        
        results.append({
            "code": code,
            "name": info['name'],
            "sector": sector,
            "price": fund_data['price'],
            "pe": fund_data['pe'],
            "pb": fund_data['pb'],
            "roe": fund_data['roe'],
            "profit_margin": fund_data['profit_margin'],
            "earnings_growth": fund_data['earnings_growth'],
            "debt_equity": fund_data['debt_equity'],
            "intrinsic": intrinsic['avg_value'],
            "upside": intrinsic['upside'],
            "fund_score": fund_score['score'],
            "fund_rating": fund_score['rating'],
            "tech_score": tech_score,
            "tech_signal": tech_data['signal'] if tech_data else "N/A",
            "rsi": tech_data['rsi'] if tech_data else 50,
            "combined_score": combined['combined_score'],
            "rating": combined['rating'],
            "rating_class": combined['rating_class']
        })
        
        time.sleep(0.15)
    
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results


# ======================= CHART =======================

def create_chart(df: pd.DataFrame, code: str) -> go.Figure:
    """Create price chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.7, 0.3], subplot_titles=(f'{code}', 'RSI'))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Giá', increasing_line_color='#00ff88', decreasing_line_color='#ff4757'
    ), row=1, col=1)
    
    # EMAs
    if 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name='EMA20',
                                  line=dict(color='#ffd93d', width=1)), row=1, col=1)
    if 'EMA50' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], name='EMA50',
                                  line=dict(color='#6c5ce7', width=1)), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI',
                                  line=dict(color='#a855f7', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.5)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,36,0.8)',
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )
    return fig


# ======================= MAIN APP =======================

def main():
    st.markdown('<h1 class="main-title">🤖 Phân tích Đầu tư AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Kết hợp Phân tích Cơ bản + Kỹ thuật + Trí tuệ Nhân tạo</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔍 Phân tích Chi tiết", "🏆 Top 10 Đáng Đầu tư", "📖 Hướng dẫn"])
    
    # ==================== TAB 1 ====================
    with tab1:
        with st.sidebar:
            st.markdown("### ⚙️ Cài đặt")
            
            # Stock selection
            stock_options = [f"{k} - {v['name']}" for k, v in VN_STOCKS.items()]
            selected = st.selectbox("Chọn cổ phiếu", stock_options, index=0)
            code = selected.split(" - ")[0]
            
            period_map = {"3 Tháng": "3mo", "6 Tháng": "6mo", "1 Năm": "1y", "2 Năm": "2y"}
            period = period_map[st.selectbox("Thời gian", list(period_map.keys()), index=2)]
            
            st.markdown("---")
            
            # AI toggle
            use_ai = st.checkbox("🤖 Sử dụng AI phân tích", value=True)
            
            st.markdown("---")
            analyze_btn = st.button("🔍 Phân tích", type="primary", use_container_width=True)
        
        if analyze_btn:
            stock_info = VN_STOCKS[code]
            sector = stock_info['sector']
            ticker = get_ticker(code)
            
            with st.spinner(f"Đang phân tích {code}..."):
                # Get data
                fund_data = get_fundamentals(ticker)
                df = get_price_data(ticker, period)
                
                if not fund_data or fund_data['price'] == 0:
                    st.error(f"❌ Không lấy được dữ liệu cho {code}")
                    return
                
                # Analysis
                fund_score = score_fundamentals(fund_data, sector)
                intrinsic = calc_intrinsic_value(fund_data, sector)
                tech_data = analyze_technical(df)
                tech_score = tech_data['tech_score'] if tech_data else 50
                combined = calc_combined_score(fund_score['score'], tech_score, intrinsic['upside'])
            
            # Header
            st.markdown(f"## {code} - {stock_info['name']}")
            st.caption(f"Ngành: {sector}")
            
            # Main Signal
            signal_class = f"signal-{combined['rating_class']}"
            text_class = f"{combined['rating_class']}-text"
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div style="font-size: 0.9rem; color: #a0a0b0;">KHUYẾN NGHỊ ĐẦU TƯ</div>
                <div class="signal-text {text_class}">{combined['rating']}</div>
                <div style="font-size: 1.3rem; font-weight: 600;">Điểm: {combined['combined_score']}/100</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-box"><div class="metric-value">{fmt_price(fund_data["price"])}</div><div class="metric-label">Giá hiện tại</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-box"><div class="metric-value">{fund_score["score"]}</div><div class="metric-label">Điểm Cơ bản</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-box"><div class="metric-value">{tech_score:.0f}</div><div class="metric-label">Điểm Kỹ thuật</div></div>', unsafe_allow_html=True)
            
            upside_color = "good" if intrinsic['upside'] > 0 else "bad"
            c4.markdown(f'<div class="metric-box"><div class="metric-value {upside_color}">{intrinsic["upside"]:+.1f}%</div><div class="metric-label">Tiềm năng</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Two columns
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 📊 Phân tích Cơ bản")
                
                # Intrinsic Value
                st.markdown(f"""
                <div class="card">
                    <div style="text-align: center;">
                        <div style="color: #a0a0b0; font-size: 0.9rem;">GIÁ TRỊ NỘI TẠI</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #a855f7;">{fmt_price(intrinsic['avg_value'])}</div>
                        <div class="{upside_color}">Chênh lệch: {intrinsic['upside']:+.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📈 Chỉ số Tài chính", expanded=True):
                    st.markdown("**Định giá:**")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("P/E", f"{fund_data['pe']:.1f}")
                    cc2.metric("P/B", f"{fund_data['pb']:.1f}")
                    cc3.metric("PEG", f"{fund_data['peg']:.2f}" if fund_data['peg'] > 0 else "N/A")
                    
                    st.markdown("**Lợi nhuận:**")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("ROE", f"{fund_data['roe']:.1f}%")
                    cc2.metric("ROA", f"{fund_data['roa']:.1f}%")
                    cc3.metric("Biên LN", f"{fund_data['profit_margin']:.1f}%")
                    
                    st.markdown("**Tăng trưởng:**")
                    cc1, cc2 = st.columns(2)
                    cc1.metric("Tăng trưởng LN", fmt_pct(fund_data['earnings_growth']))
                    cc2.metric("Tăng trưởng DT", fmt_pct(fund_data['revenue_growth']))
                    
                    st.markdown("**Sức khỏe TC:**")
                    cc1, cc2 = st.columns(2)
                    cc1.metric("Nợ/Vốn CSH", f"{fund_data['debt_equity']:.0f}%")
                    cc2.metric("Current Ratio", f"{fund_data['current_ratio']:.2f}")
            
            with col_right:
                st.markdown("### 📈 Phân tích Kỹ thuật")
                
                if tech_data:
                    tech_class = "good" if tech_data['signal_class'] == 'buy' else ("bad" if tech_data['signal_class'] == 'sell' else "neutral")
                    st.markdown(f"""
                    <div class="card">
                        <div style="text-align: center;">
                            <div style="color: #a0a0b0; font-size: 0.9rem;">TÍN HIỆU KỸ THUẬT</div>
                            <div style="font-size: 1.8rem; font-weight: 700;" class="{tech_class}">{tech_data['signal']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart
                    st.plotly_chart(create_chart(tech_data['df'], code), use_container_width=True)
                    
                    with st.expander("📊 Chỉ báo", expanded=True):
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("Xu hướng", tech_data['trend'])
                        cc2.metric("RSI", f"{tech_data['rsi']:.1f}")
                        cc3.metric("KL/TB", f"{tech_data['vol_ratio']:.2f}x")
                        
                        cc1, cc2 = st.columns(2)
                        cc1.metric("% 5 ngày", fmt_pct(tech_data['price_5d']))
                        cc2.metric("% 20 ngày", fmt_pct(tech_data['price_20d']))
            
            # AI Analysis
            if use_ai:
                st.markdown("---")
                st.markdown("### 🤖 Phân tích AI")
                
                with st.spinner("AI đang phân tích..."):
                    ai_result = get_ai_analysis(
                        code, stock_info['name'], sector,
                        fund_data, fund_score, intrinsic, tech_data
                    )
                
                st.markdown(f"""
                <div class="ai-box">
                    <div class="ai-title">🤖 Nhận định từ AI</div>
                    <div style="color: #e0e0e0; line-height: 1.7;">{ai_result}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== TAB 2 ====================
    with tab2:
        st.markdown("### 🏆 Top 10 Cổ phiếu Đáng Đầu tư Nhất")
        st.markdown("""
        *Xếp hạng dựa trên:* **45%** Cơ bản + **35%** Kỹ thuật + **20%** Định giá
        """)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            scan_btn = st.button("🔄 Quét thị trường", type="primary", use_container_width=True)
        
        if scan_btn:
            progress = st.progress(0)
            status = st.empty()
            
            def update(pct, text):
                progress.progress(pct)
                status.text(text)
            
            results = scan_all_stocks(update)
            
            progress.empty()
            status.empty()
            
            if results:
                st.session_state['results'] = results
                st.session_state['scan_time'] = datetime.now().strftime("%H:%M %d/%m/%Y")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            st.caption(f"🕐 Cập nhật: {st.session_state.get('scan_time', '')}")
            
            # Top 10
            for i, stock in enumerate(results[:10]):
                rank = i + 1
                rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
                rating_color = "#00ff88" if stock['rating_class'] == 'buy' else ("#ff4757" if stock['rating_class'] == 'sell' else "#ffd93d")
                upside_color = "#00ff88" if stock['upside'] > 0 else "#ff4757"
                
                st.markdown(f"""
                <div class="stock-row">
                    <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                        <div class="rank-badge {rank_class}">{rank}</div>
                        <div style="min-width: 140px;">
                            <div style="font-weight: 700; font-size: 1.1rem;">{stock['code']}</div>
                            <div style="color: #a0a0b0; font-size: 0.75rem;">{stock['name']}</div>
                        </div>
                        <div style="min-width: 90px;">
                            <div style="font-weight: 600;">{fmt_price(stock['price'])}</div>
                            <div style="color: {upside_color}; font-size: 0.85rem;">{stock['upside']:+.1f}%</div>
                        </div>
                        <div style="min-width: 70px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: {rating_color};">{stock['combined_score']:.0f}</div>
                            <div style="font-size: 0.7rem; color: #a0a0b0;">ĐIỂM</div>
                        </div>
                        <div style="min-width: 100px;">
                            <div>CB: <b>{stock['fund_score']}</b> | KT: <b>{stock['tech_score']:.0f}</b></div>
                            <div style="font-size: 0.8rem;">P/E: {stock['pe']:.1f} | ROE: {stock['roe']:.1f}%</div>
                        </div>
                        <div style="min-width: 120px;">
                            <div style="color: {rating_color}; font-weight: 600;">{stock['rating']}</div>
                            <div style="font-size: 0.8rem; color: #a0a0b0;">Giá trị: {fmt_price(stock['intrinsic'])}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Full table
            with st.expander("📊 Bảng đầy đủ"):
                df = pd.DataFrame(results)
                df_show = df[['code', 'name', 'sector', 'price', 'intrinsic', 'upside', 
                              'combined_score', 'fund_score', 'tech_score', 'pe', 'roe', 'rating']].copy()
                df_show.columns = ['Mã', 'Tên', 'Ngành', 'Giá', 'Giá trị NĐ', 'Upside%',
                                   'Điểm TH', 'Điểm CB', 'Điểm KT', 'P/E', 'ROE%', 'Khuyến nghị']
                df_show['Giá'] = df_show['Giá'].apply(lambda x: f"₫{x:,.0f}")
                df_show['Giá trị NĐ'] = df_show['Giá trị NĐ'].apply(lambda x: f"₫{x:,.0f}" if x > 0 else "N/A")
                st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Nhấn **Quét thị trường** để tìm Top 10")
    
    # ==================== TAB 3 ====================
    with tab3:
        st.markdown("""
        ## 📖 Hướng dẫn Sử dụng
        
        ### 🔬 Phương pháp Phân tích
        
        #### I. Phân tích Cơ bản (45%)
        Đánh giá doanh nghiệp có đáng đầu tư không:
        
        | Chỉ số | Tốt | Trung bình | Yếu |
        |--------|-----|------------|-----|
        | ROE | >15% | 10-15% | <10% |
        | Biên LN ròng | >15% | 8-15% | <8% |
        | Tăng trưởng LN | >15% | 5-15% | <5% |
        | Nợ/Vốn | <50% | 50-100% | >100% |
        | P/E | <TB ngành | =TB ngành | >TB ngành |
        
        #### II. Phân tích Kỹ thuật (35%)
        Xác định thời điểm mua/bán:
        
        - **Xu hướng EMA**: Giá > EMA20 > EMA50 > EMA200 = Tăng mạnh
        - **RSI**: 30-45 (Quá bán, cơ hội) | 70+ (Quá mua, cẩn thận)
        - **MACD**: Histogram tăng + MACD > Signal = Bullish
        - **Khối lượng**: >1.5x TB + xu hướng tăng = Xác nhận
        
        #### III. Định giá (20%)
        Tính giá trị nội tại bằng 4 phương pháp:
        
        1. **Graham Number**: √(22.5 × EPS × Book Value)
        2. **P/E Fair**: EPS × P/E trung bình ngành
        3. **DCF**: Chiết khấu dòng tiền 5 năm
        4. **PEG**: EPS × Tỷ lệ tăng trưởng
        
        #### IV. Phân tích AI
        Claude AI đánh giá tổng hợp và đưa ra:
        - Điểm mạnh/yếu của doanh nghiệp
        - Rủi ro cần lưu ý
        - Khuyến nghị đầu tư cụ thể
        - Mục tiêu giá và điểm cắt lỗ
        
        ---
        
        ### 📊 Cách đọc Khuyến nghị
        
        | Điểm | Khuyến nghị | Hành động |
        |------|-------------|-----------|
        | 75+ | MUA MẠNH | Tích cực mua vào |
        | 60-75 | MUA | Cân nhắc mua |
        | 45-60 | THEO DÕI | Chờ tín hiệu rõ hơn |
        | 35-45 | CẨN TRỌNG | Hạn chế mua mới |
        | <35 | TRÁNH | Không nên mua |
        """)
    
    st.markdown("---")
    st.caption("⚠️ **Miễn trừ:** Công cụ chỉ mang tính tham khảo, không phải tư vấn đầu tư. Hãy tự nghiên cứu trước khi quyết định.")


if __name__ == "__main__":
    main()
