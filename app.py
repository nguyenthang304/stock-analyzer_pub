"""
Công cụ Phân tích Đầu tư Toàn diện
Kết hợp Phân tích Cơ bản + Phân tích Kỹ thuật
Tìm Top 10 Cổ phiếu Đáng Đầu Tư Nhất
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

# Danh sách cổ phiếu Việt Nam với thông tin ngành
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

# Benchmark P/E theo ngành (trung bình thị trường VN)
SECTOR_PE_BENCHMARK = {
    "Tài chính": 12,
    "Bất động sản": 15,
    "Công nghệ": 20,
    "Tiêu dùng": 18,
    "Vật liệu": 10,
    "Năng lượng": 12,
    "Công nghiệp": 14,
    "Tiện ích": 12,
    "Bán lẻ": 16,
}

st.set_page_config(
    page_title="Phân tích Đầu tư Toàn diện",
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
        --accent-blue: #6c5ce7; --accent-purple: #a855f7; --text-primary: #ffffff;
        --text-secondary: #a0a0b0; --border-color: #2a2a3a;
    }
    .stApp { background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%); }
    .main-header {
        font-family: 'Be Vietnam Pro', sans-serif; font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: 700; background: linear-gradient(90deg, var(--accent-green), var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .sub-header { text-align: center; color: var(--text-secondary); margin-top: -0.5rem; font-size: 0.9rem; }
    .signal-card {
        font-family: 'Be Vietnam Pro', sans-serif; padding: 1.5rem; border-radius: 16px;
        text-align: center; margin: 0.5rem 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid var(--border-color);
    }
    .signal-bullish { background: linear-gradient(135deg, rgba(0,255,136,0.15) 0%, rgba(0,255,136,0.05) 100%); border-color: var(--accent-green); }
    .signal-bearish { background: linear-gradient(135deg, rgba(255,71,87,0.15) 0%, rgba(255,71,87,0.05) 100%); border-color: var(--accent-red); }
    .signal-neutral { background: linear-gradient(135deg, rgba(255,217,61,0.15) 0%, rgba(255,217,61,0.05) 100%); border-color: var(--accent-yellow); }
    .signal-text { font-size: clamp(1.5rem, 6vw, 3rem); font-weight: 700; margin: 0.5rem 0; }
    .signal-bullish .signal-text { color: var(--accent-green); }
    .signal-bearish .signal-text { color: var(--accent-red); }
    .signal-neutral .signal-text { color: var(--accent-yellow); }
    .metric-box { background: var(--bg-card); border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid var(--border-color); }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: var(--text-primary); }
    .metric-label { font-family: 'Be Vietnam Pro', sans-serif; font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.3rem; }
    .score-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(108,92,231,0.1) 100%);
        border-radius: 12px; padding: 1rem; border: 1px solid var(--accent-blue);
        margin: 0.5rem 0;
    }
    .fundamental-good { color: var(--accent-green); }
    .fundamental-bad { color: var(--accent-red); }
    .fundamental-neutral { color: var(--accent-yellow); }
    .rank-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white; font-weight: 700; font-family: 'JetBrains Mono', monospace;
    }
    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); }
    .rank-2 { background: linear-gradient(135deg, #E8E8E8, #B8B8B8); }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #8B4513); }
    .stock-row {
        background: var(--bg-card); border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
        border: 1px solid var(--border-color); transition: all 0.3s;
    }
    .stock-row:hover { border-color: var(--accent-green); transform: translateX(5px); }
    .valuation-box {
        background: linear-gradient(135deg, rgba(168,85,247,0.1) 0%, rgba(108,92,231,0.1) 100%);
        border-radius: 12px; padding: 1rem; border: 1px solid var(--accent-purple);
    }
    .intrinsic-value { font-size: 2rem; font-weight: 700; color: var(--accent-purple); }
    .upside { color: var(--accent-green); font-weight: 600; }
    .downside { color: var(--accent-red); font-weight: 600; }
    [data-testid="stSidebar"] { background: var(--bg-secondary); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card); border-radius: 8px; padding: 0.5rem 1rem;
        border: 1px solid var(--border-color);
    }
    .stTabs [aria-selected="true"] { background: var(--accent-blue); border-color: var(--accent-blue); }
</style>
""", unsafe_allow_html=True)


# ======================= UTILITY FUNCTIONS =======================

def safe_value(val, default=0):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return default
    return val

def format_price(val, currency="₫"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{currency}{val:,.0f}"

def format_number(val, suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1e12:
        return f"{val/1e12:.1f}T{suffix}"
    if abs(val) >= 1e9:
        return f"{val/1e9:.1f}B{suffix}"
    if abs(val) >= 1e6:
        return f"{val/1e6:.1f}M{suffix}"
    return f"{val:,.0f}{suffix}"

def format_percent(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.1f}%"

def get_vn_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if '.VN' in ticker:
        return ticker
    return f"{ticker}.VN"


# ======================= FUNDAMENTAL ANALYSIS =======================

@st.cache_data(ttl=600)
def get_fundamental_data(ticker: str) -> Optional[Dict]:
    """Lấy dữ liệu cơ bản từ Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Financials
        try:
            financials = stock.financials
            balance = stock.balance_sheet
            cashflow = stock.cashflow
        except:
            financials = pd.DataFrame()
            balance = pd.DataFrame()
            cashflow = pd.DataFrame()
        
        # Extract key metrics
        data = {
            # Giá & Định giá
            "price": safe_value(info.get('currentPrice') or info.get('regularMarketPrice'), 0),
            "market_cap": safe_value(info.get('marketCap'), 0),
            "pe_ratio": safe_value(info.get('trailingPE'), 0),
            "forward_pe": safe_value(info.get('forwardPE'), 0),
            "pb_ratio": safe_value(info.get('priceToBook'), 0),
            "ps_ratio": safe_value(info.get('priceToSalesTrailing12Months'), 0),
            "peg_ratio": safe_value(info.get('pegRatio'), 0),
            "ev_ebitda": safe_value(info.get('enterpriseToEbitda'), 0),
            
            # Lợi nhuận & Tăng trưởng
            "revenue": safe_value(info.get('totalRevenue'), 0),
            "revenue_growth": safe_value(info.get('revenueGrowth'), 0) * 100,
            "earnings": safe_value(info.get('netIncomeToCommon'), 0),
            "earnings_growth": safe_value(info.get('earningsGrowth'), 0) * 100,
            "gross_margin": safe_value(info.get('grossMargins'), 0) * 100,
            "operating_margin": safe_value(info.get('operatingMargins'), 0) * 100,
            "profit_margin": safe_value(info.get('profitMargins'), 0) * 100,
            "ebitda_margin": safe_value(info.get('ebitdaMargins'), 0) * 100,
            
            # Hiệu quả sử dụng vốn
            "roe": safe_value(info.get('returnOnEquity'), 0) * 100,
            "roa": safe_value(info.get('returnOnAssets'), 0) * 100,
            
            # Nợ & Thanh khoản
            "total_debt": safe_value(info.get('totalDebt'), 0),
            "total_equity": safe_value(info.get('totalStockholderEquity') or info.get('bookValue', 0) * info.get('sharesOutstanding', 0), 0),
            "debt_to_equity": safe_value(info.get('debtToEquity'), 0),
            "current_ratio": safe_value(info.get('currentRatio'), 0),
            "quick_ratio": safe_value(info.get('quickRatio'), 0),
            
            # Dòng tiền
            "free_cashflow": safe_value(info.get('freeCashflow'), 0),
            "operating_cashflow": safe_value(info.get('operatingCashflow'), 0),
            
            # Cổ tức
            "dividend_yield": safe_value(info.get('dividendYield'), 0) * 100,
            "payout_ratio": safe_value(info.get('payoutRatio'), 0) * 100,
            
            # Thông tin khác
            "beta": safe_value(info.get('beta'), 1),
            "52w_high": safe_value(info.get('fiftyTwoWeekHigh'), 0),
            "52w_low": safe_value(info.get('fiftyTwoWeekLow'), 0),
            "avg_volume": safe_value(info.get('averageVolume'), 0),
            "shares_outstanding": safe_value(info.get('sharesOutstanding'), 0),
            
            # EPS
            "eps": safe_value(info.get('trailingEps'), 0),
            "forward_eps": safe_value(info.get('forwardEps'), 0),
            
            # Book value
            "book_value": safe_value(info.get('bookValue'), 0),
        }
        
        return data
    except Exception as e:
        return None


def calculate_intrinsic_value(data: Dict, sector: str = "Công nghiệp") -> Dict:
    """Tính giá trị nội tại của cổ phiếu."""
    
    price = data['price']
    eps = data['eps']
    book_value = data['book_value']
    roe = data['roe']
    earnings_growth = data['earnings_growth']
    pe_ratio = data['pe_ratio']
    pb_ratio = data['pb_ratio']
    
    # Lấy P/E benchmark ngành
    pe_benchmark = SECTOR_PE_BENCHMARK.get(sector, 15)
    
    intrinsic_values = {}
    
    # 1. Graham Number (Benjamin Graham)
    # √(22.5 × EPS × Book Value)
    if eps > 0 and book_value > 0:
        graham = np.sqrt(22.5 * eps * book_value)
        intrinsic_values['graham'] = graham
    else:
        intrinsic_values['graham'] = 0
    
    # 2. P/E Based Valuation
    # Fair Value = EPS × Industry P/E
    if eps > 0:
        pe_fair_value = eps * pe_benchmark
        intrinsic_values['pe_fair'] = pe_fair_value
    else:
        intrinsic_values['pe_fair'] = 0
    
    # 3. PEG Based (Peter Lynch)
    # Fair P/E = Earnings Growth Rate
    if eps > 0 and earnings_growth > 0:
        peg_fair_pe = max(earnings_growth, 8)  # Minimum P/E of 8
        peg_fair_value = eps * peg_fair_pe
        intrinsic_values['peg_fair'] = peg_fair_value
    else:
        intrinsic_values['peg_fair'] = 0
    
    # 4. DCF Simple (Discounted Cash Flow đơn giản)
    # Giả định growth rate và discount rate
    if eps > 0:
        growth_rate = min(max(earnings_growth / 100, 0.03), 0.25)  # 3% - 25%
        discount_rate = 0.12  # 12% required return
        terminal_growth = 0.03  # 3% perpetual growth
        
        # 5-year DCF
        dcf_value = 0
        future_eps = eps
        for year in range(1, 6):
            future_eps *= (1 + growth_rate)
            dcf_value += future_eps / ((1 + discount_rate) ** year)
        
        # Terminal value
        terminal_value = future_eps * (1 + terminal_growth) / (discount_rate - terminal_growth)
        terminal_pv = terminal_value / ((1 + discount_rate) ** 5)
        dcf_value += terminal_pv
        
        intrinsic_values['dcf'] = dcf_value
    else:
        intrinsic_values['dcf'] = 0
    
    # 5. Book Value Based
    # Fair P/B based on ROE
    if book_value > 0 and roe > 0:
        # P/B should be higher if ROE is higher
        fair_pb = max(1, roe / 10)  # ROE 15% = P/B 1.5
        bv_fair_value = book_value * fair_pb
        intrinsic_values['bv_fair'] = bv_fair_value
    else:
        intrinsic_values['bv_fair'] = 0
    
    # Weighted Average Intrinsic Value
    valid_values = [v for v in intrinsic_values.values() if v > 0]
    if valid_values:
        # Trọng số: DCF > Graham > P/E > PEG > BV
        weights = {'dcf': 0.3, 'graham': 0.25, 'pe_fair': 0.2, 'peg_fair': 0.15, 'bv_fair': 0.1}
        weighted_sum = 0
        weight_total = 0
        for key, value in intrinsic_values.items():
            if value > 0:
                weighted_sum += value * weights.get(key, 0.1)
                weight_total += weights.get(key, 0.1)
        
        avg_intrinsic = weighted_sum / weight_total if weight_total > 0 else 0
    else:
        avg_intrinsic = 0
    
    # Margin of Safety
    if avg_intrinsic > 0 and price > 0:
        upside = ((avg_intrinsic - price) / price) * 100
    else:
        upside = 0
    
    return {
        "intrinsic_values": intrinsic_values,
        "avg_intrinsic": avg_intrinsic,
        "current_price": price,
        "upside_percent": upside,
        "pe_benchmark": pe_benchmark
    }


def score_fundamentals(data: Dict, sector: str = "Công nghiệp") -> Dict:
    """Chấm điểm phân tích cơ bản (0-100)."""
    
    score = 0
    max_score = 100
    details = []
    
    # 1. Profitability (25 điểm)
    profitability_score = 0
    
    # ROE (10 điểm)
    roe = data['roe']
    if roe >= 20:
        profitability_score += 10
        details.append(("ROE", roe, "Xuất sắc", "good"))
    elif roe >= 15:
        profitability_score += 8
        details.append(("ROE", roe, "Tốt", "good"))
    elif roe >= 10:
        profitability_score += 5
        details.append(("ROE", roe, "Trung bình", "neutral"))
    else:
        details.append(("ROE", roe, "Yếu", "bad"))
    
    # Profit Margin (8 điểm)
    margin = data['profit_margin']
    if margin >= 15:
        profitability_score += 8
        details.append(("Biên LN ròng", margin, "Cao", "good"))
    elif margin >= 8:
        profitability_score += 5
        details.append(("Biên LN ròng", margin, "Khá", "good"))
    elif margin >= 3:
        profitability_score += 2
        details.append(("Biên LN ròng", margin, "Thấp", "neutral"))
    else:
        details.append(("Biên LN ròng", margin, "Rất thấp", "bad"))
    
    # ROA (7 điểm)
    roa = data['roa']
    if roa >= 10:
        profitability_score += 7
    elif roa >= 5:
        profitability_score += 4
    elif roa >= 2:
        profitability_score += 2
    
    score += profitability_score
    
    # 2. Growth (25 điểm)
    growth_score = 0
    
    # Earnings Growth (15 điểm)
    eg = data['earnings_growth']
    if eg >= 20:
        growth_score += 15
        details.append(("Tăng trưởng LN", eg, "Cao", "good"))
    elif eg >= 10:
        growth_score += 10
        details.append(("Tăng trưởng LN", eg, "Khá", "good"))
    elif eg >= 0:
        growth_score += 5
        details.append(("Tăng trưởng LN", eg, "Ổn định", "neutral"))
    else:
        details.append(("Tăng trưởng LN", eg, "Âm", "bad"))
    
    # Revenue Growth (10 điểm)
    rg = data['revenue_growth']
    if rg >= 15:
        growth_score += 10
        details.append(("Tăng trưởng DT", rg, "Cao", "good"))
    elif rg >= 8:
        growth_score += 7
        details.append(("Tăng trưởng DT", rg, "Khá", "good"))
    elif rg >= 0:
        growth_score += 3
        details.append(("Tăng trưởng DT", rg, "Ổn định", "neutral"))
    else:
        details.append(("Tăng trưởng DT", rg, "Âm", "bad"))
    
    score += growth_score
    
    # 3. Financial Health (25 điểm)
    health_score = 0
    
    # Debt to Equity (12 điểm)
    de = data['debt_to_equity']
    if de == 0 or de < 30:
        health_score += 12
        details.append(("Nợ/Vốn CSH", de, "Rất an toàn", "good"))
    elif de < 80:
        health_score += 8
        details.append(("Nợ/Vốn CSH", de, "An toàn", "good"))
    elif de < 150:
        health_score += 4
        details.append(("Nợ/Vốn CSH", de, "Chấp nhận", "neutral"))
    else:
        details.append(("Nợ/Vốn CSH", de, "Cao", "bad"))
    
    # Current Ratio (8 điểm)
    cr = data['current_ratio']
    if cr >= 2:
        health_score += 8
        details.append(("Thanh toán hiện hành", cr, "Mạnh", "good"))
    elif cr >= 1.5:
        health_score += 6
        details.append(("Thanh toán hiện hành", cr, "Tốt", "good"))
    elif cr >= 1:
        health_score += 3
        details.append(("Thanh toán hiện hành", cr, "Đủ", "neutral"))
    else:
        details.append(("Thanh toán hiện hành", cr, "Yếu", "bad"))
    
    # Free Cash Flow (5 điểm)
    fcf = data['free_cashflow']
    if fcf > 0:
        health_score += 5
        details.append(("Dòng tiền tự do", fcf, "Dương", "good"))
    else:
        details.append(("Dòng tiền tự do", fcf, "Âm", "bad"))
    
    score += health_score
    
    # 4. Valuation (25 điểm)
    valuation_score = 0
    pe_benchmark = SECTOR_PE_BENCHMARK.get(sector, 15)
    
    # P/E (12 điểm)
    pe = data['pe_ratio']
    if 0 < pe < pe_benchmark * 0.7:
        valuation_score += 12
        details.append(("P/E", pe, "Rẻ", "good"))
    elif 0 < pe < pe_benchmark:
        valuation_score += 8
        details.append(("P/E", pe, "Hợp lý", "good"))
    elif 0 < pe < pe_benchmark * 1.3:
        valuation_score += 4
        details.append(("P/E", pe, "Khá cao", "neutral"))
    elif pe > 0:
        details.append(("P/E", pe, "Đắt", "bad"))
    else:
        details.append(("P/E", pe, "Âm/N/A", "bad"))
    
    # P/B (8 điểm)
    pb = data['pb_ratio']
    if 0 < pb < 1.5:
        valuation_score += 8
        details.append(("P/B", pb, "Rẻ", "good"))
    elif 0 < pb < 2.5:
        valuation_score += 5
        details.append(("P/B", pb, "Hợp lý", "good"))
    elif 0 < pb < 4:
        valuation_score += 2
        details.append(("P/B", pb, "Cao", "neutral"))
    elif pb > 0:
        details.append(("P/B", pb, "Rất cao", "bad"))
    
    # PEG (5 điểm)
    peg = data['peg_ratio']
    if 0 < peg < 1:
        valuation_score += 5
        details.append(("PEG", peg, "Hấp dẫn", "good"))
    elif 0 < peg < 1.5:
        valuation_score += 3
        details.append(("PEG", peg, "Hợp lý", "good"))
    elif peg > 0:
        details.append(("PEG", peg, "Cao", "neutral"))
    
    score += valuation_score
    
    # Rating
    if score >= 80:
        rating = "XUẤT SẮC"
        rating_class = "good"
    elif score >= 65:
        rating = "TỐT"
        rating_class = "good"
    elif score >= 50:
        rating = "TRUNG BÌNH"
        rating_class = "neutral"
    elif score >= 35:
        rating = "YẾU"
        rating_class = "bad"
    else:
        rating = "RỦI RO CAO"
        rating_class = "bad"
    
    return {
        "score": score,
        "max_score": max_score,
        "rating": rating,
        "rating_class": rating_class,
        "profitability_score": profitability_score,
        "growth_score": growth_score,
        "health_score": health_score,
        "valuation_score": valuation_score,
        "details": details
    }


# ======================= TECHNICAL ANALYSIS =======================

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


def analyze_technical(df: pd.DataFrame) -> Dict:
    """Phân tích kỹ thuật toàn diện."""
    if df is None or len(df) < 20:
        return None
    
    try:
        # EMAs
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['EMA50'] = ta.ema(df['Close'], length=50)
        df['EMA200'] = ta.ema(df['Close'], length=200)
        
        latest = df.iloc[-1]
        price = safe_value(latest['Close'])
        ema20 = safe_value(latest['EMA20'])
        ema50 = safe_value(latest['EMA50'])
        ema200 = safe_value(latest['EMA200'])
        
        # Trend
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
        rsi = safe_value(df['RSI'].iloc[-1], 50)
        
        rsi_score = 0
        if 30 < rsi < 45:
            rsi_score = 1.5  # Oversold recovery
        elif 45 <= rsi < 55:
            rsi_score = 1
        elif 55 <= rsi < 70:
            rsi_score = 0.5
        elif rsi >= 70:
            rsi_score = -1
        elif rsi <= 30:
            rsi_score = 0.5  # Very oversold - potential bounce
        
        # Volume
        df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
        vol_ratio = df['Volume'].iloc[-1] / df['Vol_SMA20'].iloc[-1] if df['Vol_SMA20'].iloc[-1] > 0 else 1
        
        vol_score = 0
        if vol_ratio > 1.5 and trend_score > 0:
            vol_score = 1.5
        elif vol_ratio > 1.2 and trend_score > 0:
            vol_score = 1
        elif vol_ratio > 1.5 and trend_score < 0:
            vol_score = -1
        
        # MACD
        try:
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            macd_val = safe_value(macd['MACD_12_26_9'].iloc[-1])
            macd_signal = safe_value(macd['MACDs_12_26_9'].iloc[-1])
            macd_hist = safe_value(macd['MACDh_12_26_9'].iloc[-1])
            macd_hist_prev = safe_value(macd['MACDh_12_26_9'].iloc[-2])
            
            macd_score = 0
            if macd_val > macd_signal and macd_hist > macd_hist_prev:
                macd_score = 1.5
            elif macd_val > macd_signal:
                macd_score = 1
            elif macd_val < macd_signal and macd_hist < macd_hist_prev:
                macd_score = -1.5
            elif macd_val < macd_signal:
                macd_score = -1
        except:
            macd_score = 0
            macd_hist = 0
        
        # Price momentum
        price_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100 if len(df) > 6 else 0
        price_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21] * 100 if len(df) > 21 else 0
        
        # Support/Resistance
        recent_high = df['High'].tail(50).max()
        recent_low = df['Low'].tail(50).min()
        price_position = (price - recent_low) / (recent_high - recent_low) * 100 if (recent_high - recent_low) > 0 else 50
        
        # Total technical score
        total_score = trend_score + rsi_score + vol_score + macd_score
        
        # Signal
        if total_score >= 3:
            signal = "MUA MẠNH"
            signal_class = "bullish"
        elif total_score >= 1.5:
            signal = "MUA"
            signal_class = "bullish"
        elif total_score <= -3:
            signal = "BÁN MẠNH"
            signal_class = "bearish"
        elif total_score <= -1.5:
            signal = "BÁN"
            signal_class = "bearish"
        else:
            signal = "TRUNG LẬP"
            signal_class = "neutral"
        
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
            "signal": signal,
            "signal_class": signal_class,
            "price_5d": price_5d,
            "price_20d": price_20d,
            "price_position": price_position,
            "recent_high": recent_high,
            "recent_low": recent_low
        }
    except Exception as e:
        return None


def score_technical(tech_data: Dict) -> int:
    """Chuyển đổi technical analysis sang điểm 0-100."""
    if tech_data is None:
        return 0
    
    # Map total_score (-6 to +6) to 0-100
    raw_score = tech_data['total_score']
    normalized = ((raw_score + 6) / 12) * 100
    return max(0, min(100, normalized))


# ======================= COMBINED ANALYSIS =======================

def calculate_combined_score(fundamental_score: int, technical_score: int, intrinsic_data: Dict) -> Dict:
    """Tính điểm tổng hợp và xếp hạng."""
    
    # Trọng số: 50% Fundamental, 30% Technical, 20% Valuation
    fundamental_weight = 0.50
    technical_weight = 0.30
    valuation_weight = 0.20
    
    # Valuation score based on upside
    upside = intrinsic_data.get('upside_percent', 0)
    if upside >= 50:
        valuation_score = 100
    elif upside >= 30:
        valuation_score = 85
    elif upside >= 15:
        valuation_score = 70
    elif upside >= 0:
        valuation_score = 55
    elif upside >= -15:
        valuation_score = 40
    elif upside >= -30:
        valuation_score = 25
    else:
        valuation_score = 10
    
    combined = (
        fundamental_score * fundamental_weight +
        technical_score * technical_weight +
        valuation_score * valuation_weight
    )
    
    # Final rating
    if combined >= 80:
        rating = "KHUYẾN NGHỊ MUA MẠNH"
        rating_class = "bullish"
    elif combined >= 65:
        rating = "KHUYẾN NGHỊ MUA"
        rating_class = "bullish"
    elif combined >= 50:
        rating = "TRUNG LẬP - THEO DÕI"
        rating_class = "neutral"
    elif combined >= 35:
        rating = "CẨN TRỌNG"
        rating_class = "neutral"
    else:
        rating = "TRÁNH / BÁN"
        rating_class = "bearish"
    
    return {
        "combined_score": round(combined, 1),
        "fundamental_score": fundamental_score,
        "technical_score": technical_score,
        "valuation_score": valuation_score,
        "rating": rating,
        "rating_class": rating_class,
        "upside": upside
    }


def scan_all_stocks(stock_list: Dict, progress_callback=None) -> List[Dict]:
    """Quét toàn bộ cổ phiếu và xếp hạng."""
    results = []
    total = len(stock_list)
    
    for i, (code, info) in enumerate(stock_list.items()):
        if progress_callback:
            progress_callback((i + 1) / total, f"Đang phân tích {code}... ({i+1}/{total})")
        
        ticker = get_vn_ticker(code)
        
        # Fundamental
        fund_data = get_fundamental_data(ticker)
        if fund_data is None or fund_data['price'] == 0:
            continue
        
        sector = info.get('sector', 'Công nghiệp')
        fund_score_data = score_fundamentals(fund_data, sector)
        intrinsic_data = calculate_intrinsic_value(fund_data, sector)
        
        # Technical
        df = fetch_stock_data(ticker, "6mo")
        tech_data = analyze_technical(df)
        tech_score = score_technical(tech_data) if tech_data else 50
        
        # Combined
        combined = calculate_combined_score(
            fund_score_data['score'],
            tech_score,
            intrinsic_data
        )
        
        results.append({
            "code": code,
            "name": info['name'],
            "sector": sector,
            "ticker": ticker,
            "price": fund_data['price'],
            "pe": fund_data['pe_ratio'],
            "pb": fund_data['pb_ratio'],
            "roe": fund_data['roe'],
            "profit_margin": fund_data['profit_margin'],
            "debt_equity": fund_data['debt_to_equity'],
            "earnings_growth": fund_data['earnings_growth'],
            "intrinsic_value": intrinsic_data['avg_intrinsic'],
            "upside": intrinsic_data['upside_percent'],
            "fundamental_score": fund_score_data['score'],
            "fundamental_rating": fund_score_data['rating'],
            "technical_score": tech_score,
            "technical_signal": tech_data['signal'] if tech_data else "N/A",
            "rsi": tech_data['rsi'] if tech_data else 50,
            "combined_score": combined['combined_score'],
            "final_rating": combined['rating'],
            "rating_class": combined['rating_class']
        })
        
        time.sleep(0.2)  # Rate limiting
    
    # Sort by combined score
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results


# ======================= UI COMPONENTS =======================

def create_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Tạo biểu đồ giá."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.7, 0.3], subplot_titles=(ticker, 'RSI'))
    
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='Giá',
                                  increasing_line_color='#00ff88', decreasing_line_color='#ff4757'), row=1, col=1)
    
    if 'EMA20' in df.columns and not df['EMA20'].isna().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name='EMA20',
                                  line=dict(color='#ffd93d', width=1)), row=1, col=1)
    if 'EMA50' in df.columns and not df['EMA50'].isna().all():
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], name='EMA50',
                                  line=dict(color='#6c5ce7', width=1)), row=1, col=1)
    
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI',
                                  line=dict(color='#a855f7', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,71,87,0.5)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.5)", row=2, col=1)
    
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(26,26,36,0.8)', height=500,
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_rangeslider_visible=False, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
    return fig


# ======================= MAIN APP =======================

def main():
    st.markdown('<h1 class="main-header">📊 Phân tích Đầu tư Toàn diện</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Kết hợp Phân tích Cơ bản + Kỹ thuật | Tìm Giá trị Nội tại | Top 10 Đáng Đầu tư</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔍 Phân tích Chi tiết", "🏆 Top 10 Đáng Đầu tư", "📖 Hướng dẫn"])
    
    # ==================== TAB 1: DETAILED ANALYSIS ====================
    with tab1:
        with st.sidebar:
            st.markdown("### ⚙️ Cài đặt")
            
            all_stocks = {f"{k} - {v['name']}": k for k, v in VN_STOCKS.items()}
            selected = st.selectbox("Chọn cổ phiếu", list(all_stocks.keys()))
            code = all_stocks[selected]
            ticker = get_vn_ticker(code)
            
            period_opts = {"3 Tháng": "3mo", "6 Tháng": "6mo", "1 Năm": "1y", "2 Năm": "2y"}
            period = period_opts[st.selectbox("Thời gian", list(period_opts.keys()), index=2)]
            
            st.markdown("---")
            analyze_btn = st.button("🔍 Phân tích Toàn diện", type="primary", use_container_width=True)
        
        if analyze_btn:
            stock_info = VN_STOCKS[code]
            sector = stock_info['sector']
            
            with st.spinner(f"Đang phân tích {code}..."):
                # Get data
                fund_data = get_fundamental_data(ticker)
                df = fetch_stock_data(ticker, period)
                
                if fund_data is None or fund_data['price'] == 0:
                    st.error(f"❌ Không lấy được dữ liệu cho {code}")
                    return
                
                # Analysis
                fund_score = score_fundamentals(fund_data, sector)
                intrinsic = calculate_intrinsic_value(fund_data, sector)
                tech_data = analyze_technical(df)
                tech_score = score_technical(tech_data) if tech_data else 50
                combined = calculate_combined_score(fund_score['score'], tech_score, intrinsic)
            
            # Header
            st.markdown(f"## {code} - {stock_info['name']}")
            st.caption(f"Ngành: {sector} | {stock_info['industry']}")
            
            # Main Signal Card
            st.markdown(f"""
            <div class="signal-card signal-{combined['rating_class']}">
                <div style="font-size: 0.9rem; color: #a0a0b0;">KHUYẾN NGHỊ ĐẦU TƯ</div>
                <div class="signal-text">{combined['rating']}</div>
                <div style="margin-top: 0.5rem;">
                    <span style="font-size: 1.5rem; font-weight: 700;">Điểm: {combined['combined_score']}/100</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Score breakdown
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f'<div class="metric-box"><div class="metric-value">{format_price(fund_data["price"])}</div><div class="metric-label">Giá hiện tại</div></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-box"><div class="metric-value fundamental-{fund_score["rating_class"]}">{fund_score["score"]}</div><div class="metric-label">Điểm Cơ bản</div></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="metric-box"><div class="metric-value">{tech_score:.0f}</div><div class="metric-label">Điểm Kỹ thuật</div></div>', unsafe_allow_html=True)
            
            upside_class = "upside" if combined['upside'] > 0 else "downside"
            col4.markdown(f'<div class="metric-box"><div class="metric-value {upside_class}">{combined["upside"]:+.1f}%</div><div class="metric-label">Tiềm năng tăng giá</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Two columns: Fundamental & Technical
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("### 📊 Phân tích Cơ bản")
                
                # Valuation Box
                st.markdown(f"""
                <div class="valuation-box">
                    <div style="text-align: center;">
                        <div style="font-size: 0.9rem; color: #a0a0b0;">GIÁ TRỊ NỘI TẠI ƯỚC TÍNH</div>
                        <div class="intrinsic-value">{format_price(intrinsic['avg_intrinsic'])}</div>
                        <div style="margin-top: 0.5rem;">
                            Giá hiện tại: {format_price(fund_data['price'])} |
                            <span class="{upside_class}">Chênh lệch: {combined['upside']:+.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Fundamental metrics
                with st.expander("📈 Chỉ số Tài chính Chi tiết", expanded=True):
                    st.markdown("**Định giá:**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("P/E", f"{fund_data['pe_ratio']:.1f}", f"Ngành: {intrinsic['pe_benchmark']}")
                    c2.metric("P/B", f"{fund_data['pb_ratio']:.1f}")
                    c3.metric("PEG", f"{fund_data['peg_ratio']:.2f}" if fund_data['peg_ratio'] > 0 else "N/A")
                    
                    st.markdown("**Lợi nhuận:**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ROE", f"{fund_data['roe']:.1f}%")
                    c2.metric("ROA", f"{fund_data['roa']:.1f}%")
                    c3.metric("Biên LN ròng", f"{fund_data['profit_margin']:.1f}%")
                    
                    st.markdown("**Tăng trưởng:**")
                    c1, c2 = st.columns(2)
                    c1.metric("Tăng trưởng LN", f"{fund_data['earnings_growth']:.1f}%")
                    c2.metric("Tăng trưởng DT", f"{fund_data['revenue_growth']:.1f}%")
                    
                    st.markdown("**Sức khỏe Tài chính:**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Nợ/Vốn CSH", f"{fund_data['debt_to_equity']:.0f}%")
                    c2.metric("Thanh toán HH", f"{fund_data['current_ratio']:.2f}")
                    c3.metric("Dòng tiền TD", format_number(fund_data['free_cashflow']))
                
                # Rating details
                with st.expander("📋 Chi tiết Đánh giá Cơ bản"):
                    for detail in fund_score['details']:
                        name, value, assessment, status = detail
                        color = "#00ff88" if status == "good" else ("#ff4757" if status == "bad" else "#ffd93d")
                        st.markdown(f"**{name}:** {value:.1f} - <span style='color:{color}'>{assessment}</span>", unsafe_allow_html=True)
            
            with col_right:
                st.markdown("### 📈 Phân tích Kỹ thuật")
                
                if df is not None and tech_data:
                    # Technical signal
                    tech_signal_class = tech_data['signal_class']
                    st.markdown(f"""
                    <div class="score-card">
                        <div style="text-align: center;">
                            <div style="font-size: 0.9rem; color: #a0a0b0;">TÍN HIỆU KỸ THUẬT</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: {'#00ff88' if tech_signal_class == 'bullish' else '#ff4757' if tech_signal_class == 'bearish' else '#ffd93d'};">
                                {tech_data['signal']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart
                    st.plotly_chart(create_chart(df, code), use_container_width=True)
                    
                    # Technical metrics
                    with st.expander("📊 Chỉ báo Kỹ thuật", expanded=True):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Xu hướng", tech_data['trend'])
                        c2.metric("RSI", f"{tech_data['rsi']:.1f}")
                        c3.metric("KL/TB", f"{tech_data['vol_ratio']:.2f}x")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("% 5 ngày", f"{tech_data['price_5d']:+.1f}%")
                        c2.metric("% 20 ngày", f"{tech_data['price_20d']:+.1f}%")
                        c3.metric("Vị trí giá", f"{tech_data['price_position']:.0f}%")
    
    # ==================== TAB 2: TOP 10 ====================
    with tab2:
        st.markdown("### 🏆 Top 10 Cổ phiếu Đáng Đầu tư Nhất")
        st.markdown("""
        *Xếp hạng dựa trên:*
        - **50%** Phân tích Cơ bản (ROE, Tăng trưởng, Nợ, Định giá)
        - **30%** Phân tích Kỹ thuật (Xu hướng, RSI, MACD, Khối lượng)
        - **20%** Định giá (So với Giá trị Nội tại)
        """)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            scan_btn = st.button("🔄 Quét & Xếp hạng", type="primary", use_container_width=True)
        
        if scan_btn:
            progress = st.progress(0)
            status = st.empty()
            
            def update(pct, text):
                progress.progress(pct)
                status.text(text)
            
            results = scan_all_stocks(VN_STOCKS, update)
            
            progress.empty()
            status.empty()
            
            if results:
                st.session_state['scan_results'] = results
                st.session_state['scan_time'] = datetime.now().strftime("%H:%M %d/%m/%Y")
        
        if 'scan_results' in st.session_state:
            results = st.session_state['scan_results']
            st.caption(f"🕐 Cập nhật: {st.session_state.get('scan_time', 'N/A')}")
            
            # Top 10
            top_10 = results[:10]
            
            for i, stock in enumerate(top_10):
                rank = i + 1
                rank_class = f"rank-{rank}" if rank <= 3 else ""
                
                rating_color = "#00ff88" if stock['rating_class'] == 'bullish' else "#ff4757" if stock['rating_class'] == 'bearish' else "#ffd93d"
                upside_color = "#00ff88" if stock['upside'] > 0 else "#ff4757"
                
                with st.container():
                    st.markdown(f"""
                    <div class="stock-row">
                        <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                            <div class="rank-badge {rank_class}">{rank}</div>
                            <div style="min-width: 150px;">
                                <div style="font-weight: 700; font-size: 1.1rem;">{stock['code']}</div>
                                <div style="color: #a0a0b0; font-size: 0.8rem;">{stock['name']}</div>
                            </div>
                            <div style="min-width: 100px;">
                                <div style="font-weight: 600;">{format_price(stock['price'])}</div>
                                <div style="color: {upside_color}; font-size: 0.9rem;">Upside: {stock['upside']:+.1f}%</div>
                            </div>
                            <div style="min-width: 80px; text-align: center;">
                                <div style="font-size: 1.5rem; font-weight: 700; color: {rating_color};">{stock['combined_score']:.0f}</div>
                                <div style="font-size: 0.7rem; color: #a0a0b0;">ĐIỂM</div>
                            </div>
                            <div style="min-width: 120px;">
                                <div>Cơ bản: <b>{stock['fundamental_score']}</b></div>
                                <div>Kỹ thuật: <b>{stock['technical_score']:.0f}</b></div>
                            </div>
                            <div style="min-width: 100px;">
                                <div>P/E: {stock['pe']:.1f}</div>
                                <div>ROE: {stock['roe']:.1f}%</div>
                            </div>
                            <div style="min-width: 150px;">
                                <div style="color: {rating_color}; font-weight: 600; font-size: 0.9rem;">{stock['final_rating']}</div>
                                <div style="font-size: 0.8rem;">Giá trị: {format_price(stock['intrinsic_value'])}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Full table
            with st.expander("📊 Bảng Tổng hợp Đầy đủ"):
                df_results = pd.DataFrame(results)
                df_display = df_results[[
                    'code', 'name', 'sector', 'price', 'intrinsic_value', 'upside',
                    'combined_score', 'fundamental_score', 'technical_score',
                    'pe', 'pb', 'roe', 'earnings_growth', 'final_rating'
                ]].copy()
                df_display.columns = [
                    'Mã', 'Tên', 'Ngành', 'Giá', 'Giá trị NĐ', 'Upside%',
                    'Điểm TH', 'Điểm CB', 'Điểm KT',
                    'P/E', 'P/B', 'ROE%', 'Tăng trưởng%', 'Khuyến nghị'
                ]
                df_display['Giá'] = df_display['Giá'].apply(lambda x: f"₫{x:,.0f}")
                df_display['Giá trị NĐ'] = df_display['Giá trị NĐ'].apply(lambda x: f"₫{x:,.0f}" if x > 0 else "N/A")
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Sector breakdown
            with st.expander("📈 Phân bổ theo Ngành"):
                sector_df = pd.DataFrame(results)
                sector_avg = sector_df.groupby('sector')['combined_score'].mean().sort_values(ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=sector_avg.values,
                    y=sector_avg.index,
                    orientation='h',
                    marker_color='#6c5ce7'
                ))
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26,26,36,0.8)',
                    height=400,
                    xaxis_title="Điểm TB",
                    yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Nhấn **Quét & Xếp hạng** để tìm Top 10 cổ phiếu đáng đầu tư nhất")
    
    # ==================== TAB 3: GUIDE ====================
    with tab3:
        st.markdown("""
        ## 📖 Hướng dẫn Sử dụng
        
        ### I. Phân tích Cơ bản (Fundamental Analysis)
        👉 *Trả lời câu hỏi: Doanh nghiệp này có đáng đầu tư không?*
        
        **1. Chỉ số Lợi nhuận:**
        - **ROE (Return on Equity):** > 15% là tốt
        - **ROA (Return on Assets):** > 5% là tốt
        - **Biên lợi nhuận ròng:** Càng cao càng tốt
        
        **2. Chỉ số Tăng trưởng:**
        - **Tăng trưởng EPS:** > 10% là tốt
        - **Tăng trưởng Doanh thu:** > 8% là tốt
        
        **3. Sức khỏe Tài chính:**
        - **Nợ/Vốn CSH:** < 100% là an toàn
        - **Current Ratio:** > 1.5 là tốt
        - **Dòng tiền tự do:** Phải dương
        
        **4. Định giá:**
        - **P/E:** So với trung bình ngành
        - **P/B:** < 2 là rẻ
        - **PEG:** < 1 là hấp dẫn
        
        ---
        
        ### II. Phân tích Kỹ thuật (Technical Analysis)
        👉 *Trả lời câu hỏi: Thời điểm nào nên mua/bán?*
        
        **1. Xu hướng (EMA):**
        - Giá > EMA20 > EMA50 > EMA200: Xu hướng tăng mạnh
        
        **2. RSI:**
        - < 30: Quá bán (cơ hội mua)
        - > 70: Quá mua (cẩn trọng)
        - 30-70: Vùng trung lập
        
        **3. MACD:**
        - MACD cắt lên Signal: Tín hiệu mua
        - Histogram tăng: Động lực tăng
        
        **4. Khối lượng:**
        - > 1.5x trung bình + xu hướng tăng: Xác nhận mạnh
        
        ---
        
        ### III. Tính Giá trị Nội tại
        
        Công cụ sử dụng nhiều phương pháp:
        
        1. **Graham Number:** √(22.5 × EPS × Book Value)
        2. **P/E Fair Value:** EPS × P/E ngành
        3. **DCF đơn giản:** Chiết khấu dòng tiền 5 năm
        4. **PEG Fair Value:** EPS × Growth Rate
        
        **Kết quả:** Giá trị nội tại trung bình có trọng số
        
        ---
        
        ### IV. Cách Đọc Kết quả
        
        | Điểm | Khuyến nghị |
        |------|-------------|
        | 80+ | MUA MẠNH |
        | 65-80 | MUA |
        | 50-65 | THEO DÕI |
        | 35-50 | CẨN TRỌNG |
        | <35 | TRÁNH |
        
        **Upside (Tiềm năng tăng giá):**
        - > 30%: Rất hấp dẫn
        - 15-30%: Hấp dẫn
        - 0-15%: Hợp lý
        - < 0%: Định giá cao
        """)
    
    st.markdown("---")
    st.caption("⚠️ **Miễn trừ trách nhiệm:** Công cụ chỉ mang tính chất tham khảo và giáo dục. Không phải tư vấn đầu tư. Hãy tự nghiên cứu kỹ trước khi ra quyết định.")


if __name__ == "__main__":
    main()
