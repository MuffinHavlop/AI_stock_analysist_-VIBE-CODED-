#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║         VN STOCK ANALYST BOT  ·  Powered by AI              ║
║         Phân tích kỹ thuật thị trường chứng khoán VN        ║
╚══════════════════════════════════════════════════════════════╝

Yêu cầu:
    pip install vnstock pandas numpy ta rich requests

Cách chạy:
    python vn_stock_bot.py                    # Quét toàn bộ watchlist
    python vn_stock_bot.py --symbol ACB       # Phân tích 1 mã
    python vn_stock_bot.py --top 10           # Top N cổ phiếu tốt nhất
    python vn_stock_bot.py --source KBS VCI   # Chỉ định nguồn dữ liệu
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import box
    from rich.rule import Rule
except ImportError:
    print("Cài đặt rich: pip install rich")
    sys.exit(1)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WATCHLIST = [
    # Ngân hàng
    "VCB", "BID", "CTG", "MBB", "ACB", "TCB", "HDB", "VPB", "STB", "TPB",
    # Bất động sản
    "VIC", "VHM", "NVL", "PDR", "DXG", "KDH",
    # Tiêu dùng & Bán lẻ
    "VNM", "MSN", "MWG", "SAB", "PNJ",
    # Công nghiệp & Năng lượng
    "GAS", "PLX", "PVD", "PVS", "BSR",
    # Hàng không & Logistics
    "HVN", "GMD", "VSC",
    # Thép & Vật liệu
    "HPG", "NKG", "HSG",
    # Chứng khoán
    "SSI", "VND", "HCM",
    # Dược phẩm
    "DHG", "IMP",
    #Công nghệ
    "FPT"
]

# Nguồn dữ liệu theo thứ tự ưu tiên (thay đổi bằng --source)
DATA_SOURCES = ["VCI", "KBS"]

# Số ngày lịch sử cần lấy
HISTORY_DAYS = 300

INDICATORS_CONFIG = {
    "RSI_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "BB_PERIOD": 20,
    "BB_STD": 2,
    "EMA_SHORT": 9,
    "EMA_MID": 21,
    "EMA_LONG": 50,
    "ATR_PERIOD": 14,
    "STOCH_K": 14,
    "STOCH_D": 3,
    "VOLUME_MA": 20,
    "OBV_EMA": 20,
    "ADX_PERIOD": 14,
    "CCI_PERIOD": 20,
    "WILLIAMS_R_PERIOD": 14,
    "MFI_PERIOD": 14,
}

SIGNAL_THRESHOLDS = {
    "RSI_OVERSOLD": 35,
    "RSI_OVERBOUGHT": 65,
    "RSI_NEUTRAL_LOW": 45,
    "RSI_NEUTRAL_HIGH": 55,
    "ADX_STRONG_TREND": 25,
    "ADX_WEAK_TREND": 20,
    "SCORE_STRONG_BUY": 75,
    "SCORE_BUY": 55,
    "SCORE_HOLD": 40,
    "SCORE_SELL": 25,
}


# ─────────────────────────────────────────────────────────────────────────────
# LẤY DỮ LIỆU THỜI GIAN THỰC
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Chuẩn hóa DataFrame từ vnstock:
    - Tên cột về lowercase
    - Đảm bảo có đủ cột OHLCV
    - Index là DatetimeIndex tăng dần
    - Nhân 1000 nếu giá đang ở đơn vị nghìn đồng (vnstock trả về x1000 VNĐ)
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    required = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in required):
        return None

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # vnstock trả về giá theo đơn vị nghìn VNĐ (vd: 60.5 = 60,500đ)
    # Nếu close < 1000 thì đang ở đơn vị nghìn → nhân 1000
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["close"].median() < 1000:
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * 1000

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=required)

    return df


def fetch_stock_data(symbol: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """
    Lấy dữ liệu giá lịch sử từ vnstock.
    Thử lần lượt các source trong DATA_SOURCES (VCI → KBS).
    Raise RuntimeError nếu tất cả source đều thất bại.
    """
    try:
        from vnstock.api.quote import Quote
    except ImportError:
        raise RuntimeError(
            "Không tìm thấy thư viện vnstock.\n"
            "Cài đặt: pip install vnstock"
        )

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days + 60)).strftime("%Y-%m-%d")

    errors = {}
    for source in DATA_SOURCES:
        try:
            q  = Quote(symbol=symbol, source=source)
            df = q.history(start=start_date, end=end_date, interval="1D")

            if df is None or len(df) < 60:
                errors[source] = f"Dữ liệu quá ít ({len(df) if df is not None else 0} nến)"
                continue

            df = _normalize_df(df)
            if df is None:
                errors[source] = "Thiếu cột OHLCV sau chuẩn hóa"
                continue

            if len(df) < 60:
                errors[source] = f"Sau chuẩn hóa chỉ còn {len(df)} nến"
                continue

            return df.tail(days)

        except Exception as e:
            # Lọc bỏ traceback dài của vnstock, chỉ lấy dòng cuối
            msg = str(e).split("\n")[-1].strip() or str(e)
            errors[source] = msg
            continue

    err_detail = " | ".join(f"{k}: {v}" for k, v in errors.items())
    raise RuntimeError(
        f"[{symbol}] Không lấy được dữ liệu từ {DATA_SOURCES}.\n"
        f"Chi tiết: {err_detail}\n"
        f"Gợi ý: kiểm tra kết nối mạng hoặc đặt API key: vnai.setup_api_key('key')"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TÍNH TOÁN CHỈ BÁO KỸ THUẬT
# ─────────────────────────────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> dict:
    c   = df["close"]
    h   = df["high"]
    l   = df["low"]
    v   = df["volume"]
    cfg = INDICATORS_CONFIG
    ind = {}

    # EMA
    ind["ema9"]  = c.ewm(span=cfg["EMA_SHORT"], adjust=False).mean()
    ind["ema21"] = c.ewm(span=cfg["EMA_MID"],   adjust=False).mean()
    ind["ema50"] = c.ewm(span=cfg["EMA_LONG"],  adjust=False).mean()

    # MACD
    ema_fast    = c.ewm(span=cfg["MACD_FAST"],   adjust=False).mean()
    ema_slow    = c.ewm(span=cfg["MACD_SLOW"],   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=cfg["MACD_SIGNAL"], adjust=False).mean()
    ind["macd"]        = macd_line
    ind["macd_signal"] = signal_line
    ind["macd_hist"]   = macd_line - signal_line

    # RSI
    delta    = c.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/cfg["RSI_PERIOD"], adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/cfg["RSI_PERIOD"], adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    ind["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = c.rolling(cfg["BB_PERIOD"]).mean()
    bb_std = c.rolling(cfg["BB_PERIOD"]).std()
    ind["bb_upper"] = bb_mid + cfg["BB_STD"] * bb_std
    ind["bb_mid"]   = bb_mid
    ind["bb_lower"] = bb_mid - cfg["BB_STD"] * bb_std
    ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / bb_mid
    bb_range = (ind["bb_upper"] - ind["bb_lower"]).replace(0, 1e-10)
    ind["bb_pct"] = (c - ind["bb_lower"]) / bb_range

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    ind["atr"]     = tr.ewm(span=cfg["ATR_PERIOD"], adjust=False).mean()
    ind["atr_pct"] = ind["atr"] / c * 100

    # Stochastic
    lowest_l    = l.rolling(cfg["STOCH_K"]).min()
    highest_h   = h.rolling(cfg["STOCH_K"]).max()
    stoch_range = (highest_h - lowest_l).replace(0, 1e-10)
    ind["stoch_k"] = (c - lowest_l) / stoch_range * 100
    ind["stoch_d"] = ind["stoch_k"].rolling(cfg["STOCH_D"]).mean()

    # ADX
    up_move   = h - h.shift()
    down_move = l.shift() - l
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_adx   = tr.ewm(span=cfg["ADX_PERIOD"], adjust=False).mean()
    di_plus  = 100 * pos_dm.ewm(span=cfg["ADX_PERIOD"], adjust=False).mean() / tr_adx.replace(0, 1e-10)
    di_minus = 100 * neg_dm.ewm(span=cfg["ADX_PERIOD"], adjust=False).mean() / tr_adx.replace(0, 1e-10)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1e-10)
    ind["adx"]      = dx.ewm(span=cfg["ADX_PERIOD"], adjust=False).mean()
    ind["di_plus"]  = di_plus
    ind["di_minus"] = di_minus

    # CCI
    tp     = (h + l + c) / 3
    tp_ma  = tp.rolling(cfg["CCI_PERIOD"]).mean()
    tp_std = tp.rolling(cfg["CCI_PERIOD"]).std().replace(0, 1e-10)
    ind["cci"] = (tp - tp_ma) / (0.015 * tp_std)

    # Williams %R
    period_h = h.rolling(cfg["WILLIAMS_R_PERIOD"]).max()
    period_l = l.rolling(cfg["WILLIAMS_R_PERIOD"]).min()
    hl_range = (period_h - period_l).replace(0, 1e-10)
    ind["williams_r"] = -100 * (period_h - c) / hl_range

    # Volume & OBV
    ind["vol_ma"]    = v.rolling(cfg["VOLUME_MA"]).mean()
    ind["vol_ratio"] = v / ind["vol_ma"].replace(0, 1e-10)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    ind["obv"]     = obv
    ind["obv_ema"] = obv.ewm(span=cfg["OBV_EMA"], adjust=False).mean()

    # MFI
    mf     = tp * v
    pos_mf = mf.where(tp > tp.shift(), 0.0)
    neg_mf = mf.where(tp < tp.shift(), 0.0).replace(0, 1e-10)
    mf_ratio = (pos_mf.rolling(cfg["MFI_PERIOD"]).sum()
                / neg_mf.rolling(cfg["MFI_PERIOD"]).sum())
    ind["mfi"] = 100 - (100 / (1 + mf_ratio))

    # Ichimoku
    ind["tenkan"] = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    ind["kijun"]  = (h.rolling(26).max() + l.rolling(26).min()) / 2

    # Momentum ROC
    ind["momentum_10"] = c / c.shift(10) * 100 - 100
    ind["momentum_20"] = c / c.shift(20) * 100 - 100

    return ind


# ─────────────────────────────────────────────────────────────────────────────
# CHẤM ĐIỂM & TẠO TÍN HIỆU
# ─────────────────────────────────────────────────────────────────────────────

def score_stock(df: pd.DataFrame, ind: dict) -> dict:
    c    = df["close"]
    last = c.iloc[-1]
    thr  = SIGNAL_THRESHOLDS
    scores  = {}
    signals = {}

    # ── 1. TREND (30%) ─────────────────────────────────────────────────────
    trend_pts  = 0.0
    trend_sigs = []
    e9  = ind["ema9"].iloc[-1]
    e21 = ind["ema21"].iloc[-1]
    e50 = ind["ema50"].iloc[-1]

    if last > e9 > e21 > e50:
        trend_pts += 30; trend_sigs.append("EMA Bullish Stack ↑")
    elif last > e21 > e50:
        trend_pts += 20; trend_sigs.append("EMA Partial Bullish")
    elif last < e9 < e21 < e50:
        trend_pts -= 30; trend_sigs.append("EMA Bearish Stack ↓")
    elif last < e21 < e50:
        trend_pts -= 20; trend_sigs.append("EMA Partial Bearish")

    macd_v      = ind["macd"].iloc[-1]
    macd_s      = ind["macd_signal"].iloc[-1]
    macd_h      = ind["macd_hist"].iloc[-1]
    macd_h_prev = ind["macd_hist"].iloc[-2] if len(ind["macd_hist"]) > 1 else 0

    if macd_v > macd_s and macd_h > macd_h_prev:
        trend_pts += 25; trend_sigs.append("MACD Bullish ↑")
    elif macd_v > macd_s:
        trend_pts += 15; trend_sigs.append("MACD Above Signal")
    elif macd_v < macd_s and macd_h < macd_h_prev:
        trend_pts -= 25; trend_sigs.append("MACD Bearish ↓")
    else:
        trend_pts -= 10; trend_sigs.append("MACD Weak")

    adx_v = ind["adx"].iloc[-1]
    dip   = ind["di_plus"].iloc[-1]
    dim   = ind["di_minus"].iloc[-1]

    if adx_v > thr["ADX_STRONG_TREND"] and dip > dim:
        trend_pts += 20; trend_sigs.append(f"ADX {adx_v:.0f} Strong Uptrend")
    elif adx_v > thr["ADX_WEAK_TREND"] and dip > dim:
        trend_pts += 10; trend_sigs.append(f"ADX {adx_v:.0f} Moderate Trend")
    elif adx_v > thr["ADX_STRONG_TREND"] and dip < dim:
        trend_pts -= 20; trend_sigs.append(f"ADX {adx_v:.0f} Strong Downtrend")

    tenkan = ind["tenkan"].iloc[-1]
    kijun  = ind["kijun"].iloc[-1]
    if last > tenkan > kijun:
        trend_pts += 15; trend_sigs.append("Ichimoku Bullish")
    elif last < tenkan < kijun:
        trend_pts -= 15; trend_sigs.append("Ichimoku Bearish")

    mom10 = ind["momentum_10"].iloc[-1]
    mom20 = ind["momentum_20"].iloc[-1]
    if mom10 > 3 and mom20 > 5:
        trend_pts += 10; trend_sigs.append(f"Momentum +{mom10:.1f}%/{mom20:.1f}%")
    elif mom10 < -3 and mom20 < -5:
        trend_pts -= 10

    scores["trend"]  = max(-100, min(100, trend_pts))
    signals["trend"] = trend_sigs

    # ── 2. MOMENTUM (25%) ──────────────────────────────────────────────────
    mom_pts  = 0.0
    mom_sigs = []
    rsi_v    = ind["rsi"].iloc[-1]
    rsi_prev = ind["rsi"].iloc[-2] if len(ind["rsi"]) > 1 else rsi_v

    if thr["RSI_OVERSOLD"] < rsi_v < thr["RSI_NEUTRAL_HIGH"] and rsi_v > rsi_prev:
        mom_pts += 30; mom_sigs.append(f"RSI {rsi_v:.1f} Phục hồi ↑")
    elif rsi_v < thr["RSI_OVERSOLD"]:
        mom_pts += 15; mom_sigs.append(f"RSI {rsi_v:.1f} Quá bán ⚡")
    elif rsi_v > thr["RSI_OVERBOUGHT"]:
        mom_pts -= 30; mom_sigs.append(f"RSI {rsi_v:.1f} Quá mua ⚠")
    elif thr["RSI_NEUTRAL_LOW"] < rsi_v < thr["RSI_NEUTRAL_HIGH"]:
        mom_sigs.append(f"RSI {rsi_v:.1f} Trung tính")

    sk      = ind["stoch_k"].iloc[-1]
    sd      = ind["stoch_d"].iloc[-1]
    sk_prev = ind["stoch_k"].iloc[-2] if len(ind["stoch_k"]) > 1 else sk

    if sk < 25 and sk > sd and sk > sk_prev:
        mom_pts += 25; mom_sigs.append(f"Stoch {sk:.0f} Bullish Cross")
    elif sk > 75 and sk < sd:
        mom_pts -= 25; mom_sigs.append(f"Stoch {sk:.0f} Bearish Cross")
    elif sk < 30:
        mom_pts += 10; mom_sigs.append(f"Stoch {sk:.0f} Vùng quá bán")
    elif sk > 70:
        mom_pts -= 10

    cci_v = ind["cci"].iloc[-1]
    if -100 < cci_v < 0 and cci_v > ind["cci"].iloc[-2]:
        mom_pts += 20; mom_sigs.append(f"CCI {cci_v:.0f} Tăng tốc")
    elif cci_v < -200:
        mom_pts += 10; mom_sigs.append(f"CCI {cci_v:.0f} Cực kỳ quá bán")
    elif cci_v > 200:
        mom_pts -= 20; mom_sigs.append(f"CCI {cci_v:.0f} Cực kỳ quá mua")

    wr = ind["williams_r"].iloc[-1]
    if -80 < wr < -50:
        mom_pts += 10; mom_sigs.append(f"W%R {wr:.0f} Vùng tích lũy")
    elif wr > -20:
        mom_pts -= 10; mom_sigs.append(f"W%R {wr:.0f} Quá mua")

    scores["momentum"]  = max(-100, min(100, mom_pts))
    signals["momentum"] = mom_sigs

    # ── 3. VOLATILITY (15%) ────────────────────────────────────────────────
    vol_pts  = 0.0
    vol_sigs = []
    bb_pct    = ind["bb_pct"].iloc[-1]
    bb_w      = ind["bb_width"].iloc[-1]
    bb_w_prev = ind["bb_width"].iloc[-5] if len(ind["bb_width"]) > 5 else bb_w
    atr_pct   = ind["atr_pct"].iloc[-1]

    if bb_pct < 0.2:
        vol_pts += 25; vol_sigs.append(f"BB %B {bb_pct:.2f} Gần Lower Band")
    elif bb_pct > 0.8:
        vol_pts -= 25; vol_sigs.append(f"BB %B {bb_pct:.2f} Gần Upper Band")
    else:
        vol_sigs.append(f"BB %B {bb_pct:.2f} Mid Zone")

    if bb_w < bb_w_prev * 0.8:
        vol_pts += 20; vol_sigs.append("BB Squeeze ⚡ Sắp bứt phá")

    if 1.0 < atr_pct < 3.0:
        vol_pts += 15; vol_sigs.append(f"ATR {atr_pct:.1f}% Biến động lành mạnh")
    elif atr_pct > 5.0:
        vol_pts -= 15; vol_sigs.append(f"ATR {atr_pct:.1f}% Biến động cao ⚠")

    scores["volatility"]  = max(-100, min(100, vol_pts))
    signals["volatility"] = vol_sigs

    # ── 4. VOLUME (20%) ────────────────────────────────────────────────────
    vl_pts  = 0.0
    vl_sigs = []
    vol_ratio = ind["vol_ratio"].iloc[-1]
    obv_v     = ind["obv"].iloc[-1]
    obv_ema_v = ind["obv_ema"].iloc[-1]
    mfi_v     = ind["mfi"].iloc[-1]

    if vol_ratio > 2.0 and c.iloc[-1] > c.iloc[-2]:
        vl_pts += 30; vl_sigs.append(f"Volume x{vol_ratio:.1f} Tăng mạnh ↑")
    elif vol_ratio > 1.5 and c.iloc[-1] > c.iloc[-2]:
        vl_pts += 20; vl_sigs.append(f"Volume x{vol_ratio:.1f} Tích cực")
    elif vol_ratio > 2.0 and c.iloc[-1] < c.iloc[-2]:
        vl_pts -= 30; vl_sigs.append(f"Volume x{vol_ratio:.1f} Phân phối ↓")
    elif vol_ratio < 0.6:
        vl_pts -= 10; vl_sigs.append(f"Volume x{vol_ratio:.1f} Thấp bất thường")

    if obv_v > obv_ema_v:
        vl_pts += 20; vl_sigs.append("OBV > EMA Tích lũy")
    else:
        vl_pts -= 10; vl_sigs.append("OBV < EMA Phân phối")

    if mfi_v < 25:
        vl_pts += 25; vl_sigs.append(f"MFI {mfi_v:.0f} Dòng tiền vào mạnh")
    elif mfi_v > 80:
        vl_pts -= 25; vl_sigs.append(f"MFI {mfi_v:.0f} Dòng tiền ra mạnh")
    else:
        vl_sigs.append(f"MFI {mfi_v:.0f} Trung tính")

    scores["volume"]  = max(-100, min(100, vl_pts))
    signals["volume"] = vl_sigs

    # ── 5. PRICE ACTION (10%) ──────────────────────────────────────────────
    pa_pts  = 0.0
    pa_sigs = []
    price_vs_ema50 = (last / e50 - 1) * 100

    if 0 < price_vs_ema50 < 5:
        pa_pts += 20; pa_sigs.append(f"Giá +{price_vs_ema50:.1f}% trên EMA50")
    elif price_vs_ema50 > 10:
        pa_pts -= 10; pa_sigs.append(f"Giá +{price_vs_ema50:.1f}% xa EMA50")
    elif price_vs_ema50 < 0:
        pa_pts -= 20; pa_sigs.append(f"Giá {price_vs_ema50:.1f}% dưới EMA50")

    body         = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    candle_range = df["high"].iloc[-1] - df["low"].iloc[-1]
    if candle_range > 0:
        body_pct = body / candle_range
        if body_pct > 0.7 and df["close"].iloc[-1] > df["open"].iloc[-1]:
            pa_pts += 20; pa_sigs.append("Nến Marubozu tăng mạnh")
        elif body_pct > 0.7 and df["close"].iloc[-1] < df["open"].iloc[-1]:
            pa_pts -= 20; pa_sigs.append("Nến Marubozu giảm mạnh")

    chg_5d = (last / c.iloc[-6] - 1) * 100 if len(c) > 5 else 0
    if 0 < chg_5d < 5:
        pa_pts += 10; pa_sigs.append(f"5d: +{chg_5d:.1f}%")
    elif chg_5d > 10:
        pa_pts -= 5
    elif chg_5d < -5:
        pa_pts -= 15

    scores["price_action"]  = max(-100, min(100, pa_pts))
    signals["price_action"] = pa_sigs

    # ── COMPOSITE SCORE ────────────────────────────────────────────────────
    weights = {"trend": 0.30, "momentum": 0.25, "volume": 0.20,
               "volatility": 0.15, "price_action": 0.10}
    raw_score = sum(scores[k] * weights[k] for k in weights)
    composite = (raw_score + 100) / 2

    if composite >= thr["SCORE_STRONG_BUY"]:
        signal = "STRONG BUY"; signal_color = "bold green"
    elif composite >= thr["SCORE_BUY"]:
        signal = "BUY";        signal_color = "green"
    elif composite >= thr["SCORE_HOLD"]:
        signal = "HOLD";       signal_color = "yellow"
    elif composite >= thr["SCORE_SELL"]:
        signal = "WATCH";      signal_color = "dark_orange"
    else:
        signal = "SELL";       signal_color = "red"

    atr_abs     = ind["atr"].iloc[-1]
    stop_loss   = round((last - 1.5 * atr_abs) / 100) * 100
    target_1    = round((last + 2.0 * atr_abs) / 100) * 100
    target_2    = round((last + 3.5 * atr_abs) / 100) * 100
    risk_reward = (target_1 - last) / max(last - stop_loss, 1)

    return {
        "composite_score": composite,
        "signal":          signal,
        "signal_color":    signal_color,
        "scores":          scores,
        "signals":         signals,
        "stop_loss":       stop_loss,
        "target_1":        target_1,
        "target_2":        target_2,
        "risk_reward":     risk_reward,
        "rsi":             ind["rsi"].iloc[-1],
        "macd_hist":       ind["macd_hist"].iloc[-1],
        "vol_ratio":       ind["vol_ratio"].iloc[-1],
        "adx":             ind["adx"].iloc[-1],
        "atr_pct":         ind["atr_pct"].iloc[-1],
        "bb_pct":          ind["bb_pct"].iloc[-1],
        "mfi":             ind["mfi"].iloc[-1],
    }


def get_price_change(df: pd.DataFrame) -> tuple[float, float]:
    if len(df) < 2:
        return 0.0, 0.0
    last  = df["close"].iloc[-1]
    prev  = df["close"].iloc[-2]
    chg   = last - prev
    chg_p = (chg / prev) * 100
    return chg, chg_p


# ─────────────────────────────────────────────────────────────────────────────
# HIỂN THỊ
# ─────────────────────────────────────────────────────────────────────────────

def format_price(price: float) -> str:
    """Format giá VNĐ theo chuẩn thị trường chứng khoán VN."""
    if price >= 1_000_000:
        return f"{price/1_000_000:.2f}M"
    return f"{price:,.0f}"


def score_bar(score: float, width: int = 10) -> str:
    normalized = max(0, min(width, int((score / 100) * width)))
    color = "green" if score >= 65 else ("yellow" if score >= 45 else "red")
    return f"[{color}]{'█' * normalized}[/{color}][dim]{'░' * (width - normalized)}[/dim]"


def print_header():
    console.print()
    console.print(Panel(
        f"[bold cyan]▸ VN STOCK ANALYST BOT[/bold cyan]  [dim]· LIVE DATA[/dim]\n"
        f"[dim]Phân tích kỹ thuật đa chỉ báo · {datetime.now().strftime('%d/%m/%Y %H:%M')}[/dim]\n"
        f"[dim]Nguồn dữ liệu: {' → '.join(DATA_SOURCES)} (fallback tự động)[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))


def print_summary_table(results: list[dict], top_n: Optional[int] = None):
    if top_n:
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

    table = Table(
        title=f"[bold]{'TOP ' + str(top_n) if top_n else 'WATCHLIST'} · PHÂN TÍCH KỸ THUẬT[/bold]",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        show_lines=False,
        min_width=112,
    )
    table.add_column("Mã",       style="bold white", width=7)
    table.add_column("Giá (đ)",  justify="right",    width=11)
    table.add_column("Δ%",       justify="right",    width=7)
    table.add_column("Score",    justify="center",   width=14)
    table.add_column("Điểm",     justify="right",    width=6)
    table.add_column("Tín hiệu", justify="center",   width=13)
    table.add_column("RSI",      justify="right",    width=6)
    table.add_column("ADX",      justify="right",    width=6)
    table.add_column("MFI",      justify="right",    width=6)
    table.add_column("Vol×",     justify="right",    width=6)
    table.add_column("R:R",      justify="right",    width=6)
    table.add_column("ATR%",     justify="right",    width=6)

    sig_colors = {
        "STRONG BUY": "bold bright_green",
        "BUY":        "green",
        "HOLD":       "yellow",
        "WATCH":      "dark_orange",
        "SELL":       "red",
    }

    for r in results:
        chg_color = "green" if r["chg_pct"] >= 0 else "red"
        sig_style = sig_colors.get(r["signal"], "white")

        rsi = r["rsi"]
        adx = r["adx"]
        mfi = r["mfi"]
        vol = r["vol_ratio"]

        table.add_row(
            r["symbol"],
            format_price(r["price"]),
            f"[{'green' if r['chg_pct'] >= 0 else 'red'}]{r['chg_pct']:+.1f}%[/]",
            score_bar(r["score"]),
            f"{r['score']:.0f}",
            f"[{sig_style}]{r['signal']}[/{sig_style}]",
            f"[{'red' if rsi>65 else 'green' if rsi<35 else 'white'}]{rsi:.0f}[/]",
            f"[{'green' if adx>25 else 'white'}]{adx:.0f}[/]",
            f"[{'red' if mfi>80 else 'green' if mfi<25 else 'white'}]{mfi:.0f}[/]",
            f"[{'green' if vol>1.5 else 'dim' if vol<0.7 else 'white'}]{vol:.1f}[/]",
            f"{r['rr']:.1f}",
            f"{r['atr_pct']:.1f}%",
        )

    console.print(table)


def print_detail_analysis(symbol: str, df: pd.DataFrame, result: dict):
    chg, chg_pct = get_price_change(df)
    price  = df["close"].iloc[-1]
    volume = df["volume"].iloc[-1]
    high52 = df["high"].rolling(min(252, len(df))).max().iloc[-1]
    low52  = df["low"].rolling(min(252, len(df))).min().iloc[-1]

    chg_color = "green" if chg >= 0 else "red"
    chg_arrow = "▲" if chg >= 0 else "▼"

    console.print()
    console.print(Rule(f"[bold cyan]  {symbol}  ·  PHÂN TÍCH CHI TIẾT  [/bold cyan]", style="cyan"))

    console.print(Panel(
        f"  [bold white]Giá:[/bold white] [bold]{format_price(price)}[/bold] đ  "
        f"[{chg_color}]{chg_arrow} {format_price(abs(chg))} đ ({chg_pct:+.2f}%)[/{chg_color}]\n"
        f"  [dim]52W High: {format_price(high52)} đ  ·  52W Low: {format_price(low52)} đ[/dim]\n"
        f"  [dim]Volume: {volume:,.0f}  ·  Ngày: {df.index[-1].strftime('%d/%m/%Y')}[/dim]",
        border_style="dim", padding=(0, 1),
    ))

    score     = result["composite_score"]
    sig       = result["signal"]
    sig_color = result["signal_color"]
    console.print(
        f"\n  [bold]ĐIỂM TỔNG HỢP:[/bold] {score_bar(score, 20)} "
        f"[bold]{score:.1f}/100[/bold]  →  [{sig_color}]{sig}[/{sig_color}]\n"
    )

    groups = [
        ("🔵 Xu hướng",    "trend",        0.30),
        ("🟡 Động lượng",  "momentum",     0.25),
        ("🟢 Volume",      "volume",       0.20),
        ("🔴 Biến động",   "volatility",   0.15),
        ("⚪ Price Action", "price_action", 0.10),
    ]
    score_table = Table(box=box.SIMPLE, show_header=True, header_style="bold", min_width=80)
    score_table.add_column("Nhóm chỉ báo",   width=22)
    score_table.add_column("Score",          width=16, justify="center")
    score_table.add_column("Điểm",           width=8,  justify="right")
    score_table.add_column("Trọng số",       width=10, justify="right")
    score_table.add_column("Tín hiệu chính", width=40)

    for label, key, wt in groups:
        s         = (result["scores"][key] + 100) / 2
        sigs_text = " · ".join(result["signals"][key][:2]) if result["signals"][key] else "—"
        score_table.add_row(label, score_bar(s, 12), f"{s:.0f}", f"{wt*100:.0f}%",
                            f"[dim]{sigs_text}[/dim]")
    console.print(score_table)

    console.print(f"  [bold]QUẢN LÝ RỦI RO:[/bold]")
    rr       = result["risk_reward"]
    rr_color = "green" if rr >= 2 else ("yellow" if rr >= 1.5 else "red")

    rm_table = Table(box=box.SIMPLE, show_header=False, min_width=60)
    rm_table.add_column("Label", style="dim", width=22)
    rm_table.add_column("Value", width=22)
    rm_table.add_column("Extra", width=20)
    rm_table.add_row("Giá hiện tại",  f"[bold]{format_price(price)} đ[/bold]", "")
    rm_table.add_row("Stop Loss",     f"[red]{format_price(result['stop_loss'])} đ[/red]",
                     f"[red]({(result['stop_loss']/price-1)*100:.1f}%)[/red]")
    rm_table.add_row("Mục tiêu 1",   f"[green]{format_price(result['target_1'])} đ[/green]",
                     f"[green](+{(result['target_1']/price-1)*100:.1f}%)[/green]")
    rm_table.add_row("Mục tiêu 2",   f"[green]{format_price(result['target_2'])} đ[/green]",
                     f"[green](+{(result['target_2']/price-1)*100:.1f}%)[/green]")
    rm_table.add_row("Tỉ lệ R:R",    f"[{rr_color}]{rr:.2f}[/{rr_color}]",
                     f"[{rr_color}]{'✓ Tốt' if rr>=2 else '~ Chấp nhận' if rr>=1.5 else '✗ Kém'}[/{rr_color}]")
    console.print(rm_table)

    console.print(f"\n  [bold]CHỈ SỐ KỸ THUẬT:[/bold]")
    ki_table = Table(box=box.SIMPLE, show_header=False, min_width=80)
    for _ in range(6):
        ki_table.add_column(width=13)

    rsi_v = result["rsi"]; adx_v = result["adx"]; mfi_v = result["mfi"]
    vol_v = result["vol_ratio"]; atr_v = result["atr_pct"]; bb_v = result["bb_pct"]

    ki_table.add_row(
        "[dim]RSI(14)[/dim]",
        f"[{'red' if rsi_v>65 else 'green' if rsi_v<35 else 'white'}]{rsi_v:.1f}[/]",
        "[dim]ADX(14)[/dim]",
        f"[{'green' if adx_v>25 else 'white'}]{adx_v:.1f}[/]",
        "[dim]MFI(14)[/dim]",
        f"[{'red' if mfi_v>80 else 'green' if mfi_v<25 else 'white'}]{mfi_v:.1f}[/]",
    )
    ki_table.add_row(
        "[dim]BB %B[/dim]",    f"{bb_v:.2f}",
        "[dim]ATR %[/dim]",    f"{atr_v:.1f}%",
        "[dim]Vol Ratio[/dim]", f"[{'green' if vol_v>1.5 else 'white'}]{vol_v:.2f}x[/]",
    )
    console.print(ki_table)

    console.print(f"\n  [bold]TÍN HIỆU CHI TIẾT:[/bold]")
    all_sigs = [f"  [dim]·[/dim] {s}" for sigs in result["signals"].values() for s in sigs]
    for i in range(0, len(all_sigs), 2):
        line = all_sigs[i] + ("    " + all_sigs[i+1] if i+1 < len(all_sigs) else "")
        console.print(line)


def print_legend():
    console.print()
    console.print(Panel(
        "[bold cyan]HƯỚNG DẪN ĐỌC TÍN HIỆU[/bold cyan]\n\n"
        "[bold green]STRONG BUY[/bold green]  Score ≥75  ·  Đủ điều kiện vào lệnh mạnh\n"
        "[green]BUY[/green]         Score ≥55  ·  Tín hiệu mua, xác nhận thêm trước khi vào\n"
        "[yellow]HOLD[/yellow]        Score ≥40  ·  Giữ vị thế, chưa rõ ràng\n"
        "[dark_orange]WATCH[/dark_orange]       Score ≥25  ·  Theo dõi, nguy cơ giảm\n"
        "[red]SELL[/red]        Score <25  ·  Tránh hoặc cắt lỗ\n\n"
        "[dim]⚠ Phân tích kỹ thuật chỉ mang tính tham khảo. "
        "Không phải lời khuyên đầu tư.[/dim]",
        border_style="dim", padding=(0, 2),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CHÍNH
# ─────────────────────────────────────────────────────────────────────────────

def analyze_symbol(symbol: str) -> dict:
    """Phân tích 1 mã. Luôn trả về dict, có key 'error' nếu thất bại."""
    try:
        df = fetch_stock_data(symbol)
    except RuntimeError as e:
        return {"error": str(e), "symbol": symbol}

    try:
        ind          = calculate_indicators(df)
        result       = score_stock(df, ind)
        chg, chg_pct = get_price_change(df)

        return {
            "symbol":    symbol,
            "price":     df["close"].iloc[-1],
            "chg":       chg,
            "chg_pct":   chg_pct,
            "score":     result["composite_score"],
            "signal":    result["signal"],
            "rsi":       result["rsi"],
            "adx":       result["adx"],
            "mfi":       result["mfi"],
            "vol_ratio": result["vol_ratio"],
            "rr":        result["risk_reward"],
            "atr_pct":   result["atr_pct"],
            "_result":   result,
            "_df":       df,
        }
    except Exception as e:
        return {"error": f"Lỗi tính toán: {e}", "symbol": symbol}


def run_scan(symbols: list[str], top_n: Optional[int], detail_symbol: Optional[str]):
    print_header()

    results = []
    errors  = []

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=30),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        transient=True,
    ) as progress:
        task = progress.add_task("Đang tải dữ liệu...", total=len(symbols))

        for sym in symbols:
            progress.update(task, description=f"[bold]{sym}[/bold]  đang tải...")
            r = analyze_symbol(sym)
            if "error" in r:
                errors.append(r)
            else:
                results.append(r)
            progress.advance(task)

    if not results:
        console.print("[red bold]Không lấy được dữ liệu cho bất kỳ mã nào.[/red bold]")
        console.print(
            "[dim]Kiểm tra:\n"
            "  1. Kết nối internet\n"
            "  2. API key vnstock: python -c \"import vnai; vnai.setup_api_key('KEY_CUA_BAN')\"\n"
            "  3. Mã CK hợp lệ trên HOSE/HNX[/dim]"
        )
        if errors:
            console.print("\n[dim]Lỗi chi tiết (5 đầu tiên):[/dim]")
            for e in errors[:5]:
                console.print(f"  [red]{e['symbol']}[/red]: [dim]{e['error'][:120]}[/dim]")
        return

    print_summary_table(results, top_n=top_n)

    buy_cnt  = sum(1 for r in results if "BUY"  in r["signal"])
    sell_cnt = sum(1 for r in results if r["signal"] == "SELL")
    hold_cnt = sum(1 for r in results if r["signal"] in ("HOLD", "WATCH"))
    top3     = sorted(results, key=lambda x: x["score"], reverse=True)[:3]

    console.print(
        f"\n  [dim]Tổng:[/dim] {len(results)} mã  "
        f"[green]▲ {buy_cnt} BUY[/green]  "
        f"[yellow]— {hold_cnt} HOLD[/yellow]  "
        f"[red]▼ {sell_cnt} SELL[/red]"
    )
    console.print(
        "  [dim]Top picks:[/dim] " +
        "  ·  ".join(f"[bold]{r['symbol']}[/bold] [green]{r['score']:.0f}[/green]" for r in top3)
    )

    if errors:
        console.print(
            f"\n  [dim]Không tải được ({len(errors)} mã): "
            + ", ".join(e["symbol"] for e in errors) + "[/dim]"
        )

    if detail_symbol:
        sym_upper = detail_symbol.upper()
        found = next((r for r in results if r["symbol"] == sym_upper), None)
        if found:
            print_detail_analysis(sym_upper, found["_df"], found["_result"])
        else:
            r = analyze_symbol(sym_upper)
            if "error" not in r:
                print_detail_analysis(sym_upper, r["_df"], r["_result"])
            else:
                console.print(f"\n[yellow]Không thể tải chi tiết cho {sym_upper}: {r['error'][:80]}[/yellow]")

    print_legend()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VN Stock Analyst Bot · Phân tích kỹ thuật thị trường VN (LIVE DATA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python vn_stock_bot.py                     # Quét toàn bộ watchlist\n"
            "  python vn_stock_bot.py -s ACB              # Phân tích chi tiết ACB\n"
            "  python vn_stock_bot.py -t 10               # Top 10 cổ phiếu tốt nhất\n"
            "  python vn_stock_bot.py -w VCB ACB HPG      # Watchlist tùy chỉnh\n"
            "  python vn_stock_bot.py -S KBS VCI          # Ưu tiên nguồn KBS\n"
        )
    )
    parser.add_argument("--symbol",    "-s", type=str,    help="Phân tích chi tiết 1 mã (vd: ACB)")
    parser.add_argument("--top",       "-t", type=int,    help="Chỉ hiển thị top N mã tốt nhất")
    parser.add_argument("--watchlist", "-w", nargs="+",   help="Danh sách mã tùy chỉnh")
    parser.add_argument("--no-detail",       action="store_true", help="Bỏ qua phân tích chi tiết")
    parser.add_argument("--source",    "-S", nargs="+",
                        choices=["VCI", "KBS", "vci", "kbs"],
                        default=["VCI", "KBS"],
                        help="Nguồn dữ liệu theo thứ tự ưu tiên (mặc định: VCI KBS)")
    args = parser.parse_args()

    global DATA_SOURCES
    DATA_SOURCES = [s.upper() for s in args.source]

    symbols = [s.upper() for s in args.watchlist] if args.watchlist else DEFAULT_WATCHLIST
    if args.symbol and args.symbol.upper() not in symbols:
        symbols = [args.symbol.upper()] + symbols

    detail_sym = (
        None if args.no_detail
        else args.symbol or (symbols[0] if args.top else None)
    )

    run_scan(symbols=symbols, top_n=args.top, detail_symbol=detail_sym)


if __name__ == "__main__":
    main()
