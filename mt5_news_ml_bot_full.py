"""
MT5 ML Reversal Bot — Single File, Interactive Setup, MarketWatch Symbols
- Terminal prompts for config (no file editing required)
- Optional JSON save/load for setup values (including Telegram)
- Auto-select symbols from Market Watch (or manual CSV)
- ML training if models missing (RandomForest)
- Reversal logic: Sweep -> CHoCH -> Confirm + Wick setups A/B
- Gold vs FX gates: spread/ATR thresholds differ
- Confidence scaling: Gold looser thresholds than FX
- Controlled pyramiding (scale-in): multiple positions on SAME symbol only when in profit
- Session filter, max trades/day, cooldown, min-hold bars before reversal
- Live trading or alerts-only mode
- Telegram heartbeat + scan summary + top skip reasons

NOTE:
- Storing Telegram token in JSON saves time but is plaintext. Keep the JSON private.
"""

import os
import json
import time
import math
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# =========================================================
# Utilities
# =========================================================

def now_utc():
    return datetime.now(timezone.utc)

def prompt_str(msg, default=""):
    s = input(f"{msg} [{default}]: ").strip()
    return s if s else default

def prompt_int(msg, default):
    while True:
        s = input(f"{msg} [{default}]: ").strip()
        if not s:
            return int(default)
        try:
            return int(s)
        except:
            print("Enter a whole number.")

def prompt_float(msg, default):
    while True:
        s = input(f"{msg} [{default}]: ").strip()
        if not s:
            return float(default)
        try:
            return float(s)
        except:
            print("Enter a number (e.g., 0.005).")

def prompt_yesno(msg, default="no"):
    d = default.strip().lower()
    s = input(f"{msg} (yes/no) [{default}]: ").strip().lower()
    if not s:
        s = d
    return s in ("y", "yes", "true", "1")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# =========================================================
# Config dataclass (filled via terminal prompts or JSON)
# =========================================================

@dataclass
class Config:
    # Telegram
    tg_token: str
    tg_chat_id: str

    # runtime
    live_trading: bool
    chatty: bool

    # symbols & timeframes
    use_marketwatch: bool
    symbols_csv: str
    entry_tf: str
    bias_tf: str

    # risk & trade rules
    risk_per_trade: float              # base entry risk fraction (0.005 = 0.5%)
    max_trades_per_day: int
    cooldown_seconds: int
    min_bars_before_reversal: int

    # session filter
    use_session_filter: bool
    session_start_hour: int
    session_end_hour: int

    # portfolio / positions behavior
    one_symbol_at_a_time: bool         # if True, bot will not open trades on multiple symbols simultaneously
    allow_multiple_positions_per_symbol: bool  # pyramiding uses multiple positions on same symbol

    # per-asset gates (Gold vs FX)
    fx_max_spread_points: int
    gold_max_spread_points: int
    fx_min_atr_points: float
    gold_min_atr_points: float

    # SL/TP ATR
    sl_atr: float
    tp_atr: float

    # structure & setup params
    ema_fast: int
    ema_slow: int
    swing_lookback: int
    level_atr_dist: float
    wick_pct: float
    body_max_pct: float

    # confidence scaling (probability thresholds)
    fx_enter_th: float
    gold_enter_th: float
    fx_reverse_th: float
    gold_reverse_th: float
    fx_reverse_margin: float
    gold_reverse_margin: float

    # pyramiding (scale-in) settings
    scale_in_enabled: bool
    max_positions_per_symbol: int              # total positions (including base)
    scale_in_min_profit_atr: float             # only add after price moves >= this * ATR in your favor
    scale_in_min_seconds_between_adds: int     # prevents spam adds
    scale_in_confidence_bump: float            # add only if p_dir >= enter_th + bump
    scale_in_risk_factors_csv: str             # e.g. "1.0,0.5,0.3" base then adds
    max_total_risk_per_symbol: float           # cap (e.g., 0.01 = 1%)

    # heartbeat
    heartbeat_seconds: int
    scan_lines_max: int

    # training
    history_bars_train: int
    live_bars: int
    horizon: int

    # model files
    model_long_path: str
    model_short_path: str

    # magic/deviation
    magic: int
    deviation: int

    # config json
    config_json_path: str


# =========================================================
# Telegram
# =========================================================

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = (token or "").strip()
        self.chat_id = (str(chat_id) if chat_id is not None else "").strip()

    @property
    def enabled(self):
        return bool(self.token) and bool(self.chat_id)

    def send(self, msg: str):
        msg = str(msg)
        if not self.enabled:
            print("[TG DISABLED]", msg)
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            r = requests.post(url, json={"chat_id": self.chat_id, "text": msg[:3900]}, timeout=10)
            return r.status_code == 200
        except Exception as e:
            print("[TG ERROR]", e)
            return False


# =========================================================
# MT5 helpers
# =========================================================

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}

def mt5_init_or_die():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

def account_info_or_die():
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("MT5 account_info unavailable. Is MT5 logged in?")
    return acc

def marketwatch_symbols():
    syms = mt5.symbols_get()
    if not syms:
        return []
    return [s.name for s in syms if getattr(s, "visible", False)]

def symbol_select(symbol: str) -> bool:
    try:
        return bool(mt5.symbol_select(symbol, True))
    except:
        return False

def fetch_rates(symbol: str, tf: str, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, TF_MAP[tf], 0, count)
    if rates is None or len(rates) < 300:
        raise RuntimeError(f"Not enough rates {symbol} {tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["time", "open", "high", "low", "close", "volume"]]

def get_spread_points(symbol: str) -> int:
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None or info.point == 0:
        return 999999
    return int(round((tick.ask - tick.bid) / info.point))

def atr_to_points(symbol: str, atr_val: float) -> float:
    info = mt5.symbol_info(symbol)
    if info is None or info.point == 0:
        return 0.0
    return float(atr_val / info.point)

def positions_by_magic(magic: int):
    pos = mt5.positions_get()
    if pos is None:
        return []
    return [p for p in pos if getattr(p, "magic", None) == magic]

def positions_by_magic_and_symbol(magic: int, symbol: str):
    return [p for p in positions_by_magic(magic) if p.symbol == symbol]

def position_side(pos) -> str:
    return "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"

def avg_entry_price(positions) -> float:
    if not positions:
        return 0.0
    vol = sum(float(p.volume) for p in positions)
    if vol <= 0:
        return 0.0
    return sum(float(p.price_open) * float(p.volume) for p in positions) / vol

def current_price_for_side(symbol: str, side: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 0.0
    return float(tick.bid) if side == "BUY" else float(tick.ask)  # for PnL check, use exit-side price

def build_sl_tp_from_atr(symbol: str, direction: str, atr_val: float, sl_atr: float, tp_atr: float):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("No tick")
    if direction == "BUY":
        entry = tick.ask
        sl = entry - sl_atr * atr_val
        tp = entry + tp_atr * atr_val
        sl_dist = entry - sl
    else:
        entry = tick.bid
        sl = entry + sl_atr * atr_val
        tp = entry - tp_atr * atr_val
        sl_dist = sl - entry
    return float(entry), float(sl), float(tp), float(sl_dist)

def calc_lot_size(symbol: str, sl_price_distance: float, risk_fraction: float) -> float:
    acc = mt5.account_info()
    info = mt5.symbol_info(symbol)
    if acc is None or info is None:
        return 0.01

    risk_amount = float(acc.balance) * float(risk_fraction)

    tick_value = float(getattr(info, "trade_tick_value", 0.0))
    tick_size = float(getattr(info, "trade_tick_size", 0.0))
    if tick_value <= 0 or tick_size <= 0:
        return max(float(getattr(info, "volume_min", 0.01)), 0.01)

    value_per_price_unit = tick_value / tick_size
    loss_per_lot = sl_price_distance * value_per_price_unit
    if loss_per_lot <= 0:
        return max(float(getattr(info, "volume_min", 0.01)), 0.01)

    lots = risk_amount / loss_per_lot
    vmin = float(getattr(info, "volume_min", 0.01))
    vmax = float(getattr(info, "volume_max", 100.0))
    step = float(getattr(info, "volume_step", 0.01))

    lots = clamp(lots, vmin, vmax)
    lots = math.floor(lots / step) * step
    return float(round(max(vmin, lots), 2))

def place_order(symbol: str, direction: str, sl: float, tp: float, lots: float, magic: int, deviation: int, comment: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("No tick")

    if direction == "BUY":
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": int(deviation),
        "magic": int(magic),
        "comment": str(comment)[:30],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"Order failed ret={result.retcode} comment={result.comment}")
    return result

def close_position(pos, magic: int, deviation: int):
    symbol = pos.symbol
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError("No tick to close")

    if pos.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "position": pos.ticket,
        "volume": float(pos.volume),
        "type": order_type,
        "price": float(price),
        "deviation": int(deviation),
        "magic": int(magic),
        "comment": "MLbot close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"close order None: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"Close failed ret={result.retcode} comment={result.comment}")
    return result

def close_all_positions_for_symbol(symbol: str, magic: int, deviation: int):
    ps = positions_by_magic_and_symbol(magic, symbol)
    for p in ps:
        close_position(p, magic, deviation)


# =========================================================
# Gold vs FX helpers (Gates + Confidence scaling)
# =========================================================

def is_gold_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return s.startswith("XAU") or s == "XAUUSD" or s == "XAUUSDm"

def max_spread_for_symbol(symbol: str, cfg: Config) -> int:
    return cfg.gold_max_spread_points if is_gold_symbol(symbol) else cfg.fx_max_spread_points

def min_atr_for_symbol(symbol: str, cfg: Config) -> float:
    return cfg.gold_min_atr_points if is_gold_symbol(symbol) else cfg.fx_min_atr_points

def enter_th_for_symbol(symbol: str, cfg: Config) -> float:
    return cfg.gold_enter_th if is_gold_symbol(symbol) else cfg.fx_enter_th

def reverse_th_for_symbol(symbol: str, cfg: Config) -> float:
    return cfg.gold_reverse_th if is_gold_symbol(symbol) else cfg.fx_reverse_th

def reverse_margin_for_symbol(symbol: str, cfg: Config) -> float:
    return cfg.gold_reverse_margin if is_gold_symbol(symbol) else cfg.fx_reverse_margin


# =========================================================
# Indicators & structure features
# =========================================================

def ema(series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period: int = 14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index).rolling(period).mean()
    loss = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, period: int = 14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], cfg.ema_fast)
    df["ema_slow"] = ema(df["close"], cfg.ema_slow)
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / (df["close"] + 1e-9)
    df["rsi_14"] = rsi(df["close"], 14)
    df["atr_14"] = atr(df, 14)
    df["atr_pct"] = df["atr_14"] / (df["close"] + 1e-9)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = rng
    df["body_pct"] = body / rng
    df["upper_wick_pct"] = upper_wick / rng
    df["lower_wick_pct"] = lower_wick / rng
    return df

def add_structure(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    lb = int(cfg.swing_lookback)
    df["swing_high"] = df["high"].rolling(lb).max()
    df["swing_low"] = df["low"].rolling(lb).min()

    pivot = max(5, lb // 5)
    df["pivot_high"] = df["high"].rolling(pivot).max()
    df["pivot_low"] = df["low"].rolling(pivot).min()

    df["last_lower_high"] = df["pivot_high"].shift(1)
    df["last_higher_low"] = df["pivot_low"].shift(1)

    df["dist_swing_high_atr"] = (df["swing_high"] - df["close"]) / (df["atr_14"] + 1e-9)
    df["dist_swing_low_atr"] = (df["close"] - df["swing_low"]) / (df["atr_14"] + 1e-9)
    return df

def add_reversal_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sweep_low"] = (
        (df["low"] < df["swing_low"].shift(1)) &
        (df["close"] > df["swing_low"].shift(1))
    ).astype(int)

    df["sweep_high"] = (
        (df["high"] > df["swing_high"].shift(1)) &
        (df["close"] < df["swing_high"].shift(1))
    ).astype(int)

    df["choch_bull"] = (df["close"] > df["last_lower_high"]).astype(int)
    df["choch_bear"] = (df["close"] < df["last_higher_low"]).astype(int)

    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_pct = (body / rng).fillna(0)

    df["bull_confirm"] = ((df["close"] > df["open"]) & (body_pct >= 0.55)).astype(int)
    df["bear_confirm"] = ((df["close"] < df["open"]) & (body_pct >= 0.55)).astype(int)

    return df

def build_frame(symbol: str, tf: str, bars: int, cfg: Config) -> pd.DataFrame:
    df = fetch_rates(symbol, tf, bars)
    df = add_indicators(df, cfg)
    df = add_structure(df, cfg)
    df = add_reversal_flags(df)
    df = df.dropna().reset_index(drop=True)
    return df

def bias_from_tf(df_bias: pd.DataFrame) -> str:
    last = df_bias.iloc[-1]
    if last["ema_fast"] > last["ema_slow"]:
        return "BUY"
    if last["ema_fast"] < last["ema_slow"]:
        return "SELL"
    return "NONE"

def detect_wick_setups(df: pd.DataFrame, cfg: Config):
    """
    Uses last two CLOSED candles:
      c = signal candle (-2), n = confirm candle (-1)
    Returns (setupA_long, setupB_short)
    """
    if len(df) < cfg.swing_lookback + 5:
        return False, False

    c = df.iloc[-2]
    n = df.iloc[-1]
    rng = max(float(c["high"] - c["low"]), 1e-9)

    body_pct = float(c["body_pct"])
    upper_wick_pct = float(c["upper_wick_pct"])
    lower_wick_pct = float(c["lower_wick_pct"])
    atr_val = float(c["atr_14"]) if not np.isnan(c["atr_14"]) else 0.0

    recent = df.iloc[-(cfg.swing_lookback + 2):-2]
    near_support = float(c["low"]) <= float(recent["low"].min()) + cfg.level_atr_dist * atr_val
    near_resist  = float(c["high"]) >= float(recent["high"].max()) - cfg.level_atr_dist * atr_val

    bullish_confirm = (n["close"] > n["open"]) and (n["close"] > c["high"] - 0.2 * rng)
    bearish_confirm = (n["close"] < n["open"]) and (n["close"] < c["low"] + 0.2 * rng)

    setup_A = (lower_wick_pct >= cfg.wick_pct) and (body_pct <= cfg.body_max_pct) and near_support and bullish_confirm
    setup_B = (upper_wick_pct >= cfg.wick_pct) and (body_pct <= cfg.body_max_pct) and near_resist and bearish_confirm

    return bool(setup_A), bool(setup_B)

def reversal_signals(df_entry: pd.DataFrame, cfg: Config):
    """
    Uses last TWO closed candles:
      signal candle = -2
      confirm candle = -1
    """
    c = df_entry.iloc[-2]
    n = df_entry.iloc[-1]

    bull_reversal = (int(c["sweep_low"]) == 1) and (int(n["choch_bull"]) == 1) and (int(n["bull_confirm"]) == 1)
    bear_reversal = (int(c["sweep_high"]) == 1) and (int(n["choch_bear"]) == 1) and (int(n["bear_confirm"]) == 1)

    setupA_long, setupB_short = detect_wick_setups(df_entry, cfg)

    return {
        "bull_reversal": bool(bull_reversal),
        "bear_reversal": bool(bear_reversal),
        "setupA_long": bool(setupA_long),
        "setupB_short": bool(setupB_short),
    }


# =========================================================
# ML: Features + labels + training
# =========================================================

FEATURE_COLS = [
    "ema_diff", "rsi_14", "atr_pct",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "dist_swing_low_atr", "dist_swing_high_atr",
    "sweep_low", "sweep_high",
    "choch_bull", "choch_bear",
    "bull_confirm", "bear_confirm",
]

def build_feature_row(df_entry: pd.DataFrame) -> np.ndarray:
    # Use signal candle (-2)
    return df_entry.iloc[-2][FEATURE_COLS].values.astype(float).reshape(1, -1)

def label_outcome(df: pd.DataFrame, i: int, side: str, sl_atr: float, tp_atr: float, horizon: int) -> int:
    """
    Label=1 if TP hit before SL within horizon candles after entry.
    Entry assumed at next candle open (i+1).
    Worst-case ordering: if both SL/TP touched in one candle, assume SL first.
    """
    if i + 1 >= len(df) - 2:
        return 0

    entry = float(df.loc[i + 1, "open"])
    atr_val = float(df.loc[i, "atr_14"])
    if atr_val <= 0 or np.isnan(atr_val):
        return 0

    if side == "LONG":
        sl = entry - sl_atr * atr_val
        tp = entry + tp_atr * atr_val
    else:
        sl = entry + sl_atr * atr_val
        tp = entry - tp_atr * atr_val

    end = min(len(df) - 1, i + 1 + horizon)
    for j in range(i + 1, end + 1):
        hi = float(df.loc[j, "high"])
        lo = float(df.loc[j, "low"])

        if side == "LONG":
            hit_sl = lo <= sl
            hit_tp = hi >= tp
        else:
            hit_sl = hi >= sl
            hit_tp = lo <= tp

        if hit_sl:
            return 0
        if hit_tp:
            return 1
    return 0

def train_models_if_needed(symbols, cfg: Config, tg: Telegram, set_status):
    if os.path.exists(cfg.model_long_path) and os.path.exists(cfg.model_short_path):
        tg.send("✅ Models found. Skipping training.")
        return

    set_status("Training models (models not found).")
    tg.send("🧠 Models not found. Starting training from MT5 history...")

    X_long, y_long = [], []
    X_short, y_short = [], []

    train_syms = symbols[:min(len(symbols), 12)]

    for sym in train_syms:
        try:
            df = fetch_rates(sym, cfg.entry_tf, cfg.history_bars_train)
            df = add_indicators(df, cfg)
            df = add_structure(df, cfg)
            df = add_reversal_flags(df)
            df = df.dropna().reset_index(drop=True)

            for i in range(cfg.swing_lookback + 10, len(df) - (cfg.horizon + 5)):
                window = df.iloc[:i+1].copy()
                sigs = reversal_signals(window, cfg)
                feats = window.iloc[-2][FEATURE_COLS].values.astype(float)

                if sigs["bull_reversal"] or sigs["setupA_long"]:
                    y = label_outcome(df, i-1, "LONG", cfg.sl_atr, cfg.tp_atr, cfg.horizon)
                    X_long.append(feats)
                    y_long.append(y)

                if sigs["bear_reversal"] or sigs["setupB_short"]:
                    y = label_outcome(df, i-1, "SHORT", cfg.sl_atr, cfg.tp_atr, cfg.horizon)
                    X_short.append(feats)
                    y_short.append(y)

            tg.send(f"📚 {sym}: samples LONG={len(y_long)} SHORT={len(y_short)}")
        except Exception as e:
            tg.send(f"⚠️ Training skipped for {sym}: {e}")

    if len(y_long) < 200 or len(y_short) < 200:
        raise RuntimeError(
            f"Insufficient training samples. LONG={len(y_long)} SHORT={len(y_short)}. "
            f"Increase history_bars_train or loosen wick/structure params."
        )

    def train_one(X, y, out_path, name):
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=12,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rep = classification_report(y_test, preds, digits=3)
        tg.send(f"🧠 {name} model trained.\n{rep[:3000]}")
        joblib.dump(model, out_path)
        tg.send(f"✅ Saved: {out_path}")

    train_one(X_long, y_long, cfg.model_long_path, "LONG")
    train_one(X_short, y_short, cfg.model_short_path, "SHORT")
    tg.send("✅ Training complete.")


# =========================================================
# Decision logic (scan + reverse + scale-in)
# =========================================================

def in_allowed_session_local(cfg: Config) -> bool:
    if not cfg.use_session_filter:
        return True
    h = datetime.now().hour
    return cfg.session_start_hour <= h <= cfg.session_end_hour

def decide_reverse(symbol: str, pos_side: str, sigs: dict, p_long: float, p_short: float, cfg: Config):
    rev_th = reverse_th_for_symbol(symbol, cfg)
    rev_margin = reverse_margin_for_symbol(symbol, cfg)

    if pos_side == "BUY":
        bearish_setup = sigs["bear_reversal"] or sigs["setupB_short"]
        if bearish_setup and p_short >= rev_th and (p_short - p_long) >= rev_margin:
            return "CLOSE_AND_OPEN_SELL"
    else:
        bullish_setup = sigs["bull_reversal"] or sigs["setupA_long"]
        if bullish_setup and p_long >= rev_th and (p_long - p_short) >= rev_margin:
            return "CLOSE_AND_OPEN_BUY"
    return "HOLD"

def parse_risk_factors(cfg: Config):
    parts = [p.strip() for p in (cfg.scale_in_risk_factors_csv or "").split(",") if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except:
            pass
    if not vals:
        vals = [1.0, 0.5, 0.3]
    # ensure base exists
    if len(vals) < 1:
        vals = [1.0]
    return vals

def effective_risk_for_leg(cfg: Config, leg_index: int, factors):
    # leg_index: 0=base, 1=first add, ...
    f = factors[min(leg_index, len(factors)-1)]
    return float(cfg.risk_per_trade) * float(f)

def total_risk_used_for_symbol(cfg: Config, symbol: str, magic: int, factors):
    # approximate: each open position counts as one leg risk factor (order of opens unknown on restart)
    ps = positions_by_magic_and_symbol(magic, symbol)
    used = 0.0
    for i, _ in enumerate(ps):
        used += effective_risk_for_leg(cfg, i, factors)
    return used

def can_scale_in(symbol: str, side: str, cfg: Config, df_entry: pd.DataFrame, p_long: float, p_short: float,
                 last_add_time_by_symbol: dict, factors):
    if not cfg.scale_in_enabled:
        return (False, "scale_in disabled")

    if not cfg.allow_multiple_positions_per_symbol:
        return (False, "multiple positions disabled")

    ps = positions_by_magic_and_symbol(cfg.magic, symbol)
    if not ps:
        return (False, "no open position")

    # only scale into same direction
    cur_side = position_side(ps[0])
    if cur_side != side:
        return (False, "side mismatch")

    if len(ps) >= cfg.max_positions_per_symbol:
        return (False, f"max positions reached ({len(ps)}/{cfg.max_positions_per_symbol})")

    # spacing between adds
    last_add = last_add_time_by_symbol.get(symbol, 0)
    if time.time() - last_add < cfg.scale_in_min_seconds_between_adds:
        return (False, "add cooldown")

    # gates still apply
    spr = get_spread_points(symbol)
    if spr > max_spread_for_symbol(symbol, cfg):
        return (False, f"spread too high for add ({spr})")

    atr_val = float(df_entry.iloc[-1]["atr_14"])
    atr_pts = atr_to_points(symbol, atr_val)
    if atr_pts < min_atr_for_symbol(symbol, cfg):
        return (False, f"atr too low for add ({atr_pts:.1f})")

    # only add if price is in profit by >= X * ATR
    avg_entry = avg_entry_price(ps)
    px = current_price_for_side(symbol, cur_side)
    # For BUY: profit if px > avg_entry; for SELL: profit if px < avg_entry
    if atr_val <= 0:
        return (False, "atr invalid")

    if cur_side == "BUY":
        profit_dist = px - avg_entry
        required = cfg.scale_in_min_profit_atr * atr_val
        if profit_dist < required:
            return (False, f"not enough profit to add ({profit_dist:.5f}<{required:.5f})")
    else:
        profit_dist = avg_entry - px
        required = cfg.scale_in_min_profit_atr * atr_val
        if profit_dist < required:
            return (False, f"not enough profit to add ({profit_dist:.5f}<{required:.5f})")

    # confidence bump for adds
    enter_th = enter_th_for_symbol(symbol, cfg)
    bump = float(cfg.scale_in_confidence_bump)
    if cur_side == "BUY":
        if p_long < (enter_th + bump):
            return (False, f"p_long below add threshold ({p_long:.3f}<{enter_th+bump:.3f})")
    else:
        if p_short < (enter_th + bump):
            return (False, f"p_short below add threshold ({p_short:.3f}<{enter_th+bump:.3f})")

    # risk cap per symbol
    used = total_risk_used_for_symbol(cfg, symbol, cfg.magic, factors)
    next_leg_idx = len(ps)  # next leg index = existing count
    next_risk = effective_risk_for_leg(cfg, next_leg_idx, factors)
    if used + next_risk > cfg.max_total_risk_per_symbol:
        return (False, f"risk cap reached ({used+next_risk:.4f}>{cfg.max_total_risk_per_symbol:.4f})")

    return (True, "ok")

def score_symbol(sym: str, cfg: Config, model_long, model_short):
    # Spread gate
    spread = get_spread_points(sym)
    max_spread = max_spread_for_symbol(sym, cfg)
    if spread > max_spread:
        return {"symbol": sym, "ok": False, "reason": f"Spread too high ({spread}>{max_spread})"}

    df_entry = build_frame(sym, cfg.entry_tf, cfg.live_bars, cfg)
    df_bias = build_frame(sym, cfg.bias_tf, cfg.live_bars, cfg)

    bias = bias_from_tf(df_bias)
    sigs = reversal_signals(df_entry, cfg)

    # ATR gate
    atr_val = float(df_entry.iloc[-1]["atr_14"])
    ap = atr_to_points(sym, atr_val)
    min_atr = min_atr_for_symbol(sym, cfg)
    if ap < min_atr:
        return {"symbol": sym, "ok": False, "reason": f"ATR too low ({ap:.1f}<{min_atr} pts)"}

    # ML probabilities
    x = build_feature_row(df_entry)
    p_long = float(model_long.predict_proba(x)[0, 1])
    p_short = float(model_short.predict_proba(x)[0, 1])

    # confidence scaling
    enter_th = enter_th_for_symbol(sym, cfg)

    long_setup = sigs["bull_reversal"] or sigs["setupA_long"]
    short_setup = sigs["bear_reversal"] or sigs["setupB_short"]

    long_ready = long_setup and (p_long >= enter_th) and (bias in ["NONE", "BUY"])
    short_ready = short_setup and (p_short >= enter_th) and (bias in ["NONE", "SELL"])

    options = []
    if long_ready:
        options.append(("BUY", p_long))
    if short_ready:
        options.append(("SELL", p_short))

    best = None
    if options:
        best_dir, best_p = sorted(options, key=lambda t: t[1], reverse=True)[0]
        best_score = abs(best_p - 0.5)
        best = (best_dir, best_p, best_score)

    return {
        "symbol": sym, "ok": True,
        "spread": spread, "bias": bias,
        "atr_pts": ap, "atr_val": atr_val,
        "p_long": p_long, "p_short": p_short,
        "signals": sigs, "best": best,
        "df_entry": df_entry,
        "enter_th_used": enter_th,
        "max_spread_used": max_spread,
        "min_atr_used": min_atr
    }


# =========================================================
# JSON config helpers
# =========================================================

def load_config_json(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config_json(path: str, cfg: Config):
    data = asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def cfg_from_dict(d: dict) -> Config:
    # allow missing keys by providing safe defaults
    base = interactive_setup(use_json=False)  # build default baseline quickly
    base_dict = asdict(base)
    base_dict.update(d or {})
    return Config(**base_dict)


# =========================================================
# Interactive setup
# =========================================================

def interactive_setup(use_json=True) -> Config:
    print("\n=== MT5 ML BOT SETUP ===\n")

    # JSON load
    config_path = "bot_config.json"
    if use_json and os.path.exists(config_path):
        if prompt_yesno(f"Config file found ({config_path}). Load it?", "yes"):
            try:
                d = load_config_json(config_path)
                cfg = cfg_from_dict(d)
                cfg.config_json_path = config_path
                print("✅ Loaded config from JSON.")
                # You can still override a few key items quickly:
                if prompt_yesno("Override LIVE trading / timeframes quickly?", "no"):
                    cfg.live_trading = prompt_yesno("Enable LIVE trading? (no = alerts only)", "no")
                    cfg.entry_tf = prompt_str("Entry timeframe (M1/M5/M15/M30/H1)", cfg.entry_tf).upper()
                    cfg.bias_tf = prompt_str("Bias timeframe (M5/M15/M30/H1)", cfg.bias_tf).upper()
                return cfg
            except Exception as e:
                print("⚠️ Failed to load JSON, switching to manual setup:", e)

    # Telegram
    tg_token = prompt_str("Telegram bot token (empty disables Telegram)", "")
    tg_chat_id = prompt_str("Telegram chat ID (empty disables Telegram)", "")

    live_trading = prompt_yesno("Enable LIVE trading? (no = alerts only)", "no")
    chatty = prompt_yesno("Chatty mode? (more updates / reasons)", "yes")

    use_mw = prompt_yesno("Use symbols from MT5 Market Watch automatically?", "yes")
    symbols_csv = ""
    if not use_mw:
        symbols_csv = prompt_str("Symbols CSV (example: XAUUSD,EURUSD,USDJPY)", "EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD")

    entry_tf = prompt_str("Entry timeframe (M1/M5/M15/M30/H1)", "M5").upper()
    bias_tf = prompt_str("Bias timeframe (M5/M15/M30/H1)", "M15").upper()
    if entry_tf not in TF_MAP or bias_tf not in TF_MAP:
        raise RuntimeError("Unsupported timeframe. Use: " + ", ".join(TF_MAP.keys()))

    # Risk
    risk = prompt_float("Risk per BASE trade (0.005 = 0.5%)", 0.005)
    max_day = prompt_int("Max trades per day", 3)
    cooldown = prompt_int("Cooldown seconds between actions", 90)
    min_bars_rev = prompt_int("Min bars before reversal (anti-chop)", 3)

    # Session
    use_session = prompt_yesno("Use session filter (trade only certain hours)?", "yes")
    start_h = prompt_int("Session start hour (local time, 0-23)", 7)
    end_h = prompt_int("Session end hour (local time, 0-23)", 23)

    # Portfolio behavior
    one_symbol = prompt_yesno("One symbol at a time? (prevents multi-pair exposure)", "yes")
    allow_multi_pos_symbol = prompt_yesno("Allow multiple positions per symbol (controlled pyramiding)?", "yes")

    # Gates
    print("\n=== Per-Asset Filters (Gold vs FX) ===")
    fx_max_spread = prompt_int("FX Max spread (points)", 20)
    gold_max_spread = prompt_int("GOLD Max spread (points) (XAU)", 100)

    fx_min_atr_pts = prompt_float("FX Min ATR (points)", 15)
    gold_min_atr_pts = prompt_float("GOLD Min ATR (points) (XAU)", 25)

    # SL/TP
    sl_atr = prompt_float("StopLoss ATR multiple", 1.2)
    tp_atr = prompt_float("TakeProfit ATR multiple", 1.8)

    # Confidence scaling
    print("\n=== Confidence Scaling (Gold vs FX) ===")
    fx_enter_th = prompt_float("FX Enter probability threshold", 0.61)
    gold_enter_th = prompt_float("GOLD Enter probability threshold (XAU)", 0.58)

    fx_reverse_th = prompt_float("FX Reverse probability threshold", 0.66)
    gold_reverse_th = prompt_float("GOLD Reverse probability threshold (XAU)", 0.62)

    fx_rev_margin = prompt_float("FX Reverse margin", 0.10)
    gold_rev_margin = prompt_float("GOLD Reverse margin (XAU)", 0.06)

    # Pyramiding
    print("\n=== Controlled Pyramiding (Scale-In) ===")
    scale_in_enabled = prompt_yesno("Enable controlled pyramiding (adds only in profit)?", "yes" if allow_multi_pos_symbol else "no")
    max_pos = prompt_int("Max positions per symbol (including base)", 3)
    scale_profit_atr = prompt_float("Min profit before add (ATR multiple)", 0.6)
    add_gap = prompt_int("Min seconds between adds", 180)
    conf_bump = prompt_float("Confidence bump required for adds (enter_th + bump)", 0.03)
    risk_factors = prompt_str("Risk factors CSV (base,add1,add2...) e.g. 1.0,0.5,0.3", "1.0,0.5,0.3")
    max_risk_sym = prompt_float("Max total risk per symbol (0.01 = 1%)", 0.010)

    # Heartbeat
    heartbeat = prompt_int("Heartbeat interval seconds", 300)
    scan_lines_max = prompt_int("Heartbeat scan lines to show", 8)

    # Training
    history_train = prompt_int("Training history bars per symbol", 9000)
    live_bars = prompt_int("Live bars to fetch per symbol", 700)
    horizon = prompt_int("Training horizon (candles forward)", 12)

    # Advanced defaults
    ema_fast = 50
    ema_slow = 200
    swing_lookback = 50
    level_atr_dist = 0.25
    wick_pct = 0.45
    body_max_pct = 0.35

    model_long_path = "model_long.pkl"
    model_short_path = "model_short.pkl"
    magic = 20260116
    deviation = 20

    cfg = Config(
        tg_token=tg_token,
        tg_chat_id=tg_chat_id,
        live_trading=live_trading,
        chatty=chatty,

        use_marketwatch=use_mw,
        symbols_csv=symbols_csv,
        entry_tf=entry_tf,
        bias_tf=bias_tf,

        risk_per_trade=risk,
        max_trades_per_day=max_day,
        cooldown_seconds=cooldown,
        min_bars_before_reversal=min_bars_rev,

        use_session_filter=use_session,
        session_start_hour=start_h,
        session_end_hour=end_h,

        one_symbol_at_a_time=one_symbol,
        allow_multiple_positions_per_symbol=allow_multi_pos_symbol,

        fx_max_spread_points=fx_max_spread,
        gold_max_spread_points=gold_max_spread,
        fx_min_atr_points=fx_min_atr_pts,
        gold_min_atr_points=gold_min_atr_pts,

        sl_atr=sl_atr,
        tp_atr=tp_atr,

        ema_fast=ema_fast,
        ema_slow=ema_slow,
        swing_lookback=swing_lookback,
        level_atr_dist=level_atr_dist,
        wick_pct=wick_pct,
        body_max_pct=body_max_pct,

        fx_enter_th=fx_enter_th,
        gold_enter_th=gold_enter_th,
        fx_reverse_th=fx_reverse_th,
        gold_reverse_th=gold_reverse_th,
        fx_reverse_margin=fx_rev_margin,
        gold_reverse_margin=gold_rev_margin,

        scale_in_enabled=scale_in_enabled,
        max_positions_per_symbol=max_pos,
        scale_in_min_profit_atr=scale_profit_atr,
        scale_in_min_seconds_between_adds=add_gap,
        scale_in_confidence_bump=conf_bump,
        scale_in_risk_factors_csv=risk_factors,
        max_total_risk_per_symbol=max_risk_sym,

        heartbeat_seconds=heartbeat,
        scan_lines_max=scan_lines_max,

        history_bars_train=history_train,
        live_bars=live_bars,
        horizon=horizon,

        model_long_path=model_long_path,
        model_short_path=model_short_path,

        magic=magic,
        deviation=deviation,

        config_json_path=config_path
    )

    # Save config?
    if use_json:
        if prompt_yesno(f"Save these settings to {config_path} for next time?", "yes"):
            save_config_json(config_path, cfg)
            print(f"✅ Saved config: {config_path}")

    return cfg


# =========================================================
# Main
# =========================================================

def main():
    cfg = interactive_setup(use_json=True)
    tg = Telegram(cfg.tg_token, cfg.tg_chat_id)

    mt5_init_or_die()
    acc = account_info_or_die()

    # symbols
    if cfg.use_marketwatch:
        symbols = marketwatch_symbols()
    else:
        symbols = [s.strip() for s in (cfg.symbols_csv or "").split(",") if s.strip()]
        symbols = [s.upper() for s in symbols]

    if not symbols:
        raise RuntimeError("No symbols available.")

    ok_syms, bad_syms = [], []
    for s in symbols:
        if symbol_select(s):
            ok_syms.append(s)
        else:
            bad_syms.append(s)

    if not ok_syms:
        raise RuntimeError("No symbols could be selected. Add symbols to Market Watch.")
    if bad_syms:
        tg.send(f"⚠️ Some symbols unavailable: {', '.join(bad_syms)}")

    tg.send(
        "✅ Bot started\n"
        f"Account: {acc.login} | Server: {acc.server}\n"
        f"Balance: {acc.balance}\n"
        f"Mode: {'LIVE TRADING' if cfg.live_trading else 'ALERTS ONLY'}\n"
        f"Symbols: {', '.join(ok_syms[:20])}{'...' if len(ok_syms) > 20 else ''}\n"
        f"EntryTF={cfg.entry_tf} BiasTF={cfg.bias_tf}\n"
        f"Chatty={cfg.chatty} Heartbeat={cfg.heartbeat_seconds}s\n"
        f"FX gate: spr<={cfg.fx_max_spread_points}, ATR>={cfg.fx_min_atr_points}, enter>={cfg.fx_enter_th}\n"
        f"GOLD gate: spr<={cfg.gold_max_spread_points}, ATR>={cfg.gold_min_atr_points}, enter>={cfg.gold_enter_th}\n"
        f"Pyramiding: {cfg.scale_in_enabled} maxPos/sym={cfg.max_positions_per_symbol} maxRisk/sym={cfg.max_total_risk_per_symbol}"
    )

    last_status = "Initialized. First scan pending..."
    last_scan_summary = ""
    last_heartbeat_time = time.time()
    last_action_time = 0

    def set_status(s: str):
        nonlocal last_status
        last_status = s

    # train/load models
    train_models_if_needed(ok_syms, cfg, tg, set_status)
    model_long = joblib.load(cfg.model_long_path)
    model_short = joblib.load(cfg.model_short_path)

    trades_today = 0
    day = now_utc().date()

    # for min-hold & adds
    entry_time_by_symbol = {}         # base entry time per symbol (best-effort)
    last_add_time_by_symbol = {}      # last add time per symbol

    # parse factors once
    factors = parse_risk_factors(cfg)

    tg.send("🚀 Live loop running.")

    while True:
        try:
            # Heartbeat
            if time.time() - last_heartbeat_time >= cfg.heartbeat_seconds:
                tg.send(
                    f"💓 Heartbeat\n"
                    f"UTC: {now_utc():%Y-%m-%d %H:%M:%S}\n"
                    f"Status: {last_status}\n\n"
                    f"Scan:\n{last_scan_summary}"
                )
                last_heartbeat_time = time.time()

            # Session filter
            if not in_allowed_session_local(cfg):
                set_status("Outside session hours (sleeping).")
                time.sleep(30)
                continue

            # Daily reset
            if now_utc().date() != day:
                day = now_utc().date()
                trades_today = 0
                tg.send("🗓 New day: trades/day counter reset.")

            # Cooldown
            if time.time() - last_action_time < cfg.cooldown_seconds:
                time.sleep(2)
                continue

            # Max trades/day
            if trades_today >= cfg.max_trades_per_day:
                set_status(f"Max trades reached ({trades_today}/{cfg.max_trades_per_day}).")
                time.sleep(15)
                continue

            # ---------------------------
            # Manage open positions first
            # ---------------------------
            all_pos = positions_by_magic(cfg.magic)
            open_symbols = sorted(list({p.symbol for p in all_pos}))

            if open_symbols and cfg.one_symbol_at_a_time:
                # only manage the first open symbol (single idea at a time)
                open_symbols = [open_symbols[0]]

            did_action = False

            for sym in open_symbols:
                ps = positions_by_magic_and_symbol(cfg.magic, sym)
                if not ps:
                    continue

                # Determine current side (assume all are same side)
                cur_side = position_side(ps[0])

                # Build frame and probs for management
                df_entry = build_frame(sym, cfg.entry_tf, cfg.live_bars, cfg)
                sigs = reversal_signals(df_entry, cfg)
                x = build_feature_row(df_entry)
                p_long = float(model_long.predict_proba(x)[0, 1])
                p_short = float(model_short.predict_proba(x)[0, 1])

                # min-hold bars protection (time approximation)
                can_reverse = True
                entry_t = entry_time_by_symbol.get(sym)
                if entry_t is not None:
                    bar_minutes = 5 if cfg.entry_tf == "M5" else 15 if cfg.entry_tf == "M15" else 1
                    min_hold = timedelta(minutes=bar_minutes * cfg.min_bars_before_reversal)
                    if now_utc() - entry_t < min_hold:
                        can_reverse = False

                # Reversal decision (close all then open opposite)
                action = decide_reverse(sym, cur_side, sigs, p_long, p_short, cfg)
                if action != "HOLD" and not can_reverse:
                    set_status(f"Reverse blocked (min-hold) {sym} pos={cur_side}")
                    if cfg.chatty:
                        tg.send(f"⛔ Reverse blocked (min-hold)\n{sym} pos={cur_side}\nSignals={sigs}\npL={p_long:.3f} pS={p_short:.3f}")
                    continue

                if action.startswith("CLOSE_AND_OPEN"):
                    new_dir = "BUY" if action.endswith("BUY") else "SELL"
                    set_status(f"Reversing {sym} {cur_side}->{new_dir}")

                    tg.send(
                        f"🔁 Reverse triggered\n"
                        f"{sym} {cfg.entry_tf}\n"
                        f"From {cur_side} to {new_dir}\n"
                        f"Signals={sigs}\n"
                        f"pL={p_long:.3f} pS={p_short:.3f}"
                    )

                    if cfg.live_trading:
                        close_all_positions_for_symbol(sym, cfg.magic, cfg.deviation)

                        atr_val = float(df_entry.iloc[-1]["atr_14"])
                        entry, sl, tp, sl_dist = build_sl_tp_from_atr(sym, new_dir, atr_val, cfg.sl_atr, cfg.tp_atr)

                        # base risk for reversal entry
                        lots = calc_lot_size(sym, sl_dist, cfg.risk_per_trade)
                        place_order(sym, new_dir, sl, tp, lots, cfg.magic, cfg.deviation, comment=f"MLrev {new_dir}")

                        trades_today += 1
                        last_action_time = time.time()
                        entry_time_by_symbol[sym] = now_utc()
                        last_add_time_by_symbol[sym] = time.time()

                        tg.send(
                            f"✅ Reverse trade placed\n"
                            f"{sym} {new_dir} lots={lots}\n"
                            f"Entry~{entry:.5f} SL={sl:.5f} TP={tp:.5f}\n"
                            f"Trades today: {trades_today}/{cfg.max_trades_per_day}"
                        )
                    else:
                        last_action_time = time.time()
                        tg.send("ℹ️ Alerts-only mode: reverse not executed, alert only.")

                    did_action = True
                    break  # action taken, restart loop

                # If no reversal: consider scale-in (adds)
                if cfg.scale_in_enabled and cfg.allow_multiple_positions_per_symbol:
                    # Add only in same direction as current
                    want_side = cur_side
                    ok_add, reason = can_scale_in(
                        sym, want_side, cfg, df_entry, p_long, p_short, last_add_time_by_symbol, factors
                    )
                    if ok_add:
                        # Use existing SL/TP from first position if set, otherwise infer from ATR now.
                        ref_sl = float(ps[0].sl) if ps[0].sl else 0.0
                        ref_tp = float(ps[0].tp) if ps[0].tp else 0.0

                        # If broker didn’t store SL/TP, fallback to ATR-based now (rare).
                        atr_val = float(df_entry.iloc[-1]["atr_14"])
                        tick = mt5.symbol_info_tick(sym)
                        if tick is None:
                            continue

                        if ref_sl == 0.0 or ref_tp == 0.0:
                            _, ref_sl, ref_tp, _ = build_sl_tp_from_atr(sym, want_side, atr_val, cfg.sl_atr, cfg.tp_atr)

                        # Compute risk for this leg (bounded)
                        leg_index = len(ps)  # next leg
                        risk_leg = effective_risk_for_leg(cfg, leg_index, factors)

                        # SL distance for sizing: distance from current entry price to shared SL
                        entry_price = float(tick.ask) if want_side == "BUY" else float(tick.bid)
                        sl_dist = abs(entry_price - ref_sl)
                        lots = calc_lot_size(sym, sl_dist, risk_leg)

                        set_status(f"Scale-in {sym} {want_side} leg={leg_index+1}/{cfg.max_positions_per_symbol}")
                        tg.send(
                            f"➕ Scale-in add\n"
                            f"{sym} side={want_side} leg={leg_index+1}/{cfg.max_positions_per_symbol}\n"
                            f"RiskLeg={risk_leg:.4f} Spread={get_spread_points(sym)}\n"
                            f"pL={p_long:.3f} pS={p_short:.3f}\n"
                            f"Using SL/TP: SL={ref_sl:.5f} TP={ref_tp:.5f}"
                        )

                        if cfg.live_trading:
                            place_order(sym, want_side, ref_sl, ref_tp, lots, cfg.magic, cfg.deviation, comment=f"MLadd {want_side}")
                            last_action_time = time.time()
                            last_add_time_by_symbol[sym] = time.time()
                            did_action = True
                            break
                        else:
                            last_action_time = time.time()
                            last_add_time_by_symbol[sym] = time.time()
                            tg.send("ℹ️ Alerts-only mode: scale-in not executed, alert only.")
                            did_action = True
                            break
                    else:
                        if cfg.chatty:
                            set_status(f"Holding {sym} {cur_side} (no add: {reason})")

            if did_action:
                time.sleep(6)
                continue

            # ---------------------------
            # If positions exist and one_symbol_at_a_time is True, do not open new symbols
            # ---------------------------
            if open_symbols and cfg.one_symbol_at_a_time:
                set_status("Managing open symbol; skipping new entries.")
                time.sleep(6)
                continue

            # ---------------------------
            # No open positions (or multi-symbol allowed): scan all for new entry
            # ---------------------------
            scan_lines = []
            top_reasons = {}
            best = None

            for sym in ok_syms:
                try:
                    cand = score_symbol(sym, cfg, model_long, model_short)
                except Exception as e:
                    cand = {"symbol": sym, "ok": False, "reason": f"Data error: {e}"}

                if not cand.get("ok"):
                    reason = cand.get("reason", "skip")
                    top_reasons[reason] = top_reasons.get(reason, 0) + 1
                    scan_lines.append(f"{sym}: ❌ {reason}")
                    continue

                if cand.get("best") is None:
                    scan_lines.append(
                        f"{sym}: ⛔ no-entry | bias={cand['bias']} spr={cand['spread']}/{cand['max_spread_used']} "
                        f"atr={cand['atr_pts']:.1f}/{cand['min_atr_used']} pL={cand['p_long']:.2f} pS={cand['p_short']:.2f} "
                        f"th={cand['enter_th_used']:.2f}"
                    )
                    continue

                d, p, _ = cand["best"]
                scan_lines.append(
                    f"{sym}: ✅ {d} p={p:.3f} | bias={cand['bias']} spr={cand['spread']}/{cand['max_spread_used']} "
                    f"atr={cand['atr_pts']:.1f}/{cand['min_atr_used']} th={cand['enter_th_used']:.2f}"
                )

                if best is None or cand["best"][2] > best["best"][2]:
                    best = cand

            last_scan_summary = "\n".join(scan_lines[:cfg.scan_lines_max])

            if best is None:
                common_reason = None
                if top_reasons:
                    common_reason = sorted(top_reasons.items(), key=lambda x: x[1], reverse=True)[0][0]
                set_status(f"No trade. Top reason: {common_reason or 'No valid setups / prob too low'}")

                if cfg.chatty:
                    tg.send(
                        "🚫 No trade this scan\n"
                        f"Top reason: {common_reason or 'No valid setups / prob too low'}\n\n"
                        f"{last_scan_summary}"
                    )

                time.sleep(10)
                continue

            sym = best["symbol"]
            direction, best_p, _ = best["best"]
            atr_val = float(best["df_entry"].iloc[-1]["atr_14"])

            entry, sl, tp, sl_dist = build_sl_tp_from_atr(sym, direction, atr_val, cfg.sl_atr, cfg.tp_atr)
            lots = calc_lot_size(sym, sl_dist, cfg.risk_per_trade)

            set_status(f"Candidate {sym} {direction} p={best_p:.3f} lots={lots}")

            tg.send(
                f"🏆 Candidate selected\n"
                f"{sym} EntryTF={cfg.entry_tf} BiasTF={cfg.bias_tf} Bias={best['bias']}\n"
                f"Dir={direction} p={best_p:.3f} Spread={best['spread']}/{best['max_spread_used']} ATR={best['atr_pts']:.1f}/{best['min_atr_used']}\n"
                f"EnterTH={best['enter_th_used']:.2f}\n"
                f"Signals={best['signals']}\n"
                f"Plan: lots={lots} entry~{entry:.5f} SL={sl:.5f} TP={tp:.5f}"
            )

            if cfg.live_trading:
                place_order(sym, direction, sl, tp, lots, cfg.magic, cfg.deviation, comment=f"ML {direction}")
                trades_today += 1
                last_action_time = time.time()
                entry_time_by_symbol[sym] = now_utc()
                last_add_time_by_symbol[sym] = time.time()

                tg.send(
                    f"✅ Trade placed\n"
                    f"{sym} {direction} lots={lots}\n"
                    f"Entry~{entry:.5f} SL={sl:.5f} TP={tp:.5f}\n"
                    f"Trades today: {trades_today}/{cfg.max_trades_per_day}"
                )
            else:
                tg.send("ℹ️ Alerts-only mode: trade not executed.")
                last_action_time = time.time()

            time.sleep(6)

        except Exception as e:
            set_status(f"Error: {str(e)[:120]}")
            tg.send("❌ ERROR\n" + str(e) + "\n" + traceback.format_exc()[:3200])
            time.sleep(20)


if __name__ == "__main__":
    main()
