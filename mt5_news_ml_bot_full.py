"""
MT5 ML Reversal + Pyramiding Bot (v1 Production Rules) — Single File

⚠️ Risk note:
- This is research/automation code, NOT a guarantee of profit.
- Use a demo account first. Tighten limits. Expect losses.

Core features:
- Interactive terminal setup + optional JSON save/load
- MarketWatch symbols (visible only) + broker suffix-safe (e.g., XAUUSDm)
- ML probabilities (RandomForest) + reversal/wick setup gates
- Per-asset gates (FX vs GOLD) for Spread/ATR/Thresholds
- Optional pyramiding with:
  - confidence ladder (each add needs higher confidence)
  - session-specific aggressiveness (London/NY behavior)
  - auto-disable pyramiding after N losses
  - dynamic add cap based on drawdown
  - per-add logging (“why it added / why it didn’t”)
- SL/TP auto-set (ATR-based); optional basket TP/SL sync after adds
- Telegram alerts + heartbeat + scan summary + top skip reasons
- Optional manual news blackout (news_events.json)

Dependencies:
  pip install numpy pandas joblib scikit-learn MetaTrader5 requests
"""

import os
import sys
import json
import time
import math
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple

# --- dependency check (friendlier errors) ---
def _missing(mod: str) -> None:
    print(f"\n❌ Missing module: {mod}")
    print("Install dependencies with:")
    print(f'  "{sys.executable}" -m pip install numpy pandas joblib scikit-learn MetaTrader5 requests')
    print("Then re-run the bot.\n")
    raise SystemExit(1)

try:
    import requests
except Exception:
    _missing("requests")

try:
    import numpy as np
except Exception:
    _missing("numpy")

try:
    import pandas as pd
except Exception:
    _missing("pandas")

try:
    import joblib
except Exception:
    _missing("joblib")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
except Exception:
    _missing("scikit-learn")

try:
    import MetaTrader5 as mt5
except Exception:
    _missing("MetaTrader5")


# =========================================================
# Utilities
# =========================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def is_gold_symbol(symbol: str) -> bool:
    # Handles XAUUSD, XAUUSDm, XAUEUR, etc.
    s = symbol.upper()
    return ("XAU" in s) or s.startswith("GOLD")

def is_fx_symbol(symbol: str) -> bool:
    return not is_gold_symbol(symbol)

def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def prompt_str(msg, default=""):
    s = input(f"{msg} [{default}]: ").strip()
    return s if s else default

def prompt_int(msg, default: int) -> int:
    while True:
        s = input(f"{msg} [{default}]: ").strip()
        if not s:
            return int(default)
        try:
            return int(s)
        except:
            print("Enter a whole number.")

def prompt_float(msg, default: float) -> float:
    while True:
        s = input(f"{msg} [{default}]: ").strip()
        if not s:
            return float(default)
        try:
            return float(s)
        except:
            print("Enter a number (e.g., 0.005).")

def prompt_yesno(msg, default="no") -> bool:
    s = input(f"{msg} (yes/no) [{default}]: ").strip().lower()
    if not s:
        s = default
    return s in ("y", "yes", "true", "1")

def safe_json_write(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def safe_json_read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Telegram
# =========================================================

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = (token or "").strip()
        self.chat_id = str(chat_id or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.token) and bool(self.chat_id)

    def send(self, msg: str) -> bool:
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

def marketwatch_symbols() -> List[str]:
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

def positions_for_symbol(symbol: str, magic: int):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return []
    return [p for p in pos if getattr(p, "magic", None) == magic]

def build_sl_tp(symbol: str, direction: str, atr_val: float, sl_atr: float, tp_atr: float):
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

def calc_lot_size(symbol: str, sl_price_distance: float, risk_per_trade: float) -> float:
    acc = mt5.account_info()
    info = mt5.symbol_info(symbol)
    if acc is None or info is None:
        return 0.01

    risk_amount = float(acc.balance) * float(risk_per_trade)

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

def modify_position_sl_tp(pos, sl: float, tp: float, deviation: int, magic: int):
    # TRADE_ACTION_SLTP modifies SL/TP for existing position
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": pos.ticket,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": int(deviation),
        "magic": int(magic),
        "comment": "MLbot sync",
    }
    result = mt5.order_send(request)
    return result


# =========================================================
# Config
# =========================================================

@dataclass
class GateConfig:
    max_spread_points: int
    min_atr_points: float
    enter_th: float
    reverse_th: float
    reverse_margin: float
    # confidence scaling weights (simple + controllable)
    use_conf_scaling: bool = True
    spread_penalty_k: float = 0.10     # bigger = more penalty as spread approaches max
    atr_bonus_k: float = 0.05          # bigger = more bonus when ATR is comfortably above min
    conf_floor: float = 0.50           # don’t scale below this unless raw is below it

@dataclass
class SessionConfig:
    use_session_filter: bool
    session_start_hour: int
    session_end_hour: int
    # session aggressiveness (UTC)
    use_session_aggression: bool
    london_start_utc: int
    london_end_utc: int
    ny_start_utc: int
    ny_end_utc: int
    london_add_delta_mult: float
    ny_add_delta_mult: float
    stop_pyramiding_in_ny: bool

@dataclass
class PyramidingConfig:
    enabled: bool
    gold_only: bool
    max_adds_per_symbol: int
    add_cooldown_seconds: int
    min_move_atr_to_add: float          # must be at least this many ATRs in profit before adding
    ladder_mode: str                    # conservative/aggressive/custom
    ladder_base_delta: float            # how much higher than entry threshold the first add needs
    ladder_step_delta: float            # additional delta per add
    add_lot_schedule: List[float]       # multipliers vs base lot [1.0, 0.7, 0.5, ...]
    # safety:
    auto_disable_after_loss_streak: bool
    loss_streak_n: int
    disable_minutes: int
    dynamic_cap_by_drawdown: bool
    dd_soft: float                      # e.g., 0.02 = 2%
    dd_hard: float                      # e.g., 0.05 = 5%
    allow_reverse_while_pyramiding: bool
    basket_sync_sl_tp: bool             # sync SL/TP across symbol after each add (basket logic)

@dataclass
class NewsConfig:
    use_news_blackout: bool
    blackout_minutes_before: int
    blackout_minutes_after: int
    events_file: str  # json list of UTC times

@dataclass
class Config:
    # Telegram + mode
    tg_token: str
    tg_chat_id: str
    live_trading: bool
    chatty: bool

    # symbols & timeframes
    use_marketwatch: bool
    symbols_csv: str
    entry_tf: str
    bias_tf: str
    gold_entry_tf_override: str  # optional override
    use_gold_tf_override: bool

    # risk & trade rules
    risk_per_trade: float
    max_trades_per_day: int
    cooldown_seconds: int
    min_bars_before_reversal: int
    one_symbol_focus: bool                 # if True, only trades one symbol at a time (global focus)

    # SL/TP ATR
    sl_atr: float
    tp_atr: float

    # structure params
    ema_fast: int
    ema_slow: int
    swing_lookback: int
    level_atr_dist: float
    wick_pct: float
    body_max_pct: float

    # gates
    gate_fx: GateConfig
    gate_gold: GateConfig

    # sessions + pyramiding + news
    session: SessionConfig
    pyramiding: PyramidingConfig
    news: NewsConfig

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
    df["body_pct"] = (body / rng).fillna(0)
    df["upper_wick_pct"] = (upper_wick / rng).fillna(0)
    df["lower_wick_pct"] = (lower_wick / rng).fillna(0)
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

def detect_wick_setups(df: pd.DataFrame, cfg: Config) -> Tuple[bool, bool]:
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

def reversal_signals(df_entry: pd.DataFrame, cfg: Config) -> dict:
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
    return df_entry.iloc[-2][FEATURE_COLS].astype(float).to_numpy().reshape(1, -1)

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

    # limit training set if too many MarketWatch symbols
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
                feats = window.iloc[-2][FEATURE_COLS].astype(float).to_numpy()

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
# Session + News blackout
# =========================================================

def in_allowed_session_local(cfg: Config) -> bool:
    if not cfg.session.use_session_filter:
        return True
    h = datetime.now().hour
    return cfg.session.session_start_hour <= h <= cfg.session.session_end_hour

def current_session_utc(cfg: Config) -> str:
    if not cfg.session.use_session_aggression:
        return "NONE"
    h = now_utc().hour
    if cfg.session.london_start_utc <= h <= cfg.session.london_end_utc:
        return "LONDON"
    if cfg.session.ny_start_utc <= h <= cfg.session.ny_end_utc:
        return "NY"
    return "OFF"

def load_news_events(cfg: Config) -> List[datetime]:
    if not cfg.news.use_news_blackout:
        return []
    path = cfg.news.events_file.strip()
    if not path or not os.path.exists(path):
        return []
    try:
        raw = safe_json_read(path)
        # expected: [{"time_utc": "2026-01-18 13:30"}, ...]
        events = []
        for item in raw:
            t = item.get("time_utc", "")
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            events.append(dt)
        return events
    except:
        return []

def in_news_blackout(cfg: Config, events_utc: List[datetime]) -> Optional[str]:
    if not cfg.news.use_news_blackout or not events_utc:
        return None
    now = now_utc()
    b = timedelta(minutes=cfg.news.blackout_minutes_before)
    a = timedelta(minutes=cfg.news.blackout_minutes_after)
    for ev in events_utc:
        if (ev - b) <= now <= (ev + a):
            return f"News blackout window around {fmt_dt(ev)}"
    return None


# =========================================================
# Gates + confidence scaling
# =========================================================

def gate_for_symbol(cfg: Config, symbol: str) -> GateConfig:
    return cfg.gate_gold if is_gold_symbol(symbol) else cfg.gate_fx

def adjust_confidence(p_raw: float, spread: int, atr_pts: float, gate: GateConfig) -> float:
    if not gate.use_conf_scaling:
        return float(p_raw)

    # Spread quality: 0.0 (perfect) to ~1.0 (at max)
    q_spread = clamp(spread / max(1.0, float(gate.max_spread_points)), 0.0, 2.0)
    # ATR quality: <1.0 means below min ATR; >1 means above
    q_atr = atr_pts / max(1e-9, float(gate.min_atr_points))

    penalty = max(0.0, q_spread - 0.60) * gate.spread_penalty_k
    bonus = max(0.0, q_atr - 1.00) * gate.atr_bonus_k

    p = float(p_raw) - penalty + bonus
    p = clamp(p, gate.conf_floor if p_raw >= gate.conf_floor else 0.0, 0.99)
    return float(p)


# =========================================================
# Decision logic (entry/reverse/pyramiding)
# =========================================================

def decide_reverse(pos_side: str, sigs: dict, p_long: float, p_short: float, gate: GateConfig):
    if pos_side == "BUY":
        bearish_setup = sigs["bear_reversal"] or sigs["setupB_short"]
        if bearish_setup and p_short >= gate.reverse_th and (p_short - p_long) >= gate.reverse_margin:
            return "CLOSE_AND_OPEN_SELL"
    else:
        bullish_setup = sigs["bull_reversal"] or sigs["setupA_long"]
        if bullish_setup and p_long >= gate.reverse_th and (p_long - p_short) >= gate.reverse_margin:
            return "CLOSE_AND_OPEN_BUY"
    return "HOLD"

def ladder_required_threshold(cfg: Config, symbol: str, add_index: int) -> float:
    """
    add_index: 1 for first add, 2 for second, etc.
    required = enter_th + base_delta + (add_index-1)*step_delta
    plus session multiplier (London/NY).
    """
    gate = gate_for_symbol(cfg, symbol)
    base = gate.enter_th
    pyr = cfg.pyramiding

    base_delta = pyr.ladder_base_delta
    step_delta = pyr.ladder_step_delta

    # mode presets (only if user picks those)
    if pyr.ladder_mode.lower() == "conservative":
        base_delta = max(base_delta, 0.05)
        step_delta = max(step_delta, 0.03)
    elif pyr.ladder_mode.lower() == "aggressive":
        base_delta = min(base_delta, 0.03)
        step_delta = min(step_delta, 0.02)

    req = base + base_delta + (max(0, add_index - 1) * step_delta)

    sess = current_session_utc(cfg)
    if cfg.session.use_session_aggression:
        if sess == "LONDON":
            req = base + (req - base) * cfg.session.london_add_delta_mult
        elif sess == "NY":
            req = base + (req - base) * cfg.session.ny_add_delta_mult

    return float(clamp(req, 0.50, 0.95))

def avg_entry_and_volume(positions) -> Tuple[float, float]:
    tv = 0.0
    te = 0.0
    for p in positions:
        tv += float(p.volume)
        te += float(p.price_open) * float(p.volume)
    if tv <= 0:
        return 0.0, 0.0
    return te / tv, tv

def sync_basket_sl_tp(cfg: Config, symbol: str, direction: str, atr_val: float, tg: Telegram) -> None:
    """
    Basket SL/TP sync across positions for this symbol.
    Never widens SL (risk). If SL would widen, keep current SL.
    """
    if not cfg.pyramiding.basket_sync_sl_tp:
        return

    pos = positions_for_symbol(symbol, cfg.magic)
    if not pos:
        return

    avg_entry, tot_vol = avg_entry_and_volume(pos)
    if avg_entry <= 0 or tot_vol <= 0:
        return

    if direction == "BUY":
        new_sl = avg_entry - cfg.sl_atr * atr_val
        new_tp = avg_entry + cfg.tp_atr * atr_val
        # never widen: SL must be >= current SL
        for p in pos:
            cur_sl = float(p.sl) if float(p.sl) > 0 else None
            if cur_sl is not None and new_sl < cur_sl:
                new_sl = cur_sl
    else:
        new_sl = avg_entry + cfg.sl_atr * atr_val
        new_tp = avg_entry - cfg.tp_atr * atr_val
        # never widen: for sells SL must be <= current SL
        for p in pos:
            cur_sl = float(p.sl) if float(p.sl) > 0 else None
            if cur_sl is not None and new_sl > cur_sl:
                new_sl = cur_sl

    for p in pos:
        modify_position_sl_tp(p, new_sl, new_tp, cfg.deviation, cfg.magic)

    tg.send(
        f"🧺 Basket SL/TP synced ({symbol})\n"
        f"Dir={direction} avg_entry={avg_entry:.5f} vol={tot_vol}\n"
        f"SL={new_sl:.5f} TP={new_tp:.5f}"
    )

def can_add_pyramiding(cfg: Config, symbol: str) -> Tuple[bool, str]:
    pyr = cfg.pyramiding
    if not pyr.enabled:
        return False, "Pyramiding disabled"
    if pyr.gold_only and not is_gold_symbol(symbol):
        return False, "Pyramiding set to GOLD-only"
    sess = current_session_utc(cfg)
    if cfg.session.use_session_aggression and sess == "NY" and cfg.session.stop_pyramiding_in_ny:
        return False, "NY session: pyramiding disabled"
    return True, "OK"


# =========================================================
# Scoring
# =========================================================

def entry_timeframe(cfg: Config, symbol: str) -> str:
    if cfg.use_gold_tf_override and is_gold_symbol(symbol):
        return cfg.gold_entry_tf_override
    return cfg.entry_tf

def score_symbol(sym: str, cfg: Config, model_long, model_short, events_utc: List[datetime]):
    gate = gate_for_symbol(cfg, sym)

    # news blackout (optional)
    nb = in_news_blackout(cfg, events_utc)
    if nb:
        return {"symbol": sym, "ok": False, "reason": nb}

    spread = get_spread_points(sym)
    if spread > gate.max_spread_points:
        return {"symbol": sym, "ok": False, "reason": f"Spread too high ({spread}>{gate.max_spread_points})"}

    tf_entry = entry_timeframe(cfg, sym)
    df_entry = build_frame(sym, tf_entry, cfg.live_bars, cfg)
    df_bias = build_frame(sym, cfg.bias_tf, cfg.live_bars, cfg)

    bias = bias_from_tf(df_bias)
    sigs = reversal_signals(df_entry, cfg)

    atr_val = float(df_entry.iloc[-1]["atr_14"])
    atr_pts = atr_to_points(sym, atr_val)
    if atr_pts < gate.min_atr_points:
        return {"symbol": sym, "ok": False, "reason": f"ATR too low ({atr_pts:.1f}<{gate.min_atr_points} pts)"}

    x = build_feature_row(df_entry)
    p_long_raw = float(model_long.predict_proba(x)[0, 1])
    p_short_raw = float(model_short.predict_proba(x)[0, 1])

    p_long = adjust_confidence(p_long_raw, spread, atr_pts, gate)
    p_short = adjust_confidence(p_short_raw, spread, atr_pts, gate)

    long_setup = sigs["bull_reversal"] or sigs["setupA_long"]
    short_setup = sigs["bear_reversal"] or sigs["setupB_short"]

    long_ready = long_setup and (p_long >= gate.enter_th) and (bias in ["NONE", "BUY"])
    short_ready = short_setup and (p_short >= gate.enter_th) and (bias in ["NONE", "SELL"])

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
        "atr_pts": atr_pts, "atr_val": atr_val,
        "p_long": p_long, "p_short": p_short,
        "p_long_raw": p_long_raw, "p_short_raw": p_short_raw,
        "signals": sigs, "best": best,
        "df_entry": df_entry,
        "tf_entry": tf_entry,
        "gate": gate
    }


# =========================================================
# Config load/save + guide
# =========================================================

def cfg_to_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    return d

def cfg_from_dict(d: dict) -> Config:
    # nested dataclasses
    d = dict(d)
    d["gate_fx"] = GateConfig(**d["gate_fx"])
    d["gate_gold"] = GateConfig(**d["gate_gold"])
    d["session"] = SessionConfig(**d["session"])
    d["pyramiding"] = PyramidingConfig(**d["pyramiding"])
    d["news"] = NewsConfig(**d["news"])
    return Config(**d)

def write_behavior_guide(cfg: Config, symbols: List[str]) -> None:
    lines = []
    lines.append("MT5 BOT v1 — Behavior Guide")
    lines.append("")
    lines.append("1) Entry gating:")
    lines.append("- Trades ONLY when reversal/wick setup triggers AND ML confidence passes threshold.")
    lines.append("- FX and GOLD use different gates (spread/ATR/threshold).")
    lines.append("")
    lines.append("2) SL/TP:")
    lines.append("- Always ATR-based SL/TP (set automatically).")
    lines.append("- If pyramiding enabled, basket sync may adjust TP (and SL without widening risk).")
    lines.append("")
    lines.append("3) Pyramiding (adds):")
    lines.append("- Adds only when trade is in profit by a minimum ATR move.")
    lines.append("- Each add requires a higher confidence threshold (ladder).")
    lines.append("- Can be disabled by NY session, loss streak, or drawdown cap.")
    lines.append("")
    lines.append("4) Safety:")
    lines.append("- Max trades/day, cooldown, session filter, spread/ATR filters.")
    lines.append("")
    lines.append(f"Symbols: {', '.join(symbols[:30])}{'...' if len(symbols)>30 else ''}")
    lines.append("")
    lines.append("NOTE: No live news scraping by default. Optional manual news blackout via news_events.json.")

    with open("bot_behavior_guide.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# Interactive setup
# =========================================================

def interactive_setup() -> Config:
    print("\n=== MT5 ML BOT SETUP (v1) ===\n")

    use_json = prompt_yesno("Load settings from JSON config file?", "no")
    if use_json:
        path = prompt_str("Config JSON path", "bot_config.json")
        if os.path.exists(path):
            d = safe_json_read(path)
            cfg = cfg_from_dict(d)
            print(f"✅ Loaded config: {path}")
            return cfg
        print("⚠️ Config file not found, switching to interactive setup.\n")

    # Telegram
    tg_token = prompt_str("Telegram bot token (empty disables Telegram)", "")
    tg_chat_id = prompt_str("Telegram chat ID (empty disables Telegram)", "")

    live_trading = prompt_yesno("Enable LIVE trading? (no = alerts only)", "no")
    chatty = prompt_yesno("Chatty mode? (more updates / reasons)", "yes")

    # Symbols
    use_mw = prompt_yesno("Use symbols from MT5 Market Watch automatically?", "yes")
    symbols_csv = ""
    if not use_mw:
        symbols_csv = prompt_str("Symbols CSV (example: XAUUSD,EURUSD,USDJPY)", "EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD")

    # TF
    entry_tf = prompt_str("Entry timeframe (M1/M5/M15/M30/H1)", "M5").upper()
    bias_tf = prompt_str("Bias timeframe (M5/M15/M30/H1)", "M15").upper()
    if entry_tf not in TF_MAP or bias_tf not in TF_MAP:
        raise RuntimeError("Unsupported timeframe. Use: " + ", ".join(TF_MAP.keys()))

    use_gold_tf_override = prompt_yesno("Use GOLD entry timeframe override?", "no")
    gold_entry_tf_override = entry_tf
    if use_gold_tf_override:
        gold_entry_tf_override = prompt_str("GOLD entry timeframe override (M1/M5/M15/M30/H1)", "M5").upper()
        if gold_entry_tf_override not in TF_MAP:
            raise RuntimeError("Unsupported GOLD timeframe.")

    # Risk
    risk = prompt_float("Risk per trade (0.005 = 0.5%)", 0.005)
    max_day = prompt_int("Max trades per day", 2)
    cooldown = prompt_int("Cooldown seconds between actions", 90)
    min_bars_rev = prompt_int("Min bars before reversal (anti-chop)", 3)
    one_symbol_focus = prompt_yesno("One-symbol focus? (manage only one symbol at a time)", "yes")

    # SL/TP
    sl_atr = prompt_float("StopLoss ATR multiple", 1.2)
    tp_atr = prompt_float("TakeProfit ATR multiple", 1.8)

    # Session filter
    use_session = prompt_yesno("Use session filter (trade only certain hours, LOCAL time)?", "yes")
    start_h = prompt_int("Session start hour (local time, 0-23)", 7)
    end_h = prompt_int("Session end hour (local time, 0-23)", 20)

    # Session aggression
    use_sess_aggr = prompt_yesno("Use session-specific aggressiveness (London vs NY)?", "yes")
    london_add_mult = prompt_float("London add threshold multiplier (1.0 = normal)", 1.0)
    ny_add_mult = prompt_float("NY add threshold multiplier (>1 = more strict)", 1.15)
    stop_pyr_ny = prompt_yesno("Auto-stop pyramiding in NY session?", "yes")

    # FX Gate
    print("\n--- FX Gate ---")
    fx_spread = prompt_int("FX max spread (points)", 20)
    fx_atr = prompt_float("FX min ATR (points)", 15.0)
    fx_enter = prompt_float("FX entry threshold", 0.61)
    fx_rev = prompt_float("FX reverse threshold", 0.68)
    fx_margin = prompt_float("FX reverse margin", 0.10)
    fx_conf = prompt_yesno("FX confidence scaling?", "yes")

    # GOLD Gate
    print("\n--- GOLD Gate ---")
    gold_spread = prompt_int("GOLD max spread (points)", 100)
    gold_atr = prompt_float("GOLD min ATR (points)", 25.0)
    gold_enter = prompt_float("GOLD entry threshold", 0.58)
    gold_rev = prompt_float("GOLD reverse threshold", 0.66)
    gold_margin = prompt_float("GOLD reverse margin", 0.08)
    gold_conf = prompt_yesno("GOLD confidence scaling?", "yes")

    # Pyramiding
    print("\n--- Pyramiding (adds) ---")
    pyr_enabled = prompt_yesno("Enable pyramiding (adds)?", "no")
    pyr_gold_only = prompt_yesno("If pyramiding enabled: GOLD-only?", "no")
    pyr_max_adds = prompt_int("Max adds per symbol (0 disables adds)", 3)
    pyr_add_cd = prompt_int("Add cooldown seconds", 120)
    pyr_min_move_atr = prompt_float("Min move (in ATRs) in profit before add", 0.6)

    ladder_mode = prompt_str("Confidence ladder mode (conservative/aggressive/custom)", "conservative")
    ladder_base_delta = prompt_float("Ladder base delta (first add needs +delta)", 0.05)
    ladder_step_delta = prompt_float("Ladder step delta (each add needs +step)", 0.03)

    # lot schedule
    schedule_raw = prompt_str("Add lot schedule multipliers (comma list)", "1.0,0.7,0.5,0.4,0.3")
    add_sched = []
    for x in schedule_raw.split(","):
        try:
            add_sched.append(float(x.strip()))
        except:
            pass
    if not add_sched:
        add_sched = [1.0, 0.7, 0.5, 0.4, 0.3]

    auto_disable = prompt_yesno("Auto-disable pyramiding after loss streak?", "yes")
    loss_n = prompt_int("Loss streak N to disable adds", 2)
    disable_mins = prompt_int("Disable pyramiding minutes after streak", 60)

    dyn_cap = prompt_yesno("Dynamic add cap based on drawdown?", "yes")
    dd_soft = prompt_float("Drawdown soft limit (e.g. 0.02 = 2%)", 0.02)
    dd_hard = prompt_float("Drawdown hard limit (e.g. 0.05 = 5%)", 0.05)

    allow_rev_pyr = prompt_yesno("Allow reverse while pyramiding?", "no")
    basket_sync = prompt_yesno("Basket sync SL/TP after adds?", "yes")

    # News blackout
    print("\n--- News Blackout (optional) ---")
    use_news_blackout = prompt_yesno("Use manual news blackout windows?", "no")
    blackout_before = prompt_int("Blackout minutes BEFORE event", 15)
    blackout_after = prompt_int("Blackout minutes AFTER event", 15)
    events_file = prompt_str("Events file path (news_events.json)", "news_events.json")

    # Heartbeat & training
    heartbeat = prompt_int("Heartbeat interval seconds", 300)
    scan_lines_max = prompt_int("Heartbeat scan lines to show", 6)
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
        gold_entry_tf_override=gold_entry_tf_override,
        use_gold_tf_override=use_gold_tf_override,

        risk_per_trade=risk,
        max_trades_per_day=max_day,
        cooldown_seconds=cooldown,
        min_bars_before_reversal=min_bars_rev,
        one_symbol_focus=one_symbol_focus,

        sl_atr=sl_atr,
        tp_atr=tp_atr,

        ema_fast=ema_fast,
        ema_slow=ema_slow,
        swing_lookback=swing_lookback,
        level_atr_dist=level_atr_dist,
        wick_pct=wick_pct,
        body_max_pct=body_max_pct,

        gate_fx=GateConfig(
            max_spread_points=fx_spread,
            min_atr_points=fx_atr,
            enter_th=fx_enter,
            reverse_th=fx_rev,
            reverse_margin=fx_margin,
            use_conf_scaling=fx_conf,
        ),
        gate_gold=GateConfig(
            max_spread_points=gold_spread,
            min_atr_points=gold_atr,
            enter_th=gold_enter,
            reverse_th=gold_rev,
            reverse_margin=gold_margin,
            use_conf_scaling=gold_conf,
        ),

        session=SessionConfig(
            use_session_filter=use_session,
            session_start_hour=start_h,
            session_end_hour=end_h,
            use_session_aggression=use_sess_aggr,
            london_start_utc=7,
            london_end_utc=16,
            ny_start_utc=12,
            ny_end_utc=21,
            london_add_delta_mult=london_add_mult,
            ny_add_delta_mult=ny_add_mult,
            stop_pyramiding_in_ny=stop_pyr_ny,
        ),

        pyramiding=PyramidingConfig(
            enabled=pyr_enabled and pyr_max_adds > 0,
            gold_only=pyr_gold_only,
            max_adds_per_symbol=max(0, pyr_max_adds),
            add_cooldown_seconds=pyr_add_cd,
            min_move_atr_to_add=pyr_min_move_atr,
            ladder_mode=ladder_mode,
            ladder_base_delta=ladder_base_delta,
            ladder_step_delta=ladder_step_delta,
            add_lot_schedule=add_sched,
            auto_disable_after_loss_streak=auto_disable,
            loss_streak_n=max(1, loss_n),
            disable_minutes=max(1, disable_mins),
            dynamic_cap_by_drawdown=dyn_cap,
            dd_soft=max(0.0, dd_soft),
            dd_hard=max(dd_soft, dd_hard),
            allow_reverse_while_pyramiding=allow_rev_pyr,
            basket_sync_sl_tp=basket_sync,
        ),

        news=NewsConfig(
            use_news_blackout=use_news_blackout,
            blackout_minutes_before=blackout_before,
            blackout_minutes_after=blackout_after,
            events_file=events_file,
        ),

        heartbeat_seconds=heartbeat,
        scan_lines_max=scan_lines_max,

        history_bars_train=history_train,
        live_bars=live_bars,
        horizon=horizon,

        model_long_path=model_long_path,
        model_short_path=model_short_path,

        magic=magic,
        deviation=deviation
    )

    save_json = prompt_yesno("Save these settings to JSON for easy reuse?", "yes")
    if save_json:
        path = prompt_str("Save path", "bot_config.json")
        safe_json_write(path, cfg_to_dict(cfg))
        print(f"✅ Saved config: {path}")

    return cfg


# =========================================================
# Pyramiding runtime state
# =========================================================

@dataclass
class SymbolState:
    base_lot: float = 0.0
    adds: int = 0
    last_add_ts: float = 0.0
    disabled_until_utc: Optional[datetime] = None
    loss_streak: int = 0

def effective_max_adds(cfg: Config, acc) -> int:
    pyr = cfg.pyramiding
    if not pyr.dynamic_cap_by_drawdown:
        return pyr.max_adds_per_symbol

    # drawdown based on equity vs peak
    # (we’ll compute peak in main loop; this helper just clamps)
    return pyr.max_adds_per_symbol

def drawdown_ratio(peak_equity: float, equity: float) -> float:
    if peak_equity <= 0:
        return 0.0
    return clamp((peak_equity - equity) / peak_equity, 0.0, 1.0)

def dynamic_add_cap(cfg: Config, dd: float) -> int:
    pyr = cfg.pyramiding
    if not pyr.dynamic_cap_by_drawdown:
        return pyr.max_adds_per_symbol
    if dd >= pyr.dd_hard:
        return 0
    if dd <= pyr.dd_soft:
        return pyr.max_adds_per_symbol
    # linear reduce between soft and hard
    t = (dd - pyr.dd_soft) / max(1e-9, (pyr.dd_hard - pyr.dd_soft))
    cap = int(round((1.0 - t) * pyr.max_adds_per_symbol))
    return max(0, min(pyr.max_adds_per_symbol, cap))


# =========================================================
# Main loop
# =========================================================

def main():
    cfg = interactive_setup()
    tg = Telegram(cfg.tg_token, cfg.tg_chat_id)

    mt5_init_or_die()
    acc = account_info_or_die()

    # Symbols
    if cfg.use_marketwatch:
        symbols = marketwatch_symbols()
    else:
        symbols = [s.strip().upper() for s in cfg.symbols_csv.split(",") if s.strip()]

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

    events_utc = load_news_events(cfg)

    write_behavior_guide(cfg, ok_syms)

    tg.send(
        "✅ Bot started\n"
        f"Account: {acc.login} | Server: {acc.server}\n"
        f"Balance: {acc.balance}\n"
        f"Mode: {'LIVE TRADING' if cfg.live_trading else 'ALERTS ONLY'}\n"
        f"Symbols: {', '.join(ok_syms[:20])}{'...' if len(ok_syms) > 20 else ''}\n"
        f"EntryTF={cfg.entry_tf} BiasTF={cfg.bias_tf}\n"
        f"Chatty={cfg.chatty} Heartbeat={cfg.heartbeat_seconds}s\n"
        f"FX gate: spr<={cfg.gate_fx.max_spread_points}, ATR>={cfg.gate_fx.min_atr_points}, enter>={cfg.gate_fx.enter_th}\n"
        f"GOLD gate: spr<={cfg.gate_gold.max_spread_points}, ATR>={cfg.gate_gold.min_atr_points}, enter>={cfg.gate_gold.enter_th}\n"
        f"Pyramiding: {cfg.pyramiding.enabled} max_adds={cfg.pyramiding.max_adds_per_symbol} ladder={cfg.pyramiding.ladder_mode}"
    )

    last_status = "Initialized. First scan pending..."
    last_scan_summary = ""
    last_heartbeat_time = time.time()

    def set_status(s: str):
        nonlocal last_status
        last_status = s

    # Train/load models
    train_models_if_needed(ok_syms, cfg, tg, set_status)
    model_long = joblib.load(cfg.model_long_path)
    model_short = joblib.load(cfg.model_short_path)

    trades_today = 0
    day = now_utc().date()
    last_action_time = 0.0

    entry_time_by_symbol: Dict[str, datetime] = {}
    sym_state: Dict[str, SymbolState] = {s: SymbolState() for s in ok_syms}

    peak_equity = float(getattr(acc, "equity", acc.balance))

    tg.send("🚀 Live loop running.")

    while True:
        try:
            # refresh account for dd cap
            acc = account_info_or_die()
            equity = float(getattr(acc, "equity", acc.balance))
            if equity > peak_equity:
                peak_equity = equity
            dd = drawdown_ratio(peak_equity, equity)
            max_adds_dd = dynamic_add_cap(cfg, dd)

            # Heartbeat
            if time.time() - last_heartbeat_time >= cfg.heartbeat_seconds:
                tg.send(
                    f"💓 Heartbeat\n"
                    f"UTC: {fmt_dt(now_utc())}\n"
                    f"Session: {current_session_utc(cfg)}\n"
                    f"Equity: {equity:.2f} Peak: {peak_equity:.2f} DD: {dd*100:.1f}% (max_adds_now={max_adds_dd})\n"
                    f"Status: {last_status}\n\n"
                    f"Scan:\n{last_scan_summary}"
                )
                last_heartbeat_time = time.time()

            # Session filter
            if not in_allowed_session_local(cfg):
                set_status("Outside session hours (sleeping).")
                time.sleep(20)
                continue

            # Daily reset
            if now_utc().date() != day:
                day = now_utc().date()
                trades_today = 0
                peak_equity = float(getattr(acc, "equity", acc.balance))
                tg.send("🗓 New day: trades/day counter reset.")

            # Cooldown
            if time.time() - last_action_time < cfg.cooldown_seconds:
                time.sleep(2)
                continue

            # Max trades/day
            if trades_today >= cfg.max_trades_per_day:
                set_status(f"Max trades reached ({trades_today}/{cfg.max_trades_per_day}).")
                time.sleep(12)
                continue

            # --- Manage existing positions (one-symbol focus = always manage before scanning) ---
            open_positions = positions_by_magic(cfg.magic)
            if open_positions and cfg.one_symbol_focus:
                # choose first symbol
                sym = open_positions[0].symbol
                pos_list = positions_for_symbol(sym, cfg.magic)
                if not pos_list:
                    time.sleep(3)
                    continue

                # Determine net direction from first position type
                pos_side = "BUY" if pos_list[0].type == mt5.POSITION_TYPE_BUY else "SELL"
                gate = gate_for_symbol(cfg, sym)

                tf_entry = entry_timeframe(cfg, sym)
                df_entry = build_frame(sym, tf_entry, cfg.live_bars, cfg)
                sigs = reversal_signals(df_entry, cfg)
                atr_val = float(df_entry.iloc[-1]["atr_14"])
                atr_pts = atr_to_points(sym, atr_val)
                spr = get_spread_points(sym)

                x = build_feature_row(df_entry)
                pL_raw = float(model_long.predict_proba(x)[0, 1])
                pS_raw = float(model_short.predict_proba(x)[0, 1])
                pL = adjust_confidence(pL_raw, spr, atr_pts, gate)
                pS = adjust_confidence(pS_raw, spr, atr_pts, gate)

                # reversal hold protection
                can_reverse = True
                entry_t = entry_time_by_symbol.get(sym)
                if entry_t is not None:
                    bar_minutes = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60}.get(tf_entry, 5)
                    min_hold = timedelta(minutes=bar_minutes * cfg.min_bars_before_reversal)
                    if now_utc() - entry_t < min_hold:
                        can_reverse = False

                # pyramiding stop rules
                state = sym_state.get(sym, SymbolState())
                if state.disabled_until_utc and now_utc() < state.disabled_until_utc:
                    pyr_ok = False
                    pyr_reason = f"Pyramiding disabled until {fmt_dt(state.disabled_until_utc)}"
                else:
                    pyr_ok, pyr_reason = can_add_pyramiding(cfg, sym)

                # reversal decision (optional when pyramiding)
                action = decide_reverse(pos_side, sigs, pL, pS, gate)
                if (not cfg.pyramiding.allow_reverse_while_pyramiding) and state.adds > 0 and action != "HOLD":
                    action = "HOLD"

                if action != "HOLD" and not can_reverse:
                    set_status(f"Reverse blocked (min-hold) {sym} pos={pos_side}")
                    if cfg.chatty:
                        tg.send(f"⛔ Reverse blocked (min-hold)\n{sym} pos={pos_side}\nSignals={sigs}\npL={pL:.3f} pS={pS:.3f}")
                    time.sleep(6)
                    continue

                if action.startswith("CLOSE_AND_OPEN"):
                    new_dir = "BUY" if action.endswith("BUY") else "SELL"
                    set_status(f"Reversing {sym} {pos_side}->{new_dir} (pL={pL:.3f} pS={pS:.3f})")

                    tg.send(
                        f"🔁 Reverse triggered\n"
                        f"{sym} {tf_entry}\n"
                        f"From {pos_side} to {new_dir}\n"
                        f"Signals={sigs}\n"
                        f"p_long={pL:.3f} p_short={pS:.3f}"
                    )

                    if cfg.live_trading:
                        # close all positions on symbol
                        for p in pos_list:
                            close_position(p, cfg.magic, cfg.deviation)

                        entry, sl, tp, sl_dist = build_sl_tp(sym, new_dir, atr_val, cfg.sl_atr, cfg.tp_atr)
                        lots = calc_lot_size(sym, sl_dist, cfg.risk_per_trade)
                        place_order(sym, new_dir, sl, tp, lots, cfg.magic, cfg.deviation, comment=f"MLrev {new_dir}")

                        trades_today += 1
                        last_action_time = time.time()
                        entry_time_by_symbol[sym] = now_utc()
                        sym_state[sym] = SymbolState(base_lot=lots, adds=0, last_add_ts=0.0)

                        tg.send(
                            f"✅ Reverse trade placed\n"
                            f"{sym} {new_dir} lots={lots}\n"
                            f"Entry~{entry:.5f} SL={sl:.5f} TP={tp:.5f}\n"
                            f"Trades today: {trades_today}/{cfg.max_trades_per_day}"
                        )
                    else:
                        last_action_time = time.time()
                        tg.send("ℹ️ Alerts-only mode: reverse not executed, alert only.")

                    time.sleep(6)
                    continue

                # --- pyramiding add logic (controlled) ---
                if cfg.pyramiding.enabled and pyr_ok and max_adds_dd > 0:
                    # respect symbol state + cooldown + cap
                    add_cap = min(cfg.pyramiding.max_adds_per_symbol, max_adds_dd)
                    if state.adds < add_cap:
                        if (time.time() - state.last_add_ts) < cfg.pyramiding.add_cooldown_seconds:
                            set_status(f"Manage {sym}: add cooldown")
                        else:
                            # ensure ATR still ok + spread ok
                            if spr > gate.max_spread_points:
                                set_status(f"Manage {sym}: spread too high for add")
                            elif atr_pts < gate.min_atr_points:
                                set_status(f"Manage {sym}: ATR too low for add")
                            else:
                                # profit-in-ATR requirement
                                # approximate using avg entry of current position(s)
                                avg_entry, tot_vol = avg_entry_and_volume(pos_list)
                                tick = mt5.symbol_info_tick(sym)
                                if tick:
                                    cur_price = tick.bid if pos_side == "BUY" else tick.ask
                                    move = (cur_price - avg_entry) if pos_side == "BUY" else (avg_entry - cur_price)
                                    move_atr = move / max(1e-9, atr_val)
                                else:
                                    move_atr = 0.0

                                add_index = state.adds + 1
                                required = ladder_required_threshold(cfg, sym, add_index)
                                cur_conf = pL if pos_side == "BUY" else pS

                                if move_atr < cfg.pyramiding.min_move_atr_to_add:
                                    if cfg.chatty:
                                        tg.send(
                                            f"➕ Add blocked (not enough move)\n{sym} pos={pos_side}\n"
                                            f"move_atr={move_atr:.2f} < {cfg.pyramiding.min_move_atr_to_add:.2f}\n"
                                            f"conf={cur_conf:.3f} req={required:.3f}"
                                        )
                                elif cur_conf < required:
                                    if cfg.chatty:
                                        tg.send(
                                            f"➕ Add blocked (confidence ladder)\n{sym} pos={pos_side}\n"
                                            f"level={add_index}/{add_cap}\n"
                                            f"conf={cur_conf:.3f} < req={required:.3f}\n"
                                            f"Signals={sigs}"
                                        )
                                else:
                                    # compute lot to add
                                    if state.base_lot <= 0:
                                        # fallback base lot = current total volume
                                        state.base_lot = float(pos_list[0].volume)

                                    sched = cfg.pyramiding.add_lot_schedule
                                    mult = sched[min(add_index, len(sched)) - 1] if sched else 0.5
                                    add_lot = max(0.01, round(state.base_lot * float(mult), 2))

                                    # place add
                                    direction = "BUY" if pos_side == "BUY" else "SELL"
                                    entry, sl, tp, sl_dist = build_sl_tp(sym, direction, atr_val, cfg.sl_atr, cfg.tp_atr)

                                    if cfg.live_trading:
                                        place_order(sym, direction, sl, tp, add_lot, cfg.magic, cfg.deviation, comment=f"ADD{add_index}")
                                        state.adds += 1
                                        state.last_add_ts = time.time()
                                        sym_state[sym] = state
                                        last_action_time = time.time()

                                        tg.send(
                                            f"✅ Add executed\n{sym} {direction}\n"
                                            f"Add level {add_index}/{add_cap}\n"
                                            f"conf={cur_conf:.3f} req={required:.3f} move_atr={move_atr:.2f}\n"
                                            f"lots={add_lot}\n"
                                            f"SL/TP re-evaluated"
                                        )

                                        # sync basket SL/TP after add
                                        sync_basket_sl_tp(cfg, sym, direction, atr_val, tg)
                                    else:
                                        tg.send(
                                            f"📣 Add signal (alerts-only)\n{sym} {direction}\n"
                                            f"level={add_index}/{add_cap} conf={cur_conf:.3f} req={required:.3f}"
                                        )
                                        state.adds += 1
                                        state.last_add_ts = time.time()
                                        sym_state[sym] = state
                                        last_action_time = time.time()

                    else:
                        set_status(f"Manage {sym}: add cap reached ({state.adds}/{add_cap})")
                else:
                    set_status(f"Managing {sym} pos={pos_side} (no adds: {pyr_reason})")

                time.sleep(6)
                continue

            # --- If not one-symbol focus or no positions: scan entries ---
            scan_lines = []
            top_reasons = {}
            best = None

            for sym in ok_syms:
                try:
                    cand = score_symbol(sym, cfg, model_long, model_short, events_utc)
                except Exception as e:
                    cand = {"symbol": sym, "ok": False, "reason": f"Data error: {e}"}

                if not cand.get("ok"):
                    reason = cand.get("reason", "skip")
                    top_reasons[reason] = top_reasons.get(reason, 0) + 1
                    scan_lines.append(f"{sym}: ❌ {reason}")
                    continue

                gate = cand["gate"]
                th = gate.enter_th

                if cand.get("best") is None:
                    scan_lines.append(
                        f"{sym}: ⛔ no-entry | bias={cand['bias']} spr={cand['spread']}/{gate.max_spread_points} "
                        f"atr={cand['atr_pts']:.1f}/{gate.min_atr_points} "
                        f"pL={cand['p_long']:.2f} pS={cand['p_short']:.2f} th={th:.2f}"
                    )
                    continue

                d, p, _ = cand["best"]
                scan_lines.append(
                    f"{sym}: ✅ {d} p={p:.3f} | bias={cand['bias']} spr={cand['spread']}/{gate.max_spread_points} "
                    f"atr={cand['atr_pts']:.1f}/{gate.min_atr_points} th={th:.2f}"
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

            entry, sl, tp, sl_dist = build_sl_tp(sym, direction, atr_val, cfg.sl_atr, cfg.tp_atr)
            lots = calc_lot_size(sym, sl_dist, cfg.risk_per_trade)

            set_status(f"Candidate {sym} {direction} p={best_p:.3f} lots={lots}")

            tg.send(
                f"🏆 Candidate selected\n"
                f"{sym} EntryTF={best['tf_entry']} BiasTF={cfg.bias_tf} Bias={best['bias']}\n"
                f"Dir={direction} p={best_p:.3f} Spread={best['spread']} ATR={best['atr_pts']:.1f}\n"
                f"Signals={best['signals']}\n"
                f"Plan: lots={lots} entry~{entry:.5f} SL={sl:.5f} TP={tp:.5f}"
            )

            if cfg.live_trading:
                place_order(sym, direction, sl, tp, lots, cfg.magic, cfg.deviation, comment=f"ML {direction}")
                trades_today += 1
                last_action_time = time.time()
                entry_time_by_symbol[sym] = now_utc()
                sym_state[sym] = SymbolState(base_lot=lots, adds=0, last_add_ts=0.0)

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
            time.sleep(15)


if __name__ == "__main__":
    main()
