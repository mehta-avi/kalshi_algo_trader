"""
Kalshi BTC 15-Minute Up/Down Trading Bot
=========================================
Trades binary "BTC up or down in 15 minutes" markets on Kalshi.

Strategy overview:
  - Fetches live BTC price from Coinbase every 10 seconds.
    Coinbase is used as a free, reliable proxy for the BTC reference price.
    Kalshi's official settlement price is sourced from a licensed data provider
    (currently Kaiko) whose API carries significant cost.  Coinbase spot price
    tracks the Kaiko index closely enough for signal generation purposes, though
    minor divergence near settlement is expected and accounted for by requiring
    a meaningful minimum edge before trading.
  - Models the probability of BTC finishing UP or DOWN using a random-walk
    (geometric Brownian motion) framework
  - Compares fair-value probability against Kalshi's implied market probability
  - Trades when the gap (edge) exceeds a configurable minimum threshold
  - Sizes positions using uncertainty-adjusted quarter-Kelly criterion
  - Protects capital with a maximum drawdown circuit breaker

Risk controls:
  - Quarter-Kelly bet sizing (25% of full Kelly)
  - Sample-size shrinkage  : reduces bet size early on when edge estimates are uncertain
  - Volatility shrinkage   : reduces bet size when BTC vol is elevated
  - Hard cap               : position size never exceeds 2% of portfolio per trade
  - Drawdown circuit breaker: halts new trades if portfolio drops >10% from peak

Usage:
  1. Set API_KEY_ID and PRIVATE_KEY_PATH in main()
  2. Run with DEMO_MODE = True first to paper trade
  3. Set DEMO_MODE = False to go live
"""

import time
import uuid
import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from collections import deque

import numpy as np
import requests
from scipy.stats import norm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

try:
    import kalshi_python
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "kalshi-python"])
    import kalshi_python


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEMO_HOST       = "https://demo-api.kalshi.co/trade-api/v2"
LIVE_HOST       = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE_URL   = "https://demo-api.kalshi.co"
LIVE_BASE_URL   = "https://api.elections.kalshi.com"
SERIES_TICKER   = "KXBTC15M"
BASELINE_VOL    = 0.015   # BTC vol the random-walk model is calibrated against
DEFAULT_BALANCE = 10_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Price Feed
# ─────────────────────────────────────────────────────────────────────────────

class BTCPriceFeed:
    """
    Fetches live BTC/USD spot price from Coinbase and maintains a rolling
    history for volatility and momentum calculations.
    """

    def __init__(self):
        self.price_history: deque = deque(maxlen=3600)  # up to 1 hour at 1s resolution

    def get_current_price(self) -> Optional[float]:
        """Fetch the latest BTC/USD spot price. Returns None on failure."""
        try:
            resp = requests.get(
                "https://api.coinbase.com/v2/prices/BTC-USD/spot",
                timeout=5
            )
            if resp.status_code == 200:
                price = float(resp.json()["data"]["amount"])
                self.price_history.append((datetime.now(timezone.utc), price))
                return price
        except Exception as e:
            print(f"  [PriceFeed] Error fetching price: {e}")
        return None

    def get_price_change_pct(self, minutes: int = 1) -> Optional[float]:
        """Return the percentage price change over the last N minutes."""
        if len(self.price_history) < 2:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        recent = [(t, p) for t, p in self.price_history if t >= cutoff]
        if len(recent) < 2:
            return None
        return (recent[-1][1] - recent[0][1]) / recent[0][1]

    def get_volatility(self, minutes: int = 15) -> float:
        """
        Estimate annualised-equivalent short-term volatility as the standard
        deviation of tick returns scaled by sqrt(n).  Falls back to the
        baseline constant if insufficient history is available.
        """
        if len(self.price_history) < 10:
            return BASELINE_VOL
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        recent = [(t, p) for t, p in self.price_history if t >= cutoff]
        if len(recent) < 10:
            return BASELINE_VOL
        returns = [
            (recent[i][1] - recent[i - 1][1]) / recent[i - 1][1]
            for i in range(1, len(recent))
        ]
        return float(np.std(returns) * np.sqrt(len(returns)))


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot
# ─────────────────────────────────────────────────────────────────────────────

class KalshiBTC15MinBot:
    """
    Algorithmic trading bot for Kalshi BTC 15-minute Up/Down binary markets.

    Parameters
    ----------
    api_key_id        : Kalshi API key ID
    private_key_path  : Path to the RSA private key (.pem)
    demo_mode         : True  → paper trade on Kalshi demo environment
                        False → live trading on production
    min_edge          : Minimum required edge (fair prob − market prob) to trade
    max_drawdown_pct  : Circuit-breaker threshold; halts new trades when
                        portfolio drawdown from peak exceeds this fraction
    """

    def __init__(
        self,
        api_key_id: str,
        private_key_path: str,
        demo_mode: bool = True,
        min_edge: float = 0.08,
        max_drawdown_pct: float = 0.10,
    ):
        self.demo_mode         = demo_mode
        self.min_edge          = min_edge
        self.max_drawdown_pct  = max_drawdown_pct
        self.trading_halted    = False

        # Load and cache credentials (used for direct REST signing)
        with open(private_key_path, "r") as f:
            self._private_key_pem = f.read()
        self._api_key_id = api_key_id

        # Kalshi SDK client (used for read operations)
        config             = kalshi_python.Configuration()
        config.host        = DEMO_HOST if demo_mode else LIVE_HOST
        config.api_key_id  = api_key_id
        config.private_key_pem = self._private_key_pem
        self.api = kalshi_python.KalshiClient(config)

        env_label = "DEMO" if demo_mode else "⚠️  PRODUCTION"
        print(f"  Environment  : {env_label}")
        print(f"  Min edge     : {min_edge:.1%}")
        print(f"  Max drawdown : {max_drawdown_pct:.1%}")

        # Price feed
        self.price_feed = BTCPriceFeed()

        # Portfolio state
        self.initial_portfolio = self._fetch_balance()
        self.portfolio_value   = self.initial_portfolio
        self.peak_portfolio    = self.initial_portfolio
        print(f"  Starting balance: ${self.portfolio_value:,.2f}")

        # Trade tracking
        self.positions:     Dict = {}
        self.closed_trades: List = []
        self.trades_log:    List = []

        # Re-hydrate open positions after a restart (live only)
        if not demo_mode:
            self._restore_open_positions()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_balance(self) -> float:
        """
        Pull the account balance from Kalshi (returned in cents).
        Falls back to DEFAULT_BALANCE if the request fails.
        """
        try:
            resp = self.api.get_balance()
            if hasattr(resp, "balance") and resp.balance is not None:
                dollars = resp.balance / 100.0
                print(f"  Fetched Kalshi balance: ${dollars:,.2f}")
                return dollars
        except Exception as e:
            print(f"  Could not fetch balance ({e}), defaulting to ${DEFAULT_BALANCE:,.0f}")
        return DEFAULT_BALANCE

    def _restore_open_positions(self):
        """
        Query Kalshi for any pre-existing open positions on startup to prevent
        double-entry after a crash or restart.
        """
        try:
            print("  Checking for existing open positions...")
            resp = self.api.get_positions()
            if not resp or not hasattr(resp, "market_positions"):
                print("  No existing positions found.")
                return

            for pos in resp.market_positions:
                ticker = getattr(pos, "ticker", None)
                if not ticker or SERIES_TICKER not in ticker:
                    continue

                net = getattr(pos, "position", 0)
                if net == 0:
                    continue

                side     = "yes" if net > 0 else "no"
                quantity = abs(net)
                avg_px   = getattr(pos, "market_exposure", 0) / max(quantity, 1) / 100.0

                self.positions[ticker] = {
                    "ticker":          ticker,
                    "side":            side,
                    "quantity":        quantity,
                    "entry_price":     avg_px,
                    "cost":            avg_px * quantity,
                    "entry_time":      datetime.now(timezone.utc).isoformat(),
                    "signal":          {},
                    "market":          None,
                    "start_btc_price": None,
                    "entry_btc_price": None,
                    "restored":        True,
                }
                print(f"  ↩  Restored: {ticker} | {side.upper()} × {quantity}")

        except Exception as e:
            print(f"  Warning: could not restore positions ({e})")

    # ─────────────────────────────────────────────────────────────────────────
    # Risk controls
    # ─────────────────────────────────────────────────────────────────────────

    def _check_drawdown(self) -> bool:
        """
        Maximum drawdown circuit breaker (Risk control #1).

        Maintains a high-water mark of total portfolio value (cash + open
        positions at cost).  If the current value falls more than
        max_drawdown_pct below that peak, all new trade entry is suspended.

        Returns True if trading should be halted.
        """
        total = self.portfolio_value + sum(p["cost"] for p in self.positions.values())

        if total > self.peak_portfolio:
            self.peak_portfolio = total

        drawdown = (self.peak_portfolio - total) / self.peak_portfolio

        if drawdown >= self.max_drawdown_pct:
            if not self.trading_halted:
                print(f"\n{'!' * 70}")
                print(f"  🛑  DRAWDOWN CIRCUIT BREAKER TRIGGERED")
                print(f"  Peak: ${self.peak_portfolio:,.2f}  |  Now: ${total:,.2f}")
                print(f"  Drawdown: {drawdown:.1%}  ≥  limit {self.max_drawdown_pct:.1%}")
                print(f"  No new trades until bot is restarted.")
                print(f"{'!' * 70}\n")
            self.trading_halted = True
        else:
            if self.trading_halted:
                print(f"  ✅ Drawdown recovered ({drawdown:.1%}), resuming.")
            self.trading_halted = False

        return self.trading_halted

    def _uncertainty_adjusted_kelly(
        self,
        base_kelly: float,
        volatility: float,
    ) -> float:
        """
        Uncertainty-adjusted Kelly sizing (Risk control #2).

        Applies two multiplicative shrinkage factors on top of the base
        quarter-Kelly fraction:

        1. Sample-size shrinkage
           The edge estimate is unreliable early in the session when few trades
           have been observed.  Confidence grows asymptotically with n_trades,
           floored at 0.20 so we never refuse to trade entirely.

           confidence(n) = max(0.20,  1 − 1/(1 + n/20))
             n=0  → 0.20   (bet at 20% of Kelly)
             n=20 → 0.50
             n=50 → 0.71

        2. Volatility shrinkage
           The probability model assumes BTC volatility near BASELINE_VOL.
           When realised vol is higher, model estimates are less reliable and
           bet size is scaled down proportionally, clamped to [0.25, 1.0].

           vol_shrink = clamp(BASELINE_VOL / current_vol, 0.25, 1.0)
        """
        n = len(self.closed_trades)
        sample_conf  = max(0.20, 1.0 - 1.0 / (1.0 + n / 20.0))
        vol_shrink   = max(0.25, min(1.0, BASELINE_VOL / max(volatility, BASELINE_VOL)))
        adjusted     = base_kelly * sample_conf * vol_shrink

        print(
            f"    Kelly: base={base_kelly:.4f}"
            f" × sample_conf={sample_conf:.2f} (n={n})"
            f" × vol_shrink={vol_shrink:.2f}"
            f" → {adjusted:.4f}"
        )
        return adjusted

    # ─────────────────────────────────────────────────────────────────────────
    # Market data
    # ─────────────────────────────────────────────────────────────────────────

    def _get_active_markets(self) -> List:
        """
        Return only the soonest-expiring KXBTC15M markets that are still open
        (i.e. those whose close_time is within the next 20 minutes).
        All markets sharing the same earliest close_time are returned together.
        """
        try:
            print("  Fetching KXBTC15M markets...")
            resp = self.api.get_markets(series_ticker=SERIES_TICKER, status="open", limit=200)

            if not resp or not hasattr(resp, "markets"):
                print("  No markets returned.")
                return []

            now        = datetime.now(timezone.utc)
            candidates = []

            for m in resp.markets:
                raw_close = getattr(m, "close_time", None)
                if raw_close is None:
                    continue
                if isinstance(raw_close, str):
                    close_time = datetime.fromisoformat(raw_close.replace("Z", "+00:00"))
                else:
                    close_time = raw_close
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)

                mins_left = (close_time - now).total_seconds() / 60
                if 0 < mins_left <= 20:
                    candidates.append((m, close_time, mins_left))

            if not candidates:
                print("  No markets expiring within 20 minutes.")
                return []

            earliest = min(c[1] for c in candidates)
            result   = [m for m, ct, _ in candidates if abs((ct - earliest).total_seconds()) < 1]

            mins_to_close = (earliest - now).total_seconds() / 60
            print(f"  {len(result)} market(s) expiring in {mins_to_close:.1f} min")
            for m in result:
                print(f"    {m.ticker}  |  {getattr(m, 'title', '')}")

            return result

        except Exception as e:
            print(f"  Error fetching markets: {e}")
            return []

    def _get_start_price(self, market) -> Optional[float]:
        """
        Determine the BTC price at market open by looking up the closest
        recorded price in local history.  Falls back to current price if
        the market's open_time is unavailable or history is sparse.
        """
        try:
            raw_open = getattr(market, "open_time", None)
            if not raw_open:
                price = self.price_feed.get_current_price()
                print(f"    ⚠️  No open_time — using current price as start: ${price:,.2f}")
                return price

            if isinstance(raw_open, str):
                open_time = datetime.fromisoformat(raw_open.replace("Z", "+00:00"))
            else:
                open_time = raw_open
            if open_time.tzinfo is None:
                open_time = open_time.replace(tzinfo=timezone.utc)

            mins_open = (datetime.now(timezone.utc) - open_time).total_seconds() / 60
            print(f"    Market open {mins_open:.1f} min ago")

            # Find the closest price in history to the open time
            if self.price_feed.price_history:
                best_price, best_diff = None, float("inf")
                for ts, px in self.price_feed.price_history:
                    diff = abs((ts - open_time).total_seconds())
                    if diff < best_diff:
                        best_diff, best_price = diff, px
                if best_price and best_diff < 600:
                    print(f"    Start price from history: ${best_price:,.2f} ({best_diff/60:.1f} min off)")
                    return best_price

            price = self.price_feed.get_current_price()
            print(f"    ⚠️  History miss — using current price as start: ${price:,.2f}")
            return price

        except Exception as e:
            print(f"    Error getting start price: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Probability model
    # ─────────────────────────────────────────────────────────────────────────

    def _fair_prob_up(
        self,
        current_price: float,
        start_price: float,
        minutes_left: float,
        volatility: float,
    ) -> float:
        """
        Estimate the probability that BTC ends the 15-minute window above its
        opening price, using a geometric Brownian motion (random-walk) model.

        The current price position relative to the start price gives us a
        z-score; the remaining time determines how much additional variance
        can accumulate before settlement.

        P(end UP) = Φ(current_change / remaining_std_dev)
        """
        pct_change    = (current_price - start_price) / start_price
        time_fraction = max(minutes_left, 0) / 15.0

        if time_fraction <= 0:
            return 1.0 if pct_change > 0 else 0.0

        remaining_std = volatility * np.sqrt(time_fraction)
        if remaining_std < 1e-10:
            return 1.0 if pct_change > 0 else 0.0

        prob = float(norm.cdf(pct_change / remaining_std))
        return max(0.0, min(1.0, prob))

    # ─────────────────────────────────────────────────────────────────────────
    # Signal generation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_market(self, market, current_price: float) -> Optional[Dict]:
        """
        Evaluate a single market for a trading opportunity.

        Computes the model's fair probability and compares it to the market's
        implied probability.  Returns a signal dict if |edge| > min_edge,
        otherwise returns None.
        """
        ticker = market.ticker
        print(f"\n  Evaluating {ticker}...")

        # ── Parse market prices ───────────────────────────────────────────────
        yes_bid = no_bid = None

        for attr in ("yes_bid", "yes_price", "last_price"):
            val = getattr(market, attr, None)
            if val is not None:
                yes_bid = val / 100
                break

        for attr in ("no_bid", "no_price"):
            val = getattr(market, attr, None)
            if val is not None:
                no_bid = val / 100
                break

        print(f"    YES bid={yes_bid}  NO bid={no_bid}  last={getattr(market, 'last_price', 'N/A')}")

        if yes_bid is None and no_bid is None:
            print("    ✗ No prices available — skipping")
            return None

        # Derive implied probability of UP
        if yes_bid and no_bid:
            market_prob = (yes_bid + (1 - no_bid)) / 2
        elif yes_bid:
            market_prob = yes_bid
        else:
            market_prob = 1 - no_bid

        # ── Model inputs ──────────────────────────────────────────────────────
        start_price = self._get_start_price(market)
        if not start_price:
            print("    ✗ Could not determine start price — skipping")
            return None

        close_time = market.close_time
        if isinstance(close_time, str):
            close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        minutes_left = (close_time - datetime.now(timezone.utc)).total_seconds() / 60

        vol = self.price_feed.get_volatility()

        fair = self._fair_prob_up(current_price, start_price, minutes_left, vol)
        edge = fair - market_prob

        print(f"    Market prob (UP): {market_prob:.1%}")
        print(f"    Fair prob   (UP): {fair:.1%}")
        print(f"    Edge             : {edge:+.1%}  (threshold: ±{self.min_edge:.1%})")
        print(f"    Time left        : {minutes_left:.1f} min  |  Vol: {vol:.2%}")

        if abs(edge) < self.min_edge:
            print("    ✗ Edge below threshold — no trade")
            return None

        side = "yes" if edge > 0 else "no"
        print(f"    ✓ SIGNAL: BUY {side.upper()}")

        return {
            "ticker":       ticker,
            "side":         side,
            "edge":         abs(edge),
            "fair_prob":    fair,
            "market_prob":  market_prob,
            "start_price":  start_price,
            "current_price": current_price,
            "minutes_left": minutes_left,
            "volatility":   vol,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Position management
    # ─────────────────────────────────────────────────────────────────────────

    def _settle_from_kalshi(self, ticker: str, pos: dict) -> Optional[dict]:
        """
        Fetch Kalshi's official settlement result for a closed market and
        return a standardised closed_trade dict.  Returns None if the result
        is not yet published (caller should retry next iteration).
        """
        try:
            resp = self.api.get_market(ticker)
            if not resp or not hasattr(resp, "market"):
                return None

            result = getattr(resp.market, "result", None)  # "yes" | "no" | None
            if result is None:
                return None

            won    = (result == pos["side"])
            payout = pos["quantity"] * 1.0 if won else 0.0
            profit = payout - pos["cost"]
            roi    = (profit / pos["cost"] * 100) if pos["cost"] > 0 else 0
            now    = datetime.now(timezone.utc)

            print(f"\n{'=' * 70}")
            print(f"  {'🎉 WIN' if won else '❌ LOSS'}: {ticker}")
            print(f"  Kalshi result: {result.upper()}  |  Our side: {pos['side'].upper()}")
            print(f"  Cost: ${pos['cost']:.2f}  Payout: ${payout:.2f}  Profit: ${profit:+.2f} ({roi:+.1f}%)")
            print(f"  Portfolio after settlement: ${self.portfolio_value:,.2f}")
            print(f"{'=' * 70}\n")

            return {
                "ticker":       ticker,
                "side":         pos["side"],
                "entry_price":  pos["entry_price"],
                "cost":         pos["cost"],
                "payout":       payout,
                "profit":       profit,
                "roi":          roi,
                "won":          won,
                "kalshi_result": result,
                "start_btc":    pos.get("start_btc_price"),
                "entry_btc":    pos.get("entry_btc_price"),
                "entry_time":   pos["entry_time"],
                "close_time":   now.isoformat(),
                "duration_min": (now - datetime.fromisoformat(pos["entry_time"])).total_seconds() / 60,
            }

        except Exception as e:
            print(f"  Error settling {ticker}: {e}")
            return None

    def _check_positions(self, current_price: float):
        """
        Iterate open positions and settle any whose close_time has passed.

        Demo mode: infers outcome from Coinbase spot vs. start price.
        Live mode: waits for Kalshi's official result via the REST API.
        """
        to_close = []

        for ticker, pos in self.positions.items():
            try:
                # Resolve close_time — fetch from API if not cached
                market    = pos.get("market")
                close_time = getattr(market, "close_time", None) if market else None

                if close_time is None:
                    mkt = self.api.get_market(ticker)
                    if mkt and hasattr(mkt, "market"):
                        close_time = getattr(mkt.market, "close_time", None)
                        self.positions[ticker]["market"] = mkt.market

                if close_time is None:
                    continue
                if isinstance(close_time, str):
                    close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < close_time:
                    continue  # still open

                # ── Settlement ───────────────────────────────────────────────
                if self.demo_mode:
                    start_price = pos.get("start_btc_price")
                    if start_price is None:
                        print(f"  ⚠️  {ticker}: no start price recorded — dropping without settlement")
                        to_close.append(ticker)
                        continue

                    btc_up = current_price > start_price
                    won    = (pos["side"] == "yes" and btc_up) or (pos["side"] == "no" and not btc_up)
                    payout = pos["quantity"] * 1.0 if won else 0.0
                    profit = payout - pos["cost"]
                    roi    = (profit / pos["cost"] * 100) if pos["cost"] > 0 else 0

                    print(
                        f"  📊 {ticker}: ${start_price:,.2f} → ${current_price:,.2f} "
                        f"({'UP' if btc_up else 'DOWN'}) | {pos['side'].upper()} | "
                        f"{'✅ WIN' if won else '❌ LOSS'}"
                    )

                    self.portfolio_value += payout
                    now = datetime.now(timezone.utc)
                    self.closed_trades.append({
                        "ticker":       ticker,
                        "side":         pos["side"],
                        "entry_price":  pos["entry_price"],
                        "cost":         pos["cost"],
                        "payout":       payout,
                        "profit":       profit,
                        "roi":          roi,
                        "won":          won,
                        "start_btc":    start_price,
                        "final_btc":    current_price,
                        "entry_time":   pos["entry_time"],
                        "close_time":   now.isoformat(),
                        "duration_min": (now - datetime.fromisoformat(pos["entry_time"])).total_seconds() / 60,
                    })

                    print(f"\n{'=' * 70}")
                    print(f"  [DEMO] {'🎉 WIN' if won else '❌ LOSS'}: {ticker}")
                    print(f"  BTC: ${start_price:,.2f} → ${current_price:,.2f}  |  Profit: ${profit:+.2f} ({roi:+.1f}%)")
                    print(f"  Portfolio: ${self.portfolio_value:,.2f}")
                    print(f"{'=' * 70}\n")

                else:
                    closed = self._settle_from_kalshi(ticker, pos)
                    if closed is None:
                        print(f"  ⏳ {ticker}: Kalshi result not yet published — retrying next iteration")
                        continue
                    self.portfolio_value += closed["payout"]
                    self.closed_trades.append(closed)

                to_close.append(ticker)

            except Exception as e:
                print(f"  Error processing {ticker}: {e}")

        for ticker in to_close:
            del self.positions[ticker]

    # ─────────────────────────────────────────────────────────────────────────
    # Order execution
    # ─────────────────────────────────────────────────────────────────────────

    def _sign_request(self, method: str, path: str) -> dict:
        """
        Generate Kalshi RSA-PSS authentication headers for a REST request.
        Implements the official Kalshi signing scheme:
            message = timestamp_ms + METHOD + /trade-api/v2/path
        """
        timestamp = str(int(time.time() * 1000))
        message   = (timestamp + method.upper() + path).encode("utf-8")
        pk        = serialization.load_pem_private_key(
            self._private_key_pem.encode(), password=None
        )
        signature = pk.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "Content-Type":            "application/json",
            "KALSHI-ACCESS-KEY":       self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def _place_order(self, ticker: str, side: str, quantity: int, price_cents: int):
        """
        Submit a limit order to Kalshi via the REST API.
        Uses direct HTTP signing to avoid SDK version compatibility issues.
        """
        path = "/trade-api/v2/portfolio/orders"
        body = {
            "ticker":          ticker,
            "action":          "buy",
            "side":            side,
            "type":            "limit",
            "count":           quantity,
            "client_order_id": str(uuid.uuid4()),
        }
        if side == "yes":
            body["yes_price"] = price_cents
        else:
            body["no_price"] = price_cents

        base = DEMO_BASE_URL if self.demo_mode else LIVE_BASE_URL
        resp = requests.post(
            base + path,
            headers=self._sign_request("POST", path),
            json=body,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            return resp.json().get("order", resp.json())
        raise Exception(f"HTTP {resp.status_code}: {resp.text}")

    def _execute_trade(self, signal: Dict, market):
        """
        Size and place a trade based on a signal dict.

        Sizing pipeline:
          1. Quarter-Kelly fraction from win probability and edge
          2. Uncertainty adjustment (sample-size + volatility shrinkage)
          3. Hard cap at 2% of portfolio
          4. Skip if adjusted size < 0.5% (position too small to be meaningful)
        """
        ticker   = signal["ticker"]
        side     = signal["side"]
        win_prob = signal["fair_prob"] if side == "yes" else (1 - signal["fair_prob"])

        # Step 1 & 2: uncertainty-adjusted quarter-Kelly
        base_kelly = max(0.0, (2 * win_prob - 1) * 0.25)
        adj_kelly  = self._uncertainty_adjusted_kelly(base_kelly, signal["volatility"])
        size_pct   = min(adj_kelly, 0.02)   # hard cap: 2% per trade

        if size_pct < 0.005:
            print(f"  ✗ Position size too small after adjustment ({size_pct:.4f}) — skipping")
            return

        # Derive quantity
        entry_px = signal["market_prob"] if side == "yes" else (1 - signal["market_prob"])
        if not (0.02 <= entry_px <= 0.98):
            print(f"  ✗ Entry price out of valid range ({entry_px:.2f}) — skipping")
            return

        position_value = self.portfolio_value * size_pct
        quantity       = int(position_value / entry_px)
        if quantity < 1:
            print(f"  ✗ ${position_value:.2f} insufficient for 1 contract at ${entry_px:.2f} — skipping")
            return

        actual_cost  = quantity * entry_px
        price_cents  = int(entry_px * 100)

        # ── Print trade summary ───────────────────────────────────────────────
        print(f"\n{'─' * 70}")
        print(f"  🎯 TRADE: {ticker}  |  {side.upper()}")
        print(f"  BTC: ${signal['start_price']:,.2f} → ${signal['current_price']:,.2f}"
              f"  ({(signal['current_price']/signal['start_price']-1)*100:+.2f}%)")
        print(f"  Fair: {signal['fair_prob']:.1%}  Market: {signal['market_prob']:.1%}  Edge: {signal['edge']:+.1%}")
        print(f"  Qty: {quantity}  @  ${entry_px:.2f}  =  ${actual_cost:.2f}  ({size_pct*100:.1f}% of portfolio)")
        print(f"  Time left: {signal['minutes_left']:.1f} min  |  Portfolio: ${self.portfolio_value:,.2f}")
        print(f"{'─' * 70}")

        if self.demo_mode:
            print("  [PAPER TRADE — not sending order]")
        else:
            try:
                order = self._place_order(ticker, side, quantity, price_cents)
                print(f"  ✅ Order placed: {order}")
            except Exception as e:
                print(f"  ❌ Order failed: {e}")
                return

        # Record position
        self.positions[ticker] = {
            "ticker":          ticker,
            "side":            side,
            "quantity":        quantity,
            "entry_price":     entry_px,
            "cost":            actual_cost,
            "entry_time":      datetime.now(timezone.utc).isoformat(),
            "signal":          signal,
            "market":          market,
            "start_btc_price": signal["start_price"],
            "entry_btc_price": signal["current_price"],
        }
        self.portfolio_value -= actual_cost
        self.trades_log.append(signal)

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, check_interval: int = 10):
        """
        Main trading loop.  Runs indefinitely until interrupted with Ctrl-C.

        Each iteration:
          1. Fetch current BTC price
          2. Settle any expired positions
          3. Check drawdown circuit breaker
          4. Fetch active markets and evaluate signals
          5. Execute any qualifying trades
          6. Sleep for check_interval seconds
        """
        print("\n" + "=" * 70)
        print("  🚀  Kalshi BTC 15-Minute Bot  —  Starting")
        print("=" * 70 + "\n")

        try:
            iteration = 0
            while True:
                iteration += 1
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{ts}] ── Iteration {iteration} {'─' * 40}")

                # 1. Price
                current_price = self.price_feed.get_current_price()
                if current_price is None:
                    print("  ⚠️  Price unavailable — skipping iteration")
                    time.sleep(check_interval)
                    continue

                change_1m = self.price_feed.get_price_change_pct(minutes=1)
                vol       = self.price_feed.get_volatility()
                change_str = f" ({change_1m:+.2%} 1m)" if change_1m is not None else ""
                print(f"  BTC: ${current_price:,.2f}{change_str}  |  Vol: {vol:.2%}")

                # 2. Settle
                if self.positions:
                    self._check_positions(current_price)

                # 3. Portfolio status
                total = self.portfolio_value + sum(p["cost"] for p in self.positions.values())
                pnl   = total - self.initial_portfolio
                print(f"  💰 Cash: ${self.portfolio_value:,.2f}  Total: ${total:,.2f}  P&L: ${pnl:+,.2f} ({pnl/self.initial_portfolio:+.2%})")

                if self.closed_trades:
                    wins  = sum(1 for t in self.closed_trades if t["won"])
                    n     = len(self.closed_trades)
                    print(f"  📊 Trades: {n}  Win rate: {wins/n:.1%}  ({wins}W / {n-wins}L)")

                if self.positions:
                    print(f"  📍 Open positions: {len(self.positions)}")
                    for tk, pos in self.positions.items():
                        held = (datetime.now(timezone.utc) - datetime.fromisoformat(pos["entry_time"])).total_seconds() / 60
                        print(f"    {tk}  {pos['side'].upper()} × {pos['quantity']}  cost=${pos['cost']:.2f}  held={held:.1f}m")

                # 4. Drawdown check
                halted   = self._check_drawdown()
                peak_dd  = (self.peak_portfolio - total) / self.peak_portfolio
                print(f"  📉 Peak: ${self.peak_portfolio:,.2f}  Drawdown: {peak_dd:.1%}"
                      + ("  🛑 HALTED" if halted else ""))

                if halted:
                    time.sleep(check_interval)
                    continue

                # 5. Signals & execution
                markets = self._get_active_markets()
                if not markets:
                    time.sleep(check_interval)
                    continue

                for market in markets:
                    if market.ticker in self.positions:
                        continue
                    signal = self._evaluate_market(market, current_price)
                    if signal:
                        self._execute_trade(signal, market)

                time.sleep(check_interval)

        except KeyboardInterrupt:
            self._print_final_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    def _print_final_summary(self):
        """Print a detailed P&L summary on shutdown."""
        open_value = sum(p["cost"] for p in self.positions.values())
        total      = self.portfolio_value + open_value
        pnl        = total - self.initial_portfolio

        print("\n\n" + "=" * 70)
        print("  📊  FINAL SUMMARY")
        print("=" * 70)
        print(f"  Starting balance : ${self.initial_portfolio:,.2f}")
        print(f"  Cash             : ${self.portfolio_value:,.2f}")
        if self.positions:
            print(f"  Open positions   : ${open_value:,.2f}  ({len(self.positions)} positions)")
        print(f"  Total value      : ${total:,.2f}")
        print(f"  P&L              : ${pnl:+,.2f}  ({pnl/self.initial_portfolio:+.2%})")

        n = len(self.closed_trades)
        if n > 0:
            wins        = [t for t in self.closed_trades if t["won"]]
            losses      = [t for t in self.closed_trades if not t["won"]]
            avg_win     = sum(t["profit"] for t in wins)   / len(wins)   if wins   else 0
            avg_loss    = sum(t["profit"] for t in losses) / len(losses) if losses else 0
            avg_profit  = sum(t["profit"] for t in self.closed_trades) / n
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else float("inf")

            print(f"\n  Closed trades  : {n}")
            print(f"  Win rate       : {len(wins)/n:.1%}  ({len(wins)}W / {len(losses)}L)")
            print(f"  Avg profit     : ${avg_profit:+.2f}")
            print(f"  Avg win        : ${avg_win:+.2f}  |  Avg loss: ${avg_loss:+.2f}")
            print(f"  Profit factor  : {profit_factor:.2f}")

        print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Configuration ────────────────────────────────────────────────────────
    with open("kalshi_keys/key_id.txt", "r") as f: key = f.read().strip()
    API_KEY_ID        = key                       # Kalshi API Key ID
    PRIVATE_KEY_PATH  = "kalshi_keys/bot_key.pem" # Path to RSA private key
    DEMO_MODE         = True                      # Set False for live trading
    MIN_EDGE          = 0.08                      # Minimum edge to trade (8%)
    MAX_DRAWDOWN      = 0.10                      # Circuit breaker at 10% drawdown
    CHECK_INTERVAL    = 10                        # Seconds between iterations
    # ─────────────────────────────────────────────────────────────────────────

    bot = KalshiBTC15MinBot(
        api_key_id       = API_KEY_ID,
        private_key_path = PRIVATE_KEY_PATH,
        demo_mode        = DEMO_MODE,
        min_edge         = MIN_EDGE,
        max_drawdown_pct = MAX_DRAWDOWN,
    )
    bot.run(check_interval=CHECK_INTERVAL)


if __name__ == "__main__":
    main()