"""Unit tests for the reward:risk maths, the setup scan and the reversal trigger."""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swing_app as s  # noqa: E402

P = dict(s.DEFAULTS)  # rsi_max 35, bb_tol 1.01, min_rel_vol 1.5, rr 2.0, atr_mult 1.5, max_risk 35


def row(**kw):
    """A quiet stock: no setup, no trigger. Override fields per test."""
    base = {
        "open": 100.0, "close": 100.0, "RSI": 50.0, "BB.lower": 90.0, "relative_volume_10d_calc": 1.0,
        "open[1]": 100.0, "close[1]": 100.0, "RSI[1]": 50.0, "BB.lower[1]": 90.0,
        "relative_volume_10d_calc[1]": 1.0,
        "close[2]": 100.0, "RSI[2]": 50.0, "BB.lower[2]": 90.0, "earnings_release_next_date": None,
    }
    base.update(kw)
    return pd.Series(base)


# ---------------------------------------------------------------- reward:risk
def test_reward_risk_basic():
    assert s.reward_risk(100, 95, 110) == pytest.approx(2.0)
    assert s.reward_risk(100, 95, 107) == pytest.approx(1.4)


def test_reward_risk_target_below_entry_is_negative():
    assert s.reward_risk(100, 95, 98) < 0


def test_reward_risk_invalid_stop_is_zero():
    assert s.reward_risk(100, 100, 110) == 0.0


def test_build_plan_uses_sma20_target_and_atr_stop():
    plan = s.build_plan(100.0, 4.0, 115.0, P)  # stop = 100 - 1.5*4 = 94
    assert plan["stop"] == 94.0
    assert plan["target"] == 115.0
    assert plan["rr"] == pytest.approx(15 / 6)
    assert plan["target_pct"] == 15.0
    assert plan["shares"] == 5  # floor(35 / 6)
    assert plan["max_entry"] == 103.0  # 115 - 2.0 * 6


def test_rr_check_message_uses_live_minimum():
    plan = s.build_plan(100.0, 4.0, 108.4, dict(P, rr=3.0))  # rr = 8.4 / 6 = 1.4
    cks = s.checks(row(**{"RSI[1]": 30, "close[1]": 89}), plan, [], dict(P, rr=3.0), "closed", 1,
                   {"ok": True, "above_open": True, "above_prev": True, "rsi_up": True, "volume": True,
                    "rel_vol": 2.0, "green": True})
    rr_line = [c for c in cks if c[0].startswith("Reward:risk")][0]
    assert rr_line[0] == "Reward:risk at least 1:3 to the 20-day average"
    assert rr_line[1] == "target is only 1:1.4, below your 1:3 minimum"
    assert rr_line[2] is False


def test_rr_check_passes_at_minimum():
    plan = s.build_plan(100.0, 4.0, 112.0, P)  # exactly 1:2
    assert plan["rr"] == pytest.approx(2.0)
    assert plan["rr"] >= P["rr"] - 1e-9


# ---------------------------------------------------------------- setup
def test_setup_in_latest_session():
    r = row(RSI=30, close=89)
    assert s.find_setup(r, P, "closed") == 0


def test_setup_two_sessions_ago():
    r = row(**{"RSI[2]": 28, "close[2]": 88})
    assert s.find_setup(r, P, "closed") == 2


def test_most_recent_setup_wins():
    r = row(**{"RSI[1]": 30, "close[1]": 89, "RSI[2]": 25, "close[2]": 85})
    assert s.find_setup(r, P, "closed") == 1


def test_setup_needs_both_rsi_and_band():
    assert s.find_setup(row(RSI=30, close=95), P, "closed") is None  # RSI low, above band
    assert s.find_setup(row(RSI=40, close=89), P, "closed") is None  # at band, RSI not low


def test_setup_band_tolerance():
    r = row(RSI=30, close=90.8)  # 0.9% above the band
    assert s.find_setup(r, P, "closed") == 0
    assert s.find_setup(r, dict(P, bb_tol=1.0), "closed") is None


def test_setup_while_open_skips_forming_bar():
    # bar 0 is still forming during market hours, so only bars 1 and 2 count
    r = row(RSI=30, close=89)
    assert s.find_setup(r, P, "open") is None
    r = row(**{"RSI[2]": 30, "close[2]": 89})
    assert s.find_setup(r, P, "open") == 1  # one session before the latest completed one


def test_setup_missing_fields_are_ignored():
    r = row(**{"RSI[2]": None, "close[2]": None})
    assert s.find_setup(r, P, "closed") is None


# ---------------------------------------------------------------- trigger
def green_day(**kw):
    base = {"open": 90.0, "close": 93.0, "close[1]": 91.0, "RSI": 33.0, "RSI[1]": 28.0,
            "relative_volume_10d_calc": 1.8}
    base.update(kw)
    return row(**base)


def test_trigger_passes():
    t = s.trigger(green_day(), P, "closed")
    assert t["ok"] and t["green"] and t["rsi_up"] and t["volume"]


def test_trigger_fails_red_candle():
    t = s.trigger(green_day(open=94.0), P, "closed")
    assert not t["ok"] and not t["above_open"]
    assert "closed red (below its open)" in s.trigger_failures(t, P)


def test_trigger_fails_below_previous_close():
    t = s.trigger(green_day(**{"close[1]": 95.0}), P, "closed")
    assert not t["ok"] and t["above_open"] and not t["above_prev"]
    assert "closed below the previous close" in s.trigger_failures(t, P)


def test_trigger_fails_rsi_falling():
    t = s.trigger(green_day(**{"RSI[1]": 35.0}), P, "closed")
    assert not t["ok"] and "RSI still falling" in s.trigger_failures(t, P)


def test_trigger_fails_low_volume():
    t = s.trigger(green_day(relative_volume_10d_calc=1.2), P, "closed")
    assert not t["ok"] and "volume only 1.2x average" in s.trigger_failures(t, P)


def test_trigger_while_open_uses_previous_completed_bar():
    r = row(**{"open[1]": 90.0, "close[1]": 93.0, "close[2]": 91.0, "RSI[1]": 33.0, "RSI[2]": 28.0,
               "relative_volume_10d_calc[1]": 2.0},
            open=95.0, close=94.0)  # today's forming bar is red, but it doesn't count yet
    assert s.trigger(r, P, "open")["ok"]
    assert not s.trigger(r, P, "closed")["ok"]


# ---------------------------------------------------------------- settings
def test_load_settings_ignores_old_stop_mode(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"stop_mode": "Target %", "target_pct": 4.0, "rr": 3.0, "max_risk": 50}))
    monkeypatch.setattr(s, "SETTINGS", str(f))
    out = s.load_settings()
    assert "stop_mode" not in out and "target_pct" not in out
    assert out["rr"] == 3.0 and out["max_risk"] == 50


def test_load_settings_clamps_rr(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"rr": 9.0}))
    monkeypatch.setattr(s, "SETTINGS", str(f))
    assert s.load_settings()["rr"] == s.MAX_RR


NO_TRIG = {"ok": False, "above_open": False, "above_prev": False, "rsi_up": False, "volume": False,
           "rel_vol": 1.0, "green": False}


def test_rsi_line_without_setup_says_so():
    plan = s.build_plan(100.0, 4.0, 115.0, P)
    cks = s.checks(row(RSI=25), plan, [], P, "premarket", None, NO_TRIG)
    rsi_line = [c for c in cks if c[0].startswith("RSI between")][0]
    assert rsi_line[1] == "no setup day to check RSI on" and rsi_line[2] is False


def test_rsi_floor_on_setup_day():
    plan = s.build_plan(100.0, 4.0, 115.0, P)
    cks = s.checks(row(RSI=14, close=85), plan, [], P, "closed", 0, NO_TRIG)
    rsi_line = [c for c in cks if c[0].startswith("RSI between")][0]
    assert "below 20: falling-knife zone" in rsi_line[1] and rsi_line[2] is False
