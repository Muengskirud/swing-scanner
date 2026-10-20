"""Unit tests for the reward:risk maths, the setup scan and the reversal trigger."""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swing_app as s  # noqa: E402

P = dict(s.DEFAULTS)  # rsi_max 35, bb_tol 1.01, min_rel_vol 1.5, rr 3.0, atr_mult 1.5, max_risk 35


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


def test_build_plan_fixed_r_target_and_atr_stop():
    plan = s.build_plan(100.0, 4.0, P)  # stop = 100 - 1.5*4 = 94, target = 100 + 3*6 = 118
    assert plan["stop"] == 94.0
    assert plan["target"] == 118.0
    assert plan["rr"] == pytest.approx(3.0)
    assert plan["target_pct"] == 18.0
    assert plan["shares"] == 5  # floor(35 / 6)


def test_build_plan_follows_rr_setting():
    plan = s.build_plan(100.0, 4.0, dict(P, rr=2.0))
    assert plan["target"] == 112.0 and plan["rr"] == pytest.approx(2.0)


def test_swing_defaults_match_backtest_winner():
    assert s.DEFAULTS["rr"] == 3.0 and s.DEFAULTS["max_hold"] == 25 and s.DEFAULTS["atr_mult"] == 1.5


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
    plan = s.build_plan(100.0, 4.0, P)
    cks = s.checks(row(RSI=25), plan, [], P, "premarket", None, NO_TRIG)
    rsi_line = [c for c in cks if c[0].startswith("RSI between")][0]
    assert rsi_line[1] == "no setup day to check RSI on" and rsi_line[2] is False


def test_rsi_floor_on_setup_day():
    plan = s.build_plan(100.0, 4.0, P)
    cks = s.checks(row(RSI=14, close=85), plan, [], P, "closed", 0, NO_TRIG)
    rsi_line = [c for c in cks if c[0].startswith("RSI between")][0]
    assert "below 20: falling-knife zone" in rsi_line[1] and rsi_line[2] is False



# ---------------------------------------------------------------- position mode
def fundamentals(**kw):
    base = {"recommendation_mark": 1.4, "total_revenue_yoy_growth_ttm": 24.0,
            "price_earnings_growth_ttm": 0.3, "free_cash_flow_margin_ttm": 28.0}
    base.update(kw)
    return pd.Series(base)


def pos_line(cks, start):
    return [c for c in cks if c[0].startswith(start)][0]


def test_position_plan():
    plan = s.position_plan(100.0, 160.0, P)  # 15% stop -> 85
    assert plan["stop"] == 85.0 and plan["target"] == 160.0
    assert plan["target_pct"] == 60.0
    assert plan["rr"] == pytest.approx(4.0)
    assert plan["shares"] == 13  # floor(200 / 15): position trades use their own $200 max loss
    assert plan["risk"] == 195.0


def test_rating_label():
    assert s.rating_label(1.1) == "Strong buy"
    assert s.rating_label(2.0) == "Buy"
    assert s.rating_label(3.0) == "Hold"
    assert s.rating_label(None) == "no rating"


def test_position_checks_pass_for_a_good_name():
    plan = s.position_plan(100.0, 160.0, P)
    assert all(ok for _, _, ok in s.position_checks(fundamentals(), plan, [], P))


def test_position_checks_fail_reasons():
    plan = s.position_plan(100.0, 110.0, P)
    cks = s.position_checks(fundamentals(recommendation_mark=2.9, total_revenue_yoy_growth_ttm=3.0,
                                         price_earnings_growth_ttm=None, free_cash_flow_margin_ttm=-5.0),
                            plan, [], P)
    assert pos_line(cks, "At least 20%")[1] == "only +10% to the analysts' target"
    assert pos_line(cks, "Analysts rate")[1] == "analysts rate it Hold (2.9)"
    assert pos_line(cks, "Revenue growing")[1] == "revenue only +3% a year"
    assert pos_line(cks, "PEG")[1] == "PEG unknown (no earnings growth to measure)"
    assert pos_line(cks, "Positive free cash flow")[1] == "burning cash (free cash flow margin -5%)"


def test_position_peg_too_high():
    plan = s.position_plan(100.0, 160.0, P)
    cks = s.position_checks(fundamentals(price_earnings_growth_ttm=2.4), plan, [], P)
    assert pos_line(cks, "PEG")[1] == "PEG 2.4: pricey for its growth" and pos_line(cks, "PEG")[2] is False


# ---------------------------------------------------------------- 2x ETF option
def test_leveraged_plan_doubles_distances_and_halves_dollars():
    plan = s.build_plan(100.0, 4.0, P)  # stock: stop -6%, target +18%, $35 risk = $583 of stock
    lev = s.leveraged_plan(plan, P)
    assert lev["stop_pct"] == pytest.approx(12.0)
    assert lev["target_pct"] == pytest.approx(36.0)
    assert lev["dollars"] == pytest.approx(35 / 0.12)  # same $35 risk, half the dollars


def test_leveraged_map_has_the_users_etfs():
    assert "MULL" in s.LEVERAGED_2X["MU"] and "SNXX" in s.LEVERAGED_2X["SNDK"]
    assert all(e.isupper() and 3 <= len(e) <= 5 for v in s.LEVERAGED_2X.values() for e in v)


def test_each_mode_has_its_own_max_loss():
    assert s.DEFAULTS["max_risk"] == 35 and s.DEFAULTS["pos_max_risk"] == 200
    swing = s.build_plan(100.0, 4.0, P)
    assert swing["risk"] <= 35  # swing sizing ignores the position max loss
    assert s.DEFAULTS["pos_peg"] == 1.0
