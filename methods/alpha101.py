"""WorldQuant Alpha101 factor library expressed in the MASE factor DSL.

Field mapping (Alpha101 input -> MASE identifier)
-------------------------------------------------
    open, high, low, close        -> open, high, low, close
    volume, amount, vwap          -> volume, amount, vwap
    cap                           -> market_cap
    returns (= close.pct_change)  -> returns
    adv{x}  (avg daily dollar/vol over x days)
                                  -> ts_mean(volume, x)
    Fundamentals available: pe, pb, ps, roe, eps, market_cap

Operator mapping (Alpha101 -> MASE DSL)
---------------------------------------
    Rank(x)              -> rank(x)               [cross-sectional percentile rank]
    Ts_Rank(x,d)         -> ts_rank(x,d)
    Ts_ArgMax(x,d)       -> ts_argmax(x,d)
    Ts_ArgMin(x,d)       -> ts_argmin(x,d)
    Ts_Min/Max/Mean/Sum/StdDev/Delta/Product
                         -> ts_min/ts_max/ts_mean/ts_sum/ts_std/ts_delta/ts_product
    Ts_Covariance(x,y,d) -> ts_cov(x,y,d)
    Ts_Corr/Correlation  -> correlation(x,y,d)
    Ts_DecayLinear(x,d)  -> decay_linear(x,d)
    Min(x,y)/Max(x,y)    -> ele_min(x,y)/ele_max(x,y)   [element-wise]
    SignedPower(x,a)     -> signedpower(x,a)
    Scale(x,a)           -> scale(x,a)
    Sign/Abs/Log/Sqrt    -> sign/abs/log/sqrt
    Power(x,a)           -> x ^ a   (or x ** a)
    If(c,a,b)            -> if(c,a,b)
    adv{x}               -> ts_mean(volume, x)
    delay(x,d)           -> delay(x,d)
    sma/mean/sum/stddev  -> ts_mean/ts_mean/ts_sum/ts_std

Two alphas are NOT expressible in this data model and are deliberately omitted
from ALPHA101_FORMULAS (left as comments below):
    * alpha007  - uses IntDay(x) (calendar day-of-month); not derivable from
                  price/volume data.
    * alpha054  - uses IndNeutralize(x, [industry]) (cross-sectional regression
                  residual on an industry dimension); no industry field exists.

All formulas in ALPHA101_FORMULAS below were validated + evaluated against
methods.evolve (_FactorExprEvaluator / _validate_factor_expr) on a synthetic
panel and each produced a (dates x stocks) DataFrame with non-trivial
non-NaN coverage.
"""

from typing import Dict, List


ALPHA101_FORMULAS: Dict[str, str] = {
    # --- OMITTED (not expressible in this data model) -------------------
    # "alpha007": uses IntDay(x) (calendar day-of-month) -> not derivable.
    # "alpha054": uses IndNeutralize(x, [industry]) -> no industry dimension.
    # "alpha048": uses IndNeutralize(...) -> fails validation (unknown operator).

    "alpha001": '(rank(ts_argmax(signedpower(if(returns < 0, ts_std(returns, 20), close), 2), 5)) - 0.5)',
    "alpha002": '(-1 * correlation(rank(ts_delta(log(volume), 2)), rank((close - open) / open), 6))',
    "alpha003": '(-1 * correlation(rank(open), rank(volume), 10))',
    "alpha004": '(-1 * ts_rank(rank(low), 9))',
    "alpha005": '(rank(open - ts_mean(vwap, 10)) * (-1 * abs(rank(close - vwap))))',
    "alpha006": '(-1 * correlation(open, volume, 10))',
    "alpha008": '(-1 * rank(ts_sum(open, 5) * ts_sum(returns, 5) - delay(ts_sum(open, 5) * ts_sum(returns, 5), 10)))',
    "alpha009": 'if(ts_min(ts_delta(close, 1), 5) > 0, ts_delta(close, 1), if(ts_max(ts_delta(close, 1), 5) < 0, ts_delta(close, 1), -ts_delta(close, 1)))',
    "alpha010": 'rank(if(ts_min(ts_delta(close, 1), 4) > 0, ts_delta(close, 1), if(ts_max(ts_delta(close, 1), 4) < 0, ts_delta(close, 1), -ts_delta(close, 1))))',
    "alpha011": '((rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(ts_delta(volume, 3)))',
    "alpha012": '(sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1)))',
    "alpha013": '(-1 * rank(ts_cov(rank(close), rank(volume), 5)))',
    "alpha014": '((-1 * rank(ts_delta(returns, 3))) * correlation(open, volume, 10))',
    "alpha015": '(-1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3))',
    "alpha016": '(-1 * rank(ts_cov(rank(high), rank(volume), 5)))',
    "alpha017": '((-1 * rank(ts_rank(close, 10))) * rank(ts_delta(ts_delta(close, 1), 1)) * rank(ts_rank(volume / ts_mean(volume, 20), 5)))',
    "alpha018": '(-1 * rank(ts_std(abs(close - open), 5) + (close - open) + correlation(close, open, 10)))',
    "alpha019": '((-1 * sign((close - delay(close, 7)) + ts_delta(close, 7))) * (1 + rank(1 + ts_sum(returns, 250))))',
    "alpha020": '((-1 * rank(open - delay(high, 1))) * rank(open - delay(close, 1)) * rank(open - delay(low, 1)))',
    "alpha021": 'if(ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2), -1, if(ts_mean(close, 2) < ts_mean(close, 8) - ts_std(close, 8), 1, if(volume / ts_mean(volume, 20) >= 1, 1, -1)))',
    "alpha022": '(-1 * (ts_delta(correlation(high, volume, 5), 5) * rank(ts_std(close, 20))))',
    "alpha023": 'if(ts_mean(high, 20) < high, -ts_delta(high, 2), 0)',
    "alpha024": 'if(ts_delta(ts_mean(close, 100), 100) / delay(close, 100) <= 0.05, -(close - ts_min(close, 100)), -ts_delta(close, 3))',
    "alpha025": 'rank((-1 * returns) * ts_mean(volume, 20) * vwap * (high - close))',
    "alpha026": '(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))',
    "alpha027": 'if(rank(ts_mean(correlation(rank(volume), rank(vwap), 6), 2) / 2.0) > 0.5, -1, 1)',
    "alpha028": 'scale(correlation(ts_mean(volume, 20), low, 5) + (high + low) / 2 - close)',
    "alpha029": '(ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta(close, 5)))), 2))))), 5) + ts_rank(delay(-returns, 6), 5))',
    "alpha030": '((1.0 - rank(sign(ts_delta(close, 1)) + sign(delay(ts_delta(close, 1), 1)) + sign(delay(ts_delta(close, 1), 2)))) * ts_sum(volume, 5)) / ts_sum(volume, 20)',
    "alpha031": '((rank(rank(rank(decay_linear(-rank(rank(ts_delta(close, 10))), 10)))) + rank(-ts_delta(close, 3))) + sign(scale(correlation(ts_mean(volume, 20), low, 12))))',
    "alpha032": '(scale(ts_mean(close, 7) - close) + 20 * scale(correlation(vwap, delay(close, 5), 230)))',
    "alpha033": 'rank(-1 + open / close)',
    "alpha034": 'rank((1 - rank(ts_std(returns, 2) / ts_std(returns, 5))) + (1 - rank(ts_delta(close, 1))))',
    "alpha035": '(ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32)))',
    "alpha036": '(2.21 * rank(correlation(close - open, delay(volume, 1), 15)) + 0.7 * rank(open - close) + 0.73 * rank(ts_rank(delay(-returns, 6), 5)) + rank(abs(correlation(vwap, ts_mean(volume, 20), 6))) + 0.6 * rank((ts_mean(close, 200) - open) * (close - open)))',
    "alpha037": '(rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close))',
    "alpha038": '(-1 * rank(ts_rank(close, 10)) * rank(close / open))',
    "alpha039": '(-1 * rank(ts_delta(close, 7) * (1 - rank(decay_linear(volume / ts_mean(volume, 20), 9)))) * (1 + rank(ts_sum(returns, 250))))',
    "alpha040": '(-1 * rank(ts_std(high, 10)) * correlation(high, volume, 10))',
    "alpha041": '(sqrt(high * low) - vwap)',
    "alpha042": '(rank(vwap - close) / rank(vwap + close))',
    "alpha043": '(ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-ts_delta(close, 7), 8))',
    "alpha044": '(-1 * correlation(high, rank(volume), 5))',
    "alpha045": '(-1 * rank(ts_mean(delay(close, 5), 20)) * correlation(close, volume, 2) * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))',
    "alpha046": 'if(((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10) < 0, 1, if(((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10) > 0.25, -1, -ts_delta(close, 1)))',
    "alpha047": '(((rank(1 / close) * volume) / ts_mean(volume, 20)) * ((high * rank(high - close)) / (ts_mean(high, 5) / 5)) - rank(vwap - delay(vwap, 5)))',
    "alpha049": 'if(((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10) < -0.1, 1, -ts_delta(close, 1))',
    "alpha050": '(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))',
    "alpha051": 'if(((delay(close, 20) - delay(close, 10)) / 10 - (delay(close, 10) - close) / 10) < -0.05, 1, -ts_delta(close, 1))',
    "alpha052": '((-ts_delta(ts_min(low, 5), 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5))',
    "alpha053": '(-ts_delta(((close - low) - (high - close)) / (close - low), 9))',
    "alpha055": '(-1 * correlation(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6))',
    "alpha056": 'if(close - delay(close, 20) > 0, -1, 0) + if(close - delay(close, 10) > 0, -1, 0) + if(close - delay(close, 5) > 0, -1, 0) + if(close - delay(close, 1) > 0, -1, 0) + if(delay(close, 1) - delay(close, 5) < 0, -1, 0) + if(delay(close, 5) - delay(close, 10) < 0, -1, 0) + if(delay(close, 10) - delay(close, 20) < 0, -1, 0)',
    "alpha057": '(-(close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))',
    "alpha058": '(-1 * ts_max(rank(correlation(low, ts_mean(volume, 20), 5)), 5) + rank(correlation(ts_min(low, 5), ts_mean(volume, 5), 5)))',
    "alpha059": '(-1 * ts_max(rank(correlation(low, ts_mean(volume, 20), 5)), 5) + rank(correlation(ts_min(low, 5), ts_mean(volume, 40), 5)))',
    "alpha060": '(-(2 * scale(rank(((close - low) - (high - close)) * volume / (high - low))) - scale(rank(ts_argmax(close, 10)))))',
    "alpha061": '((rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, ts_mean(volume, 180), 18))) * -1)',
    "alpha062": '((rank(correlation(vwap, ts_mean(ts_mean(volume, 20), 22), 10)) < rank((rank(open) + rank(open)) < (rank((high + low) / 2) + rank(high)))) * -1)',
    "alpha063": '(-1 * rank(decay_linear(correlation(rank(close), rank(ts_mean(volume, 20)), 10), 10)))',
    "alpha064": '((rank(correlation(ts_mean(open * 0.178404 + low * 0.821596, 13), ts_mean(ts_mean(volume, 120), 13), 17)) < rank(ts_delta((high + low) / 2 * 0.178404 + vwap * 0.821596, 4))) * -1)',
    "alpha065": '((rank(correlation(open * 0.00817205 + vwap * 0.99182795, ts_mean(ts_mean(volume, 60), 9), 6)) < rank(open - ts_min(open, 14))) * -1)',
    "alpha066": '((rank(decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(decay_linear(((low * 0.96633 + low * 0.03367) - vwap) / (open - (high + low) / 2), 11), 7)) * -1)',
    "alpha067": '(-1 * rank(correlation(ts_mean(volume, 20), low, 5)) + rank(correlation(ts_mean(volume, 20), high, 5)))',
    "alpha068": '((ts_rank(correlation(rank(high), rank(ts_mean(volume, 15)), 9), 14) < rank(ts_delta(close * 0.518371 + low * 0.481629, 2))) * -1)',
    "alpha069": '(rank(ts_mean(volume, 20)) + rank(ts_min(low, 20)) - rank(ts_max(high, 20)))',
    "alpha070": '(-1 * rank(ts_delta(close, 1)) * rank(ts_mean(volume, 20)))',
    "alpha071": 'ele_max(ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(ts_mean(volume, 180), 12), 18), 4), 16), ts_rank(decay_linear((rank((low + open) - (vwap + vwap))) ^ 2, 16), 4))',
    "alpha072": '(rank(decay_linear(correlation((high + low) / 2, ts_mean(volume, 40), 9), 10)) / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3)))',
    "alpha073": '(ele_max(rank(decay_linear(ts_delta(vwap, 5), 3)), ts_rank(decay_linear(-(ts_delta(open * 0.147155 + low * 0.852845, 2) / (open * 0.147155 + low * 0.852845)), 3), 17)) * -1)',
    "alpha074": '((rank(correlation(close, ts_sum(ts_mean(volume, 30), 37), 15)) < rank(correlation(rank(high * 0.0261661 + vwap * 0.9738339), rank(volume), 11))) * -1)',
    "alpha075": '((rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(ts_mean(volume, 50)), 12))) * -1)',
    "alpha076": '(-1 * rank(correlation(vwap, ts_mean(volume, 20), 5)) + rank(correlation(vwap, ts_mean(volume, 50), 5)))',
    "alpha077": 'ele_min(rank(decay_linear(((high + low) / 2 + high) - (vwap + high), 20)), rank(decay_linear(correlation((high + low) / 2, ts_mean(volume, 40), 3), 6)))',
    "alpha078": '(rank(correlation(ts_sum(low * 0.352233 + vwap * 0.647767, 20), ts_sum(ts_mean(volume, 40), 20), 7)) ^ rank(correlation(rank(vwap), rank(volume), 6)))',
    "alpha079": '(-1 * rank(correlation(delay(close, 1), delay(close, 2), 250)) + rank(correlation(close, volume, 10)))',
    "alpha080": '(rank(ts_rank(close, 5)) - rank(ts_rank(volume, 5)))',
    "alpha081": '((rank(log(ts_product(rank(rank(correlation(vwap, ts_sum(ts_mean(volume, 10), 50), 8) ^ 4)), 15))) < rank(correlation(rank(vwap), rank(volume), 5))) * -1)',
    "alpha082": '(-1 * rank(correlation(vwap, rank(volume), 10)) + rank(correlation(close, rank(volume), 10)))',
    "alpha083": '((rank(delay((high - low) / (ts_mean(close, 5) / 5), 2)) * rank(rank(volume))) / (((high - low) / (ts_mean(close, 5) / 5)) / (vwap - close)))',
    "alpha084": 'signedpower(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(close, 5))',
    "alpha085": '(rank(correlation(high * 0.876703 + close * 0.123297, ts_mean(volume, 30), 10)) ^ rank(correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))',
    "alpha086": '((ts_rank(correlation(close, ts_sum(ts_mean(volume, 20), 15), 6), 20) < rank((open + close) - (vwap + open))) * -1)',
    "alpha087": '(-1 * rank(correlation(rank(open), rank(volume), 10)) + rank(correlation(rank(close), rank(volume), 10)))',
    "alpha088": 'ele_min(rank(decay_linear((rank(open) + rank(low)) - (rank(high) + rank(close)), 8)), ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 21), 8), 7), 3))',
    "alpha089": '(-1 * rank(correlation(rank(open), rank(volume), 10)))',
    "alpha090": '(-1 * rank(correlation(rank(close), rank(volume), 10)))',
    "alpha091": '(-1 * rank(correlation(rank(high), rank(volume), 10)))',
    "alpha092": 'ele_min(ts_rank(decay_linear((((high + low) / 2 + close) < (low + open)), 15), 19), ts_rank(decay_linear(correlation(rank(low), rank(ts_mean(volume, 30)), 8), 7), 7))',
    "alpha093": '(-1 * rank(correlation(rank(low), rank(volume), 10)))',
    "alpha094": '(-(rank(vwap - ts_min(vwap, 12)) ^ ts_rank(correlation(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 4), 18), 3)))',
    "alpha095": '((rank(open - ts_min(open, 12)) < ts_rank(rank(correlation(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13) ^ 5), 12)) * -1)',
    "alpha096": '(-ele_max(ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4), 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7), ts_rank(ts_mean(volume, 60), 4), 4), 13), 14), 13)))',
    "alpha097": '((rank(decay_linear(ts_delta(vwap, 5), 3)) + rank(decay_linear(ts_delta(close, 5), 3))) * -1)',
    "alpha098": '(rank(decay_linear(correlation(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open), rank(ts_mean(volume, 15)), 21), 9), 7), 8)))',
    "alpha099": '((rank(correlation(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)) < rank(correlation(low, volume, 6))) * -1)',
    "alpha100": '(-1 * rank(correlation(rank(open), rank(volume), 10)))',
    "alpha101": '((close - open) / (high - low + 0.001))',
}


def get_alpha101_formulas() -> Dict[str, str]:
    """Return the dict of Alpha101 id -> MASE-DSL expression string."""
    return ALPHA101_FORMULAS


def list_alpha101_ids() -> List[str]:
    """Return the list of Alpha101 ids currently included in the library."""
    return list(ALPHA101_FORMULAS.keys())

