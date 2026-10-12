// RAIM fast-body hot path (C++17). Mirrors src/raim/core/*_ref.py exactly; see parity tests.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace raim {

constexpr double kPi = 3.14159265358979323846;
inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }

struct SplitMix64 {
  uint64_t state;
  explicit SplitMix64(uint64_t s) : state(s) {}
  uint64_t next() {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
};

class Rolling {
 public:
  explicit Rolling(size_t n) : n_(n) {}
  void push(double x) {
    buf_.push_back(x); s += x; s2 += x * x;
    if (buf_.size() > n_) { double y = buf_.front(); buf_.pop_front(); s -= y; s2 -= y * y; }
  }
  size_t size() const { return buf_.size(); }
  size_t cap() const { return n_; }
  double mean() const { return buf_.empty() ? 0.0 : s / double(buf_.size()); }
  double var() const {
    size_t k = buf_.size(); if (k < 2) return 0.0;
    double m = s / double(k); return std::max(s2 / double(k) - m * m, 0.0);
  }
  double s = 0.0, s2 = 0.0;
 private:
  size_t n_; std::deque<double> buf_;
};

class FeatureEngine {
 public:
  FeatureEngine(int vol_window = 200, int ofi_window = 50, double vpin_bucket_volume = 5.0,
                int vpin_buckets = 50, int vr_lag = 10, int hawkes_window = 500)
      : vr_lag_(vr_lag), bucket_v_(vpin_bucket_volume), bv_(vol_window), ofi_(ofi_window),
        ofi_abs_(ofi_window), r1_(vol_window), rk_(vol_window), vpin_(vpin_buckets), hk_(hawkes_window) {}

  void on_book(int64_t ts, double bid, double bq, double ask, double aq, double depth) {
    if (!(bid > 0 && ask > 0 && ask > bid)) return;
    if (!std::isnan(bp_)) {
      double e = 0.0;
      if (bid >= bp_) e += bq;
      if (bid <= bp_) e -= bq_;
      if (ask <= ap_) e -= aq;
      if (ask >= ap_) e += aq_;
      ofi_.push(e); ofi_abs_.push(std::fabs(e));
    }
    double logm = std::log(0.5 * (bid + ask));
    if (!std::isnan(last_logm_)) {
      double r = logm - last_logm_, absr = std::fabs(r);
      if (!std::isnan(last_absr_)) bv_.push(absr * last_absr_);
      last_absr_ = absr; r1_.push(r);
    }
    hist_.push_back(logm);
    if ((int)hist_.size() > vr_lag_ + 1) hist_.pop_front();
    if ((int)hist_.size() == vr_lag_ + 1) rk_.push(hist_.back() - hist_.front());
    last_logm_ = logm; bp_ = bid; bq_ = bq; ap_ = ask; aq_ = aq; depth_ = depth;
    hawkes_advance(ts);
  }

  void on_trade(int64_t ts, double /*price*/, double qty, int sign) {
    hawkes_advance(ts);
    hk_count_ += 1.0;
    double rem = qty;
    while (rem > 0.0) {
      double room = bucket_v_ - (bb_ + bs_);
      double take = rem < room ? rem : room;
      if (sign > 0) bb_ += take; else bs_ += take;
      rem -= take;
      if (bb_ + bs_ >= bucket_v_ - 1e-12) { vpin_.push(std::fabs(bb_ - bs_) / bucket_v_); bb_ = bs_ = 0.0; }
    }
  }

  std::vector<double> values() const {
    if (std::isnan(bp_)) return std::vector<double>(9, nan());
    double mid = 0.5 * (bp_ + ap_), tot = bq_ + aq_;
    double micro = tot > 0 ? (bp_ * aq_ + ap_ * bq_) / tot : mid;
    double spread = (ap_ - bp_) / mid * 1e4;
    double ofi = ofi_abs_.s > 1e-12 ? ofi_.s / ofi_abs_.s : 0.0;
    double bv = std::sqrt(std::max(kPi / 2.0 * bv_.mean(), 0.0));
    double v1 = r1_.var();
    double vr = (v1 > 1e-18 && rk_.size() > 1) ? rk_.var() / (vr_lag_ * v1) : 1.0;
    double hk = 0.0, m = hk_.mean();
    if (hk_.size() >= 10 && m > 1e-12) {
      double fano = hk_.var() / m;
      if (fano > 1.0) hk = std::min(1.0 - 1.0 / std::sqrt(fano), 0.99);
    }
    return {mid, micro, spread, depth_, ofi, bv, vpin_.mean(), vr, hk};
  }

 private:
  void hawkes_advance(int64_t ts) {
    int64_t b = ts >= 0 ? ts / 1000 : -((-ts + 999) / 1000);
    if (hk_bin_ < 0) { hk_bin_ = b; return; }
    if (b <= hk_bin_) return;
    int64_t gap = std::min<int64_t>(b - hk_bin_, (int64_t)hk_.cap() + 1);
    hk_.push(hk_count_);
    for (int64_t i = 0; i < gap - 1; ++i) hk_.push(0.0);
    hk_count_ = 0.0; hk_bin_ = b;
  }
  int vr_lag_; double bucket_v_;
  double bp_ = nan(), bq_ = 0, ap_ = nan(), aq_ = 0, depth_ = 0;
  double last_logm_ = nan(), last_absr_ = nan();
  Rolling bv_, ofi_, ofi_abs_, r1_, rk_, vpin_, hk_;
  std::deque<double> hist_;
  double bb_ = 0, bs_ = 0, hk_count_ = 0; int64_t hk_bin_ = -1;
};

class HMMFilter {
 public:
  using Mat = std::vector<std::vector<double>>;
  HMMFilter(std::vector<double> pi0, Mat trans, Mat means, Mat vars)
      : A_(std::move(trans)), mu_(std::move(means)), var_(std::move(vars)), post_(std::move(pi0)) {
    for (auto& r : var_) for (auto& v : r) v = std::max(v, 1e-12);
  }
  void set_transition(Mat t) { A_ = std::move(t); }
  std::vector<double> step(const std::vector<double>& x) {
    size_t k = post_.size(), d = mu_[0].size();
    std::vector<double> prior(k, 0.0), logl(k, 0.0), w(k);
    for (size_t j = 0; j < k; ++j) { double s = 0; for (size_t i = 0; i < k; ++i) s += post_[i] * A_[i][j]; prior[j] = s; }
    for (size_t j = 0; j < k; ++j) {
      double acc = 0;
      for (size_t t = 0; t < d; ++t) {
        double xv = x[t]; if (std::isnan(xv)) continue;
        double v = var_[j][t], diff = xv - mu_[j][t];
        acc += -0.5 * (std::log(2.0 * kPi * v) + diff * diff / v);
      }
      logl[j] = acc;
    }
    double mx = *std::max_element(logl.begin(), logl.end()), s = 0;
    for (size_t j = 0; j < k; ++j) { w[j] = prior[j] * std::exp(logl[j] - mx); s += w[j]; }
    if (!(s > 0.0) || std::isnan(s)) post_ = prior; else { for (auto& v : w) v /= s; post_ = w; }
    return post_;
  }
 private:
  Mat A_, mu_, var_; std::vector<double> post_;
};

inline std::tuple<double, double, double, double> quote(double mid, double sigma_bps, double q, double gamma,
    double kappa, double horizon_s, double min_half, double spread_mult, double tick) {
  double var_t = sigma_bps * sigma_bps * horizon_s;
  double shift = -q * gamma * var_t;
  double half = 0.5 * gamma * var_t + std::log(1.0 + gamma / kappa) / gamma;
  half = std::max(half, min_half) * std::max(spread_mult, 1.0);
  double res = mid * (1.0 + shift * 1e-4);
  double bid = std::floor(res * (1.0 - half * 1e-4) / tick + 1e-9) * tick;
  double ask = std::ceil(res * (1.0 + half * 1e-4) / tick - 1e-9) * tick;
  return {bid, ask, res, half};
}

struct Decision {
  bool allow_bid, allow_ask, cancel_all, halted;
  double size_mult, spread_mult, qmax; std::string reason;
};

inline double clamp01(double x) { if (std::isnan(x)) return 0.0; return x < 0 ? 0.0 : (x > 1 ? 1.0 : x); }

class RiskGovernor {
 public:
  RiskGovernor(double budget, double z, double h, double c, double max_gross, double max_dd, int64_t stale_ms,
               double max_rej, double max_msg, int widen_n, int pause_n, double pause_s)
      : budget_(budget), z_(z), h_(h), c_(c), max_gross_(max_gross), max_dd_(max_dd), stale_ms_(stale_ms),
        max_rej_(max_rej), max_msg_(max_msg), widen_n_(widen_n), pause_n_(pause_n),
        pause_ms_((int64_t)(pause_s * 1000)) {}
  void set_external(double m_llm, double m_jev) { m_llm_ = clamp01(m_llm); m_jev_ = clamp01(m_jev); }
  void on_trade_result(double pnl, int64_t now) {
    if (pnl < 0) { if (++loss_streak_ >= pause_n_) { pause_until_ = now + pause_ms_; loss_streak_ = 0; } }
    else if (pnl > 0) loss_streak_ = 0;
  }
  double qmax(double sigma, double mid, double depth) const {
    if (!(sigma > 0 && mid > 0)) return 0.0;
    double qv = budget_ / (z_ * sigma * mid * std::sqrt(h_));
    return std::max(std::min(qv, c_ * std::max(depth, 0.0)), 0.0);
  }
  Decision evaluate(int64_t now, int64_t last_feed, double equity, double pos, double mid, double sigma,
                    double depth, double rej, double msg) {
    if (equity > peak_) peak_ = equity;
    if (peak_ - equity >= max_dd_) halted_ = true;
    double qm = qmax(sigma, mid, depth);
    if (halted_) return {false, false, true, true, 0.0, 1.0, qm, "drawdown_halt"};
    if (now - last_feed > stale_ms_ || !(mid > 0)) return {false, false, true, false, 0.0, 1.0, qm, "stale_feed"};
    if (now < pause_until_) return {false, false, true, false, 0.0, 1.0, qm, "loss_streak_pause"};
    double m_rules = 1.0, sm = 1.0; std::string reason = "ok";
    if (rej > max_rej_ || msg > max_msg_) { m_rules = 0.0; reason = "rate_backoff"; }
    else if (loss_streak_ >= widen_n_) { m_rules = 0.5; sm = 2.0; reason = "loss_streak_widen"; }
    double m = std::min(std::min(m_llm_, m_rules), m_jev_);
    if (m < 1.0 && reason == "ok") reason = "external_tighten";
    double lim = qm * m, gross = std::fabs(pos) * mid;
    bool ab = m > 0 && pos < lim && (gross < max_gross_ || pos < 0);
    bool aa = m > 0 && -pos < lim && (gross < max_gross_ || pos > 0);
    if (pos > lim && pos > 0) aa = true;
    if (-pos > lim && pos < 0) ab = true;
    return {ab, aa, false, false, m, sm, qm, reason};
  }
  bool halted() const { return halted_; }
  double m_llm() const { return m_llm_; }
  double m_jev() const { return m_jev_; }
  int loss_streak() const { return loss_streak_; }
 private:
  double budget_, z_, h_, c_, max_gross_, max_dd_; int64_t stale_ms_; double max_rej_, max_msg_;
  int widen_n_, pause_n_; int64_t pause_ms_;
  double m_llm_ = 1.0, m_jev_ = 1.0, peak_ = -std::numeric_limits<double>::infinity();
  bool halted_ = false; int loss_streak_ = 0; int64_t pause_until_ = -1;
};

using Fill = std::tuple<int64_t, int, double, double, double, int64_t>;  // id, side, px, qty, fee, ts

class SimExchange {
 public:
  SimExchange(double maker_bps, double rebate_bps, double ack, double jit, double cxl, double stress, uint64_t seed)
      : fee_((maker_bps - rebate_bps) * 1e-4), ack_(ack), jit_(jit), cxl_(cxl), stress_(stress), rng_(seed) {}
  int64_t place(int64_t now, int side, double px, double qty) {
    int64_t id = next_id_++;
    orders_[id] = Order{id, side, px, qty, now + lat(ack_), 0, 0.0, -1};
    ++n_msgs; return id;
  }
  void cancel(int64_t now, int64_t id) {
    auto it = orders_.find(id);
    if (it != orders_.end() && it->second.state != 2 && it->second.cancel_at < 0) { it->second.cancel_at = now + lat(cxl_); ++n_msgs; }
  }
  std::vector<std::tuple<int64_t, int, double, double, int>> open_orders() const {
    std::vector<std::tuple<int64_t, int, double, double, int>> v;
    for (auto& kv : orders_) if (kv.second.state != 2) v.emplace_back(kv.first, kv.second.side, kv.second.price, kv.second.qty, kv.second.state);
    return v;
  }
  std::vector<Fill> on_book(int64_t now, double bid, double bq, double ask, double aq) {
    bid_ = bid; bq_ = bq; ask_ = ask; aq_ = aq; housekeep(now);
    std::vector<Fill> out;
    for (auto& kv : orders_) {
      Order& o = kv.second; if (o.state != 1) continue;
      if (o.side == 1) {
        if (ask <= o.price) fill(o, o.qty, now, out);
        else if (bid == o.price) o.qa = std::min(o.qa, bq);
        else if (bid < o.price) o.qa = 0.0;
      } else {
        if (bid >= o.price) fill(o, o.qty, now, out);
        else if (ask == o.price) o.qa = std::min(o.qa, aq);
        else if (ask > o.price) o.qa = 0.0;
      }
    }
    return out;
  }
  std::vector<Fill> on_trade(int64_t now, double px, double qty, int sign) {
    housekeep(now);
    std::vector<Order*> v;
    for (auto& kv : orders_) v.push_back(&kv.second);
    std::stable_sort(v.begin(), v.end(), [](Order* a, Order* b) {
      double ka = -a->side * a->price, kb = -b->side * b->price;
      if (ka != kb) return ka < kb; return a->id < b->id; });
    std::vector<Fill> out; double rem = qty;
    for (Order* o : v) {
      if (o->state != 1 || rem <= 0.0) continue;
      bool hit = (sign < 0 && o->side == 1 && px <= o->price) || (sign > 0 && o->side == -1 && px >= o->price);
      if (!hit) continue;
      if (px != o->price) { rem -= fill(*o, rem, now, out); continue; }
      double eat = std::min(o->qa, rem); o->qa -= eat; rem -= eat;
      if (rem > 0.0) rem -= fill(*o, rem, now, out);
    }
    return out;
  }
  int64_t n_msgs = 0, n_rejects = 0;
 private:
  struct Order { int64_t id; int side; double price, qty; int64_t active_at; int state; double qa; int64_t cancel_at; };
  int64_t lat(double base) { return (int64_t)std::llround(stress_ * (base + jit_ * rng_.uniform())); }
  void housekeep(int64_t now) {
    for (auto it = orders_.begin(); it != orders_.end();) {
      Order& o = it->second;
      if (o.state == 2 || (o.cancel_at >= 0 && o.cancel_at <= now)) { it = orders_.erase(it); continue; }
      if (o.state == 0 && o.active_at <= now) {
        bool crosses = (o.side == 1 && ask_ > 0 && o.price >= ask_) || (o.side == -1 && bid_ > 0 && o.price <= bid_);
        if (crosses) { ++n_rejects; it = orders_.erase(it); continue; }
        o.state = 1;
        o.qa = o.side == 1 ? (o.price == bid_ ? bq_ : 0.0) : (o.price == ask_ ? aq_ : 0.0);
      }
      ++it;
    }
  }
  double fill(Order& o, double qty, int64_t ts, std::vector<Fill>& out) {
    double q = qty < o.qty ? qty : o.qty; if (q <= 0.0) return 0.0;
    o.qty -= q; out.emplace_back(o.id, o.side, o.price, q, o.price * q * fee_, ts);
    if (o.qty <= 1e-12) o.state = 2; return q;
  }
  double fee_, ack_, jit_, cxl_, stress_; SplitMix64 rng_;
  std::map<int64_t, Order> orders_; int64_t next_id_ = 1;
  double bid_ = 0, bq_ = 0, ask_ = 0, aq_ = 0;
};

}  // namespace raim
