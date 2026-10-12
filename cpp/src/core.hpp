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
  bool ok = gamma > 0 && kappa > 0 && mid > 0 && tick > 0 && horizon_s >= 0 && std::isfinite(mid) &&
            std::isfinite(sigma_bps) && std::isfinite(q) && std::isfinite(gamma) && std::isfinite(kappa) &&
            std::isfinite(horizon_s) && std::isfinite(min_half) && std::isfinite(spread_mult);
  if (!ok) return {nan(), nan(), nan(), nan()};
  double var_t = sigma_bps * sigma_bps * horizon_s;
  double shift = -q * gamma * var_t;
  double half = 0.5 * gamma * var_t + std::log(1.0 + gamma / kappa) / gamma;
  half = std::max(half, min_half) * std::max(spread_mult, 1.0);
  double res = mid * (1.0 + shift * 1e-4);
  double bid = std::floor(res * (1.0 - half * 1e-4) / tick + 1e-9) * tick;
  double ask = std::ceil(res * (1.0 + half * 1e-4) / tick - 1e-9) * tick;
  return {bid, ask, res, half};
}

// Beta-Bernoulli BOCPD on flow sign (Adams & MacKay 2007; Moret & Lillo arXiv:2609.11614).
class FlowCPD {
 public:
  FlowCPD(double hazard = 0.01, double a0 = 1.0, double b0 = 1.0, int max_run = 200, int short_run = 30)
      : h_(hazard), a0_(a0), b0_(b0), max_run_(max_run), short_run_(short_run), r_{1.0}, n1_{0.0}, n_{0.0} {}
  std::vector<double> update(int sign) {
    double x = sign > 0 ? 1.0 : 0.0;
    size_t k = r_.size();
    std::vector<double> r(k + 1), n1(k + 1), n(k + 1);
    double cp = 0.0;
    for (size_t i = 0; i < k; ++i) {
      double pb = (a0_ + n1_[i]) / (a0_ + b0_ + n_[i]);
      double pred = x == 1.0 ? pb : 1.0 - pb;
      double w = r_[i] * pred;
      r[i + 1] = w * (1.0 - h_);
      cp += w * h_;
      n1[i + 1] = n1_[i] + x; n[i + 1] = n_[i] + 1.0;
    }
    r[0] = cp; n1[0] = 0.0; n[0] = 0.0;
    if ((int)r.size() > max_run_) { r.resize(max_run_); n1.resize(max_run_); n.resize(max_run_); }
    double s = 0.0; for (double v : r) s += v;
    if (s > 0.0) { for (auto& v : r) v /= s; } else { std::fill(r.begin(), r.end(), 0.0); r[0] = 1.0; }
    r_ = std::move(r); n1_ = std::move(n1); n_ = std::move(n);
    double pc = 0.0, eb = 0.0;
    for (size_t i = 0; i < r_.size(); ++i) {
      if ((int)i < short_run_) pc += r_[i];
      eb += r_[i] * (a0_ + n1_[i]) / (a0_ + b0_ + n_[i]);
    }
    return {pc, 2.0 * eb - 1.0};
  }
 private:
  double h_, a0_, b0_; int max_run_, short_run_;
  std::vector<double> r_, n1_, n_;
};

// Queue-adjusted exposure asymmetry; only widens / shrinks. Mirrors quoter_ref.exposure_adjust.
inline std::tuple<double, double, double, double> exposure_adjust(double qa_bid, double qa_ask, double flow_bias,
    double p_change, double k_flow, double k_queue, double max_extra) {
  if (std::isnan(qa_bid) || std::isnan(qa_ask) || std::isnan(flow_bias) || std::isnan(p_change) || max_extra <= 0.0)
    return {max_extra, max_extra, 0.0, 0.0};
  const double cap = 1e12;
  double qb = std::min(std::max(qa_bid, 0.0), cap), qa = std::min(std::max(qa_ask, 0.0), cap);
  double tot = qb + qa, imb = tot > 0.0 ? (qb - qa) / tot : 0.0;
  double fb = std::min(std::max(flow_bias, -1.0), 1.0), pc = std::min(std::max(p_change, 0.0), 1.0);
  double amp = 1.0 + pc;
  double ax = k_flow * std::max(fb, 0.0) * amp + k_queue * std::max(imb, 0.0);
  double bx = k_flow * std::max(-fb, 0.0) * amp + k_queue * std::max(-imb, 0.0);
  bx = std::min(std::max(bx, 0.0), max_extra); ax = std::min(std::max(ax, 0.0), max_extra);
  return {bx, ax, 1.0 - bx / max_extra, 1.0 - ax / max_extra};
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
    bool finite = std::isfinite(equity) && std::isfinite(pos);
    if (finite && equity > peak_) peak_ = equity;
    if (finite && peak_ - equity >= max_dd_) halted_ = true;
    double qm = qmax(sigma, mid, depth);
    if (halted_) return {false, false, true, true, 0.0, 1.0, qm, "drawdown_halt"};
    if (!finite) return {false, false, true, false, 0.0, 1.0, qm, "bad_input"};
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

constexpr double kUnknownQueue = 1e300;
using Fill = std::tuple<int64_t, int, double, double, double, int64_t>;  // id, side, px, qty, fee, ts

class SimExchange {
 public:
  SimExchange(double maker_bps, double rebate_bps, double ack, double jit, double cxl, double stress, uint64_t seed,
              double taker_bps = 5.0)
      : fee_((maker_bps - rebate_bps) * 1e-4), taker_fee_(taker_bps * 1e-4), ack_(ack), jit_(jit), cxl_(cxl),
        stress_(stress), rng_(seed) {}
  double queue_ahead(int64_t id) const {
    auto it = orders_.find(id);
    return (it != orders_.end() && it->second.state == 1) ? it->second.qa : nan();
  }
  std::vector<Fill> take(int64_t now, int side, double qty, double limit_px) {
    auto& px = side == 1 ? apx_ : bpx_; auto& qs = side == 1 ? aqs_ : bqs_;
    std::vector<Fill> out; double rem = qty; ++n_msgs;
    for (size_t i = 0; i < px.size(); ++i) {
      if (rem <= 1e-12) break;
      if ((side == 1 && px[i] > limit_px) || (side == -1 && px[i] < limit_px)) break;
      double q = std::min(rem, qs[i]); if (q <= 0.0) continue;
      qs[i] -= q; rem -= q;
      out.emplace_back(0, side, px[i], q, px[i] * q * taker_fee_, now);
    }
    return out;
  }
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
    return on_book_l2(now, {bid}, {bq}, {ask}, {aq});
  }
  double level_qty(int side, double price) const {
    const auto& px = side == 1 ? bpx_ : apx_; const auto& qs = side == 1 ? bqs_ : aqs_;
    if (px.empty()) return kUnknownQueue;
    for (size_t i = 0; i < px.size(); ++i) if (px[i] == price) return qs[i];
    bool better = side == 1 ? price > px.front() : price < px.front();
    bool within = side == 1 ? price > px.back() : price < px.back();
    return (better || within) ? 0.0 : kUnknownQueue;
  }
  std::vector<Fill> on_book_l2(int64_t now, std::vector<double> bpx, std::vector<double> bqs,
                               std::vector<double> apx, std::vector<double> aqs) {
    bpx_ = std::move(bpx); bqs_ = std::move(bqs); apx_ = std::move(apx); aqs_ = std::move(aqs);
    bid_ = bpx_[0]; bq_ = bqs_[0]; ask_ = apx_[0]; aq_ = aqs_[0]; housekeep(now);
    std::vector<Fill> out;
    for (auto& kv : orders_) {
      Order& o = kv.second; if (o.state != 1) continue;
      bool through = (o.side == 1 && ask_ <= o.price) || (o.side == -1 && bid_ >= o.price);
      if (through) fill(o, o.qty, now, out);
      else o.qa = std::min(o.qa, level_qty(o.side, o.price));
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
        o.qa = level_qty(o.side, o.price);
      }
      ++it;
    }
  }
  double fill(Order& o, double qty, int64_t ts, std::vector<Fill>& out) {
    double q = qty < o.qty ? qty : o.qty; if (q <= 0.0) return 0.0;
    o.qty -= q; out.emplace_back(o.id, o.side, o.price, q, o.price * q * fee_, ts);
    if (o.qty <= 1e-12) o.state = 2; return q;
  }
  double fee_, taker_fee_, ack_, jit_, cxl_, stress_; SplitMix64 rng_;
  std::map<int64_t, Order> orders_; int64_t next_id_ = 1;
  double bid_ = 0, bq_ = 0, ask_ = 0, aq_ = 0;
  std::vector<double> bpx_, bqs_, apx_, aqs_;
};

}  // namespace raim
