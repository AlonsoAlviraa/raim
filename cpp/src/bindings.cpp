#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core.hpp"
namespace py = pybind11;
using namespace raim;

PYBIND11_MODULE(raim_core, m) {
  m.doc() = "RAIM fast-body hot path (C++17)";
  py::class_<FeatureEngine>(m, "FeatureEngine")
      .def(py::init<int, int, double, int, int, int>(), py::arg("vol_window") = 200, py::arg("ofi_window") = 50,
           py::arg("vpin_bucket_volume") = 5.0, py::arg("vpin_buckets") = 50, py::arg("vr_lag") = 10,
           py::arg("hawkes_window") = 500)
      .def("on_book", &FeatureEngine::on_book).def("on_trade", &FeatureEngine::on_trade)
      .def("values", &FeatureEngine::values);
  py::class_<HMMFilter>(m, "HMMFilter")
      .def(py::init<std::vector<double>, HMMFilter::Mat, HMMFilter::Mat, HMMFilter::Mat>())
      .def("set_transition", &HMMFilter::set_transition).def("step", &HMMFilter::step);
  m.def("quote", &quote);
  py::class_<Decision>(m, "Decision")
      .def_readonly("allow_bid", &Decision::allow_bid).def_readonly("allow_ask", &Decision::allow_ask)
      .def_readonly("cancel_all", &Decision::cancel_all).def_readonly("halted", &Decision::halted)
      .def_readonly("size_mult", &Decision::size_mult).def_readonly("spread_mult", &Decision::spread_mult)
      .def_readonly("qmax", &Decision::qmax).def_readonly("reason", &Decision::reason);
  py::class_<RiskGovernor>(m, "RiskGovernor")
      .def(py::init<double, double, double, double, double, double, int64_t, double, double, int, int, double>(),
           py::arg("budget_usd"), py::arg("z"), py::arg("h_unwind_s"), py::arg("depth_frac"), py::arg("max_gross_usd"),
           py::arg("max_drawdown_usd"), py::arg("stale_feed_ms"), py::arg("max_reject_rate"), py::arg("max_msgs_per_s"),
           py::arg("loss_streak_widen"), py::arg("loss_streak_pause"), py::arg("pause_s"))
      .def("set_external", &RiskGovernor::set_external).def("on_trade_result", &RiskGovernor::on_trade_result)
      .def("qmax", &RiskGovernor::qmax).def("evaluate", &RiskGovernor::evaluate)
      .def_property_readonly("halted", &RiskGovernor::halted)
      .def_property_readonly("m_llm", &RiskGovernor::m_llm).def_property_readonly("m_jev", &RiskGovernor::m_jev)
      .def_property_readonly("loss_streak", &RiskGovernor::loss_streak);
  py::class_<SimExchange>(m, "SimExchange")
      .def(py::init<double, double, double, double, double, double, uint64_t>())
      .def("place", &SimExchange::place).def("cancel", &SimExchange::cancel)
      .def("open_orders", &SimExchange::open_orders).def("on_book", &SimExchange::on_book)
      .def("on_trade", &SimExchange::on_trade)
      .def_readonly("n_msgs", &SimExchange::n_msgs).def_readonly("n_rejects", &SimExchange::n_rejects);
}
