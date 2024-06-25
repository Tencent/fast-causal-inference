// Copyright 2021-present StarRocks, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <type_traits>

#include "column/const_column.h"
#include "column/type_traits.h"
#include "column/vectorized_fwd.h"
#include "delta_method.h"
#include "exprs/agg/aggregate.h"
#include "exprs/agg/ttest_2samp.h"
#include "exprs/agg/ttest_common.h"
#include "exprs/expr.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/expr_tree.hpp"
#include "exprs/helpers/math_helpers.hpp"
#include "exprs/helpers/serialize_helpers.hpp"
#include "gutil/casts.h"
#include "gutil/integral_types.h"
#include "types/logical_type.h"

namespace starrocks {

using XexptTtest2SampUinColumnType = RunTimeColumnType<TYPE_INT>;
using XexptTtest2SampCupedColumnType = RunTimeColumnType<TYPE_VARCHAR>;
using XexptTtest2SampDataElementColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using XexptTtest2SampAlphaColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using XexptTtest2SampMDEColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using XexptTtest2SampPowerColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using XexptTtest2SampResultColumnType = RunTimeColumnType<TYPE_VARCHAR>;
using XexptTtest2SampMetricColumnType = RunTimeColumnType<TYPE_VARCHAR>;

// uin, treatment[, [numerator, denominator, cuped_data...], cuped, {alpha, mde, power}]

enum class XexptTtest2SampMetricType : uint8_t { Unknown = 0, Avg, Sum };

class XexptTtest2SampParams {
public:
    bool operator==(const XexptTtest2SampParams& other) const {
        return _num_variables == other._num_variables && _alpha == other._alpha &&
               _cuped_expression == other._cuped_expression && _mde == other._mde && _power == other._power;
    }

    bool is_uninitialized() const { return _num_variables == -1; }

    void reset() {
        _num_variables = -1;
        _alpha = TtestCommon::kDefaultAlphaValue;
        std::string().swap(_cuped_expression);
        _mde = TtestCommon::kDefaultMDEValue;
        _power = TtestCommon::kDefaultPowerValue;
        _metric_type = XexptTtest2SampMetricType::Unknown;
        _ratios = std::vector<double>{1, 1};
    }

    void init(int num_variables, std::string const& cuped_expression, double alpha, double mde, double power,
              XexptTtest2SampMetricType metric_type, std::vector<double> const& ratios) {
        _num_variables = num_variables;
        _alpha = alpha;
        _cuped_expression = cuped_expression;
        _mde = mde;
        _power = power;
        _metric_type = metric_type;
        _ratios = ratios;
    }

    void serialize(uint8_t*& data) const {
        SerializeHelpers::serialize(&_num_variables, data);
        if (is_uninitialized()) {
            return;
        }
        auto metric_tmp = static_cast<uint8_t>(_metric_type);
        SerializeHelpers::serialize(&metric_tmp, data);
        SerializeHelpers::serialize(&_alpha, data);
        uint32_t cuped_expression_length = _cuped_expression.length();
        SerializeHelpers::serialize(&cuped_expression_length, data);
        SerializeHelpers::serialize(_cuped_expression.data(), data, cuped_expression_length);
        SerializeHelpers::serialize(&_mde, data);
        SerializeHelpers::serialize(&_power, data);
        SerializeHelpers::serialize(_ratios.data(), data, 2);
    }

    void deserialize(const uint8_t*& data) {
        SerializeHelpers::deserialize(data, &_num_variables);
        if (is_uninitialized()) {
            return;
        }
        uint8_t metric_tmp;
        SerializeHelpers::deserialize(data, &metric_tmp);
        _metric_type = static_cast<XexptTtest2SampMetricType>(metric_tmp);
        SerializeHelpers::deserialize(data, &_alpha);
        uint32_t cuped_expression_length;
        SerializeHelpers::deserialize(data, &cuped_expression_length);
        _cuped_expression.resize(cuped_expression_length);
        SerializeHelpers::deserialize(data, _cuped_expression.data(), cuped_expression_length);
        SerializeHelpers::deserialize(data, &_mde);
        SerializeHelpers::deserialize(data, &_power);
        SerializeHelpers::deserialize(data, _ratios.data(), 2);
    }

    size_t serialized_size() const {
        if (is_uninitialized()) {
            return sizeof(_num_variables);
        }
        return sizeof(_num_variables) + sizeof(_metric_type) + sizeof(_alpha) + sizeof(uint32_t) +
               _cuped_expression.length() + sizeof(_mde) + sizeof(_power) + 2 * sizeof(double);
    }

    int num_variables() const { return _num_variables; }

    double alpha() const { return _alpha; }

    double mde() const { return _mde; }

    double power() const { return _power; }

    XexptTtest2SampMetricType metric_type() const { return _metric_type; }

    std::vector<double> const& ratios() const { return _ratios; }

    std::string const& cuped_expression() const { return _cuped_expression; }

    std::string const& Y_expression() const { return _Y_expression; }

private:
    std::string const _Y_expression{"x1/x2"};
    XexptTtest2SampMetricType _metric_type;
    int _num_variables{-1};
    std::string _cuped_expression;
    double _alpha{TtestCommon::kDefaultAlphaValue};
    double _mde{TtestCommon::kDefaultMDEValue};
    double _power{TtestCommon::kDefaultPowerValue};
    std::vector<double> _ratios{1, 1};
};

class XexptTtest2SampStats {
public:
    static constexpr uint32_t kNumBuckets = 128;
    static constexpr uint32_t kBucketDivisor = (1u << (32u - 7u));

    XexptTtest2SampStats() = default;
    XexptTtest2SampStats(const XexptTtest2SampStats&) = delete;
    XexptTtest2SampStats(XexptTtest2SampStats&&) = delete;

    XexptTtest2SampStats(const uint8_t* serialized_data) { deserialize(serialized_data); }

    void init(int32_t num_columns) {
        _column_buckets = ublas::matrix<double>(num_columns, kNumBuckets, 0);
        _count = 0;
        this->_num_columns = num_columns;
    }

    void update(const double* input, int num_variables, int32_t uin) {
        CHECK(num_variables == _num_columns);
        uint32_t uin_hash = _hash(uin) / XexptTtest2SampStats::kBucketDivisor;
        _count += 1;
        for (uint32_t i = 0; i < _num_columns; ++i) {
            _column_buckets(i, uin_hash) += input[i];
        }
    }

    void merge(XexptTtest2SampStats const& other) {
        CHECK(_num_columns == other._num_columns);
        _count += other._count;
        for (uint32_t col = 0; col < _num_columns; ++col) {
            for (uint32 bucket = 0; bucket < kNumBuckets; ++bucket) {
                _column_buckets(col, bucket) += other._column_buckets(col, bucket);
            }
        }
    }

    void serialize(uint8_t*& data) const {
        SerializeHelpers::serialize(&_count, data);
        SerializeHelpers::serialize(_column_buckets.data().begin(), data, _num_columns * kNumBuckets);
    }

    void deserialize(const uint8_t*& data) {
        CHECK(_num_columns != -1);
        SerializeHelpers::deserialize(data, &_count);
        SerializeHelpers::deserialize(data, _column_buckets.data().begin(), _num_columns * kNumBuckets);
    }

    size_t serialized_size() const { return sizeof(_count) + sizeof(double) * _num_columns * kNumBuckets; }

    size_t count() const { return _count; }

    ublas::matrix<double> const& data() const { return _column_buckets; }

private:
    int _num_columns{-1};
    MathHelpers::MurmurHash3 _hash{0};
    ublas::matrix<double> _column_buckets; // col * bucket
    size_t _count = 0;
};

template <typename TreatmentType>
class XexptTtest2SampAggregateState {
public:
    XexptTtest2SampAggregateState() = default;
    XexptTtest2SampAggregateState(const XexptTtest2SampAggregateState&) = delete;
    XexptTtest2SampAggregateState(XexptTtest2SampAggregateState&&) = delete;

    XexptTtest2SampAggregateState(const uint8_t* serialized_data) { deserialize(serialized_data); }

    bool is_uninitialized() const { return _ttest_params.is_uninitialized(); }

    void check_params(XexptTtest2SampAggregateState const& other) const {
        DCHECK(_ttest_params == other._ttest_params);
    }

    void init(int num_variables, std::optional<std::string> const& cuped_expression, std::optional<double> alpha,
              std::optional<double> mde, std::optional<double> power,
              std::optional<XexptTtest2SampMetricType> metric_type, std::optional<std::vector<double>> const& ratios) {
        _ttest_params.init(num_variables, cuped_expression.value_or(std::string{}),
                           alpha.value_or(TtestCommon::kDefaultAlphaValue), mde.value_or(TtestCommon::kDefaultMDEValue),
                           power.value_or(TtestCommon::kDefaultPowerValue),
                           metric_type.value_or(XexptTtest2SampMetricType::Avg),
                           ratios.value_or(std::vector<double>{1, 1}));
    }

    void update(const double* input, int num_variables, int32_t uin, TreatmentType treatment) {
        CHECK(!is_uninitialized());
        CHECK(num_variables == _ttest_params.num_variables());

        if (_all_stats.count(treatment) == 0) {
            _all_stats[treatment].init(num_variables);
        }
        _all_stats[treatment].update(input, num_variables, uin);
    }

    void merge(XexptTtest2SampAggregateState const& other) {
        if (other.is_uninitialized()) {
            reset();
            return;
        }
        check_params(other);
        for (auto const& [treatment, stats] : other._all_stats) {
            if (_all_stats.count(treatment) == 0) {
                _all_stats[treatment].init(_ttest_params.num_variables());
            }
            _all_stats[treatment].merge(stats);
        }
    }

    void serialize(uint8_t*& data) const {
        _ttest_params.serialize(data);
        if (_ttest_params.is_uninitialized()) {
            return;
        }
        uint32_t num_groups = _all_stats.size();
        SerializeHelpers::serialize(&num_groups, data);
        for (auto const& [treatment, stats] : _all_stats) {
            if constexpr (std::is_same_v<TreatmentType, std::string>) {
                uint32_t size = treatment.length();
                SerializeHelpers::serialize(&size, data);
                SerializeHelpers::serialize(treatment.data(), data, size);
            } else {
                SerializeHelpers::serialize(&treatment, data);
            }
            stats.serialize(data);
        }
    }

    void deserialize(const uint8_t*& data) {
        _ttest_params.deserialize(data);
        if (_ttest_params.is_uninitialized()) {
            return;
        }
        uint32_t num_groups;
        SerializeHelpers::deserialize(data, &num_groups);
        for (uint32_t i = 0; i < num_groups; ++i) {
            TreatmentType treatment;
            if constexpr (std::is_same_v<TreatmentType, std::string>) {
                uint32_t size;
                SerializeHelpers::deserialize(data, &size);
                treatment.resize(size);
                SerializeHelpers::deserialize(data, treatment.data(), size);
            } else {
                SerializeHelpers::deserialize(data, &treatment);
            }
            auto& stats = _all_stats[treatment];
            stats.init(_ttest_params.num_variables());
            stats.deserialize(data);
        }
    }

    void reset() {
        _ttest_params.reset();
        _all_stats.clear();
    }

    size_t serialized_size() const {
        size_t size = _ttest_params.serialized_size();
        if (_ttest_params.is_uninitialized()) {
            return size;
        }
        size += sizeof(uint32_t);
        for (auto const& [treatment, stats] : _all_stats) {
            if constexpr (std::is_same_v<TreatmentType, std::string>) {
                size += sizeof(uint32_t);
                size += treatment.size() * sizeof(char);
            } else {
                size += sizeof(treatment);
            }
            size += stats.serialized_size();
        }
        return size;
    }

    std::string get_ttest_result() const {
        if (is_uninitialized()) {
            return fmt::format("Internal error: ttest agg state is uninitialized.");
        }
        if (_all_stats.size() != 2) {
            return fmt::format("xexpt_ttest_2samp need excatly two groups, but you give ({}) groups.",
                               _all_stats.size());
        }
        for (auto const& [treatment, stats] : _all_stats) {
            if (stats.count() <= 1) {
                return fmt::format("count({}) of group({}) should be greater than 1.", stats.count(), treatment);
            }
        }

        std::vector<TreatmentType> group_names;
        std::vector<double> numerators, denominators, counts, std_samp, numerators_pre, denominators_pre, means, vars,
                std_avg, vars_avg(2), means_avg(2);

        std::vector<std::vector<double>> all_data(XexptTtest2SampStats::kNumBuckets,
                                                  std::vector<double>(_ttest_params.num_variables()));

        DeltaMethodStats delta_method_stats_avg;
        delta_method_stats_avg.init(_ttest_params.num_variables());
        std::map<TreatmentType, DeltaMethodStats> delta_method_stats_avg_sub_stats;

        DeltaMethodStats delta_method_stats_sum;
        delta_method_stats_sum.init(_ttest_params.num_variables());
        std::map<TreatmentType, DeltaMethodStats> delta_method_stats_sum_sub_stats;

        int32_t key_idx = 0;
        for (auto const& [key, stats] : _all_stats) {
            group_names.emplace_back(key);

            ublas::matrix<double> all_rows = ublas::trans(stats.data());

            delta_method_stats_avg_sub_stats[key].init(_ttest_params.num_variables());
            delta_method_stats_sum_sub_stats[key].init(_ttest_params.num_variables());

            double numerator_sum = 0, denominator_sum = 0, numerator_pre_sum = 0, denominator_pre_sum = 0;

            for (uint32_t bucket = 0; bucket < XexptTtest2SampStats::kNumBuckets; ++bucket) {
                std::vector<double> bucket_data(_ttest_params.num_variables());
                for (uint32_t column = 0; column < _ttest_params.num_variables(); ++column) {
                    bucket_data[column] = all_rows(bucket, column);
                    all_data[bucket][column] += all_rows(bucket, column);
                }
                numerator_sum += all_rows(bucket, 0);
                denominator_sum += all_rows(bucket, 1);
                if (_ttest_params.num_variables() >= 3) {
                    numerator_pre_sum += all_rows(bucket, 2);
                }
                if (_ttest_params.num_variables() >= 4) {
                    denominator_pre_sum += all_rows(bucket, 3);
                }
                delta_method_stats_avg.update(bucket_data.data(), _ttest_params.num_variables());
                delta_method_stats_avg_sub_stats[key].update(bucket_data.data(), _ttest_params.num_variables());
                bucket_data[1] = _ttest_params.ratios()[key_idx] / XexptTtest2SampStats::kNumBuckets;
                if (!_ttest_params.cuped_expression().empty() && (int)bucket_data.size() >= 4) {
                    bucket_data[3] = _ttest_params.ratios()[key_idx] / XexptTtest2SampStats::kNumBuckets;
                }
                delta_method_stats_sum.update(bucket_data.data(), _ttest_params.num_variables());
                delta_method_stats_sum_sub_stats[key].update(bucket_data.data(), _ttest_params.num_variables());
            }

            numerators.emplace_back(numerator_sum);
            denominators.emplace_back(denominator_sum);
            if (_ttest_params.num_variables() >= 4) {
                numerators_pre.emplace_back(numerator_pre_sum);
                denominators_pre.emplace_back(denominator_pre_sum);
            }
            counts.emplace_back(stats.count());

            key_idx += 1;
        }

        DeltaMethodStats theta_stats_avg;
        theta_stats_avg.init(_ttest_params.num_variables());
        DeltaMethodStats theta_stats_sum;
        theta_stats_sum.init(_ttest_params.num_variables());

        for (uint32_t i = 0; i < XexptTtest2SampStats::kNumBuckets; ++i) {
            theta_stats_avg.update(all_data[i].data(), _ttest_params.num_variables());
            all_data[i][1] =
                    (_ttest_params.ratios()[0] + _ttest_params.ratios()[1]) / XexptTtest2SampStats::kNumBuckets;
            if (!_ttest_params.cuped_expression().empty() && _ttest_params.num_variables() >= 4) {
                all_data[i][3] =
                        (_ttest_params.ratios()[0] + _ttest_params.ratios()[1]) / XexptTtest2SampStats::kNumBuckets;
            }
            theta_stats_sum.update(all_data[i].data(), _ttest_params.num_variables());
        }

        if (!TtestCommon::calc_means_and_vars(_ttest_params.Y_expression(), _ttest_params.cuped_expression(),
                                              _ttest_params.num_variables(),
                                              delta_method_stats_avg_sub_stats.at(group_names[0]),
                                              delta_method_stats_avg_sub_stats.at(group_names[1]), theta_stats_avg,
                                              means_avg[0], means_avg[1], vars_avg[0], vars_avg[1])) {
            return "InvertMatrix failed. some variables in the table are perfectly collinear.";
        }

        if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Avg) {
            means.insert(means.end(), means_avg.begin(), means_avg.end());
            vars.insert(vars.end(), vars_avg.begin(), vars_avg.end());
        }

        if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Sum) {
            means.resize(2);
            vars.resize(2);
            if (!TtestCommon::calc_means_and_vars(_ttest_params.Y_expression(), _ttest_params.cuped_expression(),
                                                  _ttest_params.num_variables(),
                                                  delta_method_stats_sum_sub_stats.at(group_names[0]),
                                                  delta_method_stats_sum_sub_stats.at(group_names[1]), theta_stats_sum,
                                                  means[0], means[1], vars[0], vars[1])) {
                return "InvertMatrix failed. some variables in the table are perfectly collinear.";
            }
        }

        std::string complete_expression = _ttest_params.Y_expression();
        if (!_ttest_params.cuped_expression().empty()) {
            complete_expression += "+" + _ttest_params.cuped_expression();
        }
        ExprTree<double> expr_tree(complete_expression, _ttest_params.num_variables());

        std::vector<double> std_samp_avg;
        if (_ttest_params.cuped_expression().empty()) {
            auto const& stat0 = delta_method_stats_avg_sub_stats.at(group_names[0]);
            auto const& stat1 = delta_method_stats_avg_sub_stats.at(group_names[1]);
            std_samp_avg.push_back(sqrt(DeltaMethodStats::calc_delta_method(expr_tree, stat0.count(), stat0.means(),
                                                                            stat0.cov_matrix(), false)) *
                                   sqrt(denominators[0]));
            std_samp_avg.push_back(sqrt(DeltaMethodStats::calc_delta_method(expr_tree, stat1.count(), stat1.means(),
                                                                            stat1.cov_matrix(), false)) *
                                   sqrt(denominators[1]));
            if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Sum) {
                auto const& stat0 = delta_method_stats_sum_sub_stats.at(group_names[0]);
                auto const& stat1 = delta_method_stats_sum_sub_stats.at(group_names[1]);
                std_samp.push_back(sqrt(DeltaMethodStats::calc_delta_method(expr_tree, stat0.count(), stat0.means(),
                                                                            stat0.cov_matrix(), false)) *
                                   sqrt(denominators[0]));
                std_samp.push_back(sqrt(DeltaMethodStats::calc_delta_method(expr_tree, stat1.count(), stat1.means(),
                                                                            stat1.cov_matrix(), false)) *
                                   sqrt(denominators[1]));
            }
        } else {
            std_samp_avg.push_back(sqrt(vars_avg[0] * denominators[0]));
            std_samp_avg.push_back(sqrt(vars_avg[1] * denominators[1]));
            if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Sum) {
                std_samp.push_back(sqrt(vars[0] * _ttest_params.ratios()[0]));
                std_samp.push_back(sqrt(vars[1] * _ttest_params.ratios()[1]));
            }
        }
        if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Avg) {
            std_samp.insert(std_samp.end(), std_samp_avg.begin(), std_samp_avg.end());
        }

        double estimate = means[1] - means[0];
        double stderr_var = sqrt(vars[0] + vars[1]);
        if (!std::isfinite(stderr_var) || stderr_var == 0) {
            return fmt::format("stderr_var({}) is not a finite value, please check your data.", stderr_var);
        }
        double diff_relative = estimate / means[0];
        double t_stat = estimate / stderr_var;
        double p_value = TtestCommon::calc_pvalue(t_stat, TtestAlternative::TwoSided);
        auto [lower, upper] = TtestCommon::calc_confidence_interval(estimate, stderr_var, counts[0] + counts[1],
                                                                    _ttest_params.alpha(), TtestAlternative::TwoSided);

        double lower_relative = lower / means[0];
        double upper_relative = upper / means[0];
        double mde = _ttest_params.mde();
        double alpha = _ttest_params.alpha();
        boost::math::normal normal_dist(0, 1);
        double power = 1 - cdf(normal_dist, quantile(normal_dist, 1 - alpha / 2) - fabs(means[0] * mde) / stderr_var) +
                       cdf(normal_dist, quantile(normal_dist, alpha / 2) - fabs(means[0] * mde) / stderr_var);
        double std_ratio = std_samp_avg[0] / std_samp_avg[1];
        double cnt_ratio = denominators[0] / denominators[1];
        double alpha_power = quantile(normal_dist, 1 - alpha / 2) - quantile(normal_dist, 1 - _ttest_params.power());
        double recommend_samples = ((std_ratio * std_ratio + cnt_ratio) / cnt_ratio) * pow(alpha_power, 2) *
                                   pow(std_samp_avg[1] / means_avg[0], 2) / pow(mde, 2);

        std::string title;
        std::string group0;
        std::string group1;

        auto add_result3 = [&title, &group0, &group1](const std::string& title_, const std::string& group0_,
                                                      const std::string& group1_) {
            title += MathHelpers::to_string_with_precision<false>(title_);
            group0 += MathHelpers::to_string_with_precision<false>(group0_);
            group1 += MathHelpers::to_string_with_precision<false>(group1_);
            size_t max_len = std::max({title.size(), group0.size(), group1.size()});
            title += std::string(max_len - title.size(), ' ');
            group0 += std::string(max_len - group0.size(), ' ');
            group1 += std::string(max_len - group1.size(), ' ');
        };

        if constexpr (std::is_same_v<std::string, TreatmentType>) {
            add_result3("groupname", group_names[0], group_names[1]);
        } else {
            add_result3("groupname", MathHelpers::to_string_with_precision<false>(group_names[0]),
                        MathHelpers::to_string_with_precision<false>(group_names[1]));
        }
        add_result3("numerator",
                    MathHelpers::to_string_with_precision<false>(static_cast<uint64_t>(floor(numerators[0] + 0.5))),
                    MathHelpers::to_string_with_precision<false>(static_cast<uint64_t>(floor(numerators[1] + 0.5))));
        if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Avg) {
            add_result3(
                    "denominator",
                    MathHelpers::to_string_with_precision<false>(static_cast<uint64_t>(floor(denominators[0] + 0.5))),
                    MathHelpers::to_string_with_precision<false>(static_cast<uint64_t>(floor(denominators[1] + 0.5))));
        } else
            add_result3("ratio", MathHelpers::to_string_with_precision<false>(_ttest_params.ratios()[0], 12, 0),
                        MathHelpers::to_string_with_precision<false>(_ttest_params.ratios()[1], 12, 0));

        if (!denominators_pre.empty()) {
            add_result3(
                    "numerator_pre",
                    MathHelpers::to_string_with_precision<false>(static_cast<uint64_t>(floor(numerators_pre[0] + 0.5))),
                    MathHelpers::to_string_with_precision<false>(
                            static_cast<uint64_t>(floor(numerators_pre[1] + 0.5))));
            if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Avg)
                add_result3("denominator_pre",
                            MathHelpers::to_string_with_precision<false>(
                                    static_cast<uint64_t>(floor(denominators_pre[0] + 0.5))),
                            MathHelpers::to_string_with_precision<false>(
                                    static_cast<uint64_t>(floor(denominators_pre[1] + 0.5))));
        }

        if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Avg) {
            add_result3("mean", MathHelpers::to_string_with_precision<false>(means[0]),
                        MathHelpers::to_string_with_precision<false>(means[1]));
            add_result3("std_samp", MathHelpers::to_string_with_precision<false>(std_samp[0]),
                        MathHelpers::to_string_with_precision<false>(std_samp[1]));
        }

        std::string ci_prefix = std::to_string((1 - _ttest_params.alpha()) * 100);
        while (!ci_prefix.empty() && ci_prefix.back() == '0') ci_prefix.pop_back();
        if (!ci_prefix.empty() && ci_prefix.back() == '.') ci_prefix.pop_back();

        std::string res = '\n' + title + '\n' + group0 + '\n' + group1 + '\n' + '\n';

        title = "";
        std::string group;

        auto add_result2 = [&title, &group](const std::string& title_, const std::string& group_) {
            title += MathHelpers::to_string_with_precision<false>(title_);
            group += MathHelpers::to_string_with_precision<false>(group_);
            size_t max_len = std::max({title.size(), group.size()});
            title += std::string(max_len - title.size(), ' ');
            group += std::string(max_len - group.size(), ' ');
        };

        add_result2("diff_relative", std::to_string(diff_relative * 100) + "%");
        add_result2(ci_prefix + "%_relative_CI",
                    "[" + std::to_string(lower_relative * 100) + "%," + std::to_string(upper_relative * 100) + "%]");
        add_result2("p-value", MathHelpers::to_string_with_precision<false>(p_value));
        add_result2("t-statistic", MathHelpers::to_string_with_precision<false>(t_stat));

        if (_ttest_params.metric_type() == XexptTtest2SampMetricType::Avg) {
            add_result2("diff", MathHelpers::to_string_with_precision<false>(estimate));
            add_result2(ci_prefix + "%_CI", "[" + std::to_string(lower) + "," + std::to_string(upper) + "]");
        }
        add_result2("power", MathHelpers::to_string_with_precision<false>(power));
        add_result2("recommend_samples", MathHelpers::to_string_with_precision<false>(
                                                 static_cast<uint64_t>(std::floor(recommend_samples + 0.5))));
        res += title + '\n' + group + '\n';
        return res;
    }

private:
    XexptTtest2SampParams _ttest_params;
    std::map<TreatmentType, XexptTtest2SampStats> _all_stats;
};

// TreatmentType should be `int` or `string`
template <typename TreatmentType>
class XexptTtest2SampAggregateFunction
        : public AggregateFunctionBatchHelper<XexptTtest2SampAggregateState<TreatmentType>,
                                              XexptTtest2SampAggregateFunction<TreatmentType>> {
public:
    using TreatmentColumnType =
            typename std::conditional_t<std::is_same_v<TreatmentType, int>, RunTimeColumnType<TYPE_INT>,
                                        RunTimeColumnType<TYPE_VARCHAR>>;

    void reset(FunctionContext* ctx, const Columns& args, AggDataPtr state) const override {
        this->data(state).reset();
    }

    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        // uin, treatment, {numerator, denominator, cuped_data...}[, cuped, {alpha, mde, power}]
        const Column* data_col = columns[2];
        auto [input, array_size] =
                FunctionHelper::get_data_of_array<DeltaMethodDataElementColumnType, double>(data_col, row_num);
        if (input == nullptr) {
            ctx->set_error("Internal Error: fail to get data.");
            return;
        }

        if (this->data(state).is_uninitialized()) {
            if (row_num > 0) {
                ctx->set_error("Internal Error: state not initialized.");
                return;
            }
            size_t num_args = ctx->get_num_args();
            std::optional<std::string> cuped_expression;
            std::optional<XexptTtest2SampMetricType> metric_type;
            std::optional<double> alpha;
            std::optional<double> mde;
            std::optional<double> power;
            std::optional<std::vector<double>> ratios;

            if (num_args >= 4) {
                std::string tmp_str;
                if (FunctionHelper::get_data_of_column<XexptTtest2SampCupedColumnType>(columns[3], row_num, tmp_str)) {
                    if (tmp_str.length() >= 2 && tmp_str.substr(0, 2) == "X=") {
                        cuped_expression = tmp_str.substr(2);
                    }
                }
            }
            double tmp;
            if (num_args >= 5) {
                if (FunctionHelper::get_data_of_column<XexptTtest2SampAlphaColumnType>(columns[4], row_num, tmp)) {
                    alpha = tmp;
                }
            }
            if (num_args >= 6) {
                if (FunctionHelper::get_data_of_column<XexptTtest2SampMDEColumnType>(columns[5], row_num, tmp)) {
                    mde = tmp;
                }
            }
            if (num_args >= 7) {
                if (FunctionHelper::get_data_of_column<XexptTtest2SampPowerColumnType>(columns[6], row_num, tmp)) {
                    power = tmp;
                }
            }
            if (num_args >= 8) {
                std::string metric_type_str;
                if (FunctionHelper::get_data_of_column<XexptTtest2SampMetricColumnType>(columns[7], row_num,
                                                                                        metric_type_str)) {
                    if (metric_type_str != "avg" || metric_type_str != "sum") {
                        ctx->set_error(
                                fmt::format("Invalid Argument: alternative({}) is not a valid ttest alternative.",
                                            metric_type_str)
                                        .c_str());
                        return;
                    }
                    metric_type =
                            metric_type_str == "avg" ? XexptTtest2SampMetricType::Avg : XexptTtest2SampMetricType::Sum;
                }
            }
            if (num_args >= 9) {
                auto [ratios_ptr, size] =
                        FunctionHelper::get_data_of_array<XexptTtest2SampAlphaColumnType, double>(columns[8], 0);
                if (ratios_ptr != nullptr && size == 2) {
                    ratios = std::vector<double>(ratios_ptr, ratios_ptr + size);
                }
            }

            LOG(INFO) << fmt::format(
                    "xexpt ttest args - cuped_expression: {}, alpha: {}, mde: {}, power: {}, metric_type: {}, ratios: "
                    "({}, {})",
                    cuped_expression.value_or("null"), alpha.value_or(TtestCommon::kDefaultAlphaValue),
                    mde.value_or(TtestCommon::kDefaultMDEValue), power.value_or(TtestCommon::kDefaultPowerValue),
                    (int)metric_type.value_or(XexptTtest2SampMetricType::Avg),
                    ratios.value_or(std::vector<double>{1, 1})[0], ratios.value_or(std::vector<double>{1, 1})[1]);
            this->data(state).init(array_size, cuped_expression, alpha, mde, power, metric_type, ratios);
        }

        const Column* uin_col = columns[0];
        int32_t uin;
        if (!FunctionHelper::get_data_of_column<XexptTtest2SampUinColumnType>(uin_col, row_num, uin)) {
            ctx->set_error("Internal Error: tail to get uin.");
            return;
        }
        const Column* treatment_col = columns[1];
        TreatmentType treatment;
        if (!FunctionHelper::get_data_of_column<TreatmentColumnType>(treatment_col, row_num, treatment)) {
            ctx->set_error("Internal Error: tail to get treatment.");
            return;
        }

        this->data(state).update(input, array_size, uin, treatment);
    }

    void merge(FunctionContext* ctx, const Column* column, AggDataPtr __restrict state, size_t row_num) const override {
        column = FunctionHelper::unwrap_if_nullable<const Column*>(column, row_num);
        if (column == nullptr) {
            ctx->set_error("Internal Error: fail to get intermediate data.");
            return;
        }
        DCHECK(column->is_binary());
        const uint8_t* serialized_data = reinterpret_cast<const uint8_t*>(column->get(row_num).get_slice().data);
        if (this->data(state).is_uninitialized()) {
            this->data(state).deserialize(serialized_data);
            return;
        }
        XexptTtest2SampAggregateState<TreatmentType> other(serialized_data);
        this->data(state).merge(other);
    }

    void serialize_to_column(FunctionContext* ctx, ConstAggDataPtr __restrict state, Column* to) const override {
        if (to->is_nullable()) {
            auto* dst_nullable_col = down_cast<NullableColumn*>(to);
            dst_nullable_col->null_column_data().emplace_back(false);
            to = dst_nullable_col->data_column().get();
        }
        auto* column = down_cast<BinaryColumn*>(to);
        Bytes& bytes = column->get_bytes();
        size_t old_size = bytes.size();
        size_t new_size = old_size + this->data(state).serialized_size();
        bytes.resize(new_size);
        column->get_offset().emplace_back(new_size);
        uint8_t* serialized_data = bytes.data() + old_size;
        this->data(state).serialize(serialized_data);
    }

    void finalize_to_column(FunctionContext* ctx, ConstAggDataPtr __restrict state, Column* to) const override {
        if (to->is_nullable()) {
            auto* dst_nullable_col = down_cast<NullableColumn*>(to);
            if (this->data(state).is_uninitialized()) {
                ctx->set_error("Internal Error: state not initialized.");
                return;
            }
            dst_nullable_col->null_column_data().emplace_back(false);
            to = dst_nullable_col->data_column().get();
        }
        std::string result = this->data(state).get_ttest_result();
        down_cast<XexptTtest2SampResultColumnType*>(to)->append(result);
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {}

    std::string get_name() const override { return std::string(AllInSqlFunctions::xexpt_ttest_2samp); }
};

} // namespace starrocks
