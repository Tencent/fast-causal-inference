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
#include <sstream>

#include "column/const_column.h"
#include "column/vectorized_fwd.h"
#include "delta_method.h"
#include "exprs/agg/aggregate.h"
#include "exprs/agg/ttest_common.h"
#include "exprs/all_in_sql_functions.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/serialize_helpers.hpp"
#include "gutil/casts.h"
#include "types/logical_type.h"

namespace starrocks {

using Ttest2SampAlternativeColumnType = RunTimeColumnType<TYPE_VARCHAR>;
using Ttest2SampDataMuColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using Ttest2SampDataAlphaColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using Ttest2SampDataArrayColumnType = RunTimeColumnType<TYPE_ARRAY>;
using Ttest2SampDataElementColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using Ttest2SampTreatmentColumnType = RunTimeColumnType<TYPE_BOOLEAN>;
using Ttest2SampResultColumnType = RunTimeColumnType<TYPE_VARCHAR>;

class Ttest2SampParams {
public:
    bool operator==(const Ttest2SampParams& other) const {
        return _alternative == other._alternative && _alpha == other._alpha &&
               _cuped_expression == other._cuped_expression && _num_pses == other._num_pses;
    }

    bool is_uninitialized() const { return _alternative == TtestAlternative::Unknown; }

    void reset() {
        _alternative = TtestAlternative::Unknown;
        _num_variables = -1;
        _alpha = TtestCommon::kDefaultAlphaValue;
        std::string().swap(_Y_expression);
        std::string().swap(_cuped_expression);
        _num_pses = 0;
    }

    void init(TtestAlternative alternative, int num_variables, std::string const& Y_expression,
              std::string const& cuped_expression, double alpha, int num_pses) {
        _alternative = alternative;
        _num_variables = num_variables;
        _alpha = alpha;
        _Y_expression = Y_expression;
        _cuped_expression = cuped_expression;
        _num_pses = num_pses;
    }

    void serialize(uint8_t*& data) const {
        SerializeHelpers::serialize(reinterpret_cast<const uint8_t*>(&_alternative), data);
        if (is_uninitialized()) {
            return;
        }
        SerializeHelpers::serialize(&_num_variables, data);
        SerializeHelpers::serialize(&_alpha, data);
        uint32_t cuped_expression_length = _cuped_expression.length();
        SerializeHelpers::serialize(&cuped_expression_length, data);
        SerializeHelpers::serialize(_cuped_expression.data(), data, cuped_expression_length);
        uint32_t Y_expression_length = _Y_expression.length();
        SerializeHelpers::serialize(&Y_expression_length, data);
        SerializeHelpers::serialize(_Y_expression.data(), data, Y_expression_length);
        SerializeHelpers::serialize(&_num_pses, data);
    }

    void deserialize(const uint8_t*& data) {
        SerializeHelpers::deserialize(data, reinterpret_cast<uint8_t*>(&_alternative));
        if (is_uninitialized()) {
            return;
        }
        SerializeHelpers::deserialize(data, &_num_variables);
        SerializeHelpers::deserialize(data, &_alpha);
        uint32_t cuped_expression_length;
        SerializeHelpers::deserialize(data, &cuped_expression_length);
        _cuped_expression.resize(cuped_expression_length);
        SerializeHelpers::deserialize(data, _cuped_expression.data(), cuped_expression_length);
        uint32_t Y_expression_length;
        SerializeHelpers::deserialize(data, &Y_expression_length);
        _Y_expression.resize(Y_expression_length);
        SerializeHelpers::deserialize(data, _Y_expression.data(), Y_expression_length);
        SerializeHelpers::deserialize(data, &_num_pses);
    }

    size_t serialized_size() const {
        if (is_uninitialized()) {
            return sizeof(_alternative);
        }
        return sizeof(_alternative) + sizeof(_num_variables) + sizeof(_alpha) + sizeof(uint32_t) +
               _cuped_expression.length() + sizeof(uint32_t) + _Y_expression.length() + sizeof(_num_pses);
    }

    TtestAlternative alternative() const { return _alternative; }

    int num_variables() const { return _num_variables; }

    double alpha() const { return _alpha; }

    std::string const& Y_expression() const { return _Y_expression; }

    std::string const& cuped_expression() const { return _cuped_expression; }

    int num_pses() const { return _num_pses; }

private:
    TtestAlternative _alternative{TtestAlternative::Unknown};
    int _num_pses;
    int _num_variables{-1};
    double _alpha{0.05};
    std::string _Y_expression;
    std::string _cuped_expression;
};

class Ttest2SampAggregateState {
public:
    Ttest2SampAggregateState() = default;
    Ttest2SampAggregateState(const Ttest2SampAggregateState&) = delete;
    Ttest2SampAggregateState(Ttest2SampAggregateState&&) = delete;

    Ttest2SampAggregateState(const uint8_t* serialized_data) { deserialize(serialized_data); }

    bool is_uninitialized() const {
        return _ttest_params.is_uninitialized() || _delta_method_stats0.is_uninitialized() ||
               _delta_method_stats1.is_uninitialized();
    }

    void check_params(Ttest2SampAggregateState const& other) const { DCHECK(_ttest_params == other._ttest_params); }

    void init(TtestAlternative alternative, int num_variables, std::string const& Y_expression,
              std::string const& cuped_expression = "", double alpha = TtestCommon::kDefaultAlphaValue,
              int num_pses = 0) {
        _ttest_params.init(alternative, num_variables, Y_expression, cuped_expression, alpha, num_pses);
        if (!_ttest_params.is_uninitialized()) {
            _delta_method_stats0.init(num_variables);
            _delta_method_stats1.init(num_variables);
        }
    }

    void update(const double* input, int num_variables, bool treatment, std::vector<std::string> const& pse) {
        DCHECK(!is_uninitialized() && !_delta_method_stats0.is_uninitialized() &&
               !_delta_method_stats1.is_uninitialized());
        DCHECK(pse.size() == _ttest_params.num_pses());
        if (treatment) {
            _delta_method_stats1.update(input, num_variables);
        } else {
            _delta_method_stats0.update(input, num_variables);
        }
        if (UNLIKELY(_ttest_params.num_pses() != 0)) {
            uint64_t group = treatment;
            uint64_t combined_hash = 0;
            for (auto& str : pse) {
                for (const auto& ch : str) {
                    combined_hash = combined_hash ^ (ch + 0x9e3779b9 + (combined_hash << 6) + (combined_hash >> 2));
                }
            }
            combined_hash = combined_hash << 8;
            auto new_group = group + combined_hash;

            _pse2index[std::make_pair(combined_hash, group)] = new_group;
            group = new_group;
            if (!_group_stats.count(group)) {
                _group_stats[group].init(num_variables);
            }
            _group_stats[group].update(input, num_variables);
        }
    }

    void merge(Ttest2SampAggregateState const& other) {
        if (other.is_uninitialized()) {
            reset();
            return;
        }
        check_params(other);
        _delta_method_stats0.merge(other._delta_method_stats0);
        _delta_method_stats1.merge(other._delta_method_stats1);
        for (auto&& [pr, index] : other._pse2index) {
            if (_pse2index.count(pr)) {
                continue;
            }
            _pse2index[pr] = index;
        }
        for (auto&& [key, stats] : other._group_stats) {
            if (!_group_stats.count(key)) {
                _group_stats[key].init(_ttest_params.num_variables());
            }
            _group_stats[key].merge(stats);
        }
    }

    void serialize(uint8_t*& data) const {
        _ttest_params.serialize(data);
        if (_ttest_params.is_uninitialized()) {
            return;
        }
        DCHECK(!_delta_method_stats0.is_uninitialized());
        _delta_method_stats0.serialize(data);
        DCHECK(!_delta_method_stats1.is_uninitialized());
        _delta_method_stats1.serialize(data);
        size_t length = _group_stats.size();
        SerializeHelpers::serialize(&length, data);
        for (auto&& [l, r] : _group_stats) {
            SerializeHelpers::serialize(&l, data);
            r.serialize(data);
        }
        length = _pse2index.size();
        SerializeHelpers::serialize(&length, data);
        for (auto&& [pr, index] : _pse2index) {
            SerializeHelpers::serialize(&pr.first, data);
            SerializeHelpers::serialize(&pr.second, data);
            SerializeHelpers::serialize(&index, data);
        }
    }

    void deserialize(const uint8_t*& data) {
        _ttest_params.deserialize(data);
        if (_ttest_params.is_uninitialized()) {
            return;
        }
        _delta_method_stats0.init(_ttest_params.num_variables());
        _delta_method_stats0.deserialize(data);
        _delta_method_stats1.init(_ttest_params.num_variables());
        _delta_method_stats1.deserialize(data);
        size_t length = 0;
        SerializeHelpers::deserialize(data, length);
        for (size_t i = 0; i < length; ++i) {
            uint64_t key;
            SerializeHelpers::deserialize(data, key);
            _group_stats[key].init(_ttest_params.num_variables());
            _group_stats[key].deserialize(data);
        }
        SerializeHelpers::deserialize(data, length);
        for (size_t i = 0; i < length; ++i) {
            uint64_t first, second, index;
            SerializeHelpers::deserialize(data, &first);
            SerializeHelpers::deserialize(data, &second);
            SerializeHelpers::deserialize(data, &index);
            _pse2index[std::make_pair(first, second)] = index;
        }
    }

    void reset() {
        _ttest_params.reset();
        _delta_method_stats0.reset();
        _delta_method_stats1.reset();
        _group_stats.clear();
    }

    size_t serialized_size() const {
        size_t size = _ttest_params.serialized_size();
        if (_ttest_params.is_uninitialized()) {
            return size;
        }
        DCHECK(!_delta_method_stats0.is_uninitialized());
        DCHECK(!_delta_method_stats1.is_uninitialized());
        size += _delta_method_stats0.serialized_size();
        size += _delta_method_stats1.serialized_size();
        size += sizeof(size_t);
        for (auto&& [l, r] : _group_stats) {
            size += sizeof(l);
            DCHECK(!r.is_uninitialized());
            size += r.serialized_size();
        }
        size += sizeof(size_t);
        size += _pse2index.size() * sizeof(uint64_t) * 3;
        return size;
    }

    Status calc_means_and_vars_with_pse(double& mean0, double& mean1, double& var0, double& var1,
                                        std::string& warning_prefix) const {
        DeltaMethodStats delta_method_stats;
        delta_method_stats.init(_ttest_params.num_variables());
        delta_method_stats.merge(_delta_method_stats0);
        delta_method_stats.merge(_delta_method_stats1);
        uint64_t total_count = delta_method_stats.count();

        bool only_one_treatment = false;
        bool only_one_sample = false;

        auto l = _pse2index.begin();
        while (l != _pse2index.end()) {
            auto [pse2group, index] = *l;
            auto [pse, group] = pse2group;
            auto r = l;
            std::vector<DeltaMethodStats> substats;
            while (r != _pse2index.end() && r->first.first == pse) {
                if (_group_stats.count(r->second) == 0) {
                    return Status::InternalError("Some covariance matrix is missing.");
                }
                substats.emplace_back(_group_stats.at(r->second));
                r = std::next(r);
            }
            l = r;
            if (substats.size() > 2) {
                return Status::InvalidArgument("Ttest_2samp only support two samples");
            }
            if (substats.size() == 1) {
                only_one_treatment = true;
                continue;
            }
            if (substats[0].count() == 1 || substats[1].count() == 1) {
                only_one_sample = true;
                continue;
            }
            DeltaMethodStats stats;
            stats.init(_ttest_params.num_variables());
            stats.merge(substats[0]);
            stats.merge(substats[1]);
            double pse_mean0 = 0, pse_mean1 = 0, pse_vars0 = 0, pse_vars1 = 0;
            if (!TtestCommon::calc_means_and_vars(_ttest_params.Y_expression(), _ttest_params.cuped_expression(),
                                                  _ttest_params.num_variables(), substats[0].count(),
                                                  substats[1].count(), substats[0].means(), substats[1].means(),
                                                  stats.means(), substats[0].cov_matrix(), substats[1].cov_matrix(),
                                                  stats.cov_matrix(), pse_mean0, pse_mean1, pse_vars0, pse_vars1)) {
                return Status::InvalidArgument(
                        "InvertMatrix failed. some variables in the table are perfectly collinear.");
            }
            double weight = 1. * stats.count() / total_count;
            mean0 += pse_mean0 * weight;
            mean1 += pse_mean1 * weight;
            var0 += pse_vars0 * weight * weight;
            var1 += pse_vars1 * weight * weight;
        }
        if (only_one_treatment) {
            warning_prefix = "Warning: Variance cannot be computed as one of the groups has only a single sample.\n";
        }
        if (only_one_sample) {
            warning_prefix += "Warning: Cannot perform t-test as it contains only one group.\n";
        }
        if (!warning_prefix.empty()) {
            warning_prefix += "\n";
        }
        return Status::OK();
    }

    std::string get_ttest_result() const {
        if (is_uninitialized()) {
            return "Internal error: ttest agg state is uninitialized.";
        }
        if (_delta_method_stats0.count() == 0) {
            return "error: at least 2 groups are required for 2-sample t-test, please check the argument of index";
        }
        if (_delta_method_stats1.count() == 0) {
            return "error: at least 2 groups are required for 2-sample t-test, please check the argument of index";
        }

        double mean0 = 0, mean1 = 0, var0 = 0, var1 = 0;
        std::string warning_prefix;

        if (_ttest_params.num_pses() == 0) {
            DeltaMethodStats delta_method_stats;
            delta_method_stats.init(_ttest_params.num_variables());
            delta_method_stats.merge(_delta_method_stats0);
            delta_method_stats.merge(_delta_method_stats1);
            if (!TtestCommon::calc_means_and_vars(
                        _ttest_params.Y_expression(), _ttest_params.cuped_expression(), _ttest_params.num_variables(),
                        _delta_method_stats0.count(), _delta_method_stats1.count(), _delta_method_stats0.means(),
                        _delta_method_stats1.means(), delta_method_stats.means(), _delta_method_stats0.cov_matrix(),
                        _delta_method_stats1.cov_matrix(), delta_method_stats.cov_matrix(), mean0, mean1, var0, var1)) {
                return "InvertMatrix failed. some variables in the table are perfectly collinear.";
            }
        } else {
            DCHECK(_ttest_params.num_pses() > 0);
            auto st = calc_means_and_vars_with_pse(mean0, mean1, var0, var1, warning_prefix);
            if (!st.ok()) {
                return st.get_error_msg();
            }
        }

        double stderr_var = std::sqrt(var0 + var1);

        if (!std::isfinite(stderr_var)) {
            return fmt::format("stderr({}) is an abnormal float value, please check your data.", stderr_var);
        }

        double estimate = mean1 - mean0;
        double t_stat = estimate / stderr_var;
        size_t count = _delta_method_stats0.count() + _delta_method_stats1.count();

        double p_value = TtestCommon::calc_pvalue(t_stat, _ttest_params.alternative());
        auto [lower, upper] = TtestCommon::calc_confidence_interval(estimate, stderr_var, count, _ttest_params.alpha(),
                                                                    _ttest_params.alternative());

        std::stringstream result_ss;
        result_ss << warning_prefix;
        result_ss << "\n";
        result_ss << MathHelpers::to_string_with_precision("mean0");
        result_ss << MathHelpers::to_string_with_precision("mean1");
        result_ss << MathHelpers::to_string_with_precision("estimate");
        result_ss << MathHelpers::to_string_with_precision("stderr");
        result_ss << MathHelpers::to_string_with_precision("t-statistic");
        result_ss << MathHelpers::to_string_with_precision("p-value");
        result_ss << MathHelpers::to_string_with_precision("lower");
        result_ss << MathHelpers::to_string_with_precision("upper");
        result_ss << "\n";
        result_ss << MathHelpers::to_string_with_precision(mean0);
        result_ss << MathHelpers::to_string_with_precision(mean1);
        result_ss << MathHelpers::to_string_with_precision(estimate);
        result_ss << MathHelpers::to_string_with_precision(stderr_var);
        result_ss << MathHelpers::to_string_with_precision(t_stat);
        result_ss << MathHelpers::to_string_with_precision(p_value);
        result_ss << MathHelpers::to_string_with_precision(lower);
        result_ss << MathHelpers::to_string_with_precision(upper);
        result_ss << "\n";

        return result_ss.str();
    }

    size_t num_pses() const { return _ttest_params.num_pses(); }

    size_t num_groups() const { return _group_stats.size(); }

private:
    Ttest2SampParams _ttest_params;
    DeltaMethodStats _delta_method_stats0;
    DeltaMethodStats _delta_method_stats1;
    std::map<uint64_t, DeltaMethodStats> _group_stats;
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> _pse2index;
};

class Ttest2SampAggregateFunction
        : public AggregateFunctionBatchHelper<Ttest2SampAggregateState, Ttest2SampAggregateFunction> {
public:
    void reset(FunctionContext* ctx, const Columns& args, AggDataPtr state) const override {
        this->data(state).reset();
    }

    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        // expression, side, treatment, data, [cuped, alpha]
        const Column* data_col = columns[3];
        auto input_opt = FunctionHelper::get_data_of_array(data_col, row_num);
        if (!input_opt) {
            // ctx->set_error("Internal Error: fail to get data.");
            return;
        }

        size_t array_size = input_opt.value().size();

        if (this->data(state).is_uninitialized()) {
            const Column* expr_col = columns[0];
            const auto* func_expr = down_cast<const DeltaMethodExprColumnType*>(expr_col);
            Slice expr_slice = func_expr->get_data()[0];
            std::string expression = expr_slice.to_string();
            std::string cuped_expression = "";
            double alpha = TtestCommon::kDefaultAlphaValue;

            if (ctx->get_num_args() >= 5) {
                auto data = columns[4]->get(0);
                if (data.is_null()) {
                    ctx->set_error("Invalid Augument: cuped cannot be null.");
                    return;
                }
                std::string cuped_expr = data.get_slice().to_string();
                if (cuped_expr.size() < 2 || cuped_expr.substr(0, 2) != "X=") {
                    ctx->set_error("Invalid Argument: cuped expression should start with 'X='.");
                    return;
                }
                cuped_expression = cuped_expr.substr(2);
            }

            if (ctx->get_num_args() >= 6) {
                auto alpha_ = columns[5]->get(0);
                if (alpha_.is_null()) {
                    ctx->set_error("Invalid Argument: alpha cannot be null.");
                    return;
                }
                alpha = alpha_.get_double();
            }

            auto alternative_datum = columns[1]->get(0);
            if (alternative_datum.is_null()) {
                ctx->set_error("Internal Error: alternative cannot be null.");
                return;
            }

            std::string alternative_str = alternative_datum.get_slice().to_string();
            if (!TtestCommon::str2alternative.count(alternative_str)) {
                ctx->set_error(fmt::format("Invalid Argument: alternative({}) is not a valid ttest alternative.",
                                           alternative_str)
                                       .c_str());
                return;
            }
            TtestAlternative alternative = TtestCommon::str2alternative.at(alternative_str);

            int num_pses = 0;
            if (ctx->get_num_args() >= 7) {
                auto datum_array = FunctionHelper::get_data_of_array(columns[6], 0);
                if (!datum_array) {
                    return;
                }
                num_pses = datum_array.value().size();
            }

            LOG(INFO) << fmt::format(
                    "ttest args - expression: {}, alternative: {}, cuped_expression: {}, alpha: {}, num_pses: {}",
                    expression, (int)alternative, cuped_expression, alpha, num_pses);
            this->data(state).init(alternative, array_size, expression, cuped_expression, alpha, num_pses);
        }

        auto treatment_datum = columns[2]->get(row_num);
        if (treatment_datum.is_null()) {
            return;
        }
        auto treatment = treatment_datum.get_int8();

        std::vector<double> input;
        input.reserve(array_size);
        for (size_t i = 0; i < array_size; ++i) {
            if (input_opt.value()[i].is_null()) {
                return;
            }
            input.emplace_back(input_opt.value()[i].get_double());
        }

        std::vector<std::string> pse;
        if (ctx->get_num_args() >= 7) {
            auto pse_datum = FunctionHelper::get_data_of_array(columns[6], row_num);
            if (!pse_datum) {
                ctx->set_error("Internal Error: pse data cannot be null.");
                return;
            }
            pse.reserve(pse_datum.value().size());
            for (auto& i : pse_datum.value()) {
                if (i.is_null()) {
                    return;
                }
                pse.emplace_back(i.get_slice().to_string());
            }
        }
        if (pse.size() != this->data(state).num_pses()) {
            ctx->set_error("Invalid Argument: pse array should be of the same length.");
            return;
        }

        this->data(state).update(input.data(), array_size, treatment, pse);

        if (this->data(state).num_groups() > 20000) {
            ctx->set_error("Data Error: Too many groups, larger than 20000.");
        }
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
        Ttest2SampAggregateState other(serialized_data);
        this->data(state).merge(other);
        if (this->data(state).num_groups() > 20000) {
            ctx->set_error("Data Error: Too many groups, larger than 20000.");
        }
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
        DCHECK_EQ(serialized_data, new_size + bytes.data());
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
        std::string result;
        if (this->data(state).is_uninitialized()) {
            ctx->set_error("Internal Error: state not initialized.");
            return;
        }
        result = this->data(state).get_ttest_result();
        down_cast<Ttest2SampResultColumnType*>(to)->append(result);
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {}

    std::string get_name() const override { return std::string(AllInSqlFunctions::ttest_2samp); }
};

} // namespace starrocks
