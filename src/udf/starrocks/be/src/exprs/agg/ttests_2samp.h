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
#include <ios>
#include <iterator>
#include <limits>
#include <sstream>

#include "column/const_column.h"
#include "column/vectorized_fwd.h"
#include "delta_method.h"
#include "exprs/agg/aggregate.h"
#include "exprs/agg/ttest_common.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/serialize_helpers.hpp"
#include "gutil/casts.h"
#include "types/logical_type.h"

namespace starrocks {

using Ttests2SampAlternativeColumnType = RunTimeColumnType<TYPE_VARCHAR>;
using Ttests2SampDataMuColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using Ttests2SampDataAlphaColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using Ttests2SampDataArrayColumnType = RunTimeColumnType<TYPE_ARRAY>;
using Ttests2SampDataElementColumnType = RunTimeColumnType<TYPE_DOUBLE>;
using Ttests2SampTreatmentColumnType = RunTimeColumnType<TYPE_INT>;
using Ttests2SampResultColumnType = RunTimeColumnType<TYPE_VARCHAR>;

class Ttests2SampParams {
public:
    bool operator==(const Ttests2SampParams& other) const {
        return _alternative == other._alternative && _alpha == other._alpha &&
               _cuped_expression == other._cuped_expression;
    }

    bool is_uninitialized() const { return _alternative == TtestAlternative::Unknown; }

    void reset() {
        _alternative = TtestAlternative::Unknown;
        _num_variables = -1;
        _alpha = TtestCommon::kDefaultAlphaValue;
        std::string().swap(_Y_expression);
        std::string().swap(_cuped_expression);
    }

    void init(TtestAlternative alternative, int num_variables, std::string const& Y_expression,
              std::string const& cuped_expression, double alpha) {
        _alternative = alternative;
        _num_variables = num_variables;
        _alpha = alpha;
        _Y_expression = Y_expression;
        _cuped_expression = cuped_expression;
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
    }

    size_t serialized_size() const {
        if (is_uninitialized()) {
            return sizeof(_alternative);
        }
        return sizeof(_alternative) + sizeof(_num_variables) + sizeof(_alpha) + sizeof(uint32_t) +
               _cuped_expression.length() + sizeof(uint32_t) + _Y_expression.length();
    }

    TtestAlternative alternative() const { return _alternative; }

    int num_variables() const { return _num_variables; }

    double alpha() const { return _alpha; }

    std::string const& Y_expression() const { return _Y_expression; }

    std::string const& cuped_expression() const { return _cuped_expression; }

private:
    TtestAlternative _alternative{TtestAlternative::Unknown};
    int _num_variables{-1};
    double _alpha{0.05};
    std::string _Y_expression;
    std::string _cuped_expression;
};

class Ttests2SampAggregateState {
public:
    Ttests2SampAggregateState() = default;
    Ttests2SampAggregateState(const Ttests2SampAggregateState&) = delete;
    Ttests2SampAggregateState(Ttests2SampAggregateState&&) = delete;

    Ttests2SampAggregateState(const uint8_t* serialized_data) { deserialize(serialized_data); }

    bool is_uninitialized() const {
        if (ttest_params.is_uninitialized()) {
            return true;
        }
        for (auto const& [_, stats] : all_stats) {
            if (stats.is_uninitialized()) {
                return true;
            }
        }
        return false;
    }

    void check_params(Ttests2SampAggregateState const& other) const { DCHECK(ttest_params == other.ttest_params); }

    void init(TtestAlternative alternative, int num_variables, std::string const& Y_expression,
              std::optional<std::string> const& cuped_expression, std::optional<double> alpha) {
        ttest_params.init(alternative, num_variables, Y_expression, cuped_expression.value_or(std::string{}),
                          alpha.value_or(TtestCommon::kDefaultAlphaValue));
    }

    void update(const double* input, int num_variables, uint32_t treatment) {
        CHECK(!is_uninitialized());
        if (all_stats.count(treatment) == 0) {
            all_stats[treatment].init(num_variables);
        }
        all_stats[treatment].update(input, num_variables);
    }

    void merge(Ttests2SampAggregateState const& other) {
        if (other.is_uninitialized()) {
            reset();
            return;
        }
        check_params(other);
        for (auto const& [treatment, stats] : other.all_stats) {
            if (all_stats.count(treatment) == 0) {
                all_stats[treatment].init(ttest_params.num_variables());
            }
            all_stats[treatment].merge(stats);
        }
    }

    void serialize(uint8_t*& data) const {
        ttest_params.serialize(data);
        if (ttest_params.is_uninitialized()) {
            return;
        }
        uint32_t num_groups = all_stats.size();
        SerializeHelpers::serialize(&num_groups, data);
        for (auto const& [treatment, stats] : all_stats) {
            SerializeHelpers::serialize(&treatment, data);
            stats.serialize(data);
        }
    }

    void deserialize(const uint8_t*& data) {
        ttest_params.deserialize(data);
        if (ttest_params.is_uninitialized()) {
            return;
        }
        uint32_t num_groups;
        SerializeHelpers::deserialize(data, &num_groups);
        for (uint32_t i = 0; i < num_groups; ++i) {
            uint32_t treatment;
            SerializeHelpers::deserialize(data, &treatment);
            auto& stats = all_stats[treatment];
            stats.init(ttest_params.num_variables());
            stats.deserialize(data);
        }
    }

    void reset() {
        ttest_params.reset();
        all_stats.clear();
    }

    size_t serialized_size() const {
        size_t size = ttest_params.serialized_size();
        if (ttest_params.is_uninitialized()) {
            return size;
        }
        size += sizeof(uint32_t);
        for (auto const& [treatment, stats] : all_stats) {
            size += sizeof(treatment);
            size += stats.serialized_size();
        }
        return size;
    }

    std::string get_ttest_result() const {
        if (is_uninitialized()) {
            return fmt::format("Internal error: ttest agg state is uninitialized.");
        }
        for (auto const& [treatment, stats] : all_stats) {
            if (stats.count() <= 1) {
                return fmt::format("count({}) of group({}) should be greater than 1.", stats.count(), treatment);
            }
        }

        std::stringstream result_ss;
        result_ss << "\n";
        result_ss << MathHelpers::to_string_with_precision("control");
        result_ss << MathHelpers::to_string_with_precision("treatment");
        result_ss << MathHelpers::to_string_with_precision("mean0");
        result_ss << MathHelpers::to_string_with_precision("mean1");
        result_ss << MathHelpers::to_string_with_precision("estimate");
        result_ss << MathHelpers::to_string_with_precision("stderr");
        result_ss << MathHelpers::to_string_with_precision("t-statistic");
        result_ss << MathHelpers::to_string_with_precision("p-value");
        result_ss << MathHelpers::to_string_with_precision("lower");
        result_ss << MathHelpers::to_string_with_precision("upper");
        result_ss << "\n";

        for (auto iter0 = all_stats.begin(); iter0 != all_stats.end(); ++iter0) {
            auto const& [control, delta_method_stats0] = *iter0;
            for (auto iter1 = std::next(iter0); iter1 != all_stats.end(); ++iter1) {
                auto const& [treatment, delta_method_stats1] = *iter1;

                DeltaMethodStats delta_method_stats;
                delta_method_stats.init(ttest_params.num_variables());
                delta_method_stats.merge(delta_method_stats0);
                delta_method_stats.merge(delta_method_stats1);

                double mean0 = 0, mean1 = 0, var0 = 0, var1 = 0;

                if (!TtestCommon::calc_means_and_vars(
                            ttest_params.Y_expression(), ttest_params.cuped_expression(), ttest_params.num_variables(),
                            delta_method_stats0.count(), delta_method_stats1.count(), delta_method_stats0.means(),
                            delta_method_stats1.means(), delta_method_stats.means(), delta_method_stats0.cov_matrix(),
                            delta_method_stats1.cov_matrix(), delta_method_stats.cov_matrix(), mean0, mean1, var0,
                            var1)) {
                    return "InvertMatrix failed. some variables in the table are perfectly collinear.";
                }

                double stderr_var = std::sqrt(var0 + var1);

                if (!std::isfinite(stderr_var)) {
                    return fmt::format("stderr({}) is an abnormal float value, please check your data.", stderr_var);
                }

                double estimate = mean1 - mean0;
                double t_stat = estimate / stderr_var;
                size_t count = delta_method_stats0.count() + delta_method_stats1.count();

                double p_value = TtestCommon::calc_pvalue(t_stat, ttest_params.alternative());
                auto [lower, upper] = TtestCommon::calc_confidence_interval(
                        estimate, stderr_var, count, ttest_params.alpha(), ttest_params.alternative());

                result_ss << MathHelpers::to_string_with_precision(control);
                result_ss << MathHelpers::to_string_with_precision(treatment);
                result_ss << MathHelpers::to_string_with_precision(mean0);
                result_ss << MathHelpers::to_string_with_precision(mean1);
                result_ss << MathHelpers::to_string_with_precision(estimate);
                result_ss << MathHelpers::to_string_with_precision(stderr_var);
                result_ss << MathHelpers::to_string_with_precision(t_stat);
                result_ss << MathHelpers::to_string_with_precision(p_value);
                result_ss << MathHelpers::to_string_with_precision(lower);
                result_ss << MathHelpers::to_string_with_precision(upper);
                result_ss << "\n";
            }
        }

        return result_ss.str();
    }

private:
    Ttests2SampParams ttest_params;
    std::map<uint32_t, DeltaMethodStats> all_stats;
};

class Ttests2SampAggregateFunction
        : public AggregateFunctionBatchHelper<Ttests2SampAggregateState, Ttests2SampAggregateFunction> {
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
            LOG(WARNING) << "ttests_2samp: fail to get data.";
            return;
        }

        if (this->data(state).is_uninitialized()) {
            const Column* expr_col = columns[0];
            const auto* func_expr = down_cast<const DeltaMethodExprColumnType*>(expr_col);
            Slice expr_slice = func_expr->get_data()[0];
            std::string expression = expr_slice.to_string();
            std::optional<std::string> cuped_expression;
            std::optional<double> alpha;

            if (ctx->get_num_args() >= 5) {
                cuped_expression = try_parse_cuped(columns[4]);
                alpha = try_parse_alpha(columns[4]);
            }

            if (ctx->get_num_args() >= 6) {
                if (!cuped_expression.has_value()) {
                    cuped_expression = try_parse_cuped(columns[5]);
                }
                if (!alpha.has_value()) {
                    alpha = try_parse_alpha(columns[5]);
                }
            }

            const Column* alternative_col = columns[1];
            const auto* alternative_column =
                    FunctionHelper::unwrap_if_const<const Ttests2SampAlternativeColumnType*>(alternative_col);
            if (alternative_col == nullptr) {
                LOG(WARNING) << fmt::format("unable to unwrap const column alternative_col");
                return;
            }
            std::string alternative_str = alternative_column->get_data()[0].to_string();
            if (!TtestCommon::str2alternative.count(alternative_str)) {
                LOG(WARNING) << fmt::format("alternative({}) is not a valid ttest alternative.", alternative_str);
                return;
            }
            TtestAlternative alternative = TtestCommon::str2alternative.at(alternative_str);

            LOG(INFO) << fmt::format("ttest args - expression: {}, alternative: {}, cuped_expression: {}, alpha: {}",
                                     expression, (int)alternative, cuped_expression.value_or("null"),
                                     alpha.value_or(TtestCommon::kDefaultAlphaValue));
            this->data(state).init(alternative, input_opt->size(), expression, cuped_expression, alpha);
        }

        const Column* treatment_col = columns[2];
        const auto* treatment_column =
                FunctionHelper::unwrap_if_nullable<const Ttests2SampTreatmentColumnType*>(treatment_col, row_num);
        CHECK(treatment_col != nullptr);
        double treatment = treatment_column->get_data()[row_num];

        std::vector<double> input;
        input.reserve(input_opt->size());
        for (size_t i = 0; i < input_opt->size(); ++i) {
            if ((*input_opt)[i].is_null()) {
                // if any element is null, skip this row
                return;
            }
            input.emplace_back((*input_opt)[i].get_double());
        }

        this->data(state).update(input.data(), input_opt->size(), treatment);
    }

    std::optional<std::string> try_parse_cuped(const Column* cuped_expr_col) const {
        cuped_expr_col = FunctionHelper::unwrap_if_const<const Column*>(cuped_expr_col);
        if (cuped_expr_col == nullptr) {
            LOG(WARNING) << fmt::format("unable to unwrap const column.");
            return std::nullopt;
        }
        if (typeid(*cuped_expr_col) != typeid(DeltaMethodExprColumnType)) {
            LOG(WARNING) << fmt::format("col is not a varchar column");
            return std::nullopt;
        }
        const auto* cuped_func_expr = down_cast<const DeltaMethodExprColumnType*>(cuped_expr_col);
        std::string cuped_expr = cuped_func_expr->get_data()[0].to_string();
        std::string const prefix = "X=";
        size_t prefix_length = prefix.length();
        if (prefix != cuped_expr.substr(0, prefix_length)) {
            return std::nullopt;
        }
        return cuped_expr.substr(prefix_length);
    }

    std::optional<double> try_parse_alpha(const Column* alpha_col) const {
        alpha_col = FunctionHelper::unwrap_if_const<const Column*>(alpha_col);
        if (alpha_col == nullptr) {
            LOG(WARNING) << fmt::format("unable to unwrap const column.");
            return std::nullopt;
        }
        if (typeid(*alpha_col) != typeid(Ttests2SampDataAlphaColumnType)) {
            LOG(WARNING) << fmt::format("col is not a double column");
            return std::nullopt;
        }
        const auto* alpha_column = down_cast<const Ttests2SampDataAlphaColumnType*>(alpha_col);
        return alpha_column->get_data()[0];
    }

    void merge(FunctionContext* ctx, const Column* column, AggDataPtr __restrict state, size_t row_num) const override {
        column = FunctionHelper::unwrap_if_nullable<const Column*>(column, row_num);
        if (column == nullptr) {
            this->data(state).reset();
            return;
        }
        DCHECK(column->is_binary());
        const uint8_t* serialized_data = reinterpret_cast<const uint8_t*>(column->get(row_num).get_slice().data);
        if (this->data(state).is_uninitialized()) {
            this->data(state).deserialize(serialized_data);
            return;
        }
        Ttests2SampAggregateState other(serialized_data);
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
                dst_nullable_col->append_nulls(1);
                return;
            }
            dst_nullable_col->null_column_data().emplace_back(false);
            to = dst_nullable_col->data_column().get();
        }
        std::string result;
        if (this->data(state).is_uninitialized()) {
            result = "Null";
        } else {
            result = this->data(state).get_ttest_result();
        }
        down_cast<Ttests2SampResultColumnType*>(to)->append(result);
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {}

    std::string get_name() const override { return std::string(AllInSqlFunctions::ttests_2samp); }
};

} // namespace starrocks
