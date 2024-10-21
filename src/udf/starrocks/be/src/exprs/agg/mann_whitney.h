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

#include <velocypack/Builder.h>
#include <velocypack/Value.h>

#include <cctype>
#include <cmath>
#include <ios>
#include <iterator>
#include <limits>
#include <sstream>

#include "column/const_column.h"
#include "column/type_traits.h"
#include "column/vectorized_fwd.h"
#include "delta_method.h"
#include "exprs/agg/aggregate.h"
#include "exprs/agg/ttest_common.h"
#include "exprs/function_context.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/math_helpers.hpp"
#include "exprs/helpers/serialize_helpers.hpp"
#include "exprs/math_functions.h"
#include "gutil/casts.h"
#include "types/logical_type.h"
#include "util/json.h"

namespace starrocks {

class MannWhitneyAggState {
public:
    MannWhitneyAggState() = default;
    MannWhitneyAggState(const uint8_t*& data) { deserialize(data); }

    void init(TtestAlternative alternative, uint64_t continuity_correction) {
        _alternative = alternative;
        _continuity_correction = continuity_correction;
    }

    bool is_uninitialized() const { return _alternative == TtestAlternative::Unknown; }

    void update(double x, bool treatment) {
        if (!(std::isinf(x) || std::isnan(x))) {
            _stats[treatment].emplace_back(x);
        }
    }

    void merge(MannWhitneyAggState const& other) {
        DCHECK(_alternative == other._alternative);
        for (size_t idx = 0; idx < 2; ++idx) {
            _stats[idx].insert(_stats[idx].end(), other._stats[idx].begin(), other._stats[idx].end());
        }
    }

    void serialize(uint8_t*& data) const {
        DCHECK(!is_uninitialized());
        SerializeHelpers::serialize_all(data, static_cast<uint8_t>(_alternative), _stats, _continuity_correction);
    }

    void deserialize(const uint8_t*& data) {
        uint8_t tmp;
        SerializeHelpers::deserialize_all(data, tmp, _stats, _continuity_correction);
        _alternative = static_cast<TtestAlternative>(tmp);
    }

    size_t serialized_size() const {
        return SerializeHelpers::serialized_size_all(static_cast<uint8_t>(_alternative), _stats,
                                                     _continuity_correction);
    }

    void build_result(vpack::Builder& builder) const {
        if (_alternative == TtestAlternative::Unknown) {
            builder.add("Error", vpack::Value("state not initialized."));
            return;
        }
        size_t size = _stats[0].size() + _stats[1].size();
        std::vector<size_t> index(size);
        std::iota(index.begin(), index.end(), 0);
        auto data = [this](size_t idx) {
            if (idx < this->_stats[0].size()) {
                return this->_stats[0][idx];
            }
            return this->_stats[1][idx - this->_stats[0].size()];
        };
        std::sort(index.begin(), index.end(), [data](size_t lhs, size_t rhs) { return data(lhs) < data(rhs); });

        const double n1 = _stats[0].size();
        const double n2 = _stats[1].size();
        double r1 = 0;
        double tie_correction = 0;
        {
            size_t left = 0;
            double tie_numenator = 0;
            while (left < size) {
                size_t right = left;
                while (right < size && data(index[left]) == data(index[right])) {
                    ++right;
                }
                auto adjusted = (left + right + 1.) / 2.;
                auto count_equal = right - left;

                /// Scipy implementation throws exception in this case too.
                if (count_equal == size) {
                    builder.add("Error", vpack::Value("All numbers in both samples are identical."));
                    return;
                }

                tie_numenator += std::pow(count_equal, 3) - count_equal;
                size_t count = 0;
                for (size_t iter = left; iter < right; ++iter) {
                    if (index[iter] < n1) {
                        count += 1;
                    }
                }
                r1 += count * adjusted;
                left = right;
            }
            tie_correction = 1 - (tie_numenator / (std::pow(size, 3) - size));
        }

        const double u1 = n1 * n2 + (n1 * (n1 + 1.)) / 2. - r1;
        const double u2 = n1 * n2 - u1;

        /// The distribution of U-statistic under null hypothesis H0  is symmetric with respect to meanrank.
        const double meanrank = n1 * n2 / 2. + 0.5 * _continuity_correction;
        const double sd = std::sqrt(tie_correction * n1 * n2 * (n1 + n2 + 1) / 12.0);

        if (std::isnan(sd) || std::isinf(sd) || std::abs(sd) < 1e-7) {
            builder.add("Error", vpack::Value(fmt::format("sd({}) is not a valid value.", sd)));
            return;
        }

        double u = 0;
        if (_alternative == TtestAlternative::TwoSided) {
            u = std::max(u1, u2);
        } else if (_alternative == TtestAlternative::Less) {
            u = u1;
        } else if (_alternative == TtestAlternative::Greater) {
            u = u2;
        } else {
            DCHECK(false);
        }

        double z = (u - meanrank) / sd;
        if (_alternative == TtestAlternative::TwoSided) {
            z = std::abs(z);
        }

        auto standart_normal_distribution = boost::math::normal_distribution<double>();
        auto cdf = boost::math::cdf(standart_normal_distribution, z);

        double p_value = 0;
        if (_alternative == TtestAlternative::TwoSided) {
            p_value = 2 - 2 * cdf;
        } else {
            p_value = 1 - cdf;
        }

        vpack::ArrayBuilder array_builder(&builder);
        builder.add(vpack::Value(u2));
        builder.add(vpack::Value(p_value));
    }

private:
    TtestAlternative _alternative{TtestAlternative::Unknown};
    std::array<std::vector<double>, 2> _stats;
    uint64_t _continuity_correction{0};
};

class MannWhitneyAggFunction : public AggregateFunctionBatchHelper<MannWhitneyAggState, MannWhitneyAggFunction> {
public:
    using MannWhitneyDataColumn = RunTimeColumnType<TYPE_DOUBLE>;
    using MannWhitneyIndexColumn = RunTimeColumnType<TYPE_BOOLEAN>;
    using MannWhitneyAlternativeColumn = RunTimeColumnType<TYPE_VARCHAR>;
    using MannWhitneyContinuityCorrectionColumn = RunTimeColumnType<TYPE_BIGINT>;
    using MannWhitneyResultColumn = RunTimeColumnType<TYPE_JSON>;

    void reset(FunctionContext* ctx, const Columns& args, AggDataPtr state) const override { DCHECK(false); }

    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        if (this->data(state).is_uninitialized()) {
            Slice alternative;
            const Column* alternative_col = columns[2];
            if (!FunctionHelper::get_data_of_column<MannWhitneyAlternativeColumn>(alternative_col, 0, alternative)) {
                ctx->set_error("Internal Error: fail to get `alternative`.");
                return;
            }
            auto alternative_str = alternative.to_string();
            if (!TtestCommon::str2alternative.count(alternative_str)) {
                ctx->set_error("Internal Error: invalid `alternative`.");
                return;
            }
            uint64_t continuity_correction = false;
            const Column* continuity_correction_col = columns[3];
            if (!FunctionHelper::get_data_of_column<MannWhitneyContinuityCorrectionColumn>(continuity_correction_col, 0,
                                                                                           continuity_correction)) {
                ctx->set_error("Internal Error: fail to get `continuity_correction`.");
                return;
            }

            this->data(state).init(TtestCommon::str2alternative.at(alternative_str), continuity_correction);
        }

        double x;
        const Column* x_col = columns[0];
        if (!FunctionHelper::get_data_of_column<MannWhitneyDataColumn>(x_col, row_num, x)) {
            // ctx->set_error("Internal Error: fail to get `x`.");
            return;
        }

        bool treatment;
        const Column* treatment_col = columns[1];
        if (!FunctionHelper::get_data_of_column<MannWhitneyIndexColumn>(treatment_col, row_num, treatment)) {
            // ctx->set_error("Internal Error: fail to get `treatment`.");
            return;
        }

        this->data(state).update(x, treatment);
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
        MannWhitneyAggState other(serialized_data);
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
        if (this->data(state).is_uninitialized()) {
            ctx->set_error("Internal Error: state not initialized.");
            return;
        }
        vpack::Builder result_builder;
        this->data(state).build_result(result_builder);
        auto slice = result_builder.slice();
        JsonValue result_json(slice);
        down_cast<MannWhitneyResultColumn*>(to)->append(std::move(result_json));
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {}

    std::string get_name() const override { return std::string(AllInSqlFunctions::mann_whitney_u_test); }
};

} // namespace starrocks
