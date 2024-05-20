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

#include <velocypack/Builder.h>
#include <velocypack/Iterator.h>
#include <velocypack/Value.h>

#include <algorithm>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <cstdint>
#include <ctime>
#include <sstream>
#include <string>
#include <unordered_set>

#include "agent/master_info.h"
#include "column/json_column.h"
#include "column/type_traits.h"
#include "column/vectorized_fwd.h"
#include "common/compiler_util.h"
#include "exprs/agg/aggregate.h"
#include "exprs/all_in_sql_functions.h"
#include "exprs/function_context.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/math_helpers.hpp"
#include "exprs/helpers/serialize_helpers.hpp"
#include "util/json.h"

namespace starrocks {

class SRMAggState {
public:
    SRMAggState() = default;
    SRMAggState(const uint8_t* data) { deserialize(data); }

    bool is_uninitialized() const { return !_is_init; }

    void init(std::vector<double> const& ratios) {
        _ratios = ratios;
        _is_init = true;
    }

    bool is_ratio_same(SRMAggState const& other) const { return other._ratios == _ratios; }

    void update(std::string const& group, double value) { _group2sum[group] += value; }

    void merge(SRMAggState const& other) {
        for (auto&& [group, value] : other._group2sum) {
            _group2sum[group] += value;
        }
    }

    void serialize(uint8_t*& data) const {
        DCHECK(_is_init);
        uint32_t ratio_length = _ratios.size();
        SerializeHelpers::serialize(&ratio_length, data);
        SerializeHelpers::serialize(_ratios.data(), data, ratio_length);
        uint32_t num_groups = _group2sum.size();
        SerializeHelpers::serialize(&num_groups, data);
        for (auto&& [group, value] : _group2sum) {
            uint32_t group_name_size = group.length();
            SerializeHelpers::serialize(&group_name_size, data);
            SerializeHelpers::serialize(group.data(), data, group_name_size);
            SerializeHelpers::serialize(&value, data);
        }
    }

    void deserialize(const uint8_t*& data) {
        uint32_t ratio_length = 0;
        SerializeHelpers::deserialize(data, &ratio_length);
        _ratios.resize(ratio_length);
        SerializeHelpers::deserialize(data, _ratios.data(), ratio_length);
        uint32_t num_groups = 0;
        SerializeHelpers::deserialize(data, &num_groups);
        _group2sum.reserve(num_groups);
        for (uint32_t i = 0; i < num_groups; ++i) {
            uint32_t group_name_size = 0;
            SerializeHelpers::deserialize(data, &group_name_size);
            std::string group(group_name_size, 0);
            SerializeHelpers::deserialize(data, group.data(), group_name_size);
            double value;
            SerializeHelpers::deserialize(data, &value);
            _group2sum[group] = value;
        }
        _is_init = true;
    }

    size_t serialized_size() const {
        size_t size = sizeof(uint32_t) + sizeof(double) * _ratios.size() + sizeof(uint32_t);
        for (auto&& [group, value] : _group2sum) {
            size += sizeof(uint32_t) + group.length() + sizeof(double);
        }
        return size;
    }

    std::string get_result() const {
        if (_group2sum.size() != _ratios.size()) {
            return fmt::format("Logical Error: the number of groups({}) must equal to the number of ratios({}).",
                               _group2sum.size(), _ratios.size());
        }
        if (_group2sum.empty()) {
            return "Logical Error: empty table";
        }
        std::vector<double> f_obs, f_exp;
        double f_obs_sum = 0;
        for (const auto& [group, ob] : _group2sum) {
            f_obs.emplace_back(ob);
            f_obs_sum += ob;
        }
        double ratio_sum = std::reduce(_ratios.begin(), _ratios.end(), 0.);
        if (fabs(ratio_sum) <= 1e-6) {
            return fmt::format("sum of ratio({}) must not equal to zero!", ratio_sum);
        }
        for (auto& ratio : _ratios) {
            double exp = ratio / ratio_sum * f_obs_sum;
            if (exp <= 1e-6) {
                return fmt::format("f_exp({}) should not contain zeros or negative.", exp);
            }
            f_exp.emplace_back(exp);
        }
        double p_value = 0;
        double chisquare = 0;
        for (size_t i = 0; i < f_obs.size(); i++) chisquare += (f_obs[i] - f_exp[i]) * (f_obs[i] - f_exp[i]) / f_exp[i];
        if (chisquare <= 1e-6) {
            return fmt::format("chisquare({}) should not equal to zero!", chisquare);
        }
        double dof = f_obs.size() - 0 - 1;
        p_value = 1 - boost::math::cdf(boost::math::chi_squared{dof}, chisquare);
        std::stringstream result;
        result << "\n"
               << MathHelpers::to_string_with_precision("groupname") << MathHelpers::to_string_with_precision("f_obs")
               << MathHelpers::to_string_with_precision("ratio") << MathHelpers::to_string_with_precision("chisquare")
               << MathHelpers::to_string_with_precision("p-value") << "\n";
        size_t pos = 0;
        for (const auto& [group, ob] : _group2sum) {
            result << MathHelpers::to_string_with_precision(group) << MathHelpers::to_string_with_precision(ob)
                   << MathHelpers::to_string_with_precision(_ratios[pos]);
            if (!pos) {
                result << MathHelpers::to_string_with_precision(chisquare)
                       << MathHelpers::to_string_with_precision(p_value);
            }
            result << "\n";
            pos++;
        }
        return result.str();
    }

private:
    bool _is_init{false};
    std::vector<double> _ratios;
    std::unordered_map<std::string, double> _group2sum;
};

class SRMAggFunction : public AggregateFunctionBatchHelper<SRMAggState, SRMAggFunction> {
public:
    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        Slice group;
        const Column* group_col = columns[1];
        if (!FunctionHelper::get_data_of_column<BinaryColumn>(group_col, row_num, group)) {
            ctx->set_error("Internal Error: fail to get `group`.");
            return;
        }
        double value = 0;
        const Column* value_col = columns[0];
        if (!FunctionHelper::get_data_of_column<DoubleColumn>(value_col, row_num, value)) {
            ctx->set_error("Internal Error: fail to get `value`.");
            return;
        }
        if (this->data(state).is_uninitialized()) {
            DCHECK(row_num == 0);
            auto ratios_col = columns[2];
            auto [input, size] = FunctionHelper::get_data_of_array<DoubleColumn, double>(ratios_col, 0);
            if (input == nullptr) {
                ctx->set_error("Internal Error: fail to get `ratios`.");
                return;
            }
            std::vector<double> ratios(input, input + size);
            this->data(state).init(std::move(ratios));
        }
        this->data(state).update(group.to_string(), value);
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
        SRMAggState other(serialized_data);
        if (!this->data(state).is_ratio_same(other)) {
            ctx->set_error("Logical Error: states are of different step.");
            return;
        }
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
        DCHECK(!this->data(state).is_uninitialized());
        if (to->is_nullable()) {
            auto* dst_nullable_col = down_cast<NullableColumn*>(to);
            dst_nullable_col->null_column_data().emplace_back(false);
            to = dst_nullable_col->data_column().get();
        }
        down_cast<BinaryColumn&>(*to).append(this->data(state).get_result());
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {
        ctx->set_error("Logical Error: `convert_to_serialize_format` not supported.");
    }

    std::string get_name() const override { return std::string(AllInSqlFunctions::srm); }
};

} // namespace starrocks
