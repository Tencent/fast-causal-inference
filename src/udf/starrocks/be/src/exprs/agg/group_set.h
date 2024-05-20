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
#include <type_traits>
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

class GroupSetAggState {
public:
    GroupSetAggState() = default;
    GroupSetAggState(const uint8_t* data) { deserialize(data); }

    bool is_uninitialized() const { return _num_columns == -1; }

    void init(int32_t num_columns, std::optional<std::vector<std::string>> column_names) {
        DCHECK(num_columns > 0);
        _num_columns = num_columns;
        if (column_names.has_value()) {
            _column_names = column_names.value();
        } else {
            for (int i = 1; i <= num_columns; ++i) {
                _column_names.emplace_back(fmt::format("col{}", i));
            }
        }
        for (int32_t treatment = 0; treatment < 2; ++treatment) {
            _stats[treatment].resize(_num_columns);
        }
    }

    bool is_params_same(GroupSetAggState const& other) const {
        return other._num_columns == _num_columns && other._column_names == _column_names;
    }

    void update(double y, bool treatment, std::vector<int32_t> const& group_ids) {
        for (int32_t i = 0; i < _num_columns; ++i) {
            auto group_id = group_ids[i];
            auto& [cnt, sum, sum2] = _stats[treatment][i][group_id];
            cnt += 1;
            sum += y;
            sum2 += y * y;
        }
    }

    void merge(GroupSetAggState const& other) {
        for (int32_t treatment = 0; treatment < 2; ++treatment) {
            for (int32_t column_idx = 0; column_idx < _num_columns; ++column_idx) {
                for (auto [group_id, value_tuple] : other._stats[treatment][column_idx]) {
                    auto [cnt_rhs, sum_rhs, sum2_rhs] = value_tuple;
                    auto& [cnt, sum, sum2] = _stats[treatment][column_idx][group_id];
                    cnt += cnt_rhs;
                    sum += sum_rhs;
                    sum2 += sum2_rhs;
                }
            }
        }
    }

    void serialize(uint8_t*& data) const { SerializeHelpers::serialize_all(data, _num_columns, _column_names, _stats); }

    void deserialize(const uint8_t*& data) {
        SerializeHelpers::deserialize_all(data, _num_columns, _column_names, _stats);
    }

    size_t serialized_size() const {
        return SerializeHelpers::serialized_size_all(_num_columns, _column_names, _stats);
    }

    void build_result_json(vpack::Builder& builder) const {
        vpack::ArrayBuilder result_builder(&builder);
        for (int32_t treatment = 0; treatment < 2; ++treatment) {
            for (int32_t column_idx = 0; column_idx < _num_columns; ++column_idx) {
                for (auto [group_id, value_tuple] : _stats[treatment][column_idx]) {
                    auto [cnt, sum, sum2] = value_tuple;
                    vpack::ArrayBuilder tuple_builder(&builder);
                    builder.add(vpack::Value(_column_names[column_idx]));
                    builder.add(vpack::Value(treatment));
                    builder.add(vpack::Value(group_id));
                    builder.add(vpack::Value(cnt));
                    builder.add(vpack::Value(sum));
                    builder.add(vpack::Value(sum2));
                }
            }
        }
    }

private:
    int32_t _num_columns{-1};
    std::vector<std::string> _column_names;
    // _stats[treatment][column_idx][group_id] -> (cnt, sum, sum2)
    std::array<std::vector<std::unordered_map<int32_t, std::tuple<size_t, double, double>>>, 2> _stats;
};

class GroupSetAggFunction : public AggregateFunctionBatchHelper<GroupSetAggState, GroupSetAggFunction> {
public:
    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        double value = 0;
        const Column* value_col = columns[0];
        if (!FunctionHelper::get_data_of_column<DoubleColumn>(value_col, row_num, value)) {
            ctx->set_error("Internal Error: fail to get `value`.");
            return;
        }
        bool treatment = false;
        const Column* treatment_col = columns[1];
        if (!FunctionHelper::get_data_of_column<BooleanColumn>(treatment_col, row_num, treatment)) {
            ctx->set_error("Internal Error: fail to get `treatment`.");
            return;
        }
        auto groups_col = columns[2];
        auto [group_ptr, group_size] = FunctionHelper::get_data_of_array<Int32Column, int>(groups_col, row_num);
        if (group_ptr == nullptr) {
            ctx->set_error("Internal Error: fail to get `groups`.");
            return;
        }
        if (this->data(state).is_uninitialized()) {
            if (row_num != 0) {
                return;
            }
            std::optional<std::vector<std::string>> column_names_opt;
            if (ctx->get_num_args() > 3) {
                auto column_names_col = columns[3];
                auto [name_ptr, name_size] =
                        FunctionHelper::get_data_of_array<BinaryColumn, Slice>(column_names_col, 0);
                if (name_ptr == nullptr) {
                    ctx->set_error("Internal Error: fail to get `column_names`.");
                    return;
                }
                if (group_size != name_size) {
                    ctx->set_error(
                            fmt::format("num_cols_group({}) is not equal to num_cols_name({}).", group_size, name_size)
                                    .c_str());
                    return;
                }
                std::vector<std::string> column_names;
                for (int i = 0; i < name_size; ++i) {
                    column_names.emplace_back(name_ptr[i].to_string());
                }
                column_names_opt = std::move(column_names);
            }
            this->data(state).init(group_size, std::move(column_names_opt));
        }
        this->data(state).update(value, treatment, std::vector<int32_t>{group_ptr, group_ptr + group_size});
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
        GroupSetAggState other(serialized_data);
        if (!this->data(state).is_params_same(other)) {
            ctx->set_error("Logical Error: states are of different params.");
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
        vpack::Builder result_builder;
        this->data(state).build_result_json(result_builder);
        auto slice = result_builder.slice();
        JsonValue result_json(slice);
        down_cast<JsonColumn&>(*to).append(std::move(result_json));
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {
        ctx->set_error("Logical Error: `convert_to_serialize_format` not supported.");
    }

    std::string get_name() const override { return std::string(AllInSqlFunctions::group_set); }
};

} // namespace starrocks
