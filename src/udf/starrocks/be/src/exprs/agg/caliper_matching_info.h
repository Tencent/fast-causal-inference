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
#include <any>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <cstdint>
#include <ctime>
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

class CaliperMatchingInfoStats {
public:
    using TupleHash = MathHelpers::TupleHash;

    void add(bool treatment, int64_t node_key, int64_t score, size_t group_hash) {
        _all_scores.emplace(score);
        _matching_info[{node_key, score, group_hash}][treatment] += 1;
    }

    // (score, group_hash) -> (count_treatment0, count_treatment1)
    std::unordered_map<std::tuple<int64_t, size_t>, std::array<size_t, 2>, TupleHash> get_total_info() const {
        std::unordered_map<std::tuple<int64_t, size_t>, std::array<size_t, 2>, TupleHash> total_info;
        for (auto&& [key, counts] : _matching_info) {
            auto [_, score, group_hash] = key;
            total_info[{score, group_hash}][0] += counts[0];
            total_info[{score, group_hash}][1] += counts[1];
        }
        return total_info;
    }

    void merge(CaliperMatchingInfoStats const& other) {
        for (auto&& [key, counts] : other._matching_info) {
            _matching_info[key][0] += counts[0];
            _matching_info[key][1] += counts[1];
            _all_scores.emplace(std::get<1>(key));
        }
    }

    void serialize(uint8_t*& data) const {
        uint32_t size = _matching_info.size();
        SerializeHelpers::serialize(&size, data);
        for (auto [key, value] : _matching_info) {
            auto&& [node_key, score, group_hash] = key;
            SerializeHelpers::serialize(&node_key, data);
            SerializeHelpers::serialize(&score, data);
            SerializeHelpers::serialize(&group_hash, data);
            SerializeHelpers::serialize(&value[0], data);
            SerializeHelpers::serialize(&value[1], data);
        }
    }

    void deserialize(const uint8_t*& data) {
        uint32_t size;
        SerializeHelpers::deserialize(data, &size);
        for (uint32_t i = 0; i < size; ++i) {
            std::tuple<int64_t, int64_t, size_t> key;
            auto& [node_key, score, group_hash] = key;
            std::array<size_t, 2> value;
            SerializeHelpers::deserialize(data, &node_key);
            SerializeHelpers::deserialize(data, &score);
            SerializeHelpers::deserialize(data, &group_hash);
            SerializeHelpers::deserialize(data, &value[0]);
            SerializeHelpers::deserialize(data, &value[1]);
            _all_scores.emplace(score);
            _matching_info[key] = value;
        }
    }

    size_t serialized_size() const {
        return sizeof(uint32_t) +
               _matching_info.size() * (sizeof(int64_t) + sizeof(int64_t) + sizeof(size_t) + sizeof(size_t) * 2);
    }

    int num_scores() const { return _all_scores.size(); }

    std::unordered_map<std::tuple<int64_t, int64_t, size_t>, std::array<size_t, 2>, TupleHash> const& get_info() const {
        return _matching_info;
    }

    void to_json(vpack::Builder& builder) const {
        auto total_info = get_total_info();
        std::unordered_map<std::tuple<int64_t, size_t>, size_t, TupleHash> end_indices;
        std::unordered_map<std::tuple<int64_t, size_t>, std::array<size_t, 2>, TupleHash> remaining_match_counts;
        size_t current_idx = 1;
        for (auto& [key, cnt] : total_info) {
            size_t match = std::min(cnt[0], cnt[1]);
            remaining_match_counts[key] = {match, match};
            current_idx += match;
            end_indices[key] = current_idx;
        }

        std::unordered_map<int64_t, std::unordered_map<int64_t, std::unordered_map<size_t, std::array<size_t, 4>>>>
                result_matching_info;

        for (auto [info_key, count] : _matching_info) {
            auto [node_key, score, group_hash] = info_key;
            auto end_index = end_indices[{score, group_hash}];
            auto& [remaining_match_0, remaining_match_1] = remaining_match_counts[{score, group_hash}];

            auto this_match_0 = std::min(remaining_match_0, count[0]);
            auto this_match_1 = std::min(remaining_match_1, count[1]);

            auto begin_0 = end_index - this_match_0;
            auto begin_1 = end_index - this_match_1;

            result_matching_info[node_key][score][group_hash] = {begin_0, this_match_0, begin_1, this_match_1};

            remaining_match_0 -= this_match_0;
            remaining_match_1 -= this_match_1;
        }

        vpack::ObjectBuilder obj_builder(&builder);
        for (auto&& [node_key, tmp_mp1] : result_matching_info) {
            vpack::ObjectBuilder obj_builder_node_key(&builder, std::to_string(node_key));
            for (auto&& [score, tmp_mp2] : tmp_mp1) {
                vpack::ObjectBuilder obj_builder_score(&builder, std::to_string(score));
                for (auto&& [group_hash, value] : tmp_mp2) {
                    vpack::ArrayBuilder array_builder(&builder, std::to_string(group_hash));
                    for (uint32_t i = 0; i < 4; ++i) {
                        builder.add(vpack::Value(value[i]));
                    }
                }
            }
        }
    }

private:
    // (node_key, score, group_hash) -> (cnt_treatment0, cnt_treatment1)
    std::unordered_map<std::tuple<int64_t, int64_t, size_t>, std::array<size_t, 2>, TupleHash> _matching_info;
    std::unordered_set<int64_t> _all_scores;
};

class CaliperMatchingInfoAggState {
public:
    CaliperMatchingInfoAggState() = default;
    CaliperMatchingInfoAggState(const uint8_t* data) { deserialize(data); }

    bool is_uninitialized() const { return !_is_init; }

    bool is_step_same(CaliperMatchingInfoAggState const& other) const { return _step == other._step; }

    void init(int64_t node_key, double step) {
        _node_key = node_key;
        _step = step;
        _is_init = true;
    }

    void update(bool treatment, double distance, size_t group_hash) {
        auto score = static_cast<int64_t>(distance / _step);
        _stats.add(treatment, _node_key, score, group_hash);
    }

    void merge(CaliperMatchingInfoAggState const& other) { _stats.merge(other._stats); }

    void serialize(uint8_t*& data) const {
        DCHECK(_is_init);
        SerializeHelpers::serialize(&_step, data);
        _stats.serialize(data);
    }

    void deserialize(const uint8_t*& data) {
        SerializeHelpers::deserialize(data, &_step);
        _stats.deserialize(data);
        _is_init = true;
    }

    size_t serialized_size() const { return sizeof(_step) + _stats.serialized_size(); }

    void build_result_json(vpack::Builder& builder) const {
        DCHECK(_is_init);
        _stats.to_json(builder);
    }

    size_t num_scores() const { return _stats.num_scores(); }

private:
    bool _is_init{false};
    int64_t _node_key{-1};
    double _step{0};
    CaliperMatchingInfoStats _stats;
};

class CaliperMatchingInfoAggFunction
        : public AggregateFunctionBatchHelper<CaliperMatchingInfoAggState, CaliperMatchingInfoAggFunction> {
public:
    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        bool treatment = false;
        const Column* treatment_col = columns[0];
        if (!FunctionHelper::get_data_of_column<BooleanColumn>(treatment_col, row_num, treatment)) {
            ctx->set_error("Internal Error: fail to get `treatment`.");
            return;
        }
        double distance = 0;
        const Column* distance_col = columns[1];
        if (!FunctionHelper::get_data_of_column<DoubleColumn>(distance_col, row_num, distance)) {
            ctx->set_error("Internal Error: fail to get `distance`.");
            return;
        }
        if (this->data(state).is_uninitialized()) {
            DCHECK(row_num == 0);
            int64_t node_key = get_backend_id().value_or(-1);
            if (UNLIKELY(node_key == -1)) {
                ctx->set_error("Internal Error: fail to get be id.");
                return;
            }
            double step = 0;
            const Column* step_col = columns[2];
            if (!FunctionHelper::get_data_of_column<DoubleColumn>(step_col, 0, step)) {
                ctx->set_error("Internal Error: fail to get `step`.");
                return;
            }
            if (step == 0) {
                ctx->set_error("Invalid Argument: step cannot be zero.");
                return;
            }
            this->data(state).init(node_key, step);
        }
        size_t group_hash = 0;
        if (ctx->get_num_args() > 3) {
            const Column* exacts_col = columns[3];
            auto [exacts, length] = FunctionHelper::get_data_of_array<DoubleColumn, double>(exacts_col, row_num);
            for (uint32_t i = 0; i < length; ++i) {
                group_hash ^= std::hash<double>()(exacts[i]);
            }
        }
        this->data(state).update(treatment, distance, group_hash);
        if (this->data(state).num_scores() > 1000) {
            ctx->set_error("Internal Error: number of scores is larger than 1000.");
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
        CaliperMatchingInfoAggState other(serialized_data);
        if (!this->data(state).is_step_same(other)) {
            ctx->set_error("Logical Error: states are of different step.");
            return;
        }
        this->data(state).merge(other);
        if (this->data(state).num_scores() > 1000) {
            ctx->set_error("Internal Error: number of scores is larger than 1000.");
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

    std::string get_name() const override { return std::string(AllInSqlFunctions::caliper_matching_info); }
};

} // namespace starrocks
