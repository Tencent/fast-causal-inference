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
#include <velocypack/Value.h>

#include <algorithm>
#include <any>
#include <cstdint>
#include <ctime>
#include <string>

#include "agent/master_info.h"
#include "column/json_column.h"
#include "column/type_traits.h"
#include "column/vectorized_fwd.h"
#include "common/compiler_util.h"
#include "exprs/agg/aggregate.h"
#include "exprs/all_in_sql_functions.h"
#include "exprs/function_context.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/serialize_helpers.hpp"

namespace starrocks {

class DistributedNodeRowNumberAggregateState {
public:
    DistributedNodeRowNumberAggregateState() = default;
    DistributedNodeRowNumberAggregateState(const uint8_t* data) { deserialize(data); }

    bool is_uninitialized() const { return !_is_init; }

    void init(int64_t node_key, int64_t seed) {
        _node_key = node_key;
        _seed = seed;
        _stats = std::unordered_map<int64_t, size_t>{};
        _is_init = true;
    }

    void update() { _stats[_node_key] += 1; }

    void merge(DistributedNodeRowNumberAggregateState const& other) {
        for (auto&& [key, value] : other._stats) {
            _stats[key] += value;
        }
    }

    void serialize(uint8_t*& data) const {
        DCHECK(_is_init);
        SerializeHelpers::serialize(&_seed, data);
        size_t num_nodes = _stats.size();
        SerializeHelpers::serialize(&num_nodes, data);
        for (auto [key, value] : _stats) {
            SerializeHelpers::serialize(&key, data);
            SerializeHelpers::serialize(&value, data);
        }
    }

    void deserialize(const uint8_t*& data) {
        SerializeHelpers::deserialize(data, &_seed);
        size_t num_nodes = 0;
        SerializeHelpers::deserialize(data, &num_nodes);
        _stats.clear();
        _stats.reserve(num_nodes);
        for (size_t i = 0; i < num_nodes; ++i) {
            int64_t key;
            SerializeHelpers::deserialize(data, &key);
            size_t value;
            SerializeHelpers::deserialize(data, &value);
            _stats[key] = value;
        }
        _is_init = true;
    }

    size_t serialized_size() const {
        return sizeof(_seed) + sizeof(size_t) + _stats.size() * (sizeof(int64_t) + sizeof(size_t));
    }

    void build_result_json(vpack::Builder& builder) const {
        DCHECK(_is_init);
        vpack::ObjectBuilder obj_builder(&builder);
        builder.add("random_seed", vpack::Value(_seed));
        for (auto [key, value] : _stats) {
            builder.add(std::to_string(key), vpack::Value(value));
        }
    }

    size_t seed() const { return _seed; }

private:
    bool _is_init{false};
    int64_t _node_key{-1};
    size_t _seed{0};
    std::unordered_map<int64_t, size_t> _stats;
};

class DistributedNodeRowNumberAggregateFunction
        : public AggregateFunctionBatchHelper<DistributedNodeRowNumberAggregateState,
                                              DistributedNodeRowNumberAggregateFunction> {
public:
    void reset(FunctionContext* ctx, const Columns& args, AggDataPtr state) const override {}

    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        if (this->data(state).is_uninitialized()) {
            int64_t node_key = get_backend_id().value_or(-1);
            if (UNLIKELY(node_key == -1)) {
                ctx->set_error("Internal Error: fail to get be id.");
                return;
            }
            size_t seed;
            if (!FunctionHelper::get_data_of_column<RunTimeColumnType<TYPE_INT>, size_t>(columns[0], 0, seed)) {
                ctx->set_error("Internal Error: fail to get `seed`.");
                return;
            }
            this->data(state).init(node_key, seed);
        }
        this->data(state).update();
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
        DistributedNodeRowNumberAggregateState other(serialized_data);
        if (this->data(state).seed() != other.seed()) {
            ctx->set_error("Logical Error: states are of different seeds.");
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
                                     ColumnPtr* dst) const override {}

    std::string get_name() const override { return std::string(AllInSqlFunctions::distributed_node_row_number); }
};

} // namespace starrocks
