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

#include <mutex>
#include <random>
#include <string>

#include "agent/master_info.h"
#include "column/array_column.h"
#include "column/column_helper.h"
#include "column/column_viewer.h"
#include "column/column_visitor.h"
#include "column/column_visitor_adapter.h"
#include "column/json_column.h"
#include "column/nullable_column.h"
#include "column/vectorized_fwd.h"
#include "exprs/expr.h"
#include "exprs/expr_context.h"
#include "exprs/function_helper.h"
#include "exprs/table_function/table_function.h"
#include "runtime/runtime_state.h"
#include "types/logical_type.h"
#include "util/json.h"

namespace starrocks {

class BootStrapSamplingCalculator {
public:
    BootStrapSamplingCalculator() = default;

    void init(size_t total_row_num, size_t rest_row_num, size_t seed) {
        _total_row_num = total_row_num;
        _sample_num = rest_row_num;
        _random_get.seed(seed);
    }

    size_t operator()(size_t row_num) {
        row_num = std::min(row_num, _total_row_num);
        size_t res = std::binomial_distribution<size_t>(_sample_num, 1. * row_num / _total_row_num)(_random_get);
        _sample_num -= res;
        _total_row_num -= row_num;
        return res;
    }

    bool is_clear() const { return _total_row_num == 0 && _sample_num == 0; }

private:
    std::default_random_engine _random_get;
    size_t _total_row_num{0};
    size_t _sample_num{0};
};

class BootStrapGlobalState {
public:
    Status init_once(starrocks::Columns const& columns) {
        std::scoped_lock lock(_mutex);
        if (_is_init) {
            return Status::OK();
        }
        _is_init = true;
        DCHECK(columns.size() >= 3);

        size_t total_row_num = 0, total_row_number_this_node = 0, seed = 0;
        int64_t node_key = get_backend_id().value_or(-1);
        RETURN_IF(node_key == -1, Status::InternalError("Unable to get backend id."));

        const Column* json_column = columns[0].get();
        JsonValue model;
        if (FunctionHelper::can_cast<JsonColumn>(json_column)) {
            model = *ColumnViewer<TYPE_JSON>(columns[0]).value(0);
        } else {
            return Status::InvalidArgument(
                    fmt::format("first col should be JsonColumn, but get {}", columns[0]->get_name()));
        }

        if (model.to_vslice().type() != vpack::ValueType::Object) {
            return Status::InvalidArgument(fmt::format("Invalid json object."));
        }
        std::vector<std::pair<int64_t, size_t>> node_row_counts;
        for (auto&& [key_slice, value_slice] : vpack::ObjectIterator(model.to_vslice())) {
            JsonValue key_json(key_slice);
            ASSIGN_OR_RETURN(auto key, key_json.get_string());
            ASSIGN_OR_RETURN(auto value, JsonValue(value_slice).get_uint());
            LOG(INFO) << fmt::format("bootstrap stats key/value({}/{})", key.to_string(), value);
            if (key == "random_seed") {
                seed = value;
                continue;
            }
            if (value == 0) {
                return Status::InternalError("Logical Error: value is not supposed to be zero.");
            }
            total_row_num += value;
            int64_t node_id = std::stol(key.to_string());
            node_row_counts.emplace_back(node_id, value);
            if (node_id == node_key) {
                total_row_number_this_node = value;
            }
        }
        _seed = seed;

        if (total_row_number_this_node == 0) {
            return Status::InvalidArgument(
                    fmt::format("total_row_number_this_node({}) cannot be zero.", total_row_number_this_node));
        }

        const Column* N_column = columns[1].get();
        size_t N = 0, B = 0;
        if (FunctionHelper::can_cast<Int64Column>(N_column)) {
            ColumnViewer<TYPE_BIGINT> N_viewer(columns[1]);
            N = N_viewer.value(0);
        } else if (FunctionHelper::can_cast<DoubleColumn>(N_column)) {
            ColumnViewer<TYPE_DOUBLE> N_viewer(columns[1]);
            N = N_viewer.value(0) * total_row_num;
        } else {
            return Status::InvalidArgument(fmt::format("second col should be Int64Column or DoubleColumn, but get {}",
                                                       columns[1]->get_name()));
        }

        if (!(N >= 1 && N <= 10'000'000'000)) {
            return Status::InvalidArgument(fmt::format("num_samples({}) should be in [1, 10,000,000,000].", N));
        }

        const Column* B_column = columns[2].get();
        if (!FunctionHelper::can_cast<Int64Column>(B_column)) {
            return Status::InvalidArgument(
                    fmt::format("third col should be Int64Column, but get {}", columns[2]->get_name()));
        }
        ColumnViewer<TYPE_BIGINT> B_viewer(columns[2]);
        B = B_viewer.value(0);

        if (!(B >= 1 && B <= 1000)) {
            return Status::InvalidArgument(fmt::format("num_batches({}) should be in [1, 1,000].", B));
        }
        _num_batches = B;

        LOG(INFO) << fmt::format("bootstrap args N/B({}/{})", N, B);

        BootStrapSamplingCalculator global_sampling_calculators;
        global_sampling_calculators.init(total_row_num, N, seed);
        std::sort(node_row_counts.begin(), node_row_counts.end());
        size_t N_this_node = 0;
        for (auto [key, num_rows_this_node] : node_row_counts) {
            size_t N_each_node = global_sampling_calculators(num_rows_this_node);
            if (key == node_key) {
                N_this_node = N_each_node;
            }
        }

        _sampling_calculators.resize(B);
        for (size_t i = 0; i < B; ++i) {
            _sampling_calculators[i].init(total_row_number_this_node, N_this_node, seed + i + 1);
        }
        return Status::OK();
    }

    size_t calc_reputation(size_t batch_idx, size_t num_rows) {
        std::scoped_lock lock(_mutex);
        return _sampling_calculators[batch_idx](num_rows);
    }

    bool is_clear() const {
        std::scoped_lock lock(_mutex);
        return std::all_of(_sampling_calculators.begin(), _sampling_calculators.end(),
                           [](BootStrapSamplingCalculator const& calculator) { return calculator.is_clear(); });
    }

    size_t num_batches() const { return _num_batches; }

    size_t seed() const { return _seed; }

    void inc_ref() { _count++; }

    int dec_ref() { return --_count; }

    int ref() { return _count; }

private:
    mutable std::mutex _mutex;
    bool _is_init{false};
    std::vector<BootStrapSamplingCalculator> _sampling_calculators;
    size_t _num_batches{0};
    size_t _seed{0};
    std::atomic<int> _count{0};
};

class BootStrapState : public TableFunctionState {
public:
    BootStrapGlobalState* global_state() const { return _state; }

    void set_global_state(BootStrapGlobalState* state) { _state = state; }

private:
    BootStrapGlobalState* _state;
};

/**
 * bootstrap can be used to sample some rows from a table.
 */
class BootStrap final : public TableFunction {
public:
    std::pair<Columns, UInt32Column::Ptr> process(TableFunctionState* state) const override {
        if (state->get_columns().size() < 4) {
            state->set_status(Status::InvalidArgument("boot_strap: num_cols is less than four."));
            return {};
        }

        auto* global_state = reinterpret_cast<BootStrapState*>(state)->global_state();

        starrocks::Columns& columns = state->get_columns();
        auto status = global_state->init_once(columns);
        if (!status.ok()) {
            state->set_status(Status::InvalidArgument(
                    fmt::format("boot_strap: unable to init state, getting {}.", status.to_string())));
            return {};
        }

        size_t row_count = columns[0]->size();
        if (row_count == 0) {
            state->set_status(
                    Status::InvalidArgument(fmt::format("boot_strap: row_count cannot be zero.", status.to_string())));
            return {};
        }
        state->set_processed_rows(row_count);

        std::vector<ColumnPtr> values;
        for (size_t i = 3; i < columns.size(); ++i) {
            Column* column = columns[i].get();
            ColumnPtr ret_column = column->clone_empty();
            values.emplace_back(ret_column);
        }

        auto offsets = UInt32Column::create();
        size_t offset = 0;
        offsets->append(offset);

        size_t num_batches = global_state->num_batches();
        size_t seed = global_state->seed();

        std::vector<uint32_t> count_map(row_count);
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t batch_sample_num = global_state->calc_reputation(batch, row_count);
            if (batch_sample_num <= 10000) {
                std::default_random_engine generator(seed + batch + 1);
                std::uniform_int_distribution<uint32_t> distribution(0, row_count - 1);
                for (size_t i = 0; i < batch_sample_num; ++i) {
                    count_map[distribution(generator)]++;
                }
            } else {
                BootStrapSamplingCalculator row_scheduler;
                row_scheduler.init(row_count, batch_sample_num, seed + batch + 1);
                for (auto& x : count_map) {
                    x = row_scheduler(1);
                }
            }
        }

        for (int row_idx = 0; row_idx < row_count; ++row_idx) {
            size_t reputation = count_map[row_idx];
            if (reputation == 0) {
                continue;
            }
            offsets->append(offset + reputation);
            offset += reputation;

            for (size_t col_idx = 3; col_idx < columns.size(); ++col_idx) {
                ColumnPtr column = columns[col_idx];
                values[col_idx - 3]->append_value_multiple_times(*column, row_idx, reputation, true);
            }
        }

        return {values, offsets};
    }

    Status init(const TFunction& fn, TableFunctionState** state) const override {
        *state = new BootStrapState;
        return Status::OK();
    }

    Status prepare(TableFunctionState* state) const override { return Status::OK(); }

    Status open(RuntimeState* runtime_state, TableFunctionState* state) const override {
        void*& global_state = runtime_state->get_query_level_function_state();
        std::mutex& mtx = runtime_state->get_query_level_function_state_lock();
        std::scoped_lock lock(mtx);
        if (global_state == nullptr) {
            global_state = new BootStrapGlobalState;
            LOG(INFO) << "creating BootStrapGlobalState of query " << runtime_state->query_id();
        }
        reinterpret_cast<BootStrapState*>(state)->set_global_state(
                reinterpret_cast<BootStrapGlobalState*>(global_state));
        reinterpret_cast<BootStrapGlobalState*>(global_state)->inc_ref();
        return Status::OK();
    };

    Status close(RuntimeState* runtime_state, TableFunctionState* state) const override {
        if (reinterpret_cast<BootStrapState*>(state)->global_state()->dec_ref() == 0) {
            void*& global_state = runtime_state->get_query_level_function_state();
            std::mutex& mtx = runtime_state->get_query_level_function_state_lock();
            std::scoped_lock lock(mtx);
            auto* boot_strap_global_state = reinterpret_cast<BootStrapState*>(state)->global_state();
            delete boot_strap_global_state;
            global_state = nullptr;
            LOG(INFO) << "destorying BootStrapGlobalState of query " << runtime_state->query_id();
        }

        delete state;
        return Status::OK();
    }
};

} // namespace starrocks
