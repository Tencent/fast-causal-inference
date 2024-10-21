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

#include "exprs/caliper_matching.h"

#include <velocypack/Iterator.h>

#include <mutex>

#include "agent/master_info.h"
#include "column/column_builder.h"
#include "column/column_viewer.h"
#include "common/status.h"
#include "exprs/all_in_sql_functions.h"
#include "exprs/function_context.h"
#include "exprs/function_helper.h"
#include "types/logical_type.h"
#include "util/slice.h"

namespace starrocks {

Status CaliperMatchingFunction::prepare(FunctionContext* context, FunctionContext::FunctionStateScope scope) {
    if (scope == FunctionContext::FunctionStateScope::THREAD_LOCAL) {
        return Status::OK();
    }
    auto* state = new CaliperMatchingFunctionState;
    context->set_function_state(scope, state);
    return Status::OK();
}

Status CaliperMatchingFunction::close(FunctionContext* context, FunctionContext::FunctionStateScope scope) {
    if (scope == FunctionContext::FunctionStateScope::THREAD_LOCAL) {
        return Status::OK();
    }
    auto* state = reinterpret_cast<CaliperMatchingFunctionState*>(context->get_function_state(scope));
    delete state;
    return {};
}

StatusOr<ColumnPtr> CaliperMatchingFunction::caliper_matching(FunctionContext* context, const Columns& columns) {
    if (columns.empty() || columns[0]->size() == 0) {
        return ColumnBuilder<TYPE_BIGINT>(0).build(false);
    }

    auto* state = reinterpret_cast<CaliperMatchingFunctionState*>(
            context->get_function_state(FunctionContext::FunctionStateScope::FRAGMENT_LOCAL));

    ColumnViewer<TYPE_JSON> json_viewer(columns[0]);
    RETURN_IF_ERROR(state->init_once(*json_viewer.value(0)));

    size_t num_rows = columns[0]->size();
    ColumnBuilder<TYPE_BIGINT> result_builder(num_rows);

    ColumnViewer<TYPE_BOOLEAN> treatment_viewer(columns[1]);
    ColumnViewer<TYPE_DOUBLE> distance_viewer(columns[2]);
    ColumnViewer<TYPE_DOUBLE> step_viewer(columns[3]);
    for (size_t row = 0; row < num_rows; ++row) {
        size_t group_hash = 0;
        if (columns.size() > 4) {
            auto exacts = FunctionHelper::get_data_of_array(columns[4].get(), row);
            if (!exacts) {
                return Status::InvalidArgument("exacts cannot be null.");
            }
            LOG(INFO) << "exacts: " << exacts->size();
            for (const auto& i : exacts.value()) {
                if (i.is_null()) {
                    return Status::InvalidArgument("exacts cannot be null.");
                }
                group_hash ^= std::hash<std::string>()(i.get_slice().to_string());
            }
        }
        auto treatment = treatment_viewer.value(row);
        auto distance = distance_viewer.value(row);
        auto step = step_viewer.value(row);
        RETURN_IF(step == 0, Status::InvalidArgument("step cannot be zero."));
        auto score = static_cast<int64_t>(distance / step);
        auto index = state->get_index(treatment, score, group_hash);
        result_builder.append(index);
    }
    return result_builder.build(false);
}

Status CaliperMatchingFunction::CaliperMatchingFunctionState::init_once(JsonValue const& stats_json) {
    std::scoped_lock lock(_mtx);
    if (_is_init) {
        return Status::OK();
    }
    _is_init = true;

    int64_t this_node_key = get_backend_id().value_or(-1);
    if (this_node_key == -1) {
        return Status::InternalError("Invalid backend id.");
    }

    if (stats_json.to_vslice().type() != vpack::ValueType::Object) {
        return Status::InvalidArgument(fmt::format("Invalid json object."));
    }
    for (auto&& [node_key_slice, json_0] : vpack::ObjectIterator(stats_json.to_vslice())) {
        auto node_key_json = JsonValue(node_key_slice);
        ASSIGN_OR_RETURN(auto node_key_str, node_key_json.get_string());
        auto node_key = std::stoll(node_key_str.get_data());

        if (node_key != this_node_key) {
            continue;
        }

        if (json_0.type() != vpack::ValueType::Object) {
            return Status::InvalidArgument(fmt::format("Invalid json object."));
        }
        for (auto&& [score_slice, json_1] : vpack::ObjectIterator(json_0)) {
            auto score_json = JsonValue(score_slice);
            ASSIGN_OR_RETURN(auto score_str, score_json.get_string());
            auto score = std::stoll(score_str.get_data());

            if (json_1.type() != vpack::ValueType::Object) {
                return Status::InvalidArgument(fmt::format("Invalid json object."));
            }
            for (auto&& [group_hash_slice, stats_array_slice] : vpack::ObjectIterator(json_1)) {
                auto group_hash_json = JsonValue(group_hash_slice);
                ASSIGN_OR_RETURN(auto group_hash_str, group_hash_json.get_string());
                auto group_hash = std::stoull(group_hash_str.get_data());

                if (!stats_array_slice.isArray()) {
                    return Status::InvalidArgument(fmt::format("Invalid json object."));
                }
                std::vector<size_t> stats;
                for (auto value_slice : vpack::ArrayIterator(stats_array_slice)) {
                    auto value_json = JsonValue(value_slice);
                    ASSIGN_OR_RETURN(auto value, value_json.get_uint());
                    stats.emplace_back(value);
                }
                if (stats.size() != 4) {
                    return Status::InvalidArgument(fmt::format("Invalid json object."));
                }
                _info[{score, group_hash}] = std::array<size_t, 4>{stats[0], stats[1], stats[2], stats[3]};
            }
        }
    }
    return Status::OK();
}

int64_t CaliperMatchingFunction::CaliperMatchingFunctionState::get_index(bool treatment, int64_t score,
                                                                         size_t group_hash) {
    std::scoped_lock lock(_mtx);
    if (!_info.count({score, group_hash})) {
        return 0;
    }
    auto& stats = _info[{score, group_hash}];
    auto& begin = stats[treatment ? 2 : 0];
    auto& count = stats[treatment ? 3 : 1];
    if (count == 0) {
        return 0;
    }
    count--;
    return static_cast<int64_t>(begin + count) * (treatment ? -1 : 1);
}

} // namespace starrocks
