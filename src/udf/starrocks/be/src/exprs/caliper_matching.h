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

#include "common/status.h"
#include "exprs/all_in_sql_functions.h"
#include "exprs/function_context.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/math_helpers.hpp"
#include "util/json.h"

namespace starrocks {

class CaliperMatchingFunction {
public:
    // eval ml method
    static Status prepare(FunctionContext* context, FunctionContext::FunctionStateScope scope);

    static Status close(FunctionContext* context, FunctionContext::FunctionStateScope scope);

    class CaliperMatchingFunctionState {
    public:
        using TupleHash = MathHelpers::TupleHash;

        CaliperMatchingFunctionState() = default;

        Status init_once(JsonValue const& stats_json);

        int64_t get_index(bool treatment, int64_t score, size_t group_hash);

    private:
        bool _is_init{false};
        std::mutex _mtx;
        std::unordered_map<std::tuple<int64_t, size_t>, std::array<size_t, 4>, TupleHash> _info;
    };

    /**
     * do caliper matching
     *
     * @param: [stats_json, treatment, distance, step, exacts]
     * @paramType: [JsonColumn, BooleanColumn, DoubleColumn, DoubleColumn, ArrayColumn]
     * @return: BigintColumn
     */
    DEFINE_VECTORIZED_FN(caliper_matching);
};

} // namespace starrocks
