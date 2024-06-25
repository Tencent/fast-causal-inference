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

namespace starrocks {

class EvalMLMethodImplBase {
public:
    virtual ~EvalMLMethodImplBase() = default;

    virtual Status load(JsonValue const& model) = 0;

    virtual StatusOr<ColumnPtr> eval(Columns const& columns) const = 0;
};

class EvalMLMethod {
public:
    // eval ml method
    static Status eval_prepare(FunctionContext* context, FunctionContext::FunctionStateScope scope);

    static Status eval_close(FunctionContext* context, FunctionContext::FunctionStateScope scope);

    static StatusOr<EvalMLMethodImplBase*> create_eval_ml_method_impl(JsonValue const& model_col);

    /**
     * eval machine learning method interface
     *
     * @param: [model, input_vector]
     * @paramType: [JsonColumn, ArrayColumn]
     * @return: DoubleColumn
     */
    DEFINE_VECTORIZED_FN(eval_ml_method);
};

} // namespace starrocks
