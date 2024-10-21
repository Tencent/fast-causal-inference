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

#include "eval_ml_method.h"

#include <velocypack/Parser.h>

#include "column/column_builder.h"
#include "column/column_viewer.h"
#include "column/vectorized_fwd.h"
#include "common/status.h"
#include "exprs/all_in_sql_functions.h"
#include "exprs/jsonpath.h"
#include "exprs/ml_eval_function/causal_forest_eval.h"
#include "exprs/ml_eval_function/ols_eval.h"
#include "types/logical_type.h"

namespace starrocks {

Status EvalMLMethod::eval_prepare(FunctionContext* context, FunctionContext::FunctionStateScope scope) {
    return Status::OK();
}

Status EvalMLMethod::eval_close(FunctionContext* context, FunctionContext::FunctionStateScope scope) {
    delete reinterpret_cast<EvalMLMethodImplBase*>(context->get_function_state(scope));
    return Status::OK();
}

StatusOr<ColumnPtr> EvalMLMethod::eval_ml_method(FunctionContext* context, const Columns& columns) {
    auto state = context->get_function_state(FunctionContext::THREAD_LOCAL);
    if (state == nullptr) {
        ColumnViewer<TYPE_JSON> viewer(columns[0]);
        JsonValue const& model = *viewer.value(0);

        // state will be deleted in eval_close, don't worry about it.
        // todo: use mempool
        ASSIGN_OR_RETURN(state, create_eval_ml_method_impl(model));
        context->set_function_state(FunctionContext::THREAD_LOCAL, state);
    }

    RETURN_IF(state == nullptr, Status::InternalError("eval_ml_method: failed to get function state"));
    auto* impl = reinterpret_cast<EvalMLMethodImplBase*>(state);
    return impl->eval(columns);
}

StatusOr<EvalMLMethodImplBase*> EvalMLMethod::create_eval_ml_method_impl(JsonValue const& model) {
    ASSIGN_OR_RETURN(JsonPath model_name_json_path, JsonPath::parse("$.name"));
    vpack::Builder builder;
    vpack::Slice slice = JsonPath::extract(&model, model_name_json_path, &builder);
    JsonValue value(slice);
    ASSIGN_OR_RETURN(Slice function_name, value.get_string());

    EvalMLMethodImplBase* impl = nullptr;
    if (function_name == AllInSqlFunctions::ols) {
        impl = new OlsEvalImpl;
    } else if (function_name == AllInSqlFunctions::causal_forest) {
        impl = new CausalForestEvalImpl;
    } else {
        return Status::InvalidArgument("unsupported model name.");
    }

    RETURN_IF_ERROR(impl->load(model));
    return impl;
}

} // namespace starrocks
