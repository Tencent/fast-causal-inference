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

#include "column/array_column.h"
#include "column/column_builder.h"
#include "column/column_helper.h"
#include "column/column_viewer.h"
#include "column/vectorized_fwd.h"
#include "common/status.h"
#include "exprs/agg/causal_forest.h"
#include "exprs/eval_ml_method.h"
#include "exprs/function_helper.h"
#include "exprs/jsonpath.h"
#include "types/logical_type.h"
#include "util/json.h"

namespace starrocks {

class CausalForestEvalImpl : public EvalMLMethodImplBase {
public:
    Status load(const JsonValue& model) override;

    StatusOr<ColumnPtr> eval(const Columns& columns) const override;

private:
    ForestTrainer _trainer;
    size_t _num_variables;

    constexpr static std::size_t OUTCOME = 0;
    constexpr static std::size_t TREATMENT = 1;
    constexpr static std::size_t INSTRUMENT = 2;
    constexpr static std::size_t OUTCOME_INSTRUMENT = 3;
    constexpr static std::size_t TREATMENT_INSTRUMENT = 4;
    constexpr static std::size_t INSTRUMENT_INSTRUMENT = 5;
    constexpr static std::size_t WEIGHT = 6;
    constexpr static std::size_t NUM_TYPES = 7;
};

Status CausalForestEvalImpl::load(const JsonValue& model) {
    JsonValue context;
    {
        ASSIGN_OR_RETURN(JsonPath model_name_json_path, JsonPath::parse("$.context"));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&model, model_name_json_path, &builder);
        context = JsonValue(slice);
        if (context.get_type() != JsonType::JSON_OBJECT) {
            return Status::InvalidArgument("Context is not an object.");
        }
    }
    {
        ASSIGN_OR_RETURN(JsonPath model_name_json_path, JsonPath::parse("$.num_variables"));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&model, model_name_json_path, &builder);
        JsonValue value(slice);
        ASSIGN_OR_RETURN(_num_variables, value.get_uint());
    }
    RETURN_IF_ERROR(_trainer.init(context, _num_variables, true));
    return Status::OK();
}

StatusOr<ColumnPtr> CausalForestEvalImpl::eval(const Columns& columns) const {
    size_t num_rows = columns[1]->size();
    if (num_rows == 0) {
        return ColumnBuilder<TYPE_DOUBLE>(0).build(false);
    }

    ColumnBuilder<TYPE_DOUBLE> result_builder(num_rows);

    for (size_t i = 0; i < num_rows; ++i) {
        auto input_opt = FunctionHelper::get_data_of_array(columns[1].get(), i);
        if (!input_opt) {
            return Status::InvalidArgument("Fail to parse input data.");
        }
        std::vector<double> input;
        input.reserve(_num_variables);
        for (const auto& datum : input_opt.value()) {
            if (datum.is_null()) {
                return Status::InvalidArgument("Input data contains null value.");
            }
            input.emplace_back(datum.get_double());
        }
        size_t length = input_opt->size();
        if (length != _num_variables) {
            return Status::InvalidArgument(fmt::format("Number of input data ({}) is not equal to num_variables({}).",
                                                       length, _num_variables));
        }
        std::vector<double> average(7, 0);
        RETURN_IF_ERROR(_trainer.predict(input, average));
        for (auto& avg : average) {
            avg /= _trainer.getNumTrees();
        }
        double instrument_effect_numerator =
                average[OUTCOME_INSTRUMENT] * average[WEIGHT] - average[OUTCOME] * average[INSTRUMENT];
        double first_stage_numerator =
                average[TREATMENT_INSTRUMENT] * average[WEIGHT] - average[TREATMENT] * average[INSTRUMENT];
        result_builder.append(instrument_effect_numerator / first_stage_numerator);
    }
    return result_builder.build(false);
}

} // namespace starrocks
