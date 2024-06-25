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

#include <boost/numeric/ublas/matrix.hpp>

#include "column/array_column.h"
#include "column/column_builder.h"
#include "column/column_helper.h"
#include "column/column_viewer.h"
#include "column/vectorized_fwd.h"
#include "common/status.h"
#include "exprs/eval_ml_method.h"
#include "exprs/function_helper.h"
#include "exprs/jsonpath.h"
#include "types/logical_type.h"

namespace starrocks {

namespace ublas = boost::numeric::ublas;

class OlsEvalImpl : public EvalMLMethodImplBase {
public:
    Status load(const JsonValue& model) override;

    StatusOr<ColumnPtr> eval(const Columns& columns) const override;

private:
    bool _use_bias{false};
    int _num_variables{false};
    ublas::vector<double> _coef;
};

Status OlsEvalImpl::load(const JsonValue& model) {
    {
        ASSIGN_OR_RETURN(JsonPath model_name_json_path, JsonPath::parse("$.num_variables"));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&model, model_name_json_path, &builder);
        JsonValue value(slice);
        ASSIGN_OR_RETURN(_num_variables, value.get_int());
    }

    {
        ASSIGN_OR_RETURN(JsonPath model_name_json_path, JsonPath::parse("$.use_bias"));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&model, model_name_json_path, &builder);
        JsonValue value(slice);
        ASSIGN_OR_RETURN(_use_bias, value.get_bool());
    }

    std::vector<double> coef_tmp;

    {
        ASSIGN_OR_RETURN(JsonPath model_name_json_path, JsonPath::parse("$.coef"));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&model, model_name_json_path, &builder);
        RETURN_IF(!slice.isArray(), Status::InvalidArgument("Invalid coef array."));
        for (const auto& element : vpack::ArrayIterator(slice)) {
            JsonValue element_value(element);
            ASSIGN_OR_RETURN(auto coef_elem, element_value.get_double());
            coef_tmp.emplace_back(coef_elem);
        }
    }

    _coef = ublas::vector<double>(_num_variables + _use_bias, 0);
    RETURN_IF(coef_tmp.size() != _coef.size(),
              Status::InvalidArgument(fmt::format("Coef length({}) is not equal to num_variables({})+use_bias({}).",
                                                  coef_tmp.size(), _num_variables, (int)_use_bias)));
    std::copy(coef_tmp.begin(), coef_tmp.end(), _coef.data().begin());
    return Status::OK();
}

StatusOr<ColumnPtr> OlsEvalImpl::eval(const Columns& columns) const {
    size_t num_rows = columns[1]->size();
    if (num_rows == 0) {
        return ColumnBuilder<TYPE_DOUBLE>(0).build(false);
    }

    auto [_, num_variables] = FunctionHelper::get_data_of_array<DoubleColumn, double>(columns[1].get(), 0);
    if (num_variables != _num_variables) {
        return Status::InvalidArgument(fmt::format("Number of input columns ({}) is not equal to num_variables({}).",
                                                   num_variables, _num_variables));
    }

    ColumnBuilder<TYPE_DOUBLE> result_builder(num_rows);

    for (size_t i = 0; i < num_rows; ++i) {
        auto [input, length] = FunctionHelper::get_data_of_array<DoubleColumn, double>(columns[1].get(), i);
        if (input == nullptr) {
            return Status::InvalidArgument("Fail to parse input columns.");
        }
        double ans = _use_bias ? _coef[length] : 0;
        for (size_t j = 0; j < length; ++j) {
            ans += _coef[j] * input[j];
        }
        result_builder.append(ans);
    }
    return result_builder.build(false);
}

} // namespace starrocks
