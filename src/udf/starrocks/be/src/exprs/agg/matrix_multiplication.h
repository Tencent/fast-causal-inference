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

#include <velocypack/Builder.h>

#include <cctype>
#include <cmath>
#include <ios>
#include <iterator>
#include <limits>
#include <sstream>

#include "column/const_column.h"
#include "column/vectorized_fwd.h"
#include "delta_method.h"
#include "exprs/agg/aggregate.h"
#include "exprs/agg/ttest_common.h"
#include "exprs/function_context.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/math_helpers.hpp"
#include "exprs/math_functions.h"
#include "gutil/casts.h"
#include "types/logical_type.h"
#include "util/json.h"

namespace starrocks {

class MatrixMultiplicationParams {
public:
    void reset() { _num_variables = -1; }

    bool operator==(MatrixMultiplicationParams const& other) const {
        return _num_variables == other._num_variables && _use_weights == other._use_weights &&
               _ret_reverse == other._ret_reverse;
    }

    void init(int num_variables, bool ret_reverse, bool use_weight) {
        _num_variables = num_variables;
        _ret_reverse = ret_reverse;
        _use_weights = use_weight;
    }

    bool is_uninitialized() const { return _num_variables == -1; }

    void serialize(uint8_t*& data) const {
        SerializeHelpers::serialize(&_num_variables, data);
        if (is_uninitialized()) {
            return;
        }
        uint8_t use_weight = _use_weights;
        SerializeHelpers::serialize(&use_weight, data);
        uint8_t ret_reverse = _ret_reverse;
        SerializeHelpers::serialize(&ret_reverse, data);
    }

    void deserialize(const uint8_t*& data) {
        SerializeHelpers::deserialize(data, &_num_variables);
        if (is_uninitialized()) {
            return;
        }
        uint8_t use_weight = false;
        SerializeHelpers::deserialize(data, &use_weight);
        _use_weights = use_weight;
        uint8_t ret_reverse = false;
        SerializeHelpers::deserialize(data, &ret_reverse);
        _ret_reverse = ret_reverse;
    }

    size_t serialized_size() const {
        if (is_uninitialized()) {
            return sizeof(_num_variables);
        }
        return sizeof(_num_variables) + sizeof(uint8_t) + sizeof(uint8_t);
    }

    int num_variables() const { return _num_variables; }

    bool use_weights() const { return _use_weights; }

    bool ret_reverse() const { return _ret_reverse; }

private:
    int _num_variables{-1};
    bool _use_weights{false};
    bool _ret_reverse{false};
};

class MatrixMultiplicationState {
public:
    MatrixMultiplicationState() = default;
    MatrixMultiplicationState(const uint8_t* data) { deserialize(data); }

    bool is_uninitialized() const { return _params.is_uninitialized(); }

    void reset() {
        _params.reset();
        _stats.reset();
    }

    void init(int num_variables, bool ret_reverse, bool use_weight) {
        CHECK(num_variables > 0);
        _params.init(num_variables, ret_reverse, use_weight);
        _stats.init(num_variables);
    }

    void update(const double* input, size_t length) {
        CHECK(length == _params.num_variables());
        _stats.update(input, length);
    }

    void merge(MatrixMultiplicationState const& other) {
        if (other.is_uninitialized()) {
            reset();
            return;
        }
        CHECK(_params == other._params);
        _stats.merge(other._stats);
    }

    void serialize(uint8_t*& data) const {
        _params.serialize(data);
        if (_params.is_uninitialized()) {
            return;
        }
        CHECK(!_stats.is_uninitialized());
        _stats.serialize(data);
    }

    void deserialize(const uint8_t*& data) {
        _params.deserialize(data);
        if (_params.is_uninitialized()) {
            return;
        }
        _stats.init(_params.num_variables());
        _stats.deserialize(data);
    }

    size_t serialized_size() const {
        size_t size = _params.serialized_size();
        if (_params.is_uninitialized()) {
            return size;
        }
        CHECK(!_stats.is_uninitialized());
        return size + _stats.serialized_size();
    }

    bool use_weights() const { return _params.use_weights(); }

    void build_result(vpack::Builder& builder) const {
        ublas::matrix<double> XTX = _stats.XTX();
        if (_params.ret_reverse()) {
            ublas::matrix<double> XTX_inv(_params.num_variables(), _params.num_variables());
            if (!MathHelpers::invert_matrix(XTX, XTX_inv)) {
                builder.add("Error", vpack::Value("XTX is invertible."));
                return;
            }
            write_matrix(XTX_inv, builder);
        } else {
            write_matrix(XTX, builder);
        }
    }

private:
    MatrixMultiplicationParams _params;
    DeltaMethodStats _stats;

    static void write_matrix(ublas::matrix<double> const& matrix, vpack::Builder& builder) {
        vpack::ArrayBuilder matrix_builder(&builder);
        for (int i = 0; i < matrix.size1(); ++i) {
            vpack::ArrayBuilder row_builder(&builder);
            for (int j = 0; j < matrix.size2(); ++j) {
                builder.add(vpack::Value(matrix(i, j)));
            }
        }
    }
};

class MatrixMultiplication : public AggregateFunctionBatchHelper<MatrixMultiplicationState, MatrixMultiplication> {
public:
    using DeltaMethodExprColumnType = RunTimeColumnType<TYPE_VARCHAR>;
    using MatrixMulUseWeightColumnType = RunTimeColumnType<TYPE_BOOLEAN>;
    using MatrixMulRetReverseColumnType = RunTimeColumnType<TYPE_BOOLEAN>;
    using MatrixMulResultColumnType = RunTimeColumnType<TYPE_JSON>;

    void reset(FunctionContext* ctx, const Columns& args, AggDataPtr state) const override {
        this->data(state).reset();
    }

    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        const Column* data_col = columns[0];
        auto input_opt = FunctionHelper::get_data_of_array(data_col, row_num);
        if (!input_opt) {
            // ctx->set_error("Internal Error: fail to get data.");
            return;
        }

        size_t array_size = input_opt->size();
        std::vector<double> input;
        input.reserve(array_size);
        for (size_t i = 0; i < array_size; ++i) {
            if ((*input_opt)[i].is_null()) {
                return;
            }
            input.emplace_back((*input_opt)[i].get_double());
        }

        if (this->data(state).is_uninitialized()) {
            bool ret_reverse = false;
            const Column* ret_reverse_col = columns[1];
            if (!FunctionHelper::get_data_of_column<MatrixMulRetReverseColumnType>(ret_reverse_col, 0, ret_reverse)) {
                ctx->set_error("Internal Error: fail to get `ret_reverse`.");
                return;
            }
            bool use_weights = false;
            const Column* use_weight_col = columns[2];
            if (!FunctionHelper::get_data_of_column<MatrixMulRetReverseColumnType>(use_weight_col, 0, use_weights)) {
                ctx->set_error("Internal Error: fail to get `use_weights`.");
                return;
            }
            this->data(state).init(array_size - use_weights, ret_reverse, use_weights);
        }

        bool use_weights = this->data(state).use_weights();

        std::vector<double> X(input.data(), input.data() + array_size - use_weights);

        if (use_weights) {
            double weight = input[array_size - 1];
            std::transform(X.begin(), X.end(), X.begin(), [weight](double x) { return x * weight; });
        }

        this->data(state).update(X.data(), array_size - use_weights);
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
        MatrixMultiplicationState other(serialized_data);
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
        if (to->is_nullable()) {
            auto* dst_nullable_col = down_cast<NullableColumn*>(to);
            if (this->data(state).is_uninitialized()) {
                ctx->set_error("Internal Error: state not initialized.");
                return;
            }
            dst_nullable_col->null_column_data().emplace_back(false);
            to = dst_nullable_col->data_column().get();
        }
        if (this->data(state).is_uninitialized()) {
            ctx->set_error("Internal Error: state not initialized.");
            return;
        }
        vpack::Builder result_builder;
        this->data(state).build_result(result_builder);
        auto slice = result_builder.slice();
        JsonValue result_json(slice);
        down_cast<MatrixMulResultColumnType*>(to)->append(std::move(result_json));
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {}

    std::string get_name() const override { return std::string(AllInSqlFunctions::matrix_multiplication); }
};

} // namespace starrocks
