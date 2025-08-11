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
#include <velocypack/Value.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

#include "column/type_traits.h"
#include "column/vectorized_fwd.h"
#include "common/compiler_util.h"
#include "exprs/agg/aggregate.h"
#include "exprs/agg/ttest_common.h"
#include "exprs/function_context.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/serialize_helpers.hpp"
#include "gutil/casts.h"
#include "gutil/integral_types.h"
#include "types/logical_type.h"
#include "util/json.h"

namespace starrocks {

class KolmogorovSmirnovAggState {
public:
    KolmogorovSmirnovAggState() = default;
    KolmogorovSmirnovAggState(const uint8_t*& data) { deserialize(data); }

    constexpr static double kTwoSidedPValueScaler = 0.398942280401432677939946059934;

    enum TestMethod : uint8_t {
        AUTO = 0,
        EXACT,
        ASYMPTOTIC,
    };

    void init(TtestAlternative alternative, TestMethod method = AUTO) {
        _alternative = alternative;
        _method = method;
    }

    bool is_uninitialized() const { return _alternative == TtestAlternative::Unknown; }

    void update(double x, bool treatment) {
        if (!(std::isinf(x) || std::isnan(x))) {
            _stats[treatment].emplace_back(x);
        }
    }

    void merge(KolmogorovSmirnovAggState& other) {
        DCHECK(_alternative == other._alternative);
        for (size_t idx = 0; idx < 2; ++idx) {
            if (!std::is_sorted(_stats[idx].begin(), _stats[idx].end())) {
                std::sort(_stats[idx].begin(), _stats[idx].end());
            }
            if (!std::is_sorted(other._stats[idx].begin(), other._stats[idx].end())) {
                std::sort(other._stats[idx].begin(), other._stats[idx].end());
            }
            std::vector<double> tmp;
            tmp.reserve(_stats[idx].size() + other._stats[idx].size());
            std::merge(_stats[idx].begin(), _stats[idx].end(), other._stats[idx].begin(), other._stats[idx].end(),
                       std::back_inserter(tmp));
            _stats[idx] = std::move(tmp);
        }
    }

    void serialize(uint8_t*& data) const {
        DCHECK(!is_uninitialized());
        uint8_t method_int = static_cast<char>(_method);
        SerializeHelpers::serialize_all(data, static_cast<uint8_t>(_alternative), _stats,
                                        static_cast<char>(method_int));
    }

    void deserialize(const uint8_t*& data) {
        uint8_t tmp;
        uint8_t method_int;
        SerializeHelpers::deserialize_all(data, tmp, _stats, method_int);
        _alternative = static_cast<TtestAlternative>(tmp);
        _method = static_cast<TestMethod>(method_int);
    }

    size_t serialized_size() const {
        return SerializeHelpers::serialized_size_all(static_cast<uint8_t>(_alternative), _stats,
                                                     static_cast<uint8_t>(_method));
    }

    class RangeIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = uint64_t;
        using difference_type = std::ptrdiff_t;
        using pointer = uint64_t*;
        using reference = uint64_t&;

        RangeIterator(uint64_t now) : _now(now) {}

        uint64_t operator*() const { return _now; }

        void operator++() { ++_now; }

        bool operator!=(RangeIterator const& other) const { return _now != other._now; }

    private:
        uint64_t _now;
    };

    void build_result(vpack::Builder& builder) {
        JsonSchemaFormatter schema;
        vpack::ObjectBuilder obj_builder(&builder);
        schema.add_field("causal-function", "string");
        builder.add("causal-function", to_json(AllInSqlFunctions::mann_whitney_u_test));
        if (_alternative == TtestAlternative::Unknown) {
            builder.add("error", to_json("state not initialized."));
            schema.add_field("error", "string");
            return;
        }
        for (size_t idx = 0; idx < 2; ++idx) {
            // in case there is only one aggstate.
            if (!std::is_sorted(_stats[idx].begin(), _stats[idx].end())) {
                std::sort(_stats[idx].begin(), _stats[idx].end());
            }
        }
        auto const& x = _stats[0];
        auto const& y = _stats[1];
        double max_s = std::numeric_limits<double>::min();
        double min_s = std::numeric_limits<double>::max();
        double now_s = 0;
        uint64_t pos_x = 0;
        uint64_t pos_y = 0;
        uint64_t pos_tmp;
        uint64_t n1 = x.size();
        uint64_t n2 = y.size();
        double n1_d = 1. / n1;
        double n2_d = 1. / n2;
        double tol = 1e-7;

        // reference: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
        while (pos_x < x.size() && pos_y < y.size()) {
            if (LIKELY(fabs(x[pos_x] - y[pos_y]) >= tol)) {
                if (x[pos_x] < y[pos_y]) {
                    now_s += n1_d;
                    ++pos_x;
                } else {
                    now_s -= n2_d;
                    ++pos_y;
                }
            } else {
                pos_tmp = pos_x + 1;
                while (pos_tmp < x.size() && UNLIKELY(fabs(x[pos_tmp] - x[pos_x]) <= tol)) {
                    pos_tmp++;
                }
                now_s += n1_d * (pos_tmp - pos_x);
                pos_x = pos_tmp;
                pos_tmp = pos_y + 1;
                while (pos_tmp < y.size() && UNLIKELY(fabs(y[pos_tmp] - y[pos_y]) <= tol)) {
                    pos_tmp++;
                }
                now_s -= n2_d * (pos_tmp - pos_y);
                pos_y = pos_tmp;
            }
            max_s = std::max(max_s, now_s);
            min_s = std::min(min_s, now_s);
        }
        now_s += n1_d * (x.size() - pos_x) - n2_d * (y.size() - pos_y);
        min_s = std::min(min_s, now_s);
        max_s = std::max(max_s, now_s);

        double d = 0;
        if (_alternative == TtestAlternative::TwoSided) {
            d = std::max(std::abs(max_s), std::abs(min_s));
        } else if (_alternative == TtestAlternative::Less) {
            d = -min_s;
        } else if (_alternative == TtestAlternative::Greater) {
            d = max_s;
        }

        double g = std::__gcd(n1, n2);
        double nx_g = n1 / g;
        double ny_g = n2 / g;

        if (_method == AUTO) {
            _method = std::max(n1, n2) <= 10000 ? EXACT : ASYMPTOTIC;
        } else if (_method == EXACT && nx_g >= std::numeric_limits<int32_t>::max() / ny_g) {
            _method = ASYMPTOTIC;
        }

        double p_value = std::numeric_limits<double>::infinity();

        if (_method == EXACT) {
            /* 
            reference:
            Gunar Schröer and Dietrich Trenkler
            Exact and Randomization Distributions of Kolmogorov-Smirnov, Tests for Two or Three Samples
            
            and
            
            Thomas Viehmann
            Numerically more stable computation of the p-values for the two-sample Kolmogorov-Smirnov test
            */
            if (n2 > n1) std::swap(n1, n2);

            double f_n1 = static_cast<double>(n1);
            double f_n2 = static_cast<double>(n2);
            double k_d = (0.5 + floor(d * f_n2 * f_n1 - tol)) / (f_n2 * f_n1);
            std::vector<double> c(n1 + 1);

            auto check = _alternative == TtestAlternative::TwoSided
                                 ? [](const double& q, const double& r, const double& s) { return fabs(r - s) >= q; }
                                 : [](const double& q, const double& r, const double& s) { return r - s >= q; };

            c[0] = 0;
            for (uint64_t j = 1; j <= n1; j++) {
                if (check(k_d, 0., j / f_n1)) {
                    c[j] = 1.;
                } else {
                    c[j] = c[j - 1];
                }
            }

            for (uint64_t i = 1; i <= n2; i++) {
                if (check(k_d, i / f_n2, 0.)) {
                    c[0] = 1.;
                }
                for (uint64_t j = 1; j <= n1; j++)
                    if (check(k_d, i / f_n2, j / f_n1)) {
                        c[j] = 1.;
                    } else {
                        double v = i / static_cast<double>(i + j);
                        double w = j / static_cast<double>(i + j);
                        c[j] = v * c[j] + w * c[j - 1];
                    }
            }
            p_value = c[n1];
        } else if (_method == ASYMPTOTIC) {
            double n = std::min(n1, n2);
            double m = std::max(n1, n2);
            double p = sqrt((n * m) / (n + m)) * d;

            if (_alternative == TtestAlternative::TwoSided) {
                /* 
                reference:
                J.DURBIN
                Distribution theory for tests based on the sample distribution function
                */
                double new_val, old_val, s, w, z;
                uint64_t k_max = static_cast<uint64_t>(sqrt(2 - log(tol)));

                if (p < 1) {
                    z = -(M_PI_2 * M_PI_4) / (p * p);
                    w = log(p);
                    s = 0;
                    for (uint64_t k = 1; k < k_max; k += 2) {
                        s += exp(k * k * z - w);
                    }
                    p = s / kTwoSidedPValueScaler;
                } else {
                    z = -2 * p * p;
                    s = -1;
                    uint64_t k = 1;
                    old_val = 0;
                    new_val = 1;
                    while (fabs(old_val - new_val) > tol) {
                        old_val = new_val;
                        new_val += 2 * s * exp(z * k * k);
                        s *= -1;
                        k++;
                    }
                    p = new_val;
                }
                p_value = 1 - p;
            } else {
                /* 
                reference:
                J. L. HODGES, Jr
                The significance probability of the Smirnov two-sample test
                */

                // Use Hodges' suggested approximation Eqn 5.3
                // Requires m to be the larger of (n1, n2)
                double expt = -2 * p * p - 2 * p * (m + 2 * n) / sqrt(m * n * (m + n)) / 3.0;
                p_value = exp(expt);
            }
        }

        builder.add("statistic", to_json(d));
        builder.add("p-value", to_json(p_value));
        schema.add_field("statistic", "double");
        schema.add_field("p-value", "double");
        builder.add("schema", to_json(schema.print()));
    }

private:
    TtestAlternative _alternative{TtestAlternative::Unknown};
    std::array<std::vector<double>, 2> _stats;
    TestMethod _method{AUTO};
};

class KolmogorovSmirnovAggFunction
        : public AggregateFunctionBatchHelper<KolmogorovSmirnovAggState, KolmogorovSmirnovAggFunction> {
public:
    using KolmogorovSmirnovDataColumn = RunTimeColumnType<TYPE_DOUBLE>;
    using KolmogorovSmirnovIndexColumn = RunTimeColumnType<TYPE_BOOLEAN>;
    using KolmogorovSmirnovAlternativeColumn = RunTimeColumnType<TYPE_VARCHAR>;
    using KolmogorovSmirnovMethodColumn = RunTimeColumnType<TYPE_VARCHAR>;
    using KolmogorovSmirnovResultColumn = RunTimeColumnType<TYPE_JSON>;

    void reset(FunctionContext* ctx, const Columns& args, AggDataPtr state) const override { DCHECK(false); }

    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        if (this->data(state).is_uninitialized()) {
            Slice alternative;
            const Column* alternative_col = columns[2];
            if (!FunctionHelper::get_data_of_column<KolmogorovSmirnovAlternativeColumn>(alternative_col, row_num,
                                                                                        alternative)) {
                ctx->set_error("Internal Error: fail to get `alternative`.");
                return;
            }
            auto alternative_str = alternative.to_string();
            if (!TtestCommon::str2alternative.count(alternative_str)) {
                ctx->set_error("Internal Error: invalid `alternative`.");
                return;
            }
            std::string method_str = "auto";
            const Column* method_col = columns[3];
            if (!FunctionHelper::get_data_of_column<KolmogorovSmirnovMethodColumn>(method_col, row_num, method_str)) {
                ctx->set_error("Internal Error: fail to get `method`.");
                return;
            }
            KolmogorovSmirnovAggState::TestMethod method = KolmogorovSmirnovAggState::AUTO;
            boost::to_lower(method_str);
            if (method_str == "auto") {
                method = KolmogorovSmirnovAggState::AUTO;
            } else if (method_str == "exact") {
                method = KolmogorovSmirnovAggState::EXACT;
            } else if (method_str == "asymp" || method_str == "asymptotic") {
                method = KolmogorovSmirnovAggState::ASYMPTOTIC;
            } else {
                ctx->set_error(fmt::format("Invalid method `{}`, which should be one of: 'auto', 'exact', 'asymp' (or "
                                           "'asymptotic')",
                                           method_str)
                                       .c_str());
            }
            this->data(state).init(TtestCommon::str2alternative.at(alternative_str), method);
        }

        double x;
        const Column* x_col = columns[0];
        if (!FunctionHelper::get_data_of_column<KolmogorovSmirnovDataColumn>(x_col, row_num, x)) {
            // ctx->set_error("Internal Error: fail to get `x`.");
            return;
        }

        bool treatment;
        const Column* treatment_col = columns[1];
        if (!FunctionHelper::get_data_of_column<KolmogorovSmirnovIndexColumn>(treatment_col, row_num, treatment)) {
            // ctx->set_error("Internal Error: fail to get `treatment`.");
            return;
        }

        this->data(state).update(x, treatment);
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
        KolmogorovSmirnovAggState other(serialized_data);
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
        const_cast<KolmogorovSmirnovAggState&>(this->data(state)).build_result(result_builder);
        auto slice = result_builder.slice();
        JsonValue result_json(slice);
        down_cast<KolmogorovSmirnovResultColumn*>(to)->append(std::move(result_json));
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {
        DCHECK((*dst)->is_binary());
        auto* dst_column = down_cast<BinaryColumn*>((*dst).get());

        std::vector<const Column*> cols;
        std::for_each(src.begin(), src.end(), [&cols](const ColumnPtr& col) { cols.emplace_back(col.get()); });
        for (size_t i = 0; i < chunk_size; ++i) {
            KolmogorovSmirnovAggState state;
            update(ctx, cols.data(), reinterpret_cast<AggDataPtr>(&state), i);
            if (ctx->has_error()) {
                return;
            }
            Bytes& bytes = dst_column->get_bytes();
            size_t old_size = bytes.size();
            size_t new_size = old_size + state.serialized_size();
            bytes.resize(new_size);
            dst_column->get_offset().emplace_back(new_size);
            uint8_t* serialized_data = bytes.data() + old_size;
            state.serialize(serialized_data);
            DCHECK_EQ(serialized_data, new_size + bytes.data());
        }
    }

    std::string get_name() const override { return std::string(AllInSqlFunctions::kolmogorov_smirnov_test); }
};

} // namespace starrocks
