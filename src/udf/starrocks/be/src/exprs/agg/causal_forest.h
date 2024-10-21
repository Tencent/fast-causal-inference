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
#include <velocypack/Iterator.h>
#include <velocypack/Value.h>

#include <algorithm>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "column/datum.h"
#include "column/json_column.h"
#include "column/nullable_column.h"
#include "column/type_traits.h"
#include "column/vectorized_fwd.h"
#include "common/status.h"
#include "common/statusor.h"
#include "exprs/agg/aggregate.h"
#include "exprs/all_in_sql_functions.h"
#include "exprs/function_helper.h"
#include "exprs/helpers/serialize_helpers.hpp"
#include "exprs/jsonpath.h"
#include "types/logical_type.h"
#include "util/json.h"
#include "util/url_coding.h"

namespace starrocks {
enum class CausalForestState : uint8_t {
    Init = 0,
    CalcNumerAndDenom,
    FindBestSplitPre,
    FindBestSplit,
    Honesty,
    Finish,
};

class CausalForestJsonHelper {
public:
    static StatusOr<JsonValue> json_extract(const JsonValue& json, std::string const& field_name) {
        ASSIGN_OR_RETURN(JsonPath mtry_json_path, JsonPath::parse("$." + field_name));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&json, mtry_json_path, &builder);
        if (slice.isNone() || slice.isNull()) {
            return Status::InvalidArgument(fmt::format("field `{}` not found", field_name));
        }
        return JsonValue(slice);
    }

    template <typename T>
    static StatusOr<T> get_json_value(const JsonValue& value) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T> || std::is_same_v<T, bool> ||
                              std::is_same_v<T, JsonValue> || std::is_same_v<T, std::string>,
                      "unsupported type");
        if constexpr (std::is_same_v<T, std::string>) {
            auto st = value.get_string();
            if (!st.ok()) {
                return st.status();
            }
            return st.value().to_string();
        } else if constexpr (std::is_same_v<T, bool>) {
            return value.get_bool();
        } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            return value.get_int();
        } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
            return value.get_uint();
        } else if constexpr (std::is_floating_point_v<T>) {
            return value.get_double();
        } else if constexpr (std::is_same_v<T, JsonValue>) {
            return value;
        }
        return Status::NotSupported("unsupported type");
    }

    template <typename T>
    static StatusOr<T> get_json_value(const JsonValue& json, std::string const& field_name) {
        ASSIGN_OR_RETURN(JsonPath mtry_json_path, JsonPath::parse("$." + field_name));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&json, mtry_json_path, &builder);
        JsonValue value(slice);
        if (value.get_type() == JsonType::JSON_NULL) {
            return Status::InvalidArgument(fmt::format("field `{}` not found", field_name));
        }
        return get_json_value<T>(value);
    }

    static StatusOr<std::vector<JsonValue>> get_json_array(const JsonValue& json) {
        if (json.get_type() != JsonType::JSON_ARRAY) {
            return Status::InvalidArgument("field is not array");
        }
        std::vector<JsonValue> result;
        vpack::ArrayIterator iter(json.to_vslice());
        while (iter.valid()) {
            result.emplace_back(iter.value());
            iter.next();
        }
        return result;
    }

    static StatusOr<std::vector<JsonValue>> get_json_array(const JsonValue& json, std::string const& field_name) {
        ASSIGN_OR_RETURN(JsonPath mtry_json_path, JsonPath::parse("$." + field_name));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&json, mtry_json_path, &builder);
        JsonValue value(slice);
        if (value.get_type() == JsonType::JSON_NULL) {
            return Status::InvalidArgument(fmt::format("field `{}` not found", field_name));
        }
        if (value.get_type() != JsonType::JSON_ARRAY) {
            return Status::InvalidArgument("field is not array");
        }
        std::vector<JsonValue> result;
        vpack::ArrayIterator iter(value.to_vslice());
        while (iter.valid()) {
            result.emplace_back(iter.value());
            iter.next();
        }
        return result;
    }

    static StatusOr<std::unordered_map<std::string, JsonValue>> get_json_object(const JsonValue& json) {
        if (json.get_type() != JsonType::JSON_OBJECT) {
            return Status::InvalidArgument("field is not object");
        }
        std::unordered_map<std::string, JsonValue> result;
        vpack::ObjectIterator iter(json.to_vslice());
        while (iter.valid()) {
            ASSIGN_OR_RETURN(std::string key, JsonValue(iter.key()).to_string());
            result[std::move(key)] = JsonValue(iter.value());
            iter.next();
        }
        return result;
    }

    static StatusOr<std::unordered_map<std::string, JsonValue>> get_json_object(const JsonValue& json,
                                                                                std::string const& field_name) {
        ASSIGN_OR_RETURN(JsonPath mtry_json_path, JsonPath::parse("$." + field_name));
        vpack::Builder builder;
        vpack::Slice slice = JsonPath::extract(&json, mtry_json_path, &builder);
        JsonValue value(slice);
        if (value.get_type() == JsonType::JSON_NULL) {
            return Status::InvalidArgument(fmt::format("field `{}` not found", field_name));
        }
        if (value.get_type() != JsonType::JSON_OBJECT) {
            return Status::InvalidArgument("field is not object");
        }
        std::unordered_map<std::string, JsonValue> result;
        vpack::ObjectIterator iter(value.to_vslice());
        while (iter.valid()) {
            ASSIGN_OR_RETURN(std::string key, JsonValue(iter.key()).to_string());
            result[std::move(key)] = JsonValue(iter.value());
            iter.next();
        }
        return result;
    }
};

struct TreeOptions {
public:
    TreeOptions() = default;

    TreeOptions(uint32_t mtry_, size_t quantile_size_, uint32_t min_node_size_, bool honesty_, double honesty_fraction_,
                bool honesty_prune_leaves_, double alpha_, double imbalance_penalty_)
            : mtry(mtry_),
              min_node_size(min_node_size_),
              honesty(honesty_),
              honesty_fraction(honesty_fraction_),
              honesty_prune_leaves(honesty_prune_leaves_),
              alpha(alpha_),
              imbalance_penalty(imbalance_penalty_),
              quantile_size(quantile_size_) {}

    void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
        vpack::ObjectBuilder ob =
                field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
        builder.add("mtry", vpack::Value(mtry));
        builder.add("quantile_size", vpack::Value(quantile_size));
        builder.add("min_node_size", vpack::Value(min_node_size));
        builder.add("honesty", vpack::Value(honesty));
        builder.add("honesty_fraction", vpack::Value(honesty_fraction));
        builder.add("honesty_prune_leaves", vpack::Value(honesty_prune_leaves));
        builder.add("alpha", vpack::Value(alpha));
        builder.add("imbalance_penalty", vpack::Value(imbalance_penalty));
    }

    Status from_json(JsonValue const& json) {
        ASSIGN_OR_RETURN(mtry, CausalForestJsonHelper::get_json_value<uint32_t>(json, "mtry"));
        ASSIGN_OR_RETURN(quantile_size, CausalForestJsonHelper::get_json_value<size_t>(json, "quantile_size"));
        ASSIGN_OR_RETURN(min_node_size, CausalForestJsonHelper::get_json_value<uint32_t>(json, "min_node_size"));
        ASSIGN_OR_RETURN(honesty, CausalForestJsonHelper::get_json_value<bool>(json, "honesty"));
        ASSIGN_OR_RETURN(honesty_fraction, CausalForestJsonHelper::get_json_value<double>(json, "honesty_fraction"));
        ASSIGN_OR_RETURN(honesty_prune_leaves,
                         CausalForestJsonHelper::get_json_value<bool>(json, "honesty_prune_leaves"));
        ASSIGN_OR_RETURN(alpha, CausalForestJsonHelper::get_json_value<double>(json, "alpha"));
        ASSIGN_OR_RETURN(imbalance_penalty, CausalForestJsonHelper::get_json_value<double>(json, "imbalance_penalty"));
        return Status::OK();
    }

    void serialize(uint8_t*& data) const {
        uint8_t honesty_ = honesty;
        uint8_t honesty_prune_leaves_ = honesty_prune_leaves;
        SerializeHelpers::serialize_all(data, mtry, quantile_size, min_node_size, honesty_, honesty_fraction,
                                        honesty_prune_leaves_, alpha, imbalance_penalty);
    }

    void deserialize(const uint8_t*& data) {
        uint8_t honesty_;
        uint8_t honesty_prune_leaves_;
        SerializeHelpers::deserialize_all(data, mtry, quantile_size, min_node_size, honesty_, honesty_fraction,
                                          honesty_prune_leaves_, alpha, imbalance_penalty);
        honesty = honesty_;
        honesty_prune_leaves = honesty_prune_leaves_;
    }

    size_t serialized_size() const {
        return SerializeHelpers::serialized_size_all(mtry, quantile_size, min_node_size, honesty, honesty_fraction,
                                                     honesty_prune_leaves, alpha, imbalance_penalty);
    }

    uint32_t mtry;
    uint32_t min_node_size;
    bool honesty;
    double honesty_fraction;
    bool honesty_prune_leaves;
    double alpha;
    double imbalance_penalty;
    size_t quantile_size;
};

struct ForestOptions {
public:
    ForestOptions() = default;
    Status init(uint32_t num_trees_, size_t ci_group_size_, double sample_fraction_, uint32_t random_seed_,
                int32_t weight_index_, int32_t outcome_index_, int32_t treatment_index_, int32_t instrument_index_,
                CausalForestState state_, uint64_t arguments_size_, size_t max_centroids_, size_t max_unmerged_,
                double epsilon_) {
        num_trees = num_trees_;
        ci_group_size = ci_group_size_;
        sample_fraction = sample_fraction_;
        random_seed = random_seed_;
        weight_index = weight_index_;
        outcome_index = outcome_index_;
        treatment_index = treatment_index_;
        instrument_index = instrument_index_;
        state = state_;
        arguments_size = arguments_size_;
        max_centroids = max_centroids_;
        max_unmerged = max_unmerged_;
        epsilon = epsilon_;
        disallowed_split_variables.insert(weight_index);
        disallowed_split_variables.insert(outcome_index);
        disallowed_split_variables.insert(treatment_index);
        disallowed_split_variables.insert(instrument_index);
        auto max_index = std::max({weight_index, outcome_index, treatment_index_, instrument_index});
        auto min_index = std::min({weight_index, outcome_index, treatment_index_, instrument_index});
        if (max_index >= static_cast<int32_t>(arguments_size) || min_index < 0) {
            return Status::InvalidArgument("params index over flow");
        }
        return Status::OK();
    }

    size_t get_max_index() { return std::max({weight_index, outcome_index, treatment_index, instrument_index}); }

    void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
        vpack::ObjectBuilder ob =
                field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
        builder.add("num_trees", vpack::Value(num_trees));
        builder.add("ci_group_size", vpack::Value(ci_group_size));
        builder.add("sample_fraction", vpack::Value(sample_fraction));
        builder.add("random_seed", vpack::Value(random_seed));
        builder.add("weight_index", vpack::Value(weight_index));
        builder.add("outcome_index", vpack::Value(outcome_index));
        builder.add("treatment_index", vpack::Value(treatment_index));
        builder.add("instrument_index", vpack::Value(instrument_index));
        builder.add("state", vpack::Value(static_cast<uint8_t>(state)));
        builder.add("arguments_size", vpack::Value(arguments_size));
        builder.add("max_centroids", vpack::Value(max_centroids));
        builder.add("max_unmerged", vpack::Value(max_unmerged));
        builder.add("epsilon", vpack::Value(epsilon));
        {
            vpack::ArrayBuilder disallowed_split_variables_array(&builder, "disallowed_split_variables");
            for (auto& index : disallowed_split_variables) {
                builder.add(vpack::Value(index));
            }
        }
    }

    Status from_json(JsonValue const& json) {
        ASSIGN_OR_RETURN(num_trees, CausalForestJsonHelper::get_json_value<uint32_t>(json, "num_trees"));
        ASSIGN_OR_RETURN(ci_group_size, CausalForestJsonHelper::get_json_value<size_t>(json, "ci_group_size"));
        ASSIGN_OR_RETURN(sample_fraction, CausalForestJsonHelper::get_json_value<double>(json, "sample_fraction"));
        ASSIGN_OR_RETURN(random_seed, CausalForestJsonHelper::get_json_value<uint32_t>(json, "random_seed"));
        ASSIGN_OR_RETURN(weight_index, CausalForestJsonHelper::get_json_value<int32_t>(json, "weight_index"));
        ASSIGN_OR_RETURN(outcome_index, CausalForestJsonHelper::get_json_value<int32_t>(json, "outcome_index"));
        ASSIGN_OR_RETURN(treatment_index, CausalForestJsonHelper::get_json_value<int32_t>(json, "treatment_index"));
        ASSIGN_OR_RETURN(instrument_index, CausalForestJsonHelper::get_json_value<int32_t>(json, "instrument_index"));
        ASSIGN_OR_RETURN(uint8_t state_, CausalForestJsonHelper::get_json_value<uint8_t>(json, "state"));
        state = static_cast<CausalForestState>(state_);
        ASSIGN_OR_RETURN(arguments_size, CausalForestJsonHelper::get_json_value<uint64_t>(json, "arguments_size"));
        ASSIGN_OR_RETURN(max_centroids, CausalForestJsonHelper::get_json_value<size_t>(json, "max_centroids"));
        ASSIGN_OR_RETURN(max_unmerged, CausalForestJsonHelper::get_json_value<size_t>(json, "max_unmerged"));
        ASSIGN_OR_RETURN(epsilon, CausalForestJsonHelper::get_json_value<double>(json, "epsilon"));
        ASSIGN_OR_RETURN(std::vector<JsonValue> disallowed_split_variables_json,
                         CausalForestJsonHelper::get_json_array(json, "disallowed_split_variables"));
        for (auto& index : disallowed_split_variables_json) {
            ASSIGN_OR_RETURN(int var, index.get_int());
            disallowed_split_variables.emplace(var);
        }
        return Status::OK();
    }

    void serialize(uint8_t*& data) const {
        auto state_ = static_cast<uint8_t>(state);
        SerializeHelpers::serialize_all(data, num_trees, ci_group_size, sample_fraction, random_seed, weight_index,
                                        outcome_index, treatment_index, instrument_index, state_, arguments_size,
                                        max_centroids, max_unmerged, epsilon, disallowed_split_variables);
    }

    void deserialize(const uint8_t*& data) {
        uint8_t state_;
        SerializeHelpers::deserialize_all(data, num_trees, ci_group_size, sample_fraction, random_seed, weight_index,
                                          outcome_index, treatment_index, instrument_index, state_, arguments_size,
                                          max_centroids, max_unmerged, epsilon, disallowed_split_variables);
        state = static_cast<CausalForestState>(state_);
    }

    size_t serialized_size() const {
        return SerializeHelpers::serialized_size_all(num_trees, ci_group_size, sample_fraction, random_seed,
                                                     weight_index, outcome_index, treatment_index, instrument_index,
                                                     uint8_t{0}, arguments_size, max_centroids, max_unmerged, epsilon,
                                                     disallowed_split_variables);
    }

    uint32_t num_trees;
    size_t ci_group_size;
    double sample_fraction;
    uint32_t random_seed;

    int32_t weight_index = -1;
    int32_t outcome_index = -1;
    int32_t treatment_index = -1;
    int32_t instrument_index = -1;
    CausalForestState state;
    uint64_t arguments_size = 0;
    size_t max_centroids = 2048;
    size_t max_unmerged = 2048;
    double epsilon = 1e-2;

    std::set<size_t> disallowed_split_variables;
};

template <typename T>
class CausalQuantileTDigest {
    using Value = float;
    using Count = float;
    using BetterFloat = double; // For intermediate results and sum(Count). Must have better precision, than Count

    /** The centroid stores the weight of points around their mean value
      */
    struct Centroid {
        Value mean;
        Count count;

        Centroid() = default;

        explicit Centroid(Value mean_, Count count_) : mean(mean_), count(count_) {}

        bool operator<(const Centroid& other) const { return mean < other.mean; }

        void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
            vpack::ObjectBuilder ob =
                    field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
            builder.add("mean", vpack::Value(mean));
            builder.add("count", vpack::Value(count));
        }

        Status from_json(JsonValue const& json) {
            ASSIGN_OR_RETURN(mean, CausalForestJsonHelper::get_json_value<Value>(json, "mean"));
            ASSIGN_OR_RETURN(count, CausalForestJsonHelper::get_json_value<Count>(json, "count"));
            return Status::OK();
        }

        void serialize(uint8_t*& data) const { SerializeHelpers::serialize_all(data, mean, count); }

        void deserialize(const uint8_t*& data) { SerializeHelpers::deserialize_all(data, mean, count); }

        size_t serialized_size() const { return SerializeHelpers::serialized_size_all(mean, count); }
    };

    /** :param epsilon: value \delta from the article - error in the range
      *                    quantile 0.5 (default is 0.01, i.e. 1%)
      *                    if you change epsilon, you must also change max_centroids
      * :param max_centroids: depends on epsilon, the better accuracy, the more centroids you need
      *                       to describe data with this accuracy. Read article before changing.
      * :param max_unmerged: when accumulating count of new points beyond this
      *                      value centroid compression is triggered
      *                      (default is 2048, the higher the value - the
      *                      more memory is required, but amortization of execution time increases)
      *                      Change freely anytime.
      */
    struct Params {
        Value epsilon = static_cast<Value>(0.01);
        size_t max_centroids = 2048;
        size_t max_unmerged = 2048;

        void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
            vpack::ObjectBuilder ob =
                    field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
            builder.add("epsilon", vpack::Value(epsilon));
            builder.add("max_centroids", vpack::Value(max_centroids));
            builder.add("max_unmerged", vpack::Value(max_unmerged));
        }

        Status from_json(JsonValue const& json) {
            ASSIGN_OR_RETURN(epsilon, CausalForestJsonHelper::get_json_value<Value>(json, "epsilon"));
            ASSIGN_OR_RETURN(max_centroids, CausalForestJsonHelper::get_json_value<size_t>(json, "max_centroids"));
            ASSIGN_OR_RETURN(max_unmerged, CausalForestJsonHelper::get_json_value<size_t>(json, "max_unmerged"));
            return Status::OK();
        }

        void serialize(uint8_t*& data) const {
            SerializeHelpers::serialize_all(data, epsilon, max_centroids, max_unmerged);
        }

        void deserialize(const uint8_t*& data) {
            SerializeHelpers::deserialize_all(data, epsilon, max_centroids, max_unmerged);
        }

        size_t serialized_size() const {
            return SerializeHelpers::serialized_size_all(epsilon, max_centroids, max_unmerged);
        }
    };
    Params params{};

    using Centroids = std::vector<Centroid>;

    Centroids centroids;
    double count = 0;
    size_t unmerged = 0;

    /** Linear interpolation at the point x on the line (x1, y1)..(x2, y2)
      */
    static Value interpolate(Value x, Value x1, Value y1, Value x2, Value y2) {
        /// Symmetric interpolation for better results with infinities.
        double k = (x - x1) / (x2 - x1);
        return static_cast<float>((1 - k) * y1 + k * y2);
    }

    /** Adds a centroid `c` to the digest
     * centroid must be valid, validity is checked in add(), deserialize() and is maintained by compress()
      */
    void addCentroid(const Centroid& c) {
        centroids.push_back(c);
        count += c.count;
        ++unmerged;
        if (unmerged > params.max_unmerged) compress();
    }

    inline bool canBeMerged(const BetterFloat& l_mean, const Value& r_mean) {
        return l_mean == r_mean || (!std::isinf(l_mean) && !std::isinf(r_mean));
    }

    void compressBrute() {
        if (centroids.size() <= params.max_centroids) return;
        const size_t batch_size = (centroids.size() + params.max_centroids - 1) / params.max_centroids; // at least 2

        auto l = centroids.begin();
        auto r = std::next(l);
        BetterFloat sum = 0;
        BetterFloat l_mean = static_cast<float>(l->mean); // We have high-precision temporaries for numeric stability
        BetterFloat l_count = static_cast<float>(l->count);
        size_t batch_pos = 0;

        for (; r != centroids.end(); ++r) {
            if (batch_pos < batch_size - 1) {
                /// The left column "eats" the right. Middle of the batch
                l_count += r->count;
                if (r->mean != l_mean) /// Handling infinities of the same sign well.
                {
                    l_mean += r->count * (r->mean - l_mean) /
                              l_count; // Symmetric algo (M1*C1 + M2*C2)/(C1+C2) is numerically better, but slower
                }
                l->mean = static_cast<float>(l_mean);
                l->count = static_cast<float>(l_count);
                batch_pos += 1;
            } else {
                // End of the batch, start the next one
                if (!std::isnan(l->mean)) /// Skip writing batch result if we compressed something to nan.
                {
                    sum += l->count; // Not l_count, otherwise actual sum of elements will be different
                    ++l;
                }

                /// We skip all the values "eaten" earlier.
                *l = *r;
                l_mean = l->mean;
                l_count = l->count;
                batch_pos = 0;
            }
        }

        if (!std::isnan(l->mean)) {
            count = sum + l_count; // Update count, it might be different due to += inaccuracy
            centroids.resize(l - centroids.begin() + 1);
        } else /// Skip writing last batch if (super unlikely) it's nan.
        {
            count = sum;
            centroids.resize(l - centroids.begin());
        }
        // Here centroids.size() <= params.max_centroids
    }

public:
    CausalQuantileTDigest() = default;

    CausalQuantileTDigest(size_t max_centroids_, size_t max_unmerged_, Value epsilon_) {
        params.max_centroids = max_centroids_;
        params.max_unmerged = max_unmerged_;
        params.epsilon = epsilon_;
    }

    /** Performs compression of accumulated centroids
      * When merging, the invariant is retained to the maximum size of each
      * centroid that does not exceed `4 q (1 - q) \ delta N`.
      */
    void compress() {
        if (unmerged > 0 || centroids.size() > params.max_centroids) {
            // unmerged > 0 implies centroids.size() > 0, hence *l is valid below
            std::sort(centroids.begin(), centroids.end());

            /// A pair of consecutive bars of the histogram.
            auto l = centroids.begin();
            auto r = std::next(l);

            const BetterFloat count_epsilon_4 =
                    count * params.epsilon * 4; // Compiler is unable to do this optimization
            BetterFloat sum = 0;
            BetterFloat l_mean = l->mean; // We have high-precision temporaries for numeric stability
            BetterFloat l_count = l->count;
            while (r != centroids.end()) {
                /// N.B. We cannot merge all the same values into single centroids because this will lead to
                /// unbalanced compression and wrong results.
                /// For more information see: https://arxiv.org/abs/1902.04023

                /// The ratio of the part of the histogram to l, including the half l to the entire histogram. That is, what level quantile in position l.
                BetterFloat ql = (sum + l_count * 0.5) / count;
                BetterFloat err = ql * (1 - ql);

                /// The ratio of the portion of the histogram to l, including l and half r to the entire histogram. That is, what level is the quantile in position r.
                BetterFloat qr = (sum + l_count + r->count * 0.5) / count;
                BetterFloat err2 = qr * (1 - qr);

                if (err > err2) err = err2;

                BetterFloat k = count_epsilon_4 * err;

                /** The ratio of the weight of the glued column pair to all values is not greater,
                  *  than epsilon multiply by a certain quadratic coefficient, which in the median is 1 (4 * 1/2 * 1/2),
                  *  and at the edges decreases and is approximately equal to the distance to the edge * 4.
                  */

                if (l_count + r->count <= k && canBeMerged(l_mean, r->mean)) {
                    // it is possible to merge left and right
                    /// The left column "eats" the right.
                    l_count += r->count;
                    if (r->mean != l_mean) /// Handling infinities of the same sign well.
                    {
                        l_mean += r->count * (r->mean - l_mean) /
                                  l_count; // Symmetric algo (M1*C1 + M2*C2)/(C1+C2) is numerically better, but slower
                    }
                    l->mean = static_cast<float>(l_mean);
                    l->count = static_cast<float>(l_count);
                } else {
                    // not enough capacity, check the next pair
                    sum += l->count; // Not l_count, otherwise actual sum of elements will be different
                    ++l;

                    /// We skip all the values "eaten" earlier.
                    if (l != r) *l = *r;
                    l_mean = l->mean;
                    l_count = l->count;
                }
                ++r;
            }
            count = sum + l_count; // Update count, it might be different due to += inaccuracy

            /// At the end of the loop, all values to the right of l were "eaten".
            centroids.resize(l - centroids.begin() + 1);
            unmerged = 0;
        }

        // Ensures centroids.size() < max_centroids, independent of unprovable floating point blackbox above
        compressBrute();
    }

    /** Adds to the digest a change in `x` with a weight of `cnt` (default 1)
      */
    void add(T x, double cnt = 1) {
        auto vx = static_cast<Value>(x);
        if (cnt == 0 || std::isnan(vx))
            return; // Count 0 breaks compress() assumptions, Nan breaks sort(). We treat them as no sample.
        addCentroid(Centroid{vx, static_cast<Count>(cnt)});
    }

    void merge(const CausalQuantileTDigest& other) {
        for (const auto& c : other.centroids) addCentroid(c);
    }

    void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
        vpack::ObjectBuilder ob =
                field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
        params.to_json(builder, "params");
        {
            vpack::ArrayBuilder centroids_array(&builder, "centroids");
            for (auto& centroid : centroids) {
                centroid.to_json(builder);
            }
        }
        builder.add("count", vpack::Value(count));
        builder.add("unmerged", vpack::Value(unmerged));
    }

    Status from_json(JsonValue const& json) {
        ASSIGN_OR_RETURN(JsonValue params_json, CausalForestJsonHelper::json_extract(json, "params"));
        RETURN_IF_ERROR(params.from_json(params_json));
        ASSIGN_OR_RETURN(std::vector<JsonValue> centroids_json,
                         CausalForestJsonHelper::get_json_array(json, "centroids"));
        centroids.resize(centroids_json.size());
        for (size_t i = 0; i < centroids_json.size(); ++i) {
            RETURN_IF_ERROR(centroids[i].from_json(centroids_json[i]));
        }
        ASSIGN_OR_RETURN(count, CausalForestJsonHelper::get_json_value<double>(json, "count"));
        ASSIGN_OR_RETURN(unmerged, CausalForestJsonHelper::get_json_value<size_t>(json, "unmerged"));
        return Status::OK();
    }

    void serialize(uint8_t*& data) const {
        params.serialize(data);
        size_t length = centroids.size();
        SerializeHelpers::serialize_all(data, length);
        for (const auto& centroid : centroids) centroid.serialize(data);
        SerializeHelpers::serialize_all(data, count, unmerged);
    }

    void deserialize(const uint8_t*& data) {
        params.deserialize(data);
        size_t length;
        SerializeHelpers::deserialize_all(data, length);
        centroids.resize(length);
        for (auto& centroid : centroids) centroid.deserialize(data);
        SerializeHelpers::deserialize_all(data, count, unmerged);
    }

    size_t serialized_size() const {
        size_t size = params.serialized_size();
        size += SerializeHelpers::serialized_size_all(centroids.size());
        for (const auto& centroid : centroids) size += centroid.serialized_size();
        size += SerializeHelpers::serialized_size_all(count, unmerged);
        return size;
    }

    std::vector<double> getQuantiles(const size_t quantiles_count = 100) {
        std::vector<double> levels(quantiles_count);
        for (size_t i = 0; i < quantiles_count; ++i) levels[i] = i / (1. * quantiles_count);
        std::vector<size_t> levels_permutation(quantiles_count);
        for (size_t i = 0; i < quantiles_count; ++i) levels_permutation[i] = i;
        std::vector<double> result(quantiles_count);
        getManyImpl(levels.data(), levels_permutation.data(), quantiles_count, result.data());
        // unique result
        result.erase(std::unique(result.begin(), result.end()), result.end());
        //result.resize(1);
        return result;
    }

    /** Get multiple quantiles (`size` parts).
      * levels - an array of levels of the desired quantiles. They are in a random order.
      * levels_permutation - array-permutation levels. The i-th position will be the index of the i-th ascending level in the `levels` array.
      * result - the array where the results are added, in order of `levels`,
      */
    template <typename ResultType>
    void getManyImpl(const double* levels, const size_t* levels_permutation, size_t size, ResultType* result) {
        if (centroids.empty()) {
            for (size_t result_num = 0; result_num < size; ++result_num)
                result[result_num] = std::is_floating_point_v<ResultType> ? NAN : 0;
            return;
        }

        compress();

        if (centroids.size() == 1) {
            for (size_t result_num = 0; result_num < size; ++result_num) result[result_num] = centroids.front().mean;
            return;
        }

        double x = levels[levels_permutation[0]] * count;
        double prev_x = 0;
        Count sum = 0;
        Value prev_mean = centroids.front().mean;
        Count prev_count = centroids.front().count;

        size_t result_num = 0;
        for (const auto& c : centroids) {
            double current_x = sum + c.count * 0.5;

            if (current_x >= x) {
                /// Special handling of singletons.
                double left = prev_x + 0.5 * (prev_count == 1);
                double right = current_x - 0.5 * (c.count == 1);

                while (current_x >= x) {
                    if (x <= left)
                        result[levels_permutation[result_num]] = prev_mean;
                    else if (x >= right)
                        result[levels_permutation[result_num]] = c.mean;
                    else
                        result[levels_permutation[result_num]] = static_cast<float>(interpolate(
                                static_cast<float>(x), static_cast<float>(left), static_cast<float>(prev_mean),
                                static_cast<float>(right), static_cast<float>(c.mean)));

                    ++result_num;
                    if (result_num >= size) return;

                    x = levels[levels_permutation[result_num]] * count;
                }
            }

            sum += c.count;
            prev_mean = c.mean;
            prev_count = c.count;
            prev_x = current_x;
        }

        auto rest_of_results = centroids.back().mean;
        for (; result_num < size; ++result_num) result[levels_permutation[result_num]] = rest_of_results;
    }
};

class Tree {
public:
    struct SplitNode // size equal to quantiles.size
    {
        uint64_t counter = 0;
        double weight_sums = 0.0;
        double sums = 0.0;
        uint64_t num_small_z = 0.0;
        double sums_z = 0.0;
        double sums_z_squared = 0.0;

        double total_outcome = 0.0;
        double total_treatment = 0.0;
        double total_outcome_treatment = 0.0;

        std::string toString() const {
            std::string res = "\n";
            res += "counter: " + std::to_string(counter) + "\n";
            res += "weight_sums: " + std::to_string(weight_sums) + "\n";
            res += "sums: " + std::to_string(sums) + "\n";
            res += "num_small_z: " + std::to_string(num_small_z) + "\n";
            res += "sums_z: " + std::to_string(sums_z) + "\n";
            res += "sums_z_squared: " + std::to_string(sums_z_squared) + "\n";
            res += "total_outcome: " + std::to_string(total_outcome) + "\n";
            res += "total_treatment: " + std::to_string(total_treatment) + "\n";
            res += "total_outcome_treatment: " + std::to_string(total_outcome_treatment) + "\n";
            return res;
        }

        void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
            vpack::ObjectBuilder ob =
                    field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
            builder.add("counter", vpack::Value(counter));
            builder.add("weight_sums", vpack::Value(weight_sums));
            builder.add("sums", vpack::Value(sums));
            builder.add("num_small_z", vpack::Value(num_small_z));
            builder.add("sums_z", vpack::Value(sums_z));
            builder.add("sums_z_squared", vpack::Value(sums_z_squared));
            builder.add("total_outcome", vpack::Value(total_outcome));
            builder.add("total_treatment", vpack::Value(total_treatment));
            builder.add("total_outcome_treatment", vpack::Value(total_outcome_treatment));
        }

        Status from_json(JsonValue const& json) {
            ASSIGN_OR_RETURN(counter, CausalForestJsonHelper::get_json_value<uint64_t>(json, "counter"));
            ASSIGN_OR_RETURN(weight_sums, CausalForestJsonHelper::get_json_value<double>(json, "weight_sums"));
            ASSIGN_OR_RETURN(sums, CausalForestJsonHelper::get_json_value<double>(json, "sums"));
            ASSIGN_OR_RETURN(num_small_z, CausalForestJsonHelper::get_json_value<uint64_t>(json, "num_small_z"));
            ASSIGN_OR_RETURN(sums_z, CausalForestJsonHelper::get_json_value<double>(json, "sums_z"));
            ASSIGN_OR_RETURN(sums_z_squared, CausalForestJsonHelper::get_json_value<double>(json, "sums_z_squared"));
            ASSIGN_OR_RETURN(total_outcome, CausalForestJsonHelper::get_json_value<double>(json, "total_outcome"));
            ASSIGN_OR_RETURN(total_treatment, CausalForestJsonHelper::get_json_value<double>(json, "total_treatment"));
            ASSIGN_OR_RETURN(total_outcome_treatment,
                             CausalForestJsonHelper::get_json_value<double>(json, "total_outcome_treatment"));
            return Status::OK();
        }

        void serialize(uint8_t*& data) const {
            SerializeHelpers::serialize_all(data, counter, weight_sums, sums, num_small_z, sums_z, sums_z_squared,
                                            total_outcome, total_treatment, total_outcome_treatment);
        }

        void deserialize(const uint8_t*& data) {
            SerializeHelpers::deserialize_all(data, counter, weight_sums, sums, num_small_z, sums_z, sums_z_squared,
                                              total_outcome, total_treatment, total_outcome_treatment);
        }

        size_t serialized_size() const {
            return SerializeHelpers::serialized_size_all(counter, weight_sums, sums, num_small_z, sums_z,
                                                         sums_z_squared, total_outcome, total_treatment,
                                                         total_outcome_treatment);
        }

        void merge(const SplitNode& other) {
            counter += other.counter;
            weight_sums += other.weight_sums;
            sums += other.sums;
            num_small_z += other.num_small_z;
            sums_z += other.sums_z;
            sums_z_squared += other.sums_z_squared;
            total_outcome += other.total_outcome;
            total_treatment += other.total_treatment;
            total_outcome_treatment += other.total_outcome_treatment;
        }
    };

    struct CalcNodeInfo {
        double sum_weight = 0.0;
        double total_outcome = 0.0;
        double total_treatment = 0.0;
        double total_instrument = 0.0;
        double total_outcome_treatment = 0.0;
        uint64_t num_samples = 0;
        std::vector<size_t> possible_split_vars;
        bool is_stop = false;
        double numerator = 0.0;
        double denominator = 0.0;
        double sum_node_z_squared = 0.0;

        double sum_node = 0.0;
        uint64_t num_node_small_z = 0;

        mutable std::vector<CausalQuantileTDigest<float>> quantiles_calcers;
        std::vector<std::vector<double>> quantiles;
        std::vector<std::vector<SplitNode>> split_nodes;

        int32_t father = -1;

        void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
            vpack::ObjectBuilder ob =
                    field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
            builder.add("sum_weight", vpack::Value(sum_weight));
            builder.add("total_outcome", vpack::Value(total_outcome));
            builder.add("total_treatment", vpack::Value(total_treatment));
            builder.add("total_instrument", vpack::Value(total_instrument));
            builder.add("total_outcome_treatment", vpack::Value(total_outcome_treatment));
            {
                vpack::ArrayBuilder possible_split_vars_array(&builder, "possible_split_vars");
                for (const auto& var : possible_split_vars) {
                    builder.add(vpack::Value(var));
                }
            }
            builder.add("numerator", vpack::Value(numerator));
            builder.add("denominator", vpack::Value(denominator));
            builder.add("is_stop", vpack::Value(is_stop));
            builder.add("num_samples", vpack::Value(num_samples));
            builder.add("sum_node_z_squared", vpack::Value(sum_node_z_squared));
            builder.add("sum_node", vpack::Value(sum_node));
            builder.add("num_node_small_z", vpack::Value(num_node_small_z));
            {
                vpack::ArrayBuilder quantiles_calcers_array(&builder, "quantiles_calcers");
                for (auto& quantile : quantiles_calcers) {
                    quantile.to_json(builder);
                }
            }
            {
                vpack::ArrayBuilder quantiles_array(&builder, "quantiles");
                for (const auto& quantile : quantiles) {
                    vpack::ArrayBuilder quantile_array(&builder);
                    for (const auto& val : quantile) {
                        builder.add(vpack::Value(val));
                    }
                }
            }
            {
                vpack::ArrayBuilder split_nodes_array(&builder, "split_nodes");
                for (const auto& split_node : split_nodes) {
                    vpack::ArrayBuilder split_node_array(&builder);
                    for (const auto& node : split_node) {
                        node.to_json(builder);
                    }
                }
            }
            builder.add("father", vpack::Value(father));
        }

        Status from_json(JsonValue const& json) {
            ASSIGN_OR_RETURN(sum_weight, CausalForestJsonHelper::get_json_value<double>(json, "sum_weight"));
            ASSIGN_OR_RETURN(total_outcome, CausalForestJsonHelper::get_json_value<double>(json, "total_outcome"));
            ASSIGN_OR_RETURN(total_treatment, CausalForestJsonHelper::get_json_value<double>(json, "total_treatment"));
            ASSIGN_OR_RETURN(total_instrument,
                             CausalForestJsonHelper::get_json_value<double>(json, "total_instrument"));
            ASSIGN_OR_RETURN(total_outcome_treatment,
                             CausalForestJsonHelper::get_json_value<double>(json, "total_outcome_treatment"));
            ASSIGN_OR_RETURN(std::vector<JsonValue> possible_split_vars_json,
                             CausalForestJsonHelper::get_json_array(json, "possible_split_vars"));
            for (const auto& var : possible_split_vars_json) {
                ASSIGN_OR_RETURN(size_t v, var.get_uint());
                possible_split_vars.push_back(v);
            }
            ASSIGN_OR_RETURN(numerator, CausalForestJsonHelper::get_json_value<double>(json, "numerator"));
            ASSIGN_OR_RETURN(denominator, CausalForestJsonHelper::get_json_value<double>(json, "denominator"));
            ASSIGN_OR_RETURN(is_stop, CausalForestJsonHelper::get_json_value<bool>(json, "is_stop"));
            ASSIGN_OR_RETURN(num_samples, CausalForestJsonHelper::get_json_value<uint64_t>(json, "num_samples"));
            ASSIGN_OR_RETURN(sum_node_z_squared,
                             CausalForestJsonHelper::get_json_value<double>(json, "sum_node_z_squared"));
            ASSIGN_OR_RETURN(sum_node, CausalForestJsonHelper::get_json_value<double>(json, "sum_node"));
            ASSIGN_OR_RETURN(num_node_small_z,
                             CausalForestJsonHelper::get_json_value<uint64_t>(json, "num_node_small_z"));
            ASSIGN_OR_RETURN(std::vector<JsonValue> quantiles_calcers_json,
                             CausalForestJsonHelper::get_json_array(json, "quantiles_calcers"));
            quantiles_calcers.resize(quantiles_calcers_json.size());
            for (size_t i = 0; i < quantiles_calcers_json.size(); ++i) {
                RETURN_IF_ERROR(quantiles_calcers[i].from_json(quantiles_calcers_json[i]));
            }
            ASSIGN_OR_RETURN(std::vector<JsonValue> quantiles_,
                             CausalForestJsonHelper::get_json_array(json, "quantiles"));
            for (auto const& json : quantiles_) {
                ASSIGN_OR_RETURN(std::vector<JsonValue> quantile_json, CausalForestJsonHelper::get_json_array(json));
                std::vector<double> quantile(quantile_json.size());
                for (size_t i = 0; i < quantile_json.size(); ++i) {
                    ASSIGN_OR_RETURN(quantile[i], quantile_json[i].get_double());
                }
                quantiles.emplace_back(std::move(quantile));
            }
            ASSIGN_OR_RETURN(std::vector<JsonValue> split_nodes_json,
                             CausalForestJsonHelper::get_json_array(json, "split_nodes"));
            split_nodes.resize(split_nodes_json.size());
            for (size_t i = 0; i < split_nodes_json.size(); ++i) {
                ASSIGN_OR_RETURN(std::vector<JsonValue> split_node_json,
                                 CausalForestJsonHelper::get_json_array(split_nodes_json[i]));
                split_nodes[i].resize(split_node_json.size());
                for (size_t j = 0; j < split_node_json.size(); ++j) {
                    RETURN_IF_ERROR(split_nodes[i][j].from_json(split_node_json[j]));
                }
            }
            ASSIGN_OR_RETURN(father, CausalForestJsonHelper::get_json_value<int32_t>(json, "father"));
            return Status::OK();
        }

        void serialize(uint8_t*& data) const {
            uint8_t is_stop_ = is_stop;
            SerializeHelpers::serialize_all(data, sum_weight, total_outcome, total_treatment, total_instrument,
                                            total_outcome_treatment, num_samples, is_stop_, numerator, denominator,
                                            sum_node_z_squared, sum_node, num_node_small_z, father);
            size_t length = possible_split_vars.size();
            SerializeHelpers::serialize_all(data, length);
            for (const auto& var : possible_split_vars) SerializeHelpers::serialize_all(data, var);
            length = quantiles_calcers.size();
            SerializeHelpers::serialize_all(data, length);
            for (const auto& quantile : quantiles_calcers) quantile.serialize(data);
            length = quantiles.size();
            SerializeHelpers::serialize_all(data, length);
            for (const auto& quantile : quantiles) {
                size_t length = quantile.size();
                SerializeHelpers::serialize_all(data, length);
                for (const auto& val : quantile) SerializeHelpers::serialize_all(data, val);
            }
            length = split_nodes.size();
            SerializeHelpers::serialize_all(data, length);
            for (const auto& split_node : split_nodes) {
                size_t length = split_node.size();
                SerializeHelpers::serialize_all(data, length);
                for (const auto& node : split_node) node.serialize(data);
            }
        }

        void deserialize(const uint8_t*& data) {
            uint8_t is_stop_;
            SerializeHelpers::deserialize_all(data, sum_weight, total_outcome, total_treatment, total_instrument,
                                              total_outcome_treatment, num_samples, is_stop_, numerator, denominator,
                                              sum_node_z_squared, sum_node, num_node_small_z, father);
            is_stop = is_stop_;
            size_t length;
            SerializeHelpers::deserialize_all(data, length);
            possible_split_vars.resize(length);
            for (auto& var : possible_split_vars) SerializeHelpers::deserialize_all(data, var);
            SerializeHelpers::deserialize_all(data, length);
            quantiles_calcers.resize(length);
            for (auto& quantile : quantiles_calcers) quantile.deserialize(data);
            SerializeHelpers::deserialize_all(data, length);
            quantiles.resize(length);
            for (auto& quantile : quantiles) {
                size_t length;
                SerializeHelpers::deserialize_all(data, length);
                quantile.resize(length);
                for (auto& val : quantile) SerializeHelpers::deserialize_all(data, val);
            }
            SerializeHelpers::deserialize_all(data, length);
            split_nodes.resize(length);
            for (auto& split_node : split_nodes) {
                size_t length;
                SerializeHelpers::deserialize_all(data, length);
                split_node.resize(length);
                for (auto& node : split_node) node.deserialize(data);
            }
        }

        size_t serialized_size() const {
            size_t size = SerializeHelpers::serialized_size_all(
                    sum_weight, total_outcome, total_treatment, total_instrument, total_outcome_treatment, num_samples,
                    uint8_t{0}, numerator, denominator, sum_node_z_squared, sum_node, num_node_small_z, father);
            size += SerializeHelpers::serialized_size_all(possible_split_vars.size());
            for (const auto& var : possible_split_vars) size += SerializeHelpers::serialized_size_all(var);
            size += SerializeHelpers::serialized_size_all(quantiles_calcers.size());
            for (const auto& quantile : quantiles_calcers) size += quantile.serialized_size();
            size += SerializeHelpers::serialized_size_all(quantiles.size());
            for (const auto& quantile : quantiles) {
                size += SerializeHelpers::serialized_size_all(quantile.size());
                for (const auto& val : quantile) size += SerializeHelpers::serialized_size_all(val);
            }
            size += SerializeHelpers::serialized_size_all(split_nodes.size());
            for (const auto& split_node : split_nodes) {
                size += SerializeHelpers::serialized_size_all(split_node.size());
                for (const auto& node : split_node) size += node.serialized_size();
            }
            return size;
        }

        Status merge(const CalcNodeInfo& other, const CausalForestState& state) {
            if (quantiles_calcers.size() != other.quantiles_calcers.size()) {
                return Status::InvalidArgument("quantiles_calcers.size() != other.quantiles_calcers.size()");
            }
            if (quantiles.size() != other.quantiles.size()) {
                return Status::InvalidArgument("quantiles.size() != other.quantiles.size()");
            }
            if (split_nodes.size() != other.split_nodes.size()) {
                return Status::InvalidArgument("split_nodes.size() != other.split_nodes.size()");
            }
            if (state == CausalForestState::Init || state == CausalForestState::Honesty) {
                total_outcome += other.total_outcome;
                total_treatment += other.total_treatment;
                total_instrument += other.total_instrument;
                total_outcome_treatment += other.total_outcome_treatment;

                sum_weight += other.sum_weight;
                num_samples += other.num_samples;
                sum_node_z_squared += other.sum_node_z_squared;
            }

            is_stop = is_stop && other.is_stop;
            if (possible_split_vars.size() != other.possible_split_vars.size()) {
                return Status::InvalidArgument("possible_split_vars.size() != other.possible_split_vars.size()");
            }

            if (state == CausalForestState::CalcNumerAndDenom) {
                numerator += other.numerator;
                denominator += other.denominator;
            } else if (state == CausalForestState::FindBestSplitPre) {
                sum_node += other.sum_node;
                num_node_small_z += other.num_node_small_z;
                for (size_t i = 0; i < quantiles_calcers.size(); ++i)
                    quantiles_calcers[i].merge(other.quantiles_calcers[i]);
            } else if (state == CausalForestState::FindBestSplit) {
                for (size_t i = 0; i < split_nodes.size(); ++i) {
                    if (split_nodes[i].size() != other.split_nodes[i].size()) {
                        return Status::InvalidArgument("split_nodes[i].size() != other.split_nodes[i].size()");
                    }
                    for (size_t j = 0; j < split_nodes[i].size(); ++j) split_nodes[i][j].merge(other.split_nodes[i][j]);
                }
            }
            return Status::OK();
        }

        void updateStop(CausalForestState state, const TreeOptions& tree_options) {
            if (std::abs(sum_weight) <= 1e-16) // NOLINT
                is_stop = true;
            if (num_samples <= tree_options.min_node_size) is_stop = true;
            if (state >= CausalForestState::CalcNumerAndDenom && std::abs(denominator) <= 1e-10) is_stop = true;
        }

        double responses_by_sample(double response, double treatment, double instrument) const // NOLINT
        {
            double regularized_instrument = (1 - reduced_form_weight) * instrument + reduced_form_weight * treatment;
            double residual = (response - average_outcome()) -
                              local_average_treatment_effect() * (treatment - average_treatment());
            return (regularized_instrument - average_regularized_instrument()) * residual;
        }

        double average_outcome() const { return total_outcome / sum_weight; }       // NOLINT
        double average_treatment() const { return total_treatment / sum_weight; }   // NOLINT
        double average_instrument() const { return total_instrument / sum_weight; } // NOLINT
        double average_regularized_instrument() const {                             // NOLINT
            return (1 - reduced_form_weight) * average_instrument() +
                   reduced_form_weight * average_treatment(); // NOLINT
        }
        double local_average_treatment_effect() const { return numerator / denominator; } // NOLINT
        double weight_sum_node() const { return sum_weight; }                             // NOLINT
        double sum_node_z() const {
            return total_instrument;
        } // NOLINT
          //
        double size_node() const {
            return sum_node_z_squared - sum_node_z() * sum_node_z() / weight_sum_node();
        } // NOLINT
        //   double min_child_size = size_node * alpha;
        double min_child_size(double alpha) const { return size_node() * alpha; } // NOLINT
        //   double mean_z_node = sum_node_z / weight_sum_node;
        double mean_z_node() const { return sum_node_z() / weight_sum_node(); } // NOLINT

        Status find_best_split_value(const size_t var, size_t& best_var, double& best_value, double& best_decrease,
                                     bool& best_send_missing_left, const TreeOptions& tree_options) {
            std::vector<double>& possible_split_values = quantiles[var];

            // Try next variable if all equal for this
            if (possible_split_values.size() < 2) return Status::OK();

            size_t num_splits = possible_split_values.size() - 1;

            std::vector<size_t> counter(num_splits, 0);
            std::vector<double> weight_sums(num_splits, 0);
            std::vector<double> sums(num_splits, 0);
            std::vector<size_t> num_small_z(num_splits, 0);
            std::vector<double> sums_z(num_splits, 0);
            std::vector<double> sums_z_squared(num_splits, 0);

            size_t split_index = 0;
            if (possible_split_values.size() != split_nodes[var].size()) {
                return Status::InvalidArgument("possible_split_values.size() != split_nodes[var].size()");
            }

            for (size_t i = 0; i < num_splits - 1; i++) {
                double sample_value = possible_split_values[i];

                weight_sums[split_index] += split_nodes[var][i].weight_sums;
                sums[split_index] += split_nodes[var][i].sums;
                counter[split_index] += split_nodes[var][i].counter;

                sums_z[split_index] += split_nodes[var][i].sums_z;

                sums_z_squared[split_index] += split_nodes[var][i].sums_z_squared;
                num_small_z[split_index] += split_nodes[var][i].num_small_z;
                double next_sample_value = possible_split_values[i + 1];
                // if the next sample value is different, including the transition (..., NaN, Xij, ...)
                // then move on to the next bucket (all logical operators with NaN evaluates to false by default)
                if (sample_value != next_sample_value && !std::isnan(next_sample_value)) ++split_index;
            }

            size_t n_left = 0;
            double weight_sum_left = 0;
            double sum_left = 0;
            double sum_left_z = 0;
            double sum_left_z_squared = 0;
            size_t num_left_small_z = 0;

            double min_size = min_child_size(tree_options.alpha);

            // Compute decrease of impurity for each possible split.
            for (size_t i = 0; i < num_splits; ++i) {
                // not necessary to evaluate sending right when splitting on NaN.
                if (i == 0) continue;

                n_left += counter[i]; // 这组样本的数量
                num_left_small_z += num_small_z[i];
                weight_sum_left += weight_sums[i];
                sum_left += sums[i];
                sum_left_z += sums_z[i];
                sum_left_z_squared += sums_z_squared[i];

                // Skip this split if the left child does not contain enough
                // z values below and above the parent's mean.
                size_t num_left_large_z = n_left - num_left_small_z;
                if (num_left_small_z < tree_options.min_node_size || num_left_large_z < tree_options.min_node_size)
                    continue;

                // Stop if the right child does not contain enough z values below
                // and above the parent's mean.
                size_t n_right = num_samples - n_left;
                size_t num_right_small_z = num_node_small_z - num_left_small_z;
                size_t num_right_large_z = n_right - num_right_small_z;
                if (num_right_small_z < tree_options.min_node_size || num_right_large_z < tree_options.min_node_size)
                    break;

                // Calculate relevant quantities for the left child.
                double size_left = sum_left_z_squared - sum_left_z * sum_left_z / weight_sum_left;
                // Skip this split if the left child's variance is too small.
                if (size_left < min_size || (tree_options.imbalance_penalty > 0.0 && size_left == 0)) continue;

                // Calculate relevant quantities for the right child.
                double weight_sum_right = weight_sum_node() - weight_sum_left;
                double sum_right = sum_node - sum_left;
                double sum_right_z_squared = sum_node_z_squared - sum_left_z_squared;
                double sum_right_z = sum_node_z() - sum_left_z;
                double size_right = sum_right_z_squared - sum_right_z * sum_right_z / weight_sum_right;

                // Skip this split if the right child's variance is too small.
                if (size_right < min_size || (tree_options.imbalance_penalty > 0.0 && size_right == 0)) continue;

                // Calculate the decrease in impurity.
                double decrease = sum_left * sum_left / weight_sum_left + sum_right * sum_right / weight_sum_right;
                // Penalize splits that are too close to the edges of the data.
                decrease -= tree_options.imbalance_penalty * (1.0 / size_left + 1.0 / size_right);

                // Save this split if it is the best seen so far.
                if (decrease > best_decrease) {
                    best_value = possible_split_values[i];
                    best_var = var;
                    best_decrease = decrease;
                    best_send_missing_left = true;
                }
            }
            return Status::OK();
        }

        constexpr static double reduced_form_weight = 0;
    };

    struct NodeInfo {
        int16_t split_var = -1;
        double split_value = 0.0;
        uint16_t left_child = 0;
        uint16_t right_child = 0;
        bool is_leaf = false;
        std::optional<CalcNodeInfo> calc_node = std::nullopt;

        bool isNeedPure() const { return is_leaf && calc_node.has_value() && calc_node.value().num_samples <= 1; }

        void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
            vpack::ObjectBuilder ob =
                    field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
            builder.add("split_var", vpack::Value(split_var));
            builder.add("split_value", vpack::Value(split_value));
            builder.add("left_child", vpack::Value(left_child));
            builder.add("right_child", vpack::Value(right_child));
            builder.add("is_leaf", vpack::Value(is_leaf));
            if (calc_node.has_value()) {
                calc_node->to_json(builder, "calc_node");
            }
        }

        Status from_json(JsonValue const& json) {
            ASSIGN_OR_RETURN(split_var, CausalForestJsonHelper::get_json_value<int16_t>(json, "split_var"));
            ASSIGN_OR_RETURN(split_value, CausalForestJsonHelper::get_json_value<double>(json, "split_value"));
            ASSIGN_OR_RETURN(left_child, CausalForestJsonHelper::get_json_value<uint16_t>(json, "left_child"));
            ASSIGN_OR_RETURN(right_child, CausalForestJsonHelper::get_json_value<uint16_t>(json, "right_child"));
            ASSIGN_OR_RETURN(is_leaf, CausalForestJsonHelper::get_json_value<bool>(json, "is_leaf"));
            if (is_leaf) {
                calc_node = CalcNodeInfo();
                ASSIGN_OR_RETURN(auto calc_node_json,
                                 CausalForestJsonHelper::get_json_value<JsonValue>(json, "calc_node"));
                RETURN_IF_ERROR(calc_node->from_json(calc_node_json));
            }
            return Status::OK();
        }

        void serialize(uint8_t*& data) const {
            uint8_t is_leaf_ = is_leaf ? 1 : 0;
            SerializeHelpers::serialize_all(data, split_var, split_value, left_child, right_child, is_leaf_);
            if (calc_node.has_value()) {
                calc_node->serialize(data);
            }
        }

        void deserialize(const uint8_t*& data) {
            uint8_t is_leaf_;
            SerializeHelpers::deserialize_all(data, split_var, split_value, left_child, right_child, is_leaf_);
            is_leaf = is_leaf_ == 1;
            if (is_leaf) {
                calc_node = CalcNodeInfo();
                calc_node->deserialize(data);
            }
        }

        size_t serialized_size() const {
            size_t size =
                    SerializeHelpers::serialized_size_all(split_var, split_value, left_child, right_child, uint8_t{0});
            if (calc_node.has_value()) {
                size += calc_node->serialized_size();
            }
            return size;
        }

        Status merge(const NodeInfo& other, const CausalForestState& state) {
            if (split_var != other.split_var) {
                return Status::InternalError("split_var != other.split_var");
            }
            if (calc_node.has_value() != other.calc_node.has_value()) {
                return Status::InternalError("calc_node.has_value() != other.calc_node.has_value()");
            }
            if (calc_node.has_value()) {
                RETURN_IF_ERROR(calc_node->merge(other.calc_node.value(), state));
            }
            return Status::OK();
        }
    };

    Tree() = default;

    Tree(uint64_t bucket_num_, uint64_t honesty_bucket_num_, uint64_t random_seed_, const ForestOptions& forest_options,
         const TreeOptions& tree_options)
            : bucket_num(bucket_num_),
              random_seed(random_seed_),
              random_number_generator(random_seed),
              honesty_bucket_num(honesty_bucket_num_) {
        calc_index = 0;
        nodes.emplace_back(NodeInfo());
        nodes[0].calc_node = CalcNodeInfo();
        nodes[0].calc_node->possible_split_vars = createSplitVariable(forest_options, tree_options);
        nodes[0].is_leaf = true;
    }

    bool isBelongToHonestyBucket(uint64_t bucket_id) const {
        bucket_id ^= random_seed;
        return bucket_id % TOTAL_BUCKET_NUM <= honesty_bucket_num;
    }

    bool isBelongToBucket(uint64_t bucket_id) const {
        bucket_id ^= random_seed * bucket_id + (bucket_id << 6) + (bucket_id >> 2);
        return bucket_id % TOTAL_BUCKET_NUM <= bucket_num;
    }

    Status predict(std::vector<double> const& data, std::vector<double>& result) const {
        size_t root = 0;
        while (root < nodes.size() && !nodes[root].is_leaf) {
            const auto& value = data[nodes[root].split_var];
            if (value > nodes[root].split_value + 1e-6) {
                root = nodes[root].right_child;
            } else
                root = nodes[root].left_child;
        }
        if (root >= nodes.size()) {
            return Status::InvalidArgument("Wrong tree structure.");
        }
        if (!nodes[root].calc_node.has_value()) {
            return Status::InvalidArgument("no calc node");
        }
        const auto& calc_node = nodes[root].calc_node.value();
        result[0] += calc_node.total_outcome / calc_node.num_samples;
        result[1] += calc_node.total_treatment / calc_node.num_samples;
        result[2] += calc_node.total_instrument / calc_node.num_samples;
        result[3] += calc_node.total_outcome_treatment / calc_node.num_samples;
        result[4] += calc_node.sum_node_z_squared / calc_node.num_samples;
        result[5] = result[4];
        result[6] += calc_node.sum_weight / calc_node.num_samples;
        return Status::OK();
    }

    Status add(double y, bool treatment, double weight, std::vector<double> const& data,
               const ForestOptions& forest_options, const TreeOptions& tree_options) {
        if (forest_options.state == CausalForestState::Init) {
            addInit(y, treatment, weight, data, forest_options, tree_options);
        } else if (forest_options.state == CausalForestState::CalcNumerAndDenom) {
            addCalcNumerAndDenom(y, treatment, weight, data, forest_options, tree_options);
        } else if (forest_options.state == CausalForestState::FindBestSplitPre) {
            RETURN_IF_ERROR(addFindBestSplitPre(y, treatment, weight, data, forest_options, tree_options));
        } else if (forest_options.state == CausalForestState::FindBestSplit) {
            RETURN_IF_ERROR(addFindBestSplit(y, treatment, weight, data, forest_options, tree_options));
        } else if (forest_options.state == CausalForestState::Honesty) {
            addHonesty(y, treatment, weight, data, forest_options, tree_options);
        } else if (forest_options.state != CausalForestState::Finish) {
            return Status::InvalidArgument("Tree::add: unknown forest_options.state");
        }
        return Status::OK();
    }

    void addInit(double y, bool treatment, double weight, std::vector<double> const& data,
                 const ForestOptions& forest_options, const TreeOptions& tree_options) {
        auto& calc_node = nodes[0].calc_node.value();
        calc_node.total_outcome += weight * data[forest_options.outcome_index];
        calc_node.total_treatment += weight * data[forest_options.treatment_index];
        calc_node.total_outcome_treatment +=
                weight * data[forest_options.outcome_index] * data[forest_options.treatment_index];
        calc_node.total_instrument += weight * data[forest_options.instrument_index];
        calc_node.sum_weight += weight;
        calc_node.num_samples++;
        calc_node.sum_node_z_squared += weight * std::pow(data[forest_options.instrument_index], 2);
    }

    void addCalcNumerAndDenom(double y, bool treatment, double weight, std::vector<double> const& data,
                              const ForestOptions& forest_options, const TreeOptions& tree_options) {
        size_t calc_node_index;
        if (!getCalcNode(y, treatment, weight, data, calc_node_index)) return;
        auto& node = nodes[calc_node_index].calc_node.value();
        double outcome = data[forest_options.outcome_index];
        double instrument = data[forest_options.instrument_index];
        double regularized_instrument =
                (1 - node.reduced_form_weight) * instrument + node.reduced_form_weight * static_cast<double>(treatment);

        node.numerator += weight * (regularized_instrument - node.average_regularized_instrument()) *
                          (outcome - node.average_outcome());
        node.denominator += weight * (regularized_instrument - node.average_regularized_instrument()) *
                            (static_cast<double>(treatment) - node.average_treatment());
    }

    Status addFindBestSplitPre(double y, bool treatment, double weight, std::vector<double> const& data,
                               const ForestOptions& forest_options, const TreeOptions& tree_options) {
        size_t calc_node_index;
        if (!getCalcNode(y, treatment, weight, data, calc_node_index)) return Status::OK();
        auto& node = nodes[calc_node_index].calc_node.value();
        double outcome = data[forest_options.outcome_index];
        double instrument = data[forest_options.instrument_index];
        node.sum_node += weight * node.responses_by_sample(outcome, static_cast<double>(treatment), instrument);

        if (instrument < node.mean_z_node()) node.num_node_small_z++;

        if (node.possible_split_vars.size() != node.quantiles_calcers.size()) {
            return Status::InvalidArgument(
                    "Tree::addFindBestSplitPre: node.possible_split_vars.size() != node.quantiles_calcers.size()");
        }
        for (size_t i = 0; i < node.possible_split_vars.size(); ++i) {
            auto var = node.possible_split_vars[i];
            auto& quantiles_calcer = node.quantiles_calcers[i];
            quantiles_calcer.add(data[var]);
        }
        return Status::OK();
    }

    Status addFindBestSplit(double y, bool treatment, double weight, std::vector<double> const& data,
                            const ForestOptions& forest_options, const TreeOptions& tree_options) {
        size_t calc_node_index;
        if (!getCalcNode(y, treatment, weight, data, calc_node_index)) {
            return Status::OK();
        }
        auto& node = nodes[calc_node_index].calc_node.value();
        const double outcome = data[forest_options.outcome_index];
        const double instrument = data[forest_options.instrument_index];

        for (size_t i = 0; i < node.possible_split_vars.size(); ++i) {
            auto var = node.possible_split_vars[i];
            auto value = data[var];
            auto& quantile = node.quantiles[i];
            int64_t left = 0, right = static_cast<int64_t>(quantile.size()) - 1, res = -1;
            while (left <= right) {
                auto mid = (left + right) / 2;
                if (value <= quantile[mid] + 1e-6) {
                    res = mid;
                    right = mid - 1;
                } else
                    left = mid + 1;
            }

            if (res == -1) res = quantile.size() - 1;
            auto& split_node = node.split_nodes[i][res];
            split_node.weight_sums += weight;
            split_node.counter++;
            split_node.sums += weight * node.responses_by_sample(outcome, static_cast<double>(treatment), instrument);
            split_node.sums_z += weight * instrument;
            split_node.sums_z_squared += weight * instrument * instrument;
            split_node.total_outcome += weight * outcome;
            split_node.total_treatment += weight * static_cast<double>(treatment);
            split_node.total_outcome_treatment += weight * outcome * static_cast<double>(treatment);
            if (instrument < node.mean_z_node()) split_node.num_small_z++;
        }
        return Status::OK();
    }

    void addHonesty(double y, bool treatment, double weight, std::vector<double> const& data,
                    const ForestOptions& forest_options, const TreeOptions& tree_options) {
        size_t calc_node_index;
        if (!getCalcNode(y, treatment, weight, data, calc_node_index)) return;
        auto& node = nodes[calc_node_index].calc_node.value(); // NOLINT
        node.total_outcome += weight * data[forest_options.outcome_index];
        node.total_treatment += weight * data[forest_options.treatment_index];
        node.total_outcome_treatment +=
                weight * data[forest_options.outcome_index] * data[forest_options.treatment_index];
        node.total_instrument += weight * data[forest_options.instrument_index];
        node.sum_weight += weight;
        node.num_samples++;
        node.sum_node_z_squared += weight * pow(data[forest_options.instrument_index], 2);
    }

    void initHonesty(TreeOptions const& tree_options) {
        if (!tree_options.honesty) {
            return;
        }
        for (auto& node : nodes) {
            if (node.calc_node.has_value()) {
                node.calc_node = CalcNodeInfo();
            }
        }
    }

    bool getCalcNode(double y, bool treatment, double weight, std::vector<double> const& data, size_t& res) {
        size_t root = 0;
        while (root < nodes.size() && !nodes[root].is_leaf) {
            const auto& value = data[nodes[root].split_var];
            if (value > nodes[root].split_value + 1e-6) {
                root = nodes[root].right_child;
            } else
                root = nodes[root].left_child;
        }
        if (root >= nodes.size()) return false;
        if (!nodes[root].calc_node.has_value() || nodes[root].calc_node.value().is_stop) return false;
        res = root;
        return true;
    }

    void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
        vpack::ObjectBuilder ob =
                field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
        builder.add("bucket_num", vpack::Value(bucket_num));
        builder.add("honesty_bucket_num", vpack::Value(honesty_bucket_num));
        builder.add("random_seed", vpack::Value(random_seed));
        builder.add("calc_index", vpack::Value(calc_index));
        {
            vpack::ArrayBuilder ab(&builder, "nodes");
            for (auto const& node : nodes) {
                node.to_json(builder);
            }
        }
        // std::ostringstream ss;
        // ss << random_number_generator;
        // builder.add("random_number_generator", vpack::Value(ss.str()));
    }

    Status from_json(JsonValue const& json) {
        ASSIGN_OR_RETURN(bucket_num, CausalForestJsonHelper::get_json_value<uint64_t>(json, "bucket_num"));
        ASSIGN_OR_RETURN(honesty_bucket_num,
                         CausalForestJsonHelper::get_json_value<uint64_t>(json, "honesty_bucket_num"));
        ASSIGN_OR_RETURN(random_seed, CausalForestJsonHelper::get_json_value<uint64_t>(json, "random_seed"));
        ASSIGN_OR_RETURN(calc_index, CausalForestJsonHelper::get_json_value<size_t>(json, "calc_index"));
        ASSIGN_OR_RETURN(auto nodes_json, CausalForestJsonHelper::get_json_array(json, "nodes"));
        nodes.resize(nodes_json.size());
        for (size_t i = 0; i < nodes_json.size(); ++i) {
            RETURN_IF_ERROR(nodes[i].from_json(nodes_json[i]));
        }
        random_number_generator.seed(random_seed);
        // ASSIGN_OR_RETURN(auto random_number_generator_str,
        //                  CausalForestJsonHelper::get_json_value<std::string>(json, "random_number_generator"));
        // std::istringstream istr(random_number_generator_str);
        // istr >> random_number_generator;
        return Status::OK();
    }

    void serialize(uint8_t*& data) const {
        SerializeHelpers::serialize_all(data, bucket_num, honesty_bucket_num, random_seed, calc_index);
        size_t length = nodes.size();
        SerializeHelpers::serialize_all(data, length);
        for (const auto& node : nodes) node.serialize(data);
    }

    void deserialize(const uint8_t*& data) {
        SerializeHelpers::deserialize_all(data, bucket_num, honesty_bucket_num, random_seed, calc_index);
        size_t length;
        SerializeHelpers::deserialize_all(data, length);
        nodes.resize(length);
        for (auto& node : nodes) node.deserialize(data);
        random_number_generator.seed(random_seed);
    }

    size_t serialized_size() const {
        size_t size = SerializeHelpers::serialized_size_all(bucket_num, honesty_bucket_num, random_seed, calc_index);
        size += SerializeHelpers::serialized_size_all(nodes.size());
        for (const auto& node : nodes) size += node.serialized_size();
        return size;
    }

    static constexpr uint64_t TOTAL_BUCKET_NUM = 1000;

    StatusOr<bool> initTrain(const ForestOptions& forest_options, const TreeOptions&) {
        if (forest_options.state == CausalForestState::FindBestSplitPre) {
            for (size_t i = calc_index; i < nodes.size(); ++i) {
                if (!nodes[i].calc_node.has_value()) {
                    return Status::InvalidArgument("node is not calc node");
                }
                auto& node = nodes[i].calc_node.value(); // NOLINT
                node.quantiles_calcers.resize(
                        node.possible_split_vars.size(),
                        CausalQuantileTDigest<float>(forest_options.max_centroids, forest_options.max_unmerged,
                                                     forest_options.epsilon));
            }
        }
        if (forest_options.state == CausalForestState::FindBestSplit) {
            for (size_t i = calc_index; i < nodes.size(); ++i) {
                if (!nodes[i].calc_node.has_value()) {
                    return Status::InvalidArgument("node is not calc node");
                }
                auto& node = nodes[i].calc_node.value(); // NOLINT
                for (auto& quantile : node.quantiles) {
                    node.split_nodes.emplace_back(std::vector<SplitNode>(quantile.size()));
                }
                node.quantiles_calcers.clear();
            }
        }
        bool stop = false;
        if (forest_options.state == CausalForestState::CalcNumerAndDenom && calc_index == nodes.size()) stop = true;
        return stop;
    }

    Status afterTrain(const ForestOptions& forest_options, const TreeOptions& tree_options) {
        auto old_state = forest_options.state;

        for (size_t i = calc_index; i < nodes.size(); i++) {
            if (nodes[i].calc_node.has_value()) {
                nodes[i].calc_node.value().updateStop(forest_options.state, tree_options);
            }
        }

        if (old_state == CausalForestState::FindBestSplit) {
            size_t new_calc_index = nodes.size();
            size_t nodes_size = nodes.size();
            for (size_t j = calc_index; j < nodes_size; ++j) {
                if (!nodes[j].calc_node.has_value()) {
                    return Status::InvalidArgument("node is not calc node");
                }
                auto& calc_node = nodes[j].calc_node.value(); // NOLINT
                size_t best_var = 0;
                double best_value = 0;
                double best_decrease = 0.0;
                bool best_send_missing_left = true;

                for (size_t i = 0; i < calc_node.possible_split_vars.size(); ++i) // 对于每个可能的分裂变量
                    RETURN_IF_ERROR(calc_node.find_best_split_value(i, best_var, best_value, best_decrease,
                                                                    best_send_missing_left, tree_options));
                if (best_decrease <= 0.0 || calc_node.is_stop) {
                    transformToLeaf(nodes[j]);
                    continue;
                }
                RETURN_IF_ERROR(splitNode(nodes[j], best_var, best_value, forest_options, tree_options));
            }
            calc_index = new_calc_index;
        }

        // 获取分位点
        if (forest_options.state == CausalForestState::FindBestSplitPre) {
            for (size_t i = calc_index; i < nodes.size(); ++i) {
                if (!nodes[i].calc_node.has_value()) {
                    return Status::InvalidArgument("node is not calc node");
                }
                auto& node = nodes[i].calc_node.value(); // NOLINT
                for (auto& quantiles_calcer : node.quantiles_calcers) {
                    node.quantiles.emplace_back(quantiles_calcer.getQuantiles(tree_options.quantile_size));
                }
                node.quantiles_calcers.clear();
            }
        }

        if (forest_options.state == CausalForestState::Honesty) {
            pureNode();
        }
        return Status::OK();
    }

    void pureNode() {
        for (int64_t i = static_cast<int64_t>(nodes.size()) - 1; i >= 0; --i) {
            auto& node = nodes[i];
            if (node.is_leaf) continue;
            auto& left_child = nodes[node.left_child];
            auto& right_child = nodes[node.right_child];
            if (left_child.isNeedPure() || right_child.isNeedPure()) {
                node.is_leaf = true;
                if (!left_child.isNeedPure())
                    node = left_child;
                else if (!right_child.isNeedPure())
                    node = right_child;
                else
                    node.calc_node = CalcNodeInfo();
            }
        }
    }

    static void transformToLeaf(NodeInfo& node) {
        node.split_var = -1;
        node.is_leaf = true;
        if (!node.calc_node.has_value()) return;
        if (node.calc_node.has_value()) {
            auto& calc_node = node.calc_node.value(); // NOLINT
            calc_node.quantiles.clear();
            calc_node.split_nodes.clear();
            calc_node.is_stop = true;
        }
    }

    Status splitNode(NodeInfo& node, size_t best_var, double best_value, const ForestOptions& forest_options,
                     const TreeOptions& tree_options) {
        auto& calc_node = node.calc_node.value(); // NOLINT
        node.split_var = calc_node.possible_split_vars[best_var];
        node.split_value = best_value;
        node.is_leaf = false;
        NodeInfo left_node;
        NodeInfo right_node;
        left_node.is_leaf = true;
        right_node.is_leaf = true;
        left_node.calc_node = CalcNodeInfo();
        right_node.calc_node = CalcNodeInfo();

        if (calc_node.quantiles[best_var].size() != calc_node.split_nodes[best_var].size()) {
            return Status::InvalidArgument("quantiles size is not equal to split_nodes size");
        }
        for (size_t i = 0; i < calc_node.quantiles[best_var].size(); ++i) {
            auto& value = calc_node.quantiles[best_var][i];
            if (value > best_value + 1e-6) {
                auto& new_calc_node = right_node.calc_node.value(); // NOLINT
                new_calc_node.sum_weight += calc_node.split_nodes[best_var][i].weight_sums;
                new_calc_node.num_samples += calc_node.split_nodes[best_var][i].counter;
                new_calc_node.sum_node_z_squared += calc_node.split_nodes[best_var][i].sums_z_squared;
                new_calc_node.total_instrument += calc_node.split_nodes[best_var][i].sums_z;
                new_calc_node.total_outcome += calc_node.split_nodes[best_var][i].total_outcome;
                new_calc_node.total_treatment += calc_node.split_nodes[best_var][i].total_treatment;
                new_calc_node.total_outcome_treatment += calc_node.split_nodes[best_var][i].total_outcome_treatment;
            } else {
                auto& new_calc_node = left_node.calc_node.value(); // NOLINT
                new_calc_node.sum_weight += calc_node.split_nodes[best_var][i].weight_sums;
                new_calc_node.num_samples += calc_node.split_nodes[best_var][i].counter;
                new_calc_node.sum_node_z_squared += calc_node.split_nodes[best_var][i].sums_z_squared;
                new_calc_node.total_instrument += calc_node.split_nodes[best_var][i].sums_z;
                new_calc_node.total_outcome += calc_node.split_nodes[best_var][i].total_outcome;
                new_calc_node.total_treatment += calc_node.split_nodes[best_var][i].total_treatment;
                new_calc_node.total_outcome_treatment += calc_node.split_nodes[best_var][i].total_outcome_treatment;
            }
        }
        node.calc_node.reset();
        node.left_child = nodes.size();
        node.right_child = nodes.size() + 1;
        left_node.calc_node.value().possible_split_vars = createSplitVariable(forest_options, tree_options);
        right_node.calc_node.value().possible_split_vars = createSplitVariable(forest_options, tree_options);
        nodes.push_back(left_node);
        nodes.push_back(right_node);
        return Status::OK();
    }

    Status merge(const Tree& other, const CausalForestState& state) {
        if (nodes.size() != other.nodes.size()) {
            return Status::InternalError("nodes.size() != other.nodes.size()");
        }
        for (size_t i = (state == CausalForestState::Honesty ? 0 : calc_index); i < nodes.size(); ++i) {
            RETURN_IF_ERROR(nodes[i].merge(other.nodes[i], state));
        }
        return Status::OK();
    }

    std::vector<size_t> createSplitVariable(const ForestOptions& forest_options, const TreeOptions& tree_options) {
        // Randomly select an mtry for this tree based on the overall setting.
        size_t num_independent_variables =
                forest_options.arguments_size - forest_options.disallowed_split_variables.size();

        std::poisson_distribution<size_t> distribution(static_cast<double>(tree_options.mtry));
        size_t mtry_sample = distribution(random_number_generator);
        size_t split_mtry = std::max<size_t>(std::min<size_t>(mtry_sample, num_independent_variables), 1uL);

        std::vector<size_t> result;
        draw(result, forest_options.arguments_size, forest_options.disallowed_split_variables, split_mtry);

        return result;
    }

    void drawFisherYates(std::vector<size_t>& result, // NOLINT
                         size_t max, const std::set<size_t>& skip, size_t num_samples) {
        // Populate result vector with 0,...,max-1
        result.resize(max);
        std::iota(result.begin(), result.end(), 0);

        // Remove values that are to be skipped
        std::for_each(skip.rbegin(), skip.rend(), [&](size_t i) { result.erase(result.begin() + i); });

        // Draw without replacement using Fisher Yates algorithm
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (size_t i = 0; i < num_samples; ++i) {
            auto j = static_cast<size_t>(i + distribution(random_number_generator) * (max - skip.size() - i));
            std::swap(result[i], result[j]);
        }

        result.resize(num_samples);
    }

    void drawSimple(std::vector<size_t>& result, // NOLINT
                    size_t max, const std::set<size_t>& skip, size_t num_samples) {
        result.resize(num_samples);

        // Set all to not selected
        std::vector<bool> temp;
        temp.resize(max, false);

        std::uniform_int_distribution<size_t> unif_dist(0, max - 1 - skip.size());
        for (size_t i = 0; i < num_samples; ++i) {
            size_t draw;
            do {
                draw = unif_dist(random_number_generator);
                for (const auto& skip_value : skip) {
                    if (draw >= skip_value) {
                        ++draw;
                    }
                }
            } while (temp[draw]);
            temp[draw] = true;
            result[i] = draw;
        }
    }

    void draw(std::vector<size_t>& result, size_t max, const std::set<size_t>& skip, size_t num_samples) {
        if (num_samples < max / 10) {
            drawSimple(result, max, skip, num_samples);
        } else {
            drawFisherYates(result, max, skip, num_samples);
        }
    }

    uint64_t bucket_num;
    uint64_t random_seed;
    std::vector<NodeInfo> nodes;
    std::mt19937_64 random_number_generator;
    size_t calc_index = 0;
    size_t honesty_bucket_num;
};

class ForestTrainer {
public:
    ForestTrainer() = default;

    Status init(const JsonValue& context, uint64_t arguments_size_, bool is_predict = false) {
        // tree options
        uint32_t mtry = 3;
        uint32_t min_node_size = 1;
        bool honesty = true;
        double honesty_fraction = 0.5;
        bool honesty_prune_leaves = false;
        double alpha = 0.05;
        double imbalance_penalty = 0.0;
        size_t quantile_size = 100;

        // forest options
        uint32_t num_trees = 50;
        size_t ci_group_size = 1;
        double sample_fraction = 0.35;
        uint32_t random_seed = 42;
        int32_t weight_index = -1;
        int32_t outcome_index = -1;
        int32_t treatment_index = -1;
        int32_t instrument_index = -1;
        CausalForestState state = CausalForestState::Init;

        size_t max_centroids = 2048;
        size_t max_unmerged = 2048;
        double epsilon = 0.01;

        auto st_or_trainer = CausalForestJsonHelper::json_extract(context, "trainer");
        if (!st_or_trainer.ok()) {
            ASSIGN_OR_RETURN(mtry, CausalForestJsonHelper::get_json_value<uint32_t>(context, "mtry"));
            ASSIGN_OR_RETURN(min_node_size, CausalForestJsonHelper::get_json_value<uint32_t>(context, "min_node_size"));
            ASSIGN_OR_RETURN(honesty, CausalForestJsonHelper::get_json_value<bool>(context, "honesty"));
            ASSIGN_OR_RETURN(honesty_fraction,
                             CausalForestJsonHelper::get_json_value<double>(context, "honesty_fraction"));
            ASSIGN_OR_RETURN(honesty_prune_leaves,
                             CausalForestJsonHelper::get_json_value<bool>(context, "honesty_prune_leaves"));
            ASSIGN_OR_RETURN(alpha, CausalForestJsonHelper::get_json_value<double>(context, "alpha"));
            ASSIGN_OR_RETURN(imbalance_penalty,
                             CausalForestJsonHelper::get_json_value<double>(context, "imbalance_penalty"));
            ASSIGN_OR_RETURN(num_trees, CausalForestJsonHelper::get_json_value<uint32_t>(context, "num_trees"));
            ASSIGN_OR_RETURN(ci_group_size, CausalForestJsonHelper::get_json_value<size_t>(context, "ci_group_size"));
            ASSIGN_OR_RETURN(sample_fraction,
                             CausalForestJsonHelper::get_json_value<double>(context, "sample_fraction"));

            ASSIGN_OR_RETURN(random_seed, CausalForestJsonHelper::get_json_value<uint32_t>(context, "random_seed"));
            ASSIGN_OR_RETURN(weight_index, CausalForestJsonHelper::get_json_value<int32_t>(context, "weight_index"));
            ASSIGN_OR_RETURN(outcome_index, CausalForestJsonHelper::get_json_value<int32_t>(context, "outcome_index"));
            ASSIGN_OR_RETURN(treatment_index,
                             CausalForestJsonHelper::get_json_value<int32_t>(context, "treatment_index"));
            ASSIGN_OR_RETURN(instrument_index,
                             CausalForestJsonHelper::get_json_value<int32_t>(context, "instrument_index"));
            ASSIGN_OR_RETURN(quantile_size, CausalForestJsonHelper::get_json_value<size_t>(context, "quantile_size"));

            ASSIGN_OR_RETURN(max_centroids, CausalForestJsonHelper::get_json_value<size_t>(context, "max_centroids"));
            ASSIGN_OR_RETURN(max_unmerged, CausalForestJsonHelper::get_json_value<size_t>(context, "max_unmerged"));
            ASSIGN_OR_RETURN(epsilon, CausalForestJsonHelper::get_json_value<double>(context, "epsilon"));

            if (instrument_index == -1) instrument_index = treatment_index;
            state = CausalForestState::Init;
            tree_options = TreeOptions(mtry, quantile_size, min_node_size, honesty, honesty_fraction,
                                       honesty_prune_leaves, alpha, imbalance_penalty);
            RETURN_IF_ERROR(forest_options.init(num_trees, ci_group_size, sample_fraction, random_seed, weight_index,
                                                outcome_index, treatment_index, instrument_index, state,
                                                arguments_size_, max_centroids, max_unmerged, epsilon));

            std::mt19937_64 random_number_generator(random_seed);
            std::uniform_int_distribution<uint32_t> udist;
            for (size_t i = 0; i < forest_options.num_trees; ++i) {
                trees.emplace_back(Tree(static_cast<int64_t>(sample_fraction * Tree::TOTAL_BUCKET_NUM),
                                        static_cast<int64_t>(honesty_fraction * Tree::TOTAL_BUCKET_NUM),
                                        udist(random_number_generator), forest_options, tree_options));
            }
        } else {
            RETURN_IF_ERROR(from_json(st_or_trainer.value()));
            uint32_t stop_tree = 0;
            if (!is_predict) {
                for (auto& tree : trees) {
                    ASSIGN_OR_RETURN(bool train_over, tree.initTrain(forest_options, tree_options));
                    if (train_over) {
                        stop_tree++;
                    }
                }
            }
            if (stop_tree > trees.size() / 2) {
                return Status::InvalidArgument("train over");
            }
        }
        if (is_predict) {
            initHonesty();
        }
        is_init_ = false;
        return Status::OK();
    }

    Status add(double y, bool treatment, double weight, std::vector<double> const& data) {
        uint64_t hash_value = 0;
        for (auto x : data) {
            hash_value ^= std::hash<double>()(x) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        }
        if (forest_options.get_max_index() >= data.size()) {
            return Status::InvalidArgument("given array is too short: data.size() <= forest_options.get_max_index()");
        }
        for (auto& tree : trees) {
            if (tree.isBelongToBucket(hash_value)) {
                if (!tree_options.honesty ||
                    (forest_options.state != CausalForestState::Honesty && tree.isBelongToHonestyBucket(hash_value)) ||
                    (forest_options.state == CausalForestState::Honesty && !tree.isBelongToHonestyBucket(hash_value)))
                    RETURN_IF_ERROR(tree.add(y, treatment, weight, data, forest_options, tree_options));
            }
        }
        return Status::OK();
    }

    void initHonesty() {
        if (!tree_options.honesty) {
            forest_options.state = CausalForestState::Finish;
            return;
        }
        if (forest_options.state == CausalForestState::Honesty) {
            return;
        }
        for (auto& tree : trees) {
            tree.initHonesty(tree_options);
        }
        forest_options.state = CausalForestState::Honesty;
    }

    void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
        vpack::ObjectBuilder ob =
                field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
        forest_options.to_json(builder, "forest_options");
        tree_options.to_json(builder, "tree_options");
        {
            vpack::ArrayBuilder ab(&builder, "trees");
            for (auto const& tree : trees) {
                tree.to_json(builder);
            }
        }
        builder.add("is_init", vpack::Value(is_init_));
    }

    Status from_json(JsonValue const& json) {
        ASSIGN_OR_RETURN(auto forest_options_json, CausalForestJsonHelper::json_extract(json, "forest_options"));
        RETURN_IF_ERROR(forest_options.from_json(forest_options_json));
        ASSIGN_OR_RETURN(auto tree_options_json, CausalForestJsonHelper::json_extract(json, "tree_options"));
        RETURN_IF_ERROR(tree_options.from_json(tree_options_json));
        ASSIGN_OR_RETURN(auto trees_json, CausalForestJsonHelper::get_json_array(json, "trees"));
        trees.resize(trees_json.size());
        for (size_t i = 0; i < trees_json.size(); ++i) {
            RETURN_IF_ERROR(trees[i].from_json(trees_json[i]));
        }
        ASSIGN_OR_RETURN(is_init_, CausalForestJsonHelper::get_json_value<bool>(json, "is_init"));
        DCHECK(!is_init_);
        return Status::OK();
    }

    void serialize(uint8_t*& data) const {
        forest_options.serialize(data);
        tree_options.serialize(data);
        size_t length = trees.size();
        SerializeHelpers::serialize_all(data, length);
        for (const auto& tree : trees) tree.serialize(data);
        uint8_t is_init = is_init_ ? 1 : 0;
        SerializeHelpers::serialize_all(data, is_init);
    }

    void deserialize(const uint8_t*& data) {
        forest_options.deserialize(data);
        tree_options.deserialize(data);
        size_t length;
        SerializeHelpers::deserialize_all(data, length);
        trees.resize(length);
        for (auto& tree : trees) tree.deserialize(data);
        uint8_t is_init;
        SerializeHelpers::deserialize_all(data, is_init);
        is_init_ = is_init ? true : false;
        DCHECK(!is_init_);
    }

    size_t serialized_size() const {
        size_t size = forest_options.serialized_size() + tree_options.serialized_size();
        size += SerializeHelpers::serialized_size_all(trees.size());
        for (const auto& tree : trees) size += tree.serialized_size();
        size += SerializeHelpers::serialized_size_all(uint8_t{0});
        return size;
    }

    Status afterTrain() {
        auto& state = forest_options.state;

        for (auto& tree : trees) RETURN_IF_ERROR(tree.afterTrain(forest_options, tree_options));

        if (state == CausalForestState::Init)
            state = CausalForestState::CalcNumerAndDenom;
        else if (state == CausalForestState::CalcNumerAndDenom)
            state = CausalForestState::FindBestSplitPre;
        else if (state == CausalForestState::FindBestSplitPre)
            state = CausalForestState::FindBestSplit;
        else if (state == CausalForestState::FindBestSplit)
            state = CausalForestState::CalcNumerAndDenom;
        return Status::OK();
    }

    Status merge(const ForestTrainer& other) {
        DCHECK(forest_options.state == other.forest_options.state);
        DCHECK(trees.size() == other.trees.size());
        DCHECK(is_init_ == other.is_init_);
        for (size_t i = 0; i < trees.size(); ++i) {
            RETURN_IF_ERROR(trees[i].merge(other.trees[i], forest_options.state));
        }
        return Status::OK();
    }

    uint16_t getNumTrees() const { return forest_options.num_trees; }

    Status predict(std::vector<double> const& data, std::vector<double>& average_value) const {
        if (data.size() != forest_options.arguments_size) {
            LOG(WARNING) << "data.size() != forest_options.arguments_size";
            return Status::InvalidArgument(
                    fmt::format("Wrong number of arguments for predict, input size: {}, "
                                "expected size: {}",
                                data.size(), forest_options.arguments_size));
        }
        for (auto& tree : trees) {
            RETURN_IF_ERROR(tree.predict(data, average_value));
        }
        return Status::OK();
    }

    bool is_init() const { return is_init_; }

private:
    std::vector<Tree> trees;

    ForestOptions forest_options;

    TreeOptions tree_options;

    bool is_init_{true};
};

struct CausalForestData {
    CausalForestData() = default;

    explicit CausalForestData(ForestTrainer trainer_) : trainer(std::move(trainer_)) {}
    explicit CausalForestData(const uint8_t*& data) { deserialize(data); }

    Status add(double y, bool treatment, double weight, std::vector<double> const& data) {
        return trainer.add(y, treatment, weight, data);
    }

    Status merge(const CausalForestData& other) { return trainer.merge(other.trainer); }

    bool is_uninitialized() const { return trainer.is_init(); }

    Status init(JsonValue const& model, int num_args, bool is_predict = false) {
        RETURN_IF_ERROR(trainer.init(model, num_args, is_predict));
        return Status::OK();
    }

    void to_json(vpack::Builder& builder, std::string const& field_name = "") const {
        vpack::ObjectBuilder ob =
                field_name == "" ? vpack::ObjectBuilder(&builder) : vpack::ObjectBuilder(&builder, field_name);
        trainer.to_json(builder, "trainer");
    }

    Status from_json(JsonValue const& json) {
        ASSIGN_OR_RETURN(auto trainer_json, CausalForestJsonHelper::json_extract(json, "trainer"));
        return trainer.from_json(trainer_json);
    }

    void serialize(uint8_t*& data) const { trainer.serialize(data); }

    void deserialize(const uint8_t*& data) { trainer.deserialize(data); }

    size_t serialized_size() const { return trainer.serialized_size(); }

    mutable ForestTrainer trainer;
};

class AggregateFunctionCausalForest
        : public AggregateFunctionBatchHelper<CausalForestData, AggregateFunctionCausalForest> {
public:
    void update(FunctionContext* ctx, const Column** columns, AggDataPtr __restrict state,
                size_t row_num) const override {
        double y{};
        const Column* y_col = columns[1];
        if (!FunctionHelper::get_data_of_column<DoubleColumn>(y_col, row_num, y)) {
            // ctx->set_error("Internal Error: fail to get `y`.");
            return;
        }
        const Column* treatment_col = columns[2];
        bool treatment = false;
        if (!FunctionHelper::get_data_of_column<BooleanColumn>(treatment_col, row_num, treatment)) {
            // ctx->set_error("Internal Error: fail to get `treatment`.");
            return;
        }
        double weight{};
        const Column* weight_col = columns[3];
        if (!FunctionHelper::get_data_of_column<DoubleColumn>(weight_col, row_num, weight)) {
            // ctx->set_error("Internal Error: fail to get `weight`.");
            return;
        }

        auto xs = FunctionHelper::get_data_of_array(columns[4], row_num);
        if (!xs) {
            // ctx->set_error("Internal Error: fail to get `data`.");
            return;
        }
        std::vector<double> data{y, static_cast<double>(treatment), weight};
        for (auto& x : xs.value()) {
            if (x.is_null()) {
                // ctx->set_error("Internal Error: fail to get `data`.");
                return;
            }
            data.emplace_back(x.get_double());
        }

        if (this->data(state).is_uninitialized()) {
            auto datum = columns[0]->get(row_num);
            if (datum.is_null()) {
                ctx->set_error("Internal Error: fail to get model json.");
                return;
            }
            const JsonValue* model = datum.get_json();
            bool is_predict = false;
            if (ctx->get_num_args() >= 6) {
                Datum dt = columns[5]->get(row_num);
                if (dt.is_null()) {
                    ctx->set_error("Internal Error: fail to get is_predict.");
                    return;
                }
                is_predict = dt.get_uint8() != 0;
            }
            auto st = this->data(state).init(*model, data.size(), is_predict);
            if (!st.ok()) {
                ctx->set_error(st.get_error_msg().c_str());
                return;
            }
        }

        auto st = this->data(state).add(y, treatment, weight, data);
        if (!st.ok()) {
            ctx->set_error(st.get_error_msg().c_str());
        }
    }

    void merge(FunctionContext* ctx, const Column* column, AggDataPtr __restrict state, size_t row_num) const override {
        column = FunctionHelper::unwrap_if_nullable<const Column*>(column, row_num);
        if (column == nullptr) {
            ctx->set_error("Internal Error: fail to get intermediate data.");
            return;
        }
        DCHECK(column->is_binary());
        auto data_slice = column->get(row_num).get_slice();
        auto data_size = data_slice.size;
        const auto* serialized_data = reinterpret_cast<const uint8_t*>(data_slice.data);
        if (this->data(state).is_uninitialized()) {
            this->data(state).deserialize(serialized_data);
            DCHECK_EQ(serialized_data, reinterpret_cast<const uint8_t*>(data_slice.data) + data_size);
            return;
        }
        CausalForestData other(serialized_data);
        DCHECK_EQ(serialized_data, reinterpret_cast<const uint8_t*>(data_slice.data) + data_size);
        this->data(state).merge(other);
    }

    void serialize_to_column(FunctionContext* ctx, ConstAggDataPtr __restrict state, Column* to) const override {
        if (to->is_nullable()) {
            auto* dst_nullable_col = down_cast<NullableColumn*>(to);
            dst_nullable_col->null_column_data().emplace_back(false);
            to = dst_nullable_col->data_column().get();
        }
        DCHECK(to->is_binary());
        auto* column = down_cast<BinaryColumn*>(to);
        Bytes& bytes = column->get_bytes();
        size_t old_size = bytes.size();
        size_t new_size = old_size + this->data(state).serialized_size();
        bytes.resize(new_size);
        column->get_offset().emplace_back(new_size);
        uint8_t* serialized_data = bytes.data() + old_size;
        this->data(state).serialize(serialized_data);
        DCHECK_EQ(serialized_data, bytes.data() + new_size);
    }

    void finalize_to_column(FunctionContext* ctx, ConstAggDataPtr __restrict state, Column* to) const override {
        DCHECK(!this->data(state).is_uninitialized());
        if (to->is_nullable()) {
            auto* dst_nullable_col = down_cast<NullableColumn*>(to);
            dst_nullable_col->null_column_data().emplace_back(false);
            to = dst_nullable_col->data_column().get();
        }
        auto st = this->data(state).trainer.afterTrain();
        if (!st.ok()) {
            ctx->set_error(st.get_error_msg().c_str());
            return;
        }
        vpack::Builder builder;
        this->data(state).to_json(builder);
        JsonValue json(builder.slice());
        down_cast<JsonColumn*>(to)->append_datum(Datum{&json});
    }

    void convert_to_serialize_format(FunctionContext* ctx, const Columns& src, size_t chunk_size,
                                     ColumnPtr* dst) const override {
        ctx->set_error("Logical Error: `convert_to_serialize_format` not supported.");
    }

    std::string get_name() const override { return std::string(AllInSqlFunctions::causal_forest); }
};

} // namespace starrocks
