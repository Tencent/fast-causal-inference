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

#include <glog/logging.h>

#include <string>
#include <unordered_set>

namespace starrocks {

class AllInSqlFunctions {
public:
    constexpr static std::string_view delta_method = "delta_method";
    constexpr static std::string_view ttest_1samp = "ttest_1samp";
    constexpr static std::string_view ttest_2samp = "ttest_2samp";
    constexpr static std::string_view ttests_2samp = "ttests_2samp";
    constexpr static std::string_view xexpt_ttest_2samp = "xexpt_ttest_2samp";
    constexpr static std::string_view ols = "ols";
    constexpr static std::string_view ols_train = "ols_train";
    constexpr static std::string_view wls = "wls";
    constexpr static std::string_view wls_train = "wls_train";
    constexpr static std::string_view eval_ml_method = "eval_ml_method";
    constexpr static std::string_view matrix_multiplication = "matrix_multiplication";
    constexpr static std::string_view distributed_node_row_number = "distributed_node_row_number";
    constexpr static std::string_view boot_strap = "boot_strap";
    constexpr static std::string_view caliper_matching_info = "caliper_matching_info";
    constexpr static std::string_view srm = "srm";
    constexpr static std::string_view group_set = "group_set";
    constexpr static std::string_view mann_whitney_u_test = "mann_whitney_u_test";
    constexpr static std::string_view causal_forest = "causal_forest";
    static bool is_all_in_sql_function(std::string_view func_name) { return all_in_sql_functions.count(func_name); }

private:
    const static std::unordered_set<std::string_view> all_in_sql_functions;
};

} // namespace starrocks
