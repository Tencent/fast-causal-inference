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

#include "all_in_sql_functions.h"

namespace starrocks {

const std::unordered_set<std::string_view> AllInSqlFunctions::all_in_sql_functions{
        "delta_method",
        "ttest_1samp",
        "ttest_2samp",
        "ttests_2samp",
        "xexpt_ttest_2samp",
        "eval_ml_method",
        "ols",
        "ols_train",
        "wls",
        "wls_train",
        "matrix_multiplication",
        "distributed_node_row_number",
        "boot_strap",
        "caliper_matching_info",
        "caliper_matching",
        "srm",
        "group_set",
        "mann_whitney_u_test",
        "causal_forest",
        "kolmogorov_smirnov_test",
};

} // namespace starrocks
