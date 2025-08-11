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

#include <vector>

#include "exprs/agg/aggregate_factory.h"
#include "exprs/agg/caliper_matching_info.h"
#include "exprs/agg/causal_forest.h"
#include "exprs/agg/delta_method.h"
#include "exprs/agg/distributed_node_row_number.h"
#include "exprs/agg/factory/aggregate_factory.hpp"
#include "exprs/agg/factory/aggregate_resolver.hpp"
#include "exprs/agg/kolmogorov_smirnov_test.h"
#include "exprs/agg/group_set.h"
#include "exprs/agg/mann_whitney.h"
#include "exprs/agg/matrix_multiplication.h"
#include "exprs/agg/ols.h"
#include "exprs/agg/srm.h"
#include "exprs/agg/ttest_1samp.h"
#include "exprs/agg/ttest_2samp.h"
#include "exprs/agg/ttests_2samp.h"
#include "exprs/agg/xexpt_ttest_2samp.h"
#include "exprs/all_in_sql_functions.h"
#include "types/logical_type.h"

namespace starrocks {

void AggregateFuncResolver::register_all_in_sql() {
    // register for delta method
    add_aggregate_mapping<TYPE_DOUBLE, DeltaMethodAggregateState>(
            std::string(AllInSqlFunctions::delta_method), std::vector{TYPE_VARCHAR, TYPE_BOOLEAN, TYPE_ARRAY}, false,
            std::make_shared<DeltaMethodAggregateFunction>());

    // register for ttest 1 samp
    add_aggregate_mapping<TYPE_VARCHAR, Ttest1SampAggregateState>(
            std::string(AllInSqlFunctions::ttest_1samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_DOUBLE, TYPE_ARRAY}, false,
            std::make_shared<Ttest1SampAggregateFunction>());

    // register for ttest 1 samp
    add_aggregate_mapping<TYPE_VARCHAR, Ttest1SampAggregateState>(
            std::string(AllInSqlFunctions::ttest_1samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_DOUBLE, TYPE_ARRAY, TYPE_VARCHAR}, false,
            std::make_shared<Ttest1SampAggregateFunction>());

    // register for ttest 1 samp
    add_aggregate_mapping<TYPE_VARCHAR, Ttest1SampAggregateState>(
            std::string(AllInSqlFunctions::ttest_1samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_DOUBLE, TYPE_ARRAY, TYPE_VARCHAR, TYPE_DOUBLE}, false,
            std::make_shared<Ttest1SampAggregateFunction>());

    // expression, side, treatment, data, [cuped, alpha]
    // register for ttest 2 samp
    add_aggregate_mapping<TYPE_VARCHAR, Ttest2SampAggregateState>(
            std::string(AllInSqlFunctions::ttest_2samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_BOOLEAN, TYPE_ARRAY}, false,
            std::make_shared<Ttest2SampAggregateFunction>());

    // register for ttest 2 samp
    add_aggregate_mapping<TYPE_VARCHAR, Ttest2SampAggregateState>(
            std::string(AllInSqlFunctions::ttest_2samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_BOOLEAN, TYPE_ARRAY, TYPE_VARCHAR}, false,
            std::make_shared<Ttest2SampAggregateFunction>());

    // register for ttest 2 samp
    add_aggregate_mapping<TYPE_VARCHAR, Ttest2SampAggregateState>(
            std::string(AllInSqlFunctions::ttest_2samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_BOOLEAN, TYPE_ARRAY, TYPE_VARCHAR, TYPE_DOUBLE}, false,
            std::make_shared<Ttest2SampAggregateFunction>());

    // expression, alternative, treatment, data[, cuped[, alpha[, pse_index, pse_data]]]
    // register for ttest 2 samp
    add_aggregate_mapping<TYPE_VARCHAR, Ttests2SampAggregateState>(
            std::string(AllInSqlFunctions::ttests_2samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_BOOLEAN, TYPE_ARRAY}, false,
            std::make_shared<Ttests2SampAggregateFunction>());

    add_aggregate_mapping<TYPE_VARCHAR, Ttests2SampAggregateState>(
            std::string(AllInSqlFunctions::ttests_2samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_BOOLEAN, TYPE_ARRAY, TYPE_VARCHAR}, false,
            std::make_shared<Ttests2SampAggregateFunction>());

    add_aggregate_mapping<TYPE_VARCHAR, Ttests2SampAggregateState>(
            std::string(AllInSqlFunctions::ttests_2samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_BOOLEAN, TYPE_ARRAY, TYPE_VARCHAR, TYPE_DOUBLE}, false,
            std::make_shared<Ttests2SampAggregateFunction>());

    add_aggregate_mapping<TYPE_VARCHAR, Ttests2SampAggregateState>(
            std::string(AllInSqlFunctions::ttests_2samp),
            std::vector{TYPE_VARCHAR, TYPE_VARCHAR, TYPE_BOOLEAN, TYPE_ARRAY, TYPE_VARCHAR, TYPE_DOUBLE, TYPE_ARRAY},
            false, std::make_shared<Ttests2SampAggregateFunction>());

    add_aggregate_mapping<TYPE_VARCHAR, XexptTtest2SampAggregateState<std::string>>(
            std::string(AllInSqlFunctions::xexpt_ttest_2samp), std::vector{TYPE_BIGINT, TYPE_VARCHAR, TYPE_ARRAY},
            false, std::make_shared<XexptTtest2SampAggregateFunction<std::string>>());

    add_aggregate_mapping<TYPE_VARCHAR, XexptTtest2SampAggregateState<std::string>>(
            std::string(AllInSqlFunctions::xexpt_ttest_2samp),
            std::vector{TYPE_BIGINT, TYPE_VARCHAR, TYPE_ARRAY, TYPE_VARCHAR}, false,
            std::make_shared<XexptTtest2SampAggregateFunction<std::string>>());

    add_aggregate_mapping<TYPE_VARCHAR, XexptTtest2SampAggregateState<std::string>>(
            std::string(AllInSqlFunctions::xexpt_ttest_2samp),
            std::vector{TYPE_BIGINT, TYPE_VARCHAR, TYPE_ARRAY, TYPE_VARCHAR, TYPE_DOUBLE, TYPE_DOUBLE, TYPE_DOUBLE},
            false, std::make_shared<XexptTtest2SampAggregateFunction<std::string>>());

    add_aggregate_mapping<TYPE_VARCHAR, XexptTtest2SampAggregateState<std::string>>(
            std::string(AllInSqlFunctions::xexpt_ttest_2samp),
            std::vector{TYPE_BIGINT, TYPE_VARCHAR, TYPE_ARRAY, TYPE_VARCHAR, TYPE_DOUBLE, TYPE_DOUBLE, TYPE_DOUBLE,
                        TYPE_VARCHAR, TYPE_ARRAY},
            false, std::make_shared<XexptTtest2SampAggregateFunction<std::string>>());

    // register for ols
    add_aggregate_mapping<TYPE_JSON, OlsState>(std::string(AllInSqlFunctions::ols_train),
                                               std::vector{TYPE_DOUBLE, TYPE_ARRAY, TYPE_BOOLEAN}, false,
                                               std::make_shared<OlsAggregateFunction<true, false>>());

    add_aggregate_mapping<TYPE_VARCHAR, OlsState>(std::string(AllInSqlFunctions::ols),
                                                  std::vector{TYPE_DOUBLE, TYPE_ARRAY, TYPE_BOOLEAN}, false,
                                                  std::make_shared<OlsAggregateFunction<false, false>>());

    add_aggregate_mapping<TYPE_VARCHAR, OlsState>(std::string(AllInSqlFunctions::ols),
                                                  std::vector{TYPE_DOUBLE, TYPE_ARRAY, TYPE_BOOLEAN, TYPE_VARCHAR},
                                                  false, std::make_shared<OlsAggregateFunction<false, false>>());

    add_aggregate_mapping<TYPE_VARCHAR, OlsState>(
            std::string(AllInSqlFunctions::ols),
            std::vector{TYPE_DOUBLE, TYPE_ARRAY, TYPE_BOOLEAN, TYPE_VARCHAR, TYPE_JSON, TYPE_JSON}, false,
            std::make_shared<OlsAggregateFunction<false, false>>());

    // register for ols
    add_aggregate_mapping<TYPE_JSON, OlsState>(std::string(AllInSqlFunctions::wls_train),
                                               std::vector{TYPE_DOUBLE, TYPE_ARRAY, TYPE_DOUBLE, TYPE_BOOLEAN}, false,
                                               std::make_shared<OlsAggregateFunction<true, true>>());

    add_aggregate_mapping<TYPE_VARCHAR, OlsState>(std::string(AllInSqlFunctions::wls),
                                                  std::vector{TYPE_DOUBLE, TYPE_ARRAY, TYPE_DOUBLE, TYPE_BOOLEAN},
                                                  false, std::make_shared<OlsAggregateFunction<false, true>>());

    add_aggregate_mapping<TYPE_VARCHAR, OlsState>(
            std::string(AllInSqlFunctions::wls),
            std::vector{TYPE_DOUBLE, TYPE_ARRAY, TYPE_DOUBLE, TYPE_BOOLEAN, TYPE_VARCHAR}, false,
            std::make_shared<OlsAggregateFunction<false, true>>());

    add_aggregate_mapping<TYPE_VARCHAR, OlsState>(
            std::string(AllInSqlFunctions::wls),
            std::vector{TYPE_DOUBLE, TYPE_ARRAY, TYPE_DOUBLE, TYPE_BOOLEAN, TYPE_VARCHAR, TYPE_JSON, TYPE_JSON}, false,
            std::make_shared<OlsAggregateFunction<false, true>>());

    add_aggregate_mapping<TYPE_JSON, MatrixMultiplicationState>(std::string(AllInSqlFunctions::matrix_multiplication),
                                                                std::vector{TYPE_ARRAY, TYPE_BOOLEAN, TYPE_BOOLEAN},
                                                                false, std::make_shared<MatrixMultiplication>());

    add_aggregate_mapping<TYPE_JSON, DistributedNodeRowNumberAggregateState>(
            std::string(AllInSqlFunctions::distributed_node_row_number), std::vector{TYPE_INT}, false,
            std::make_shared<DistributedNodeRowNumberAggregateFunction>());

    add_aggregate_mapping<TYPE_JSON, CaliperMatchingInfoAggState>(
            std::string(AllInSqlFunctions::caliper_matching_info), std::vector{TYPE_BOOLEAN, TYPE_DOUBLE, TYPE_DOUBLE},
            false, std::make_shared<CaliperMatchingInfoAggFunction>());

    add_aggregate_mapping<TYPE_JSON, CaliperMatchingInfoAggState>(
            std::string(AllInSqlFunctions::caliper_matching_info),
            std::vector{TYPE_BOOLEAN, TYPE_DOUBLE, TYPE_DOUBLE, TYPE_ARRAY}, false,
            std::make_shared<CaliperMatchingInfoAggFunction>());

    add_aggregate_mapping<TYPE_VARCHAR, SRMAggState>(std::string(AllInSqlFunctions::srm),
                                                     std::vector{TYPE_DOUBLE, TYPE_VARCHAR, TYPE_ARRAY}, false,
                                                     std::make_shared<SRMAggFunction>());

    add_aggregate_mapping<TYPE_JSON, GroupSetAggState>(std::string(AllInSqlFunctions::group_set),
                                                       std::vector{TYPE_DOUBLE, TYPE_INT, TYPE_ARRAY}, false,
                                                       std::make_shared<GroupSetAggFunction>());

    add_aggregate_mapping<TYPE_JSON, GroupSetAggState>(std::string(AllInSqlFunctions::group_set),
                                                       std::vector{TYPE_DOUBLE, TYPE_INT, TYPE_ARRAY, TYPE_ARRAY},
                                                       false, std::make_shared<GroupSetAggFunction>());

    add_aggregate_mapping<TYPE_JSON, MannWhitneyAggState>(
            std::string(AllInSqlFunctions::mann_whitney_u_test),
            std::vector{TYPE_DOUBLE, TYPE_BOOLEAN, TYPE_VARCHAR, TYPE_BIGINT}, false,
            std::make_shared<MannWhitneyAggFunction>());

    add_aggregate_mapping<TYPE_JSON, CausalForestData>(
            std::string(AllInSqlFunctions::causal_forest),
            std::vector{TYPE_JSON, TYPE_DOUBLE, TYPE_BOOLEAN, TYPE_DOUBLE, TYPE_ARRAY}, false,
            std::make_shared<AggregateFunctionCausalForest>());

    add_aggregate_mapping<TYPE_JSON, CausalForestData>(
            std::string(AllInSqlFunctions::causal_forest),
            std::vector{TYPE_JSON, TYPE_DOUBLE, TYPE_BOOLEAN, TYPE_DOUBLE, TYPE_ARRAY, TYPE_BOOLEAN}, false,
            std::make_shared<AggregateFunctionCausalForest>());

    add_aggregate_mapping<TYPE_JSON, KolmogorovSmirnovAggState>(
            std::string(AllInSqlFunctions::kolmogorov_smirnov_test),
            std::vector{TYPE_DOUBLE, TYPE_BOOLEAN, TYPE_VARCHAR, TYPE_VARCHAR}, false,
            std::make_shared<KolmogorovSmirnovAggFunction>());
}

} // namespace starrocks
