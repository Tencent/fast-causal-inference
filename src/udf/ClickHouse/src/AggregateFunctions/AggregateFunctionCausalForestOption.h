#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/StatCommon.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnString.h>
#include <Core/Field.h>
#include <base/types.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeString.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <IO/WriteBufferFromString.h>
#include <IO/ReadBufferFromString.h>
#include <pqxx/params.hxx>
#include <Common/Exception.h>
#include <regex>

namespace DB
{
struct Settings;
namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

enum class CausalForestState
{
    Init,
    CalcNumerAndDenom,
    FindBestSplitPre,
    FindBestSplit,
    Finish,
};

struct TreeOptions 
{
public:
    TreeOptions() = default;

    TreeOptions(UInt32 mtry,
                UInt32 min_node_size,
                bool honesty,
                double honesty_fraction,
                bool honesty_prune_leaves,
                double alpha,
                double imbalance_penalty);

    String toString() const;

    void serialize(WriteBuffer & buf) const;

    void deserialize(ReadBuffer & buf);

    UInt32 mtry;
    UInt32 min_node_size;
    bool honesty;
    double honesty_fraction;
    bool honesty_prune_leaves;
    double alpha;
    double imbalance_penalty;
};

struct ForestOptions 
{
public:
    ForestOptions() = default;
    ForestOptions(UInt32 num_trees_,
                  size_t ci_group_size_,
                  double sample_fraction_,
                  UInt32 random_seed_,
                  Int32 weight_index_,
                  Int32 outcome_index_,
                  Int32 treatment_index_,
                  Int32 instrument_index_,
                  CausalForestState state_,
                  UInt64 arguments_size_) :
      num_trees(num_trees_),
      ci_group_size(ci_group_size_),
      sample_fraction(sample_fraction_),
      random_seed(random_seed_),
      weight_index(weight_index_),
      outcome_index(outcome_index_),
      treatment_index(treatment_index_),
      instrument_index(instrument_index_),
      state(state_),
      arguments_size(arguments_size_)
    {
        disallowed_split_variables.insert(weight_index);
        disallowed_split_variables.insert(outcome_index);
        disallowed_split_variables.insert(treatment_index);
        disallowed_split_variables.insert(instrument_index);
        auto max_index = std::max({weight_index, outcome_index, treatment_index_, instrument_index});
        auto min_index = std::min({weight_index, outcome_index, treatment_index_, instrument_index});
        if (max_index >= static_cast<Int32>(arguments_size) || min_index < 0)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "index over flow");
    }

    String toString() const;

    void serialize(WriteBuffer & buf) const;

    void deserialize(ReadBuffer & buf);

    UInt32 num_trees;
    size_t ci_group_size;
    double sample_fraction;
    UInt32 random_seed;

    Int32 weight_index = -1;
    Int32 outcome_index = -1;
    Int32 treatment_index = -1;
    Int32 instrument_index = -1;
    CausalForestState state;
    UInt64 arguments_size = 0;

    std::set<size_t> disallowed_split_variables;
};

}
