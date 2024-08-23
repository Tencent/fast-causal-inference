#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/StatCommon.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <AggregateFunctions/AggregateFunctionCausalForestOption.h>
#include <AggregateFunctions/AggregateFunctionCausalForestTree.h>
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
#include <pqxx/params.hxx>
#include <Common/PODArray_fwd.h>
#include <Core/ColumnsWithTypeAndName.h>
#include <regex>

namespace DB
{
struct Settings;
namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}


class Tree;

class ForestTrainer 
{
public:
    ForestTrainer() = default;
    explicit ForestTrainer(const String & context, UInt64 arguments_size_, bool is_predict = false);

    void train() const;

    String toString() 
    {
      String res = "\ntree_options:\n";
      res += tree_options.toString() + "\n";

      res += "\nforest_options:\n";
      res += forest_options.toString() + "\n";

      res += "\ntrees:\n";
      for (auto & tree : trees)
        res += tree.toString() + "\n";
      return res;
    }

    void add(const IColumn ** columns, size_t row_num);

    void serialize(WriteBuffer & buf) const
    {
        forest_options.serialize(buf);
        tree_options.serialize(buf);
        writeBinary(trees.size(), buf);
        for (const auto & tree : trees)
          tree.serialize(buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        forest_options.deserialize(buf);
        tree_options.deserialize(buf);
        size_t num_trees;
        readBinary(num_trees, buf);
        trees.resize(num_trees);
        for (auto & tree : trees)
          tree.deserialize(buf);
    }

    void afterTrain() 
    {
        auto & state = forest_options.state;

        for (auto & tree : trees)
          tree.afterTrain(forest_options, tree_options);

        if (state == CausalForestState::Init)
          state = CausalForestState::CalcNumerAndDenom;
        else if (state == CausalForestState::CalcNumerAndDenom)
          state = CausalForestState::FindBestSplitPre;
        else if (state == CausalForestState::FindBestSplitPre)
          state = CausalForestState::FindBestSplit;
        else if (state == CausalForestState::FindBestSplit)
          state = CausalForestState::CalcNumerAndDenom;
    }

    void merge(const ForestTrainer & other) 
    {
        if (forest_options.state < other.forest_options.state)
            forest_options.state = other.forest_options.state;
        else if (forest_options.state == other.forest_options.state)
        {
            for (size_t i = 0; i < trees.size(); ++i)
                trees[i].merge(other.trees[i], forest_options.state);
        }
    }

    UInt16 getNumTrees() const 
    {
        return forest_options.num_trees;
    }

    void predict(const ColumnsWithTypeAndName & arguments, UInt64 row_num, PaddedPODArray<Float64> & average_value) const;

    void initHonesty();

private:

    std::vector<Tree> trees;

    ForestOptions forest_options;

    TreeOptions tree_options;

};

}
