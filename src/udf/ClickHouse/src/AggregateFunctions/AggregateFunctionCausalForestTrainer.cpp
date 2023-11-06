#include <random>
#include <type_traits>
#include <AggregateFunctions/AggregateFunctionCausalForestTrainer.h>
#include <IO/WriteBufferFromString.h>
#include <IO/ReadBufferFromString.h>
#include <boost/algorithm/string/trim.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/ptree.hpp>
#include "Common/Exception.h"
#include "Core/Types.h"
#include "base/types.h"

namespace DB
{

ForestTrainer::ForestTrainer(const String & context, UInt64 arguments_size_, bool is_predict)
{
    // tree options
    UInt32 mtry = 3;
    UInt32 min_node_size = 1;
    bool honesty = false;
    double honesty_fraction = 0.5;
    bool honesty_prune_leaves = false;
    double alpha = 0.0;
    double imbalance_penalty = 0.0;

    // forest options
    UInt32 num_trees = 50;
    size_t ci_group_size = 1;
    double sample_fraction = 0.35;
    UInt32 random_seed = 42;
    Int32 weight_index = -1;
    Int32 outcome_index = -1;
    Int32 treatment_index = -1;
    Int32 instrument_index = -1;
    CausalForestState state = CausalForestState::Init;


    boost::property_tree::ptree pt;
    std::istringstream json_stream(context);

    bool is_json = boost::algorithm::starts_with(context, "{");
    if (is_json) 
    {
        try 
        {
            boost::property_tree::read_json(json_stream, pt);
        }
        catch (...) 
        {
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Cannot parse JSON string: ");
        }
        auto fill_param = [&](String name, auto & param) 
        {
            if (pt.count(name) > 0) 
            {
                typedef std::remove_reference_t<decltype(param)> param_type; // NOLINT
                if constexpr (std::is_floating_point_v<param_type>) 
                    param = pt.get<double>(name);
                else if constexpr (std::is_integral_v<param_type>) 
                    param = pt.get<UInt32>(name);
                else 
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Unknown parameter type");
            }
        };

        fill_param("mtry", mtry);
        fill_param("min_node_size", min_node_size);
        fill_param("honesty", honesty);
        fill_param("honesty_fraction", honesty_fraction);
        fill_param("honesty_prune_leaves", honesty_prune_leaves);
        fill_param("alpha", alpha);
        fill_param("imbalance_penalty", imbalance_penalty);
        fill_param("num_trees", num_trees);
        fill_param("ci_group_size", ci_group_size);
        fill_param("sample_fraction", sample_fraction);

        fill_param("random_seed", random_seed);
        fill_param("weight_index", weight_index);
        fill_param("outcome_index", outcome_index);
        fill_param("treatment_index", treatment_index);
        fill_param("instrument_index", instrument_index);
        if (instrument_index == -1)
          instrument_index = treatment_index;
        state = CausalForestState::Init;
        tree_options = TreeOptions(mtry, min_node_size, honesty, honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty);
        forest_options = ForestOptions(num_trees, ci_group_size, sample_fraction, random_seed, weight_index, outcome_index, treatment_index, instrument_index, state, arguments_size_);

        std::mt19937_64 random_number_generator(random_seed);
        nonstd::uniform_int_distribution<uint> udist;
        std::uniform_int_distribution<UInt64> dist(0, std::numeric_limits<UInt64>::max());
        for (size_t i = 0; i < forest_options.num_trees; ++i)
        {
            trees.emplace_back(static_cast<Int64>(sample_fraction * Tree::TOTAL_BUCKET_NUM), udist(random_number_generator), forest_options, tree_options);
        }
    }
    else 
    {
        ReadBufferFromString rb{context};
        deserialize(rb);
        UInt32 stop_tree = 0;
        if (!is_predict)
          for (auto & tree : trees)
              if (tree.initTrain(forest_options, tree_options))
                  stop_tree++;
        if (stop_tree > trees.size() / 2)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "train over");
    }
}

void ForestTrainer::add(const IColumn ** columns, size_t row_num)
{
    UInt64 sample_hash = 0;
    for (size_t i = 0; i < forest_options.arguments_size; i++)
      sample_hash ^= static_cast<UInt64>(columns[i]->get64(row_num)) + 0x9e3779b9 + (sample_hash << 6) + (sample_hash >> 2);

    for (auto & tree : trees)
      if (tree.isBelongToBucket(sample_hash))
        tree.add(columns, row_num, forest_options, tree_options);
}

void ForestTrainer::predict(const ColumnsWithTypeAndName & arguments, UInt64 row_num, PaddedPODArray<Float64> & average_value) const
{
    if (arguments.size() != forest_options.arguments_size) // model 和 output 都是 1
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Wrong number of arguments for predict");
    for (const auto & tree : trees)
        tree.predict(arguments, row_num, average_value);
}

}
