#include <AggregateFunctions/AggregateFunctionCausalForestOption.h>

namespace DB
{
struct Settings;

TreeOptions::TreeOptions(UInt32 mtry_,
                         size_t quantile_size_,
                         UInt32 min_node_size_,
                         bool honesty_,
                         double honesty_fraction_,
                         bool honesty_prune_leaves_,
                         double alpha_,
                         double imbalance_penalty_):
  mtry(mtry_),
  quantile_size(quantile_size_),
  min_node_size(min_node_size_),
  honesty(honesty_),
  honesty_fraction(honesty_fraction_),
  honesty_prune_leaves(honesty_prune_leaves_),
  alpha(alpha_),
  imbalance_penalty(imbalance_penalty_) {}

String TreeOptions::toString() const 
{
    std::stringstream ss;
    ss << "mtry: " << mtry << std::endl 
      << "quantile_size: " << quantile_size << std::endl
      << "min_node_size: " << min_node_size << std::endl 
      << "honesty: " << honesty << std::endl 
      << "honesty_fraction: " << honesty_fraction << std::endl 
      << "honesty_prune_leaves: " << honesty_prune_leaves << std::endl 
      << "alpha: " << alpha << std::endl
      << "imbalance_penalty: " << imbalance_penalty << std::endl;
    return ss.str();
}

void TreeOptions::serialize(WriteBuffer & buf) const
{
    writeVarUInt(mtry, buf);
    writeVarUInt(quantile_size, buf);
    writeVarUInt(min_node_size, buf);
    writeBinary(honesty, buf);
    writeFloatBinary(honesty_fraction, buf);
    writeBinary(honesty_prune_leaves, buf);
    writeFloatBinary(alpha, buf);
    writeFloatBinary(imbalance_penalty, buf);
}

void TreeOptions::deserialize(ReadBuffer & buf)
{
    readVarUInt(mtry, buf);
    readVarUInt(quantile_size, buf);
    readVarUInt(min_node_size, buf);
    readBinary(honesty, buf);
    readFloatBinary(honesty_fraction, buf);
    readBinary(honesty_prune_leaves, buf);
    readFloatBinary(alpha, buf);
    readFloatBinary(imbalance_penalty, buf);
}

String ForestOptions::toString() const 
{
  std::stringstream ss;
  ss << "num_trees: " << num_trees << std::endl 
     << "ci_group_size: " << ci_group_size << std::endl 
     << "sample_fraction: " << sample_fraction << std::endl 
     << "random_seed: " << random_seed << std::endl
     << "weight_index: " << weight_index << std::endl 
     << "outcome_index: " << outcome_index << std::endl
     << "treatment_index: " << treatment_index << std::endl
     << "instrument_index: " << instrument_index << std::endl
     << "state: " << static_cast<Int32>(state) << std::endl
     << "arguments_size: " << arguments_size;
  return ss.str();
}

void ForestOptions::serialize(WriteBuffer & buf) const
{
    writeVarUInt(num_trees, buf);
    writeVarUInt(ci_group_size, buf);
    writeFloatBinary(sample_fraction, buf);
    writeVarUInt(random_seed, buf);
    writeVarInt(weight_index, buf);
    writeVarInt(outcome_index, buf);
    writeVarInt(treatment_index, buf);
    writeVarInt(instrument_index, buf);
    writeBinary(static_cast<Int32>(state), buf);
    writeVarUInt(arguments_size, buf);
    writeVarUInt(disallowed_split_variables.size(), buf);
    for (const auto & var : disallowed_split_variables)
        writeVarUInt(var, buf);
    writeVarUInt(max_centroids, buf);
    writeVarUInt(max_unmerged, buf);
    writeBinary(epsilon, buf);
}

void ForestOptions::deserialize(ReadBuffer & buf)
{
    readVarUInt(num_trees, buf);
    readVarUInt(ci_group_size, buf);
    readFloatBinary(sample_fraction, buf);
    readVarUInt(random_seed, buf);
    readVarInt(weight_index, buf);
    readVarInt(outcome_index, buf);
    readVarInt(treatment_index, buf);
    readVarInt(instrument_index, buf);
    Int32 state_int;
    readBinary(state_int, buf);
    state = static_cast<CausalForestState>(state_int);
    readVarUInt(arguments_size, buf);
    size_t disallowed_split_variables_size;
    readVarUInt(disallowed_split_variables_size, buf);
    for (size_t i = 0; i < disallowed_split_variables_size; ++i)
    {
        UInt32 var;
        readVarUInt(var, buf);
        disallowed_split_variables.insert(var);
    }
    readVarUInt(max_centroids, buf);
    readVarUInt(max_unmerged, buf);
    readBinary(epsilon, buf);
}

}
