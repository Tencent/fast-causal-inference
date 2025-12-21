#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/StatCommon.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <AggregateFunctions/AggregateFunctionCausalForestOption.h>
#include <AggregateFunctions/AggregateFunctionCausalForestRandom.hpp>
#include <AggregateFunctions/QuantileTDigest.h>
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
#include <boost/graph/detail/histogram_sort.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <pqxx/params.hxx>
#include <Common/PODArray_fwd.h>
#include <regex>
#include <Poco/JSON/Parser.h>
#include <Poco/JSON/Object.h>
#include <Poco/JSON/Array.h>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

class Tree {
public:
    Tree() = default;

    Tree(UInt64 bucket_num_, UInt64 honesty_bucket_num_, UInt64 random_seed_, const ForestOptions& forest_options, const TreeOptions& tree_options);

    struct CalcNodeInfo;

    bool isBelongToBucket(UInt64 bucket_id) const 
    {
        bucket_id ^= random_seed * bucket_id + (bucket_id << 6) + (bucket_id >> 2);
        return bucket_id % TOTAL_BUCKET_NUM <= bucket_num;
    } 

    bool isBelongToHonestyBucket(UInt64 bucket_id) const 
    {
        bucket_id ^= random_seed;
        return bucket_id % TOTAL_BUCKET_NUM <= honesty_bucket_num;
    }

    void predict(const ColumnsWithTypeAndName & arguments, UInt64 row_num, PaddedPODArray<Float64> & average_value) const;

    void add(const IColumn ** columns, size_t row_num, const ForestOptions & forest_options, const TreeOptions & tree_options);

    void addInit(const IColumn ** columns, size_t row_num, const ForestOptions & forest_options, const TreeOptions & tree_options);

    void addCalcNumerAndDenom(const IColumn ** columns, size_t row_num, const ForestOptions & forest_options, const TreeOptions & tree_options);

    void addFindBestSplitPre(const IColumn ** columns, size_t row_num, const ForestOptions & forest_options, const TreeOptions &); // NOLINT

    void addFindBestSplit(const IColumn ** columns, size_t row_num, const ForestOptions & forest_options, const TreeOptions & tree_options);

    void addHonesty(const IColumn ** columns, size_t row_num, const ForestOptions & forest_options, const TreeOptions & tree_options);

    bool getCalcNode(const IColumn ** columns, size_t row_num, size_t & res);

    void initHonesty(const TreeOptions & tree_options)
    {
        if (!tree_options.honesty)
            return ;
        for (auto & node : nodes)
            if (node.calc_node.has_value())
                node.calc_node = CalcNodeInfo();
    }

    String toString() { // NOLINT
      String res;
      res += "bucket_num: " + std::to_string(bucket_num) + "\n";
      res += "honesty_bucket_num: " + std::to_string(honesty_bucket_num) + "\n";
      res += "random_seed: " + std::to_string(random_seed) + "\n";
      res += "calc_index: " + std::to_string(calc_index) + "\n";
      res += "\nnode_info:\n";
      size_t i = 0;
      for (auto & node : nodes)
      {
        res += std::to_string(i++) + ": \n";
        res += node.toString() + "\n";
      }
      return res;
    }

    void serialize(WriteBuffer & buf) const;

    void deserialize(ReadBuffer & buf);

    static constexpr UInt64 TOTAL_BUCKET_NUM = 1000;

    void createEmptyNode();

    bool initTrain(const ForestOptions & forest_options, const TreeOptions &)
    {
        if (forest_options.state == CausalForestState::FindBestSplitPre)
        {
            for (size_t i = calc_index; i < nodes.size(); ++i)
            {
                if (!nodes[i].calc_node.has_value())
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "node is not calc node");
                auto & node = nodes[i].calc_node.value(); // NOLINT
                node.quantiles_calcers.resize(node.possible_split_vars.size(), CausalQuantileTDigest<Float32>(forest_options.max_centroids, forest_options.max_unmerged, forest_options.epsilon));
            }
        }
        if (forest_options.state == CausalForestState::FindBestSplit)
        {
            for (size_t i = calc_index; i < nodes.size(); ++i)
            {
              if (!nodes[i].calc_node.has_value())
                  throw Exception(ErrorCodes::BAD_ARGUMENTS, "node is not calc node");
                auto & node = nodes[i].calc_node.value(); // NOLINT
                for (auto & quantile : node.quantiles)
                    node.split_nodes.push_back(std::vector<SplitNode>(quantile.size()));
                node.quantiles_calcers.clear();
            }
        }
        bool stop = false;
        if (forest_options.state == CausalForestState::CalcNumerAndDenom && calc_index == nodes.size())
            stop = true;
        return stop;
    }

    void pureNode()
    {
        for (Int64 i = static_cast<Int64>(nodes.size()) - 1; i >= 0; --i)
        {
            auto & node = nodes[i];
            if (node.is_leaf)
                continue;
            auto & left_child = nodes[node.left_child];
            auto & right_child = nodes[node.right_child];
            if (left_child.isNeedPure() || right_child.isNeedPure())
            {
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

    void afterTrain(const ForestOptions & forest_options, const TreeOptions & tree_options)
    {
        auto old_state = forest_options.state;

        for (size_t i = calc_index; i < nodes.size(); i++) {
          if (nodes[i].calc_node.has_value()) {
            nodes[i].calc_node.value().updateStop(forest_options.state, tree_options);
          }
        }

        if (old_state == CausalForestState::FindBestSplit)
        {
            size_t new_calc_index = nodes.size();
            size_t nodes_size = nodes.size();
            for (size_t j = calc_index; j < nodes_size; ++j) 
            {
                if (!nodes[j].calc_node.has_value())
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "node is not calc node");
                auto & calc_node = nodes[j].calc_node.value(); // NOLINT
                size_t best_var = 0;
                double best_value = 0;
                double best_decrease = -1;
                bool best_send_missing_left = true;

                for (size_t i = 0; i < calc_node.possible_split_vars.size(); ++i) // 对于每个可能的分裂变量
                /*
                   calc_node.find_best_split_value(i, best_var, best_value, best_decrease, best_send_missing_left, tree_options);
                    */
                  if (tree_options.causal_tree)
                    calc_node.find_best_split_value_cf(i, best_var, best_value, best_decrease, best_send_missing_left, tree_options);
                  else
                    calc_node.find_best_split_value(i, best_var, best_value, best_decrease, best_send_missing_left, tree_options);
                if (best_decrease <= 0.0 || calc_node.is_stop)
                {
                    transformToLeaf(nodes[j]);
                    continue;
                }
                splitNode(nodes[j], best_var, best_value, forest_options, tree_options);
            }
            calc_index = new_calc_index;
        }

        // 获取分位点
        if (forest_options.state == CausalForestState::FindBestSplitPre)
        {
            for (size_t i = calc_index; i < nodes.size(); ++i)
            {
              if (!nodes[i].calc_node.has_value())
                  throw Exception(ErrorCodes::BAD_ARGUMENTS, "node is not calc node");
                auto & node = nodes[i].calc_node.value(); // NOLINT
                for (auto & quantiles_calcer : node.quantiles_calcers)
                    node.quantiles.push_back(quantiles_calcer.getQuantiles(tree_options.quantile_size));
                node.quantiles_calcers.clear();
            }
        }

        if (forest_options.state == CausalForestState::Honesty)
            pureNode();

    }

    String getTreeStruct() const
    {
        String res;
        if (nodes.empty())
            return "";
        UInt64 total_samples = nodes[0].calc_node.value().num_samples;
        Float64 ate = 0;
        std::map<size_t, size_t> node_father;
        std::map<size_t, Int32> depth;
        Poco::JSON::Array res_json;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            auto tree_id = i;
            bool is_leaf = nodes[i].is_leaf;
            UInt64 treatment_count = nodes[i].calc_node.value().cnt1();
            UInt64 control_count = nodes[i].calc_node.value().cnt0();
            Float64 y0_column = nodes[i].calc_node.value().y0();
            Float64 y1_column = nodes[i].calc_node.value().y1();
            Float64 treated_label_avg = y1_column / treatment_count;
            Float64 control_label_avg = y0_column / control_count;
            // calc treated_label_var and control_label_var, is a varsamp
            Float64 treated_label_var = nodes[i].calc_node.value().y1_squared / treatment_count - treated_label_avg * treated_label_avg;
            Float64 control_label_var = nodes[i].calc_node.value().y0_squared / control_count - control_label_avg * control_label_avg;
            Float64 prediction = treated_label_avg - control_label_avg;
            if (i == 0)
                ate = prediction;
            Float64 ratio = (treatment_count + control_count) / static_cast<Float64>(total_samples);

            Float64 est_point1 = prediction;
            Float64 std1 = std::sqrt(treated_label_var / treatment_count + control_label_var / control_count);
            Float64 z_value1 = est_point1 / std1;
            Float64 p_value1 = 2 * (1 - boost::math::cdf(boost::math::normal(), std::abs(z_value1)));
            Float64 lower_ci1 = prediction - 1.96 * std1;
            Float64 upper_ci1 = prediction + 1.96 * std1;

            Float64 est_point2 = prediction - ate;
            Float64 std2 = std1;
            Float64 z_value2 = est_point2 / std2;
            Float64 p_value2 = 2 * (1 - boost::math::cdf(boost::math::normal(), std::abs(z_value2)));
            Float64 lower_ci2 = est_point2 - 1.96 * std2;
            Float64 upper_ci2 = est_point2 + 1.96 * std2;

            Float64 est_point3 = treated_label_avg / control_label_avg - 1;
            Float64 std3 = std1;
            Float64 z_value3 = z_value1;
            Float64 p_value3 = p_value1;
            Float64 lower_ci3 = est_point3 - 1.96 * std1 * est_point3 / est_point1;
            Float64 upper_ci3 = est_point3 + 1.96 * std1 * est_point3 / est_point1;

            Float64 est_point4 = (treated_label_avg - ate) / control_label_avg - 1;
            Float64 std4 = std2;
            Float64 z_value4 = z_value2;
            Float64 p_value4 = p_value2;


            Float64 lower_ci4 = 0;
            Float64 upper_ci4 = 0;
            if (tree_id == 0)
            {
                est_point2 = 0;
                est_point4 = 0;
                lower_ci4 = 0;
                upper_ci4 = 0;
            }
            else
            {
                lower_ci4 = est_point4 - 1.96 * std2 * est_point4 / est_point2;
                upper_ci4 = est_point4 + 1.96 * std2 * est_point4 / est_point2;
            }
            
            Int32 split_var = nodes[i].split_var;
            Float64 split_value = nodes[i].split_value;
            Int32 left_child = nodes[i].left_child;
            Int32 right_child = nodes[i].right_child;
            if (!is_leaf && left_child) 
                node_father[left_child] = i;
            if (!is_leaf && right_child)
                node_father[right_child] = i;

            Int32 fahter = -1;
            if (i)
            {
                auto it = node_father.find(i);
                if (it != node_father.end())
                    fahter = static_cast<Int32>(it->second);
                depth[i] = 1 + depth[fahter];
            }


            Poco::JSON::Object node_json;
            node_json.set("node_id", static_cast<Int64>(i));
            node_json.set("nodeType", is_leaf ? "leaf" : "internal");
            node_json.set("split_var", split_var);
            node_json.set("split_value", split_value);
            node_json.set("count_ratio", ratio * 100);
            node_json.set("controlCount", static_cast<Int64>(control_count));
            node_json.set("treatedCount", static_cast<Int64>(treatment_count));
            node_json.set("controlAvg", control_label_avg);
            node_json.set("treatedAvg", treated_label_avg);
            Poco::JSON::Array pvalues_json;
            pvalues_json.add(p_value1);
            pvalues_json.add(p_value2);
            pvalues_json.add(p_value3);
            pvalues_json.add(p_value4);
            node_json.set("pvalues", pvalues_json);
            node_json.set("tau_i", prediction);
            Poco::JSON::Array tau_i_new_json;
            tau_i_new_json.add(est_point1);
            tau_i_new_json.add(est_point2);
            tau_i_new_json.add(est_point3);
            tau_i_new_json.add(est_point4);
            node_json.set("tau_i_new", tau_i_new_json);
            node_json.set("father", fahter);
            if (is_leaf)
                node_json.set("left_child", -1);
            else 
                node_json.set("left_child", left_child);
            if (is_leaf)
                node_json.set("right_child", -1);
            else
                node_json.set("right_child", right_child);
            node_json.set("est_point1", est_point1);
            node_json.set("est_point2", est_point2);
            node_json.set("est_point3", est_point3);
            node_json.set("est_point4", est_point4);
            node_json.set("std1", std1);
            node_json.set("std2", std2);
            node_json.set("std3", std3);
            node_json.set("std4", std4);
            node_json.set("z_value1", z_value1);
            node_json.set("z_value2", z_value2);
            node_json.set("z_value3", z_value3);
            node_json.set("z_value4", z_value4);
            node_json.set("lower_ci1", lower_ci1);
            node_json.set("upper_ci1", upper_ci1);
            node_json.set("lower_ci2", lower_ci2);
            node_json.set("upper_ci2", upper_ci2);
            node_json.set("lower_ci3", lower_ci3);
            node_json.set("upper_ci3", upper_ci3);
            node_json.set("lower_ci4", lower_ci4);
            node_json.set("upper_ci4", upper_ci4);
            node_json.set("prediction", prediction);
            node_json.set("ratio", ratio);
            node_json.set("level", depth[i]);
            res_json.add(node_json);
        }
        std::ostringstream json_stream;
        res_json.stringify(json_stream);
        return json_stream.str();
    }

    struct NodeInfo;

    static void transformToLeaf(NodeInfo & node)
    {
        node.split_var = -1;
        node.is_leaf = true;
        if (!node.calc_node.has_value())
            return;
        if (node.calc_node.has_value())
        {
            auto & calc_node = node.calc_node.value(); // NOLINT
            calc_node.quantiles.clear();
            calc_node.split_nodes.clear();
            calc_node.is_stop = true;
        }
    }

    void splitNode(NodeInfo & node, size_t best_var, double best_value, const ForestOptions & forest_options, const TreeOptions & tree_options)
    {
        auto & calc_node = node.calc_node.value(); // NOLINT
        node.split_var = calc_node.possible_split_vars[best_var];
        node.split_value = best_value;
        node.is_leaf = false;
        NodeInfo left_node;
        NodeInfo right_node;
        left_node.is_leaf = true;
        right_node.is_leaf = true;
        left_node.calc_node = CalcNodeInfo();
        right_node.calc_node = CalcNodeInfo();

        if (calc_node.quantiles[best_var].size() != calc_node.split_nodes[best_var].size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "quantiles size is not equal to split_nodes size");
        for (size_t i = 0; i < calc_node.quantiles[best_var].size(); ++i)
        {
            auto & value = calc_node.quantiles[best_var][i];
            if (value > best_value + 1e-6) 
            {
                auto & new_calc_node = right_node.calc_node.value(); // NOLINT
                new_calc_node.sum_weight += calc_node.split_nodes[best_var][i].weight_sums;
                new_calc_node.num_samples += calc_node.split_nodes[best_var][i].counter;
                new_calc_node.sum_node_z_squared += calc_node.split_nodes[best_var][i].sums_z_squared;
                new_calc_node.y0_squared += calc_node.split_nodes[best_var][i].y0_squared;
                new_calc_node.y1_squared += calc_node.split_nodes[best_var][i].y1_squared;
                new_calc_node.total_instrument += calc_node.split_nodes[best_var][i].sums_z;
                new_calc_node.total_outcome += calc_node.split_nodes[best_var][i].total_outcome;
                new_calc_node.total_treatment += calc_node.split_nodes[best_var][i].total_treatment;
                new_calc_node.total_outcome_treatment += calc_node.split_nodes[best_var][i].total_outcome_treatment;
            }
            else 
            {
                auto & new_calc_node = left_node.calc_node.value(); // NOLINT
                new_calc_node.sum_weight += calc_node.split_nodes[best_var][i].weight_sums;
                new_calc_node.num_samples += calc_node.split_nodes[best_var][i].counter;
                new_calc_node.sum_node_z_squared += calc_node.split_nodes[best_var][i].sums_z_squared;
                new_calc_node.y0_squared += calc_node.split_nodes[best_var][i].y0_squared;
                new_calc_node.y1_squared += calc_node.split_nodes[best_var][i].y1_squared;
                new_calc_node.total_instrument += calc_node.split_nodes[best_var][i].sums_z;
                new_calc_node.total_outcome += calc_node.split_nodes[best_var][i].total_outcome;
                new_calc_node.total_treatment += calc_node.split_nodes[best_var][i].total_treatment;
                new_calc_node.total_outcome_treatment += calc_node.split_nodes[best_var][i].total_outcome_treatment;
            }
        }
        if (!tree_options.causal_tree)
            node.calc_node.reset();
        node.left_child = nodes.size();
        node.right_child = nodes.size() + 1;
        left_node.calc_node.value().possible_split_vars = createSplitVariable(forest_options, tree_options);
        right_node.calc_node.value().possible_split_vars = createSplitVariable(forest_options, tree_options);
        nodes.push_back(left_node);
        nodes.push_back(right_node);
    }

    struct SplitNode // size equal to quantiles.size
    {
        UInt64 counter = 0;
        Float64 weight_sums = 0.0;
        Float64 sums = 0.0;
        UInt64 num_small_z = 0.0;
        Float64 sums_z = 0.0;
        Float64 sums_z_squared = 0.0;
        Float64 y0_squared = 0.0;
        Float64 y1_squared = 0.0;

        Float64 total_outcome = 0.0;
        Float64 total_treatment = 0.0;
        Float64 total_outcome_treatment = 0.0;


        String toString() const
        {
            String res;
            res += "counter: " + std::to_string(counter) + "\n";
            res += "weight_sums: " + std::to_string(weight_sums) + "\n";
            res += "sums: " + std::to_string(sums) + "\n";
            res += "num_small_z: " + std::to_string(num_small_z) + "\n";
            res += "sums_z: " + std::to_string(sums_z) + "\n";
            res += "sums_z_squared: " + std::to_string(sums_z_squared) + "\n";
            res += "y0_squared: " + std::to_string(y0_squared) + "\n";
            res += "y1_squared: " + std::to_string(y1_squared) + "\n";
            res += "total_outcome: " + std::to_string(total_outcome) + "\n";
            res += "total_treatment: " + std::to_string(total_treatment) + "\n";
            res += "total_outcome_treatment: " + std::to_string(total_outcome_treatment) + "\n";
            return res;
        }

        void serialize(WriteBuffer & buf) const
        {
            writeBinary(counter, buf);
            writeBinary(weight_sums, buf);
            writeBinary(sums, buf);
            writeBinary(num_small_z, buf);
            writeBinary(sums_z, buf);
            writeBinary(sums_z_squared, buf);
            writeBinary(y0_squared, buf);
            writeBinary(y1_squared, buf);
            writeBinary(total_outcome, buf);
            writeBinary(total_treatment, buf);
            writeBinary(total_outcome_treatment, buf);
        }

        void deserialize(ReadBuffer & buf)
        {
            readBinary(counter, buf);
            readBinary(weight_sums, buf);
            readBinary(sums, buf);
            readBinary(num_small_z, buf);
            readBinary(sums_z, buf);
            readBinary(sums_z_squared, buf);
            readBinary(y0_squared, buf);
            readBinary(y1_squared, buf);
            readBinary(total_outcome, buf);
            readBinary(total_treatment, buf);
            readBinary(total_outcome_treatment, buf);
        }

        void merge(const SplitNode & other)
        {
            counter += other.counter;
            weight_sums += other.weight_sums;
            sums += other.sums;
            num_small_z += other.num_small_z;
            sums_z += other.sums_z;
            sums_z_squared += other.sums_z_squared;
            y0_squared += other.y0_squared;
            y1_squared += other.y1_squared;
            total_outcome += other.total_outcome;
            total_treatment += other.total_treatment;
            total_outcome_treatment += other.total_outcome_treatment;
        }
    };


    struct CalcNodeInfo
    {
        Float64 sum_weight = 0.0;
        Float64 total_outcome = 0.0;
        Float64 total_treatment = 0.0;
        Float64 total_instrument = 0.0;
        Float64 total_outcome_treatment = 0.0;
        UInt64 num_samples = 0;
        std::vector<size_t> possible_split_vars;
        bool is_stop = false;
        Float64 numerator = 0.0;
        Float64 denominator = 0.0;
        Float64 sum_node_z_squared = 0.0;
        Float64 y0_squared = 0.0;
        Float64 y1_squared = 0.0;

        Float64 sum_node = 0.0;
        UInt64 num_node_small_z = 0;

        std::vector<CausalQuantileTDigest<Float32>> quantiles_calcers;
        std::vector<std::vector<Float64>> quantiles;
        std::vector<std::vector<SplitNode>> split_nodes;

        Int32 father = -1;

        String toString() const
        {
            String res;
            res += "sum_weight: " + std::to_string(sum_weight) + "\n";
            res += "total_outcome: " + std::to_string(total_outcome) + "\n";
            res += "total_treatment: " + std::to_string(total_treatment) + "\n";
            res += "total_instrument: " + std::to_string(total_instrument) + "\n";
            res += "total_outcome_treatment: " + std::to_string(total_outcome_treatment) + "\n";
            res += "cnt0:" + std::to_string(cnt0()) + "\n";
            res += "cnt1:" + std::to_string(cnt1()) + "\n";
            res += "y0_avg:" + std::to_string(y0()/cnt0()) + "\n";
            res += "y1_avg:" + std::to_string(y1()/cnt1()) + "\n";
            res += "y0_squared_avg:" + std::to_string(y0_squared / cnt0()) + "\n";
            res += "y1_squared_avg:" + std::to_string(y1_squared / cnt1()) + "\n";
            res += "effect: " + std::to_string(effect()) + "\n";
            res += "possible_split_vars: ";
            for (const auto & var : possible_split_vars)
                res += std::to_string(var) + " ";
            res += "\n";
            res += "average_outcome: " + std::to_string(average_outcome()) + "\n";
            res += "average_treatment: " + std::to_string(average_treatment()) + "\n";
            res += "average_instrument: " + std::to_string(average_instrument()) + "\n";
            res += "average_regularized_instrument: " + std::to_string(average_regularized_instrument()) + "\n";
            res += "numerator: " + std::to_string(numerator) + "\n";
            res += "denominator: " + std::to_string(denominator) + "\n";
            res += "is_stop: " + std::to_string(is_stop) + "\n";
            res += "num_samples: " + std::to_string(num_samples) + "\n";
            res += "sum_node_z_squared: " + std::to_string(sum_node_z_squared) + "\n";
            res += "y0_squared: " + std::to_string(y0_squared) + "\n";
            res += "y1_squared: " + std::to_string(y1_squared) + "\n";
            res += "sum_node: " + std::to_string(sum_node) + "\n";
            res += "num_node_small_z: " + std::to_string(num_node_small_z) + "\n";
            res += "quantiles_calcers: ";
            res += "size: " + std::to_string(quantiles_calcers.size()) + "\n";
            res += "quantiles: ";
            for (const auto & quantile : quantiles)
            {
                res += "size: " + std::to_string(quantile.size()) + "\n";
                for (const auto & val : quantile)
                    res += std::to_string(val) + " ";
                res += "\n";
            }
            res += "split_nodes: \n";
            for (const auto & split_node : split_nodes)
            {
                for (const auto & node : split_node)
                    res += node.toString();
                res += "\n";
            }

            res += "father: " + std::to_string(father) + "\n";
            return res;
        }

        void serialize(WriteBuffer & buf) const
        {
            writeBinary(sum_weight, buf);
            writeBinary(total_outcome, buf);
            writeBinary(total_treatment, buf);
            writeBinary(total_instrument, buf);
            writeBinary(total_outcome_treatment, buf);
            writeBinary(possible_split_vars, buf);
            writeBinary(numerator, buf);
            writeBinary(denominator, buf);
            writeBinary(is_stop, buf);
            writeBinary(num_samples, buf);
            writeBinary(sum_node_z_squared, buf);
            writeBinary(y0_squared, buf);
            writeBinary(y1_squared, buf);
            writeBinary(sum_node, buf);
            writeBinary(num_node_small_z, buf);

            writeVarUInt(quantiles_calcers.size(), buf);
            for (const auto & quantile : quantiles_calcers)
                const_cast<CausalQuantileTDigest<Float32> &>(quantile).serialize(buf);

            writeVarUInt(quantiles.size(), buf);
            for (const auto & quantile : quantiles)
            {
                writeVarUInt(quantile.size(), buf);
                for (const auto & val : quantile)
                    writeBinary(val, buf);
            }

            writeVarUInt(split_nodes.size(), buf);
            for (const auto & split_node : split_nodes)
            {
                writeVarUInt(split_node.size(), buf);
                for (const auto & node : split_node)
                    node.serialize(buf);
            }

            writeBinary(father, buf);
        }

        void deserialize(ReadBuffer & buf)
        {
            readBinary(sum_weight, buf);
            readBinary(total_outcome, buf);
            readBinary(total_treatment, buf);
            readBinary(total_instrument, buf);
            readBinary(total_outcome_treatment, buf);
            readBinary(possible_split_vars, buf);
            readBinary(numerator, buf);
            readBinary(denominator, buf);
            readBinary(is_stop, buf);
            readBinary(num_samples, buf);
            readBinary(sum_node_z_squared, buf);
            readBinary(y0_squared, buf);
            readBinary(y1_squared, buf);
            readBinary(sum_node, buf);
            readBinary(num_node_small_z, buf);
            size_t quantiles_size;
            readVarUInt(quantiles_size, buf);
            quantiles_calcers.resize(quantiles_size);

            for (auto & quantile : quantiles_calcers)
                quantile.deserialize(buf);

            readVarUInt(quantiles_size, buf);
            quantiles.resize(quantiles_size);
            for (auto & quantile : quantiles)
            {
                size_t quantile_size;
                readVarUInt(quantile_size, buf);
                quantile.resize(quantile_size);
                for (auto & val : quantile)
                    readBinary(val, buf);
            }

            size_t split_nodes_size;
            readVarUInt(split_nodes_size, buf);
            split_nodes.resize(split_nodes_size);
            for (auto & split_node : split_nodes)
            {
                size_t split_node_size;
                readVarUInt(split_node_size, buf);
                split_node.resize(split_node_size);
                for (auto & node : split_node)
                    node.deserialize(buf);
            }

            readBinary(father, buf);
        }

        void merge(const CalcNodeInfo & other, const CausalForestState & state)
        {
            if (quantiles_calcers.size() != other.quantiles_calcers.size())
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "quantiles_calcers.size() != other.quantiles_calcers.size()");
            if (quantiles.size() != other.quantiles.size())
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "quantiles.size() != other.quantiles.size()");
            if (split_nodes.size() != other.split_nodes.size())
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "split_nodes.size() != other.split_nodes.size()");
            if (state == CausalForestState::Init || state == CausalForestState::Honesty)
            {
                total_outcome += other.total_outcome;
                total_treatment += other.total_treatment;
                total_instrument += other.total_instrument;
                total_outcome_treatment += other.total_outcome_treatment;

                sum_weight += other.sum_weight;
                num_samples += other.num_samples;
                sum_node_z_squared += other.sum_node_z_squared;
                y0_squared += other.y0_squared;
                y1_squared += other.y1_squared;
            }

            is_stop = is_stop && other.is_stop;
            if (possible_split_vars.size() != other.possible_split_vars.size())
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "possible_split_vars.size() != other.possible_split_vars.size()");

            if (state == CausalForestState::CalcNumerAndDenom)
            {
                numerator += other.numerator;
                denominator += other.denominator;
            }
            else if (state == CausalForestState::FindBestSplitPre)
            {
                sum_node += other.sum_node;
                num_node_small_z += other.num_node_small_z;
                for (size_t i = 0; i < quantiles_calcers.size(); ++i)
                    quantiles_calcers[i].merge(other.quantiles_calcers[i]);
            }
            else if (state == CausalForestState::FindBestSplit)
            {
                for (size_t i = 0; i < split_nodes.size(); ++i)
                {
                    if (split_nodes[i].size() != other.split_nodes[i].size())
                        throw Exception(ErrorCodes::BAD_ARGUMENTS, "split_nodes[i].size() != other.split_nodes[i].size()");
                    for (size_t j = 0; j < split_nodes[i].size(); ++j)
                        split_nodes[i][j].merge(other.split_nodes[i][j]);
                }
            }
        }

        void updateStop(CausalForestState state, const TreeOptions & tree_options)
        {
            if (std::abs(sum_weight) <= 1e-16) // NOLINT
                is_stop = true;
            if (num_samples <= tree_options.min_node_size)
                is_stop = true;
            if (!tree_options.causal_tree && state >= CausalForestState::CalcNumerAndDenom && std::abs(denominator) <= 1e-10)
                is_stop = true;
        }

        Float64 responses_by_sample(Float64 response, Float64 treatment, Float64 instrument) const // NOLINT
        {
            Float64 regularized_instrument = (1 - reduced_form_weight) * instrument + reduced_form_weight * treatment;
            Float64 residual = (response - average_outcome()) - local_average_treatment_effect() * (treatment - average_treatment());
            return (regularized_instrument - average_regularized_instrument()) * residual;
        }

        Float64 average_outcome() const { return total_outcome / sum_weight; } // NOLINT
        Float64 average_treatment() const { return total_treatment / sum_weight; } // NOLINT
        Float64 average_instrument() const { return total_instrument / sum_weight; } // NOLINT
        Float64 average_regularized_instrument() const { // NOLINT
          return (1 - reduced_form_weight) * average_instrument() + reduced_form_weight * average_treatment(); // NOLINT
        }
        Float64 local_average_treatment_effect() const { return numerator / denominator; } // NOLINT
        Float64 weight_sum_node() const { return sum_weight; } // NOLINT
        Float64 sum_node_z() const { return total_instrument; } // NOLINT
                                                                //
        UInt64 cnt1() const { return static_cast<UInt64>(total_treatment); } // NOLINT
        UInt64 cnt0() const { return static_cast<UInt64>(num_samples - total_treatment); } // NOLINT

        Float64 y1() const { return total_outcome_treatment; }
        Float64 y0() const { return total_outcome - total_outcome_treatment; } // NOLINT

        Float64 size_node() const { return sum_node_z_squared - sum_node_z() * sum_node_z() / weight_sum_node(); } // NOLINT
        //   double min_child_size = size_node * alpha;
        Float64 min_child_size(Float64 alpha) const { return size_node() * alpha; } // NOLINT
        //   double mean_z_node = sum_node_z / weight_sum_node;
        Float64 mean_z_node() const { return sum_node_z() / weight_sum_node(); } // NOLINT
                                                                                 //
        Float64 effect() const {
          /*
           *         tau = y1 - y0
        tr_var = y1_square - y1 ** 2
        con_var = y0_square - y0 ** 2
        # effect = 0.5 * tau * tau * (cnt1 + cnt0) - 0.5 * 2 * (cnt1 + cnt0) * (
        #         tr_var / cnt1 + con_var / cnt0
        # )
        effect = split_alpha * 0.5 * tau * tau * (cnt1+cnt0) - (1-split_alpha) * 0.5 * (1+train_to_est_ratio) * (cnt1+cnt0) * (tr_var/cnt1 + con_var/cnt0)
        return effect
*/
            Float64 y0_avg = y0() / cnt0();
            Float64 y1_avg = y1() / cnt1();
            Float64 y0_squared_avg = y0_squared / cnt0();
            Float64 y1_squared_avg = y1_squared / cnt1();
            Float64 tau = y1_avg - y0_avg;
            Float64 tr_var = y1_squared_avg - y1_avg * y1_avg;
            Float64 con_var = y0_squared_avg - y0_avg * y0_avg;
            Float64 split_alpha = 1;
            Float64 train_to_est_ratio = 1;
            Float64 effect = split_alpha * 0.5 * tau * tau * (cnt1() + cnt0()) - (1 - split_alpha) * 0.5 * (1 + train_to_est_ratio) * (cnt1() + cnt0()) * (tr_var / cnt1() + con_var / cnt0());
            return effect;
        }

        void find_best_split_value(const size_t var, size_t & best_var, double & best_value, double & best_decrease, bool & best_send_missing_left, const TreeOptions & tree_options); // NOLINT
                                                                                                                                                                                       //
        void find_best_split_value_cf(const size_t var, size_t & best_var, double & best_value, double & best_decrease, bool & best_send_missing_left, const TreeOptions & tree_options); // NOLINT

        constexpr static Float64 reduced_form_weight = 0;
    };

    struct NodeInfo
    {
        Int16 split_var = -1;
        Float64 split_value = 0.0;
        UInt16 left_child = 0;
        UInt16 right_child = 0;
        bool is_leaf = false;
        std::optional<CalcNodeInfo> calc_node = std::nullopt;

        bool isNeedPure()
        {
            return is_leaf && calc_node.has_value() && calc_node.value().num_samples <= 1;
        }

        String toString() const
        {
            String res;
            res += "split_var: " + std::to_string(split_var) + "\n";
            res += "split_value: " + std::to_string(split_value) + "\n";
            res += "left_child: " + std::to_string(left_child) + "\n";
            res += "right_child: " + std::to_string(right_child) + "\n";
            res += "is_leaf: " + std::to_string(is_leaf) + "\n";
            res += "calc_node: " + (calc_node.has_value() ? calc_node->toString() : "null") + "\n";
            return res;
        }

        void serialize(WriteBuffer & buf) const
        {
            writeBinary(split_var, buf);
            writeBinary(split_value, buf);
            writeBinary(left_child, buf);
            writeBinary(right_child, buf);
            writeBinary(is_leaf, buf);
            writeBinary(calc_node.has_value(), buf);
            if (calc_node.has_value())
                calc_node->serialize(buf);
        }

        void deserialize(ReadBuffer & buf)
        {
            readBinary(split_var, buf);
            readBinary(split_value, buf);
            readBinary(left_child, buf);
            readBinary(right_child, buf);
            readBinary(is_leaf, buf);
            bool has_calc_node;
            readBinary(has_calc_node, buf);
            if (has_calc_node)
            {
                calc_node = CalcNodeInfo();
                calc_node->deserialize(buf);
            }
        }

        void merge(const NodeInfo & other, const CausalForestState & state)
        {
            if (split_var != other.split_var)
                throw std::runtime_error("split_var != other.split_var");
            if (calc_node.has_value() != other.calc_node.has_value())
                throw std::runtime_error("calc_node.has_value() != other.calc_node.has_value()");
            if (calc_node.has_value())
                calc_node->merge(other.calc_node.value(), state);
        }

    };

    void merge(const Tree & other, const CausalForestState & state)
    {
        if (nodes.size() != other.nodes.size())
            throw std::runtime_error("nodes.size() != other.nodes.size()");

        for (size_t i = (state == CausalForestState::Honesty ? 0 : calc_index); i < nodes.size(); ++i)
            nodes[i].merge(other.nodes[i], state);
    }


    std::vector<size_t> createSplitVariable(const ForestOptions& forest_options, const TreeOptions& tree_options);

    void drawFisherYates(std::vector<size_t>& result, // NOLINT
                                          size_t max,
                                          const std::set<size_t>& skip,
                                          size_t num_samples);

    void drawSimple(std::vector<size_t>& result, // NOLINT
                                    size_t max,
                                    const std::set<size_t>& skip,
                                    size_t num_samples);

    void draw(std::vector<size_t>& result,
                            size_t max,
                            const std::set<size_t>& skip,
                            size_t num_samples);

                                  //

    UInt64 bucket_num;
    UInt64 honesty_bucket_num;
    UInt64 random_seed;
    std::vector<NodeInfo> nodes;
    std::mt19937_64 random_number_generator;
    size_t calc_index = 0;
};

}
