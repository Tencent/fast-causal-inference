#include <type_traits>
#include <AggregateFunctions/AggregateFunctionCausalForestTree.h>
#include <boost/algorithm/string/trim.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/ptree.hpp>
#include <Common/Exception.h>
#include <AggregateFunctions/AggregateFunctionCausalForestOption.h>
#include <Columns/IColumn.h>
#include <base/types.h>

namespace DB
{

Tree::Tree(UInt64 bucket_num_, UInt64 honesty_bucket_num_, UInt64 random_seed_, const ForestOptions& forest_options, const TreeOptions& tree_options) : bucket_num(bucket_num_), honesty_bucket_num(honesty_bucket_num_), random_seed(random_seed_), // NOLINT
  random_number_generator(random_seed_) 
{
    calc_index = 0;
    nodes.push_back(NodeInfo());
    nodes[0].calc_node = CalcNodeInfo();
    nodes[0].calc_node->possible_split_vars = createSplitVariable(forest_options, tree_options);
    nodes[0].is_leaf = true;
}

void Tree::drawSimple(std::vector<size_t>& result,
                                size_t max,
                                const std::set<size_t>& skip,
                                size_t num_samples) 
{
    result.resize(num_samples);

    // Set all to not selected
    std::vector<bool> temp;
    temp.resize(max, false);

    nonstd::uniform_int_distribution<size_t> unif_dist(0, max - 1 - skip.size());
    for (size_t i = 0; i < num_samples; ++i) 
    {
        size_t draw;
        do 
        {
            draw = unif_dist(random_number_generator);
            for (const auto& skip_value : skip) 
            {
                if (draw >= skip_value) 
                {
                    ++draw;
                }
            }
        } while (temp[draw]);
        temp[draw] = true;
        result[i] = draw;
    }
}

void Tree::drawFisherYates(std::vector<size_t>& result,
                                      size_t max,
                                      const std::set<size_t>& skip,
                                      size_t num_samples) 
{

    // Populate result vector with 0,...,max-1
    result.resize(max);
    std::iota(result.begin(), result.end(), 0);

    // Remove values that are to be skipped
    std::for_each(skip.rbegin(), skip.rend(),
                  [&](size_t i) { result.erase(result.begin() + i); }
    );

    // Draw without replacement using Fisher Yates algorithm
    nonstd::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < num_samples; ++i) 
    {
        size_t j = static_cast<size_t>(i + distribution(random_number_generator) * (max - skip.size() - i));
        std::swap(result[i], result[j]);
    }

    result.resize(num_samples);
}

void Tree::draw(std::vector<size_t>& result,
                         size_t max,
                         const std::set<size_t>& skip,
                         size_t num_samples) 
{
    if (num_samples < max / 10) 
    {
      drawSimple(result, max, skip, num_samples);
    } 
    else 
    {
      drawFisherYates(result, max, skip, num_samples);
    }
}

std::vector<size_t> Tree::createSplitVariable(const ForestOptions& forest_options, const TreeOptions& tree_options)
{
    // Randomly select an mtry for this tree based on the overall setting.
    size_t num_independent_variables = forest_options.arguments_size - forest_options.disallowed_split_variables.size();

    nonstd::poisson_distribution<size_t> distribution(static_cast<double>(tree_options.mtry));
    size_t mtry_sample = distribution(random_number_generator);
    size_t split_mtry = std::max<size_t>(std::min<size_t>(mtry_sample, num_independent_variables), 1uL);

    std::vector<size_t> result;
    draw(result,
        forest_options.arguments_size,
        forest_options.disallowed_split_variables,
        split_mtry);

    return result;
}

void Tree::addInit(const IColumn ** columns, size_t row_num,
    const ForestOptions & forest_options, const TreeOptions &)
{
    auto & calc_node = nodes[0].calc_node.value();
    Float64 weight = columns[forest_options.weight_index]->getFloat64(row_num);
    calc_node.total_outcome += weight * columns[forest_options.outcome_index]->getFloat64(row_num);
    calc_node.total_treatment += weight * columns[forest_options.treatment_index]->getFloat64(row_num);
    calc_node.total_outcome_treatment += weight * columns[forest_options.outcome_index]->getFloat64(row_num) * columns[forest_options.treatment_index]->getFloat64(row_num);
    calc_node.total_instrument += weight * columns[forest_options.instrument_index]->getFloat64(row_num);
    calc_node.sum_weight += weight;
    calc_node.num_samples ++;
    calc_node.sum_node_z_squared += weight * pow(columns[forest_options.instrument_index]->getFloat64(row_num), 2);
    if (columns[forest_options.treatment_index]->get64(row_num) == 0)
        calc_node.y0_squared += weight * pow(columns[forest_options.outcome_index]->getFloat64(row_num), 2);
    else
        calc_node.y1_squared += weight * pow(columns[forest_options.outcome_index]->getFloat64(row_num), 2);
}

bool Tree::getCalcNode(const IColumn ** columns, size_t row_num, size_t & res)
{
    size_t root = 0;
    while (root < nodes.size() && !nodes[root].is_leaf)
    {
        const auto & value = columns[nodes[root].split_var]->getFloat64(row_num);
        if (value > nodes[root].split_value + 1e-6)
        {
            root = nodes[root].right_child;
        }
        else
            root = nodes[root].left_child;
    }
    if (root >= nodes.size())
        return false;
    if (!nodes[root].calc_node.has_value() || nodes[root].calc_node.value().is_stop)
        return false;
    res = root;
    return true;
} 

void Tree::predict(const ColumnsWithTypeAndName & arguments, UInt64 row_num, PaddedPODArray<Float64> & average_value) const
{
    size_t root = 0;
    while (root < nodes.size() && !nodes[root].is_leaf)
    {
        const auto & value = arguments[nodes[root].split_var].column->getFloat64(row_num);
        if (value > nodes[root].split_value + 1e-6)
        {
            root = nodes[root].right_child;
        }
        else
            root = nodes[root].left_child;
    }
    if (root >= nodes.size())
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Wrong tree structure");
    if (!nodes[root].calc_node.has_value())
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "no calc node");
    const auto & calc_node = nodes[root].calc_node.value();

    average_value[0] += calc_node.total_outcome / calc_node.num_samples;
    average_value[1] += calc_node.total_treatment / calc_node.num_samples;
    average_value[2] += calc_node.total_instrument / calc_node.num_samples;
    average_value[3] += calc_node.total_outcome_treatment / calc_node.num_samples;
    average_value[4] += calc_node.sum_node_z_squared / calc_node.num_samples;
    average_value[5] = average_value[4];
    average_value[6] += calc_node.sum_weight / calc_node.num_samples;
}

void Tree::addCalcNumerAndDenom(const IColumn ** columns, size_t row_num, // NOLINT
    const ForestOptions & forest_options, const TreeOptions &)
{
    size_t calc_node_index;
    if (!getCalcNode(columns, row_num, calc_node_index))
        return;
    auto & node = nodes[calc_node_index].calc_node.value();
    Float64 weight = columns[forest_options.weight_index]->getFloat64(row_num);
    Float64 outcome = columns[forest_options.outcome_index]->getFloat64(row_num);
    Float64 treatment = columns[forest_options.treatment_index]->getFloat64(row_num);
    Float64 instrument = columns[forest_options.instrument_index]->getFloat64(row_num);
    Float64 regularized_instrument = (1 - node.reduced_form_weight) * instrument
                                + node.reduced_form_weight * treatment;

    node.numerator += weight * (regularized_instrument - node.average_regularized_instrument()) * (outcome - node.average_outcome());
    node.denominator += weight * (regularized_instrument - node.average_regularized_instrument()) * (treatment - node.average_treatment());
}

void Tree::addFindBestSplitPre(const IColumn ** columns, size_t row_num, // NOLINT
    const ForestOptions & forest_options, const TreeOptions &)
{
    size_t calc_node_index;
    if (!getCalcNode(columns, row_num, calc_node_index))
        return;
    auto & node = nodes[calc_node_index].calc_node.value();
    Float64 weight = columns[forest_options.weight_index]->getFloat64(row_num);
    Float64 outcome = columns[forest_options.outcome_index]->getFloat64(row_num);
    Float64 treatment = columns[forest_options.treatment_index]->getFloat64(row_num);
    Float64 instrument = columns[forest_options.instrument_index]->getFloat64(row_num);
    node.sum_node += weight * node.responses_by_sample(outcome, treatment, instrument);

    if (instrument < node.mean_z_node())
        node.num_node_small_z++;

    if (node.possible_split_vars.size() != node.quantiles_calcers.size())
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Tree::addFindBestSplitPre: node.possible_split_vars.size() != node.quantiles_calcers.size()");
    }
    for (size_t i = 0; i < node.possible_split_vars.size(); ++i)
    {
        auto var = node.possible_split_vars[i];
        auto & quantiles_calcer = node.quantiles_calcers[i];
        quantiles_calcer.add(columns[var]->getFloat32(row_num));
    }
}

void Tree::addFindBestSplit(const IColumn ** columns, size_t row_num, const ForestOptions & forest_options, const TreeOptions &)
{
    size_t calc_node_index;
    if (!getCalcNode(columns, row_num, calc_node_index))
        return;
    auto & node = nodes[calc_node_index].calc_node.value();
    const Float64 weight = columns[forest_options.weight_index]->getFloat64(row_num);
    const Float64 outcome = columns[forest_options.outcome_index]->getFloat64(row_num);
    const Float64 treatment = columns[forest_options.treatment_index]->getFloat64(row_num);
    const Float64 instrument = columns[forest_options.instrument_index]->getFloat64(row_num);

    for (size_t i = 0; i < node.possible_split_vars.size(); ++i)
    {
        auto var = node.possible_split_vars[i];
        auto value = columns[var]->getFloat64(row_num);
        auto & quantile = node.quantiles[i];
        Int64 left = 0, right = quantile.size() - 1, res = -1;
        while (left <= right)
        {
            auto mid = (left + right) / 2;
            if (value <= quantile[mid] + 1e-6)
            {
                res = mid;
                right = mid - 1;
            }
            else
                left = mid + 1;
        }

        if (res == -1)
            res = quantile.size() - 1;
        auto & split_node = node.split_nodes[i][res];
        split_node.weight_sums += weight;
        split_node.counter++;
        split_node.sums += weight * node.responses_by_sample(outcome, treatment, instrument);
        split_node.sums_z += weight * instrument;
        split_node.sums_z_squared += weight * instrument * instrument;
        split_node.total_outcome += weight * outcome;
        split_node.total_treatment += weight * treatment;
        split_node.total_outcome_treatment += weight * outcome * treatment;
        if (instrument < node.mean_z_node())
            split_node.num_small_z++;
        if (treatment == 0)
            split_node.y0_squared += outcome * outcome;
        else
            split_node.y1_squared += outcome * outcome;
    }
}

void Tree::addHonesty(const IColumn ** columns, size_t row_num, const ForestOptions & forest_options, const TreeOptions & tree_options)
{
    size_t calc_node_index;
    if (!getCalcNode(columns, row_num, calc_node_index))
        return;
    CalcNodeInfo & node = nodes[calc_node_index].calc_node.value(); // NOLINT
    /*
    Float64 weight = columns[forest_options.weight_index]->getFloat64(row_num);
    node.total_outcome += weight * columns[forest_options.outcome_index]->getFloat64(row_num);
    node.total_treatment += weight * columns[forest_options.treatment_index]->getFloat64(row_num);
    node.total_outcome_treatment += weight * columns[forest_options.outcome_index]->getFloat64(row_num) * columns[forest_options.treatment_index]->getFloat64(row_num);
    node.total_instrument += weight * columns[forest_options.instrument_index]->getFloat64(row_num);
    node.sum_weight += weight;
    node.num_samples ++;
    node.sum_node_z_squared += weight * pow(columns[forest_options.instrument_index]->getFloat64(row_num), 2);
    if (columns[forest_options.treatment_index]->get64(row_num) == 0)
        node.y0_squared += weight * pow(columns[forest_options.outcome_index]->getFloat64(row_num), 2);
    else
        node.y1_squared += weight * pow(columns[forest_options.outcome_index]->getFloat64(row_num), 2);
    */
    auto add_node = [&](CalcNodeInfo & node) 
    {
        Float64 weight = columns[forest_options.weight_index]->getFloat64(row_num);
        node.total_outcome += weight * columns[forest_options.outcome_index]->getFloat64(row_num);
        node.total_treatment += weight * columns[forest_options.treatment_index]->getFloat64(row_num);
        node.total_outcome_treatment += weight * columns[forest_options.outcome_index]->getFloat64(row_num) * columns[forest_options.treatment_index]->getFloat64(row_num);
        node.total_instrument += weight * columns[forest_options.instrument_index]->getFloat64(row_num);
        node.sum_weight += weight;
        node.num_samples ++;
        node.sum_node_z_squared += weight * pow(columns[forest_options.instrument_index]->getFloat64(row_num), 2);
        if (columns[forest_options.treatment_index]->get64(row_num) == 0)
            node.y0_squared += weight * pow(columns[forest_options.outcome_index]->getFloat64(row_num), 2);
        else
            node.y1_squared += weight * pow(columns[forest_options.outcome_index]->getFloat64(row_num), 2);
    };
    if (tree_options.causal_tree) 
    {
        size_t root = 0;
        add_node(nodes[root].calc_node.value());

        while (root < nodes.size() && !nodes[root].is_leaf)
        {
            const auto & value = columns[nodes[root].split_var]->getFloat64(row_num);
            if (value > nodes[root].split_value + 1e-6)
            {
                root = nodes[root].right_child;
            }
            else
                root = nodes[root].left_child;
            if (root < nodes.size())
                add_node(nodes[root].calc_node.value());
        }
    } 
    else
        add_node(node);
}


void Tree::add(const IColumn ** columns, size_t row_num, 
    const ForestOptions & forest_options, const TreeOptions & tree_options) // NOLINT
{
    if (forest_options.state == CausalForestState::Init)
        addInit(columns, row_num, forest_options, tree_options);
    else if (forest_options.state == CausalForestState::CalcNumerAndDenom)
        addCalcNumerAndDenom(columns, row_num, forest_options, tree_options);
    else if (forest_options.state == CausalForestState::FindBestSplitPre)
        addFindBestSplitPre(columns, row_num, forest_options, tree_options);
    else if (forest_options.state == CausalForestState::FindBestSplit)
        addFindBestSplit(columns, row_num, forest_options, tree_options);
    else if (forest_options.state == CausalForestState::Honesty)
      addHonesty(columns, row_num, forest_options, tree_options);
    else if (forest_options.state != CausalForestState::Finish)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Tree::add: unknown forest_options.state");
}



void Tree::serialize(WriteBuffer & buf) const
{
    writeVarUInt(bucket_num, buf);
    writeVarUInt(honesty_bucket_num, buf);
    writeVarUInt(random_seed, buf);
    writeVarUInt(nodes.size(), buf);
    for (const auto & node : nodes)
        node.serialize(buf);
    writeVarUInt(calc_index, buf);
    std::ostringstream ostr;
    ostr << random_number_generator;
    std::string str = ostr.str();
    writeStringBinary(str, buf);
}

void Tree::deserialize(ReadBuffer & buf)
{
    readVarUInt(bucket_num, buf);
    readVarUInt(honesty_bucket_num, buf);
    readVarUInt(random_seed, buf);
    size_t nodes_size;
    readVarUInt(nodes_size, buf);
    nodes.resize(nodes_size);
    for (auto & node : nodes)
        node.deserialize(buf);
    readVarUInt(calc_index, buf);
    std::string str;
    readStringBinary(str, buf);
    std::istringstream istr(str);
    istr >> random_number_generator;
}

void Tree::CalcNodeInfo::find_best_split_value_cf(const size_t var, size_t & best_var, double & best_value, double & best_decrease, bool & best_send_missing_left, const TreeOptions &) // NOLINT 
{
    std::vector<Float64> & possible_split_values = quantiles[var];
    // Try next variable if all equal for this
    if (possible_split_values.size() < 2)
        return;

    if (possible_split_values.size() != split_nodes[var].size())
      throw Exception(ErrorCodes::BAD_ARGUMENTS, "possible_split_values.size() != split_nodes[var].size()");

    UInt64 num_splits = possible_split_values.size();
    std::vector<UInt64> cnts0(num_splits, 0), cnts1(num_splits, 0);
    std::vector<Float64> ys0(num_splits, 0), ys1(num_splits, 0);
    std::vector<Float64> ys1_squared(num_splits, 0), ys0_squared(num_splits, 0);

    UInt64 sum_cnt0 = 0, sum_cnt1 = 0;
    Float64 sum_y0 = 0, sum_y1 = 0;
    Float64 sum_y1_squared = 0, sum_y0_squared = 0;

    for (size_t i = 0; i < possible_split_values.size(); i++)
    {
        UInt64 count = split_nodes[var][i].counter;
        UInt64 cnt1 = static_cast<UInt64>(split_nodes[var][i].total_treatment);
        UInt64 cnt0 = static_cast<UInt64>(count - split_nodes[var][i].total_treatment);
        cnts1[i] = cnt1;
        cnts0[i] = cnt0;
        sum_cnt1 += cnt1;
        sum_cnt0 += cnt0;

        Float64 y1 = split_nodes[var][i].total_outcome_treatment;
        Float64 y0 = (split_nodes[var][i].total_outcome - split_nodes[var][i].total_outcome_treatment);
        ys1[i] = y1;
        ys0[i] = y0;
        sum_y1 += y1;
        sum_y0 += y0;

        Float64 y1_square = split_nodes[var][i].y1_squared;
        Float64 y0_square = split_nodes[var][i].y0_squared;
        ys1_squared[i] = y1_square;
        ys0_squared[i] = y0_square;
        sum_y1_squared += y1_square;
        sum_y0_squared += y0_square;
    }

    UInt64 left_cnt0 = 0, left_cnt1 = 0;
    Float64 left_y0 = 0, left_y1 = 0;
    Float64 left_y0_squared = 0, left_y1_squared = 0;

    const Float64 train_to_est_ratio = 1.0;
    const Float64 split_alpha = 1.0;
    for (size_t i = 0; i < possible_split_values.size(); i++)
    {
        left_cnt0 += cnts0[i];
        left_cnt1 += cnts1[i];
        left_y0 += ys0[i];
        left_y1 += ys1[i];
        left_y0_squared += ys0_squared[i];
        left_y1_squared += ys1_squared[i];

        UInt64 right_cnt0 = sum_cnt0 - left_cnt0;
        UInt64 right_cnt1 = sum_cnt1 - left_cnt1;
        Float64 right_y0 = sum_y0 - left_y0;
        Float64 right_y1 = sum_y1 - left_y1;
        Float64 right_y0_squared = sum_y0_squared - left_y0_squared;
        Float64 right_y1_squared = sum_y1_squared - left_y1_squared;

        Float64 left_y0_avg = left_cnt0 > 0 ? left_y0 / left_cnt0 : 0;
        Float64 left_y1_avg = left_cnt1 > 0 ? left_y1 / left_cnt1 : 0;
        Float64 right_y0_avg = right_cnt0 > 0 ? right_y0 / right_cnt0 : 0;
        Float64 right_y1_avg = right_cnt1 > 0 ? right_y1 / right_cnt1 : 0;
        Float64 left_y0_squared_avg = left_cnt0 > 0 ? left_y0_squared / left_cnt0 : 0;
        Float64 left_y1_squared_avg = left_cnt1 > 0 ? left_y1_squared / left_cnt1 : 0;
        Float64 right_y0_squared_avg = right_cnt0 > 0 ? right_y0_squared / right_cnt0 : 0;
        Float64 right_y1_squared_avg = right_cnt1 > 0 ? right_y1_squared / right_cnt1 : 0;

        // left_effect = split_alpha * 0.5 * tau * tau * (cnt1+cnt0) - (1-split_alpha) * 0.5 * (1+train_to_est_ratio) * (cnt1+cnt0) * (tr_var/cnt1 + con_var/cnt0)
        Float64 left_effect = split_alpha * 0.5 * (left_y1_avg - left_y0_avg) * (left_y1_avg - left_y0_avg) * (left_cnt1 + left_cnt0)
            - (1 - split_alpha) * 0.5 * (1 + train_to_est_ratio) * (left_cnt1 + left_cnt0) * ((left_y1_squared_avg / left_cnt1) + (left_y0_squared_avg / left_cnt0));
        Float64 right_effect = split_alpha * 0.5 * (right_y1_avg - right_y0_avg) * (right_y1_avg - right_y0_avg) * (right_cnt1 + right_cnt0)
            - (1 - split_alpha) * 0.5 * (1 + train_to_est_ratio) * (right_cnt1 + right_cnt0) * ((right_y1_squared_avg / right_cnt1) + (right_y0_squared_avg / right_cnt0));
        Float64 decrease = left_effect + right_effect - effect();
        if (isnan(decrease) || isinf(decrease))
            continue;
        if (decrease > best_decrease) 
        {
            best_value = possible_split_values[i];
            best_var = var;
            best_decrease = decrease;
            best_send_missing_left = true;
        }
    }
}

void Tree::CalcNodeInfo::find_best_split_value(const size_t var, size_t & best_var, double & best_value, double & best_decrease, bool & best_send_missing_left, const TreeOptions & tree_options) // NOLINT
{
    std::vector<Float64> & possible_split_values = quantiles[var];

    // Try next variable if all equal for this
    if (possible_split_values.size() < 2)
      return;

    size_t num_splits = possible_split_values.size() - 1;

    std::vector<size_t> counter(num_splits, 0);
    std::vector<Float64> weight_sums(num_splits, 0);
    std::vector<Float64> sums(num_splits, 0);
    std::vector<size_t> num_small_z(num_splits, 0);
    std::vector<Float64> sums_z(num_splits, 0);
    std::vector<Float64> sums_z_squared(num_splits, 0);

    size_t split_index = 0;
    if (possible_split_values.size() != split_nodes[var].size())
      throw Exception(ErrorCodes::BAD_ARGUMENTS, "possible_split_values.size() != split_nodes[var].size()");

    for (size_t i = 0; i < num_splits - 1; i++) 
    {
        Float64 sample_value = possible_split_values[i];

        weight_sums[split_index] += split_nodes[var][i].weight_sums;
        sums[split_index] += split_nodes[var][i].sums;
        counter[split_index] += split_nodes[var][i].counter;

        sums_z[split_index] += split_nodes[var][i].sums_z;

        sums_z_squared[split_index] += split_nodes[var][i].sums_z_squared;
        num_small_z[split_index] += split_nodes[var][i].num_small_z;
        Float64 next_sample_value = possible_split_values[i + 1];
        // if the next sample value is different, including the transition (..., NaN, Xij, ...)
        // then move on to the next bucket (all logical operators with NaN evaluates to false by default)
        if (sample_value != next_sample_value && !std::isnan(next_sample_value))
          ++split_index;
    }

    size_t n_left = 0;
    double weight_sum_left = 0;
    double sum_left = 0;
    double sum_left_z = 0;
    double sum_left_z_squared = 0;
    size_t num_left_small_z = 0;

    // Compute decrease of impurity for each possible split.
    for (bool send_left : {true, false}) 
    {
        if (!send_left) 
        {
            // A normal split with no NaNs, so we can stop early.
            break;
            // It is not necessary to adjust n_right or sum_right as the the missing
            // part is included in the total sum.
            /*
            n_left = 0;
            weight_sum_left = 0;
            sum_left = 0;
            sum_left_z = 0;
            sum_left_z_squared = 0;
            num_left_small_z = 0;
            */
        }

        for (size_t i = 0; i < num_splits; ++i) 
        {
            // not necessary to evaluate sending right when splitting on NaN.
            if (i == 0 && !send_left)
                continue;

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
            if ((!tree_options.causal_tree && size_left < min_child_size(tree_options.alpha)) || (tree_options.imbalance_penalty > 0.0 && size_left == 0))
                continue;

            // Calculate relevant quantities for the right child.
            double weight_sum_right = weight_sum_node() - weight_sum_left;
            double sum_right = sum_node - sum_left;
            double sum_right_z_squared = sum_node_z_squared - sum_left_z_squared;
            double sum_right_z = sum_node_z() - sum_left_z;
            double size_right = sum_right_z_squared - sum_right_z * sum_right_z / weight_sum_right;

            // Skip this split if the right child's variance is too small.
            if ((!tree_options.causal_tree && size_right < min_child_size(tree_options.alpha)) || (tree_options.imbalance_penalty > 0.0 && size_right == 0)) 
                continue;

            // Calculate the decrease in impurity.
            double decrease = sum_left * sum_left / weight_sum_left + sum_right * sum_right / weight_sum_right;
            // Penalize splits that are too close to the edges of the data.
            decrease -= tree_options.imbalance_penalty * (1.0 / size_left + 1.0 / size_right);

            // Save this split if it is the best seen so far.
            if (decrease > best_decrease) 
            {
                best_value = possible_split_values[i];
                best_var = var;
                best_decrease = decrease;
                best_send_missing_left = send_left;
            }
        }
    }
}

}
