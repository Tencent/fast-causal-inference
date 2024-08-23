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
#include <pqxx/params.hxx>
#include <Common/Exception.h>
#include <base/defines.h>
#include <clickhouse_grpc.pb.h>
#include <cctype>
#include <regex>
#include <fstream>
template <typename T> 
void LOG(T t, String name = "") {
  std::fstream f("log", std::ios::app | std::ios::out);
  if (!name.empty())
    f << name << ": ";
  f << t << std::endl;
  f.close();
}

template <typename Iter>
void LOG(Iter begin, Iter end, String name = "") {
  std::fstream f("log", std::ios::app | std::ios::out);
  if (!name.empty())
    f << name << ": ";
  for (auto it = begin; it != end; ++it) {
    f << *it << " ";
  }
  f << std::endl;
  f.close();
}

template <typename T>
void LOG(const boost::numeric::ublas::matrix<T>& ma, String name = "") {
  std::fstream f("log", std::ios::app | std::ios::out);
  if (!name.empty())
    f << name << ": ";
  for (size_t i = 0; i < ma.size1(); i++) {
    for (size_t j = 0; j < ma.size2(); j++) {
      f << ma(i, j) << " ";
    }
    f << std::endl;
  }
  f.close();
}

namespace DB
{
struct Settings;

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

class FunctionParser {
public:
    using polynomial = boost::math::tools::polynomial<Float64>;

    FunctionParser() = default;

    const std::unordered_map<char, int> op2rk {{'(', 0}, {')', 0}, {'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}, {'^', 3}};

    template <typename U>
    inline bool isOperator(const U &c) const
    {
        String tmp = std::string() + c;
        return tmp.size() == 1 && op2rk.find(tmp[0]) != op2rk.end();
    }

    inline int getOpRank(char c) const
    {
        return op2rk.count(c) ? op2rk.at(c) : -1;
    }

    inline static bool isIdentifier(const std::string & s)
    {
        return std::regex_match(s, std::regex("x[0-9]+"));
    }

    inline static bool isNumbers(const std::string & s)
    {
        return std::regex_match(s, std::regex("[0-9]+[.]?[0-9]*"));
    }

    inline static std::vector<size_t> getIndex(const std::string& g)
    {
        std::vector<size_t> index;
        for (size_t i = 0; i < g.size(); ++i)
        {
            if (g[i] == 'x')
            {
                size_t j = i + 1;
                while (j < g.size() && isdigit(g[j])) j++;
                index.emplace_back(std::stoul(g.substr(i + 1, j - i - 1))-1);
                i = j - 1;
            }
        }
        std::sort(index.begin(), index.end());
        index.erase(std::unique(index.begin(), index.end()), index.end());
        return index;
    }

    bool parse(std::string_view expr, int arg_num = -1);

    std::vector<Float64> getPartialDeriv(const std::vector<Float64>& means) const;

    Float64 getExpressionResult(const std::vector<Float64>& x) const;

private:
    std::vector<std::string> rpn_expr;
};

using CovarianceMatrix = ColMatrix<CovarianceSimpleData<Float64>, false>;

struct DeltaMethod
{
    static constexpr auto name = "Deltamethod";

    static inline Float64 apply(const CovarianceMatrix& cov_ma, const FunctionParser& partial_deriv)
    {
        UInt64 count = cov_ma.getCount();
        auto means = cov_ma.getMeans();
        Float64 result = 0;
        Matrix cov = cov_ma.getMatrix();
        auto partial_derivatives = partial_deriv.getPartialDeriv(means);
        if (partial_derivatives.size() != cov.size1() || partial_derivatives.size() != cov.size2())
            throw Exception("The number of partial derivatives is not equal to the number of covariance",
                ErrorCodes::BAD_ARGUMENTS);
        for (size_t i = 0; i < cov.size2(); ++i)
            for (size_t j = 0; j < cov.size1(); ++j)
                result += partial_derivatives[j] * partial_derivatives[i] * (cov(j, i) / count);
        return result;
    }
};

struct DeltaMethodData
{
    DeltaMethodData() = default;

    explicit DeltaMethodData(const size_t arguments_num) : covar_ma(arguments_num) {}

    void add(const IColumn ** column, size_t row_num)
    {
        covar_ma.add(column, row_num);
    }

    void merge(const DeltaMethodData & source)
    {
        covar_ma.merge(source.covar_ma);
    }

    void serialize(WriteBuffer & buf) const
    {
        covar_ma.serialize(buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        covar_ma.deserialize(buf);
    }

    Float64 publish(const FunctionParser& partial_deriv) const
    {
        return DeltaMethod::apply(covar_ma, partial_deriv);
    }

    CovarianceMatrix covar_ma;
};

class AggregateFunctionDeltaMethod final:
    public IAggregateFunctionDataHelper<DeltaMethodData, AggregateFunctionDeltaMethod>
{
private:
    size_t arguments_num;
    FunctionParser partial_derivatives;
    bool is_std = true;

public:
    explicit AggregateFunctionDeltaMethod(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<DeltaMethodData, AggregateFunctionDeltaMethod> ({arguments}, {})
    {
        arguments_num = arguments.size();
        if (!((params.size() == 1 && params[0].getType() == Field::Types::String) ||
              (params.size() == 2 && params[0].getType() == Field::Types::String && params[1].getType() == Field::Types::Bool)))
            throw Exception("Aggregate function " + getName() + " requires 1 or 2 parameters: String, [Bool].", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
        if (!partial_derivatives.parse(params[0].get<const String &>(), arguments_num))
            throw Exception("params of g is not valid", ErrorCodes::BAD_ARGUMENTS);
        if (params.size() > 1)
            is_std = params[1].get<bool>();
    }

    String getName() const override
    {
        return "Deltamethod";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeFloat64>();
    }

    void create(AggregateDataPtr __restrict place) const override
    {
        new (place) Data(arguments_num);
    }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        this->data(place).add(columns, row_num);
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> /* version */) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> /* version */, Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        auto result = this->data(place).publish(partial_derivatives);
        assert_cast<ColumnFloat64 &>(to).getData().emplace_back(is_std? sqrt(result) : result);
    }
};

template <typename Op, bool use_index = false>
struct TtestSampData : public DeltaMethodData
{
    TtestSampData() = default;

    explicit TtestSampData(const UInt64 arg_num_, const UInt64 pse_num_) 
      : DeltaMethodData(arg_num_ - static_cast<UInt64>(use_index) - pse_num_), arg_num(arg_num_), pse_num(pse_num_) {}


    void add(const IColumn ** column, size_t row_num)
    {
        covar_ma.add(column, row_num);
        if constexpr (use_index)
        {
            auto group = column[arg_num - 1]->getUInt(row_num);
            if (unlikely(pse_num != 0))
            {
                UInt64 combined_hash = 0;
                for (size_t i = 0; i < pse_num; ++i)
                {
                    String pse_value = column[arg_num - 2 - i]->getDataAt(row_num).toString();
                    for (const auto & ch : pse_value)
                        combined_hash = combined_hash ^ (ch + 0x9e3779b9 + (combined_hash << 6) + (combined_hash >> 2));
                }
                combined_hash = combined_hash << 8;
                auto new_group = group + combined_hash;

                pse2covs[combined_hash][group] = new_group;
                group = new_group;
            }
            if (!index2covs.contains(group))
                index2covs[group] = CovarianceMatrix{arg_num - 1 - pse_num};
            auto & index2cov = index2covs[group];
            index2cov.add(column, row_num);
        }
    }

    void addWithPODArray(const PaddedPODArray<Float64> & row)
    {
        covar_ma.add(row);
        if constexpr (use_index)
        {
            if (!index2covs.contains(row.back()))
                index2covs[row.back()] = CovarianceMatrix{arg_num - 1 - pse_num};
            index2covs[row.back()].add(row);
        }
    }

    void merge(const TtestSampData & source)
    {
        covar_ma.merge(source.covar_ma);
        if constexpr (use_index)
        {
            for (const auto & [index, cov] : source.index2covs)
            {
                if (!index2covs.count(index))
                    index2covs[index] = source.index2covs.at(index);
                else
                    index2covs[index].merge(cov);
            }
        }
        if (index2covs.size() > 20000)
            throw Exception("Too many group, more than 20000", ErrorCodes::BAD_ARGUMENTS);
        for (const auto & [pse, index2cov] : source.pse2covs)
        {
            if (!pse2covs.count(pse))
                pse2covs[pse] = index2cov;
            else
            {
                for (const auto & [index, new_index] : index2cov)
                {
                    if (!pse2covs[pse].count(index))
                        pse2covs[pse][index] = new_index;
                }
            }
        }
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(pse2covs.size(), buf);
        for (const auto & [pse, index2cov] : pse2covs)
        {
            writeBinary(pse, buf);
            writeVarUInt(index2cov.size(), buf);
            for (const auto & [index, new_index] : index2cov)
            {
                writeBinary(index, buf);
                writeBinary(new_index, buf);
            }
        }

        writeVarUInt(arg_num, buf);
        writeVarUInt(pse_num, buf);
        covar_ma.serialize(buf);
        if constexpr (use_index)
        {
            writeVarUInt(index2covs.size(), buf);
            for (const auto & [index, cov] : index2covs)
            {
                writeBinary(index, buf);
                cov.serialize(buf);
            }
        }
    }

    void deserialize(ReadBuffer & buf)
    {
        UInt64 size = 0;
        readVarUInt(size, buf);
        for (size_t i = 0; i < size; ++i)
        {
            UInt64 pse = 0;
            readBinary(pse, buf);
            UInt64 index2cov_size = 0;
            readVarUInt(index2cov_size, buf);
            for (size_t j = 0; j < index2cov_size; ++j)
            {
                UInt64 index = 0;
                UInt64 new_index = 0;
                readBinary(index, buf);
                readBinary(new_index, buf);
                pse2covs[pse][index] = new_index;
            }
        }

        readVarUInt(arg_num, buf);
        readVarUInt(pse_num, buf);
        covar_ma.deserialize(buf);
        if constexpr (use_index)
        {
            size = 0;
            readVarUInt(size, buf);
            for (size_t i = 0; i < size; ++i)
            {
                UInt64 index = 0;
                readBinary(index, buf);
                if (!index2covs.contains(index))
                    index2covs[index] = CovarianceMatrix{arg_num - 1 - pse_num};
                index2covs[index].deserialize(buf);
            }
        }
    }

    String publish(const typename Op::Params& ttest_params)
    {
        return Op::apply(covar_ma, index2covs, ttest_params, pse2covs);
    }

    UInt64 arg_num;
    UInt64 pse_num;
    std::map<UInt64, CovarianceMatrix> index2covs;
    std::map<UInt64, std::map<UInt64, UInt64>> pse2covs; // Key: hash of pse, value: {group_index, cov_index}
};

struct Ttest1Samp
{
    enum class Alternative
    {
        TwoSided,
        Less,
        Greater
    };

    enum class MetricType
    {
        Avg,
        Sum
    };

    struct Params
    {
        String g;
        Alternative alternative = Alternative::TwoSided;
        Float64 mu = 0;
        String cuped;
        size_t arguments_num;
        Float64 alpha = 0.05;
        Float64 mde = 0.005;
        Float64 power = 0.8;
        UInt64 pse = 0;
        MetricType metric_type = MetricType::Avg;
        std::vector<Float64> ratios = {1, 1};
    };

    static constexpr auto name = "Ttest_1samp";

    static inline std::vector<std::vector<Float64>> getCovs(CovarianceMatrix& cov_ma, const Params& params)
    {
        std::vector<std::pair<String, std::vector<size_t>>> calc_elements;
        calc_elements.push_back({params.g, FunctionParser::getIndex(params.g)});
        auto means = cov_ma.getMeans();
        UInt64 count = cov_ma.getCount();

        String total_g = params.g;
        if (!params.cuped.empty())
        {
            total_g += " + " + params.cuped;
            std::vector<String> cuped_elements;
            boost::split(cuped_elements, params.cuped, boost::is_any_of("+"));
            for (const auto& element : cuped_elements)
                calc_elements.push_back({element, FunctionParser::getIndex(element)});
        }
        FunctionParser total_g_parser;
        if (!total_g_parser.parse(total_g))
            throw Exception("params of g is not valid", ErrorCodes::BAD_ARGUMENTS);
        auto deriv_result = total_g_parser.getPartialDeriv(means);

        size_t n = calc_elements.size();
        std::vector<std::vector<Float64>> covs(n, std::vector<Float64>(n, 0));
        Matrix cov = cov_ma.getMatrix();
        std::vector<Float64> temp(cov.size2(), 0);

        for (const auto& [g, index] : calc_elements)
            for (const auto & i : index)
                for (const auto & j : index)
                    temp[i] += deriv_result[j] * deriv_result[i] * (cov(j, i) / count);


        for (size_t i = 0; i < n; i++)
            covs[i][i] = std::accumulate(calc_elements[i].second.begin(), calc_elements[i].second.end(), 0.0,
                         [&temp](Float64 sum, size_t index) { return sum + temp[index]; }); 


        auto get_deriv = [&](const String& g, const std::vector<size_t>& indexs)
                        -> std::vector<Float64>
        {
              FunctionParser parser;
              if (!parser.parse(g))
                  throw Exception("params of a is not valid", ErrorCodes::BAD_ARGUMENTS);
              auto derivs = parser.getPartialDeriv(means);
              std::vector<Float64> result(indexs.size(), 0);
              for (size_t i = 0; i < indexs.size(); i++)
                  result[i] = derivs[indexs[i]];
              return result;
        };

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i + 1; j < n; ++j)
            {
                auto index_ab = calc_elements[i].second;
                index_ab.insert(index_ab.end(), calc_elements[j].second.begin(), calc_elements[j].second.end());
                auto cov_tmp = cov_ma.getSubMatrix(index_ab);
                auto & index_a = calc_elements[i].second;
                auto & index_b = calc_elements[j].second;
                Matrix deriv_a_tmp(1, index_a.size() + index_b.size(), 0);
                Matrix deriv_b_tmp(index_a.size() + index_b.size(), 1, 0);
                size_t pos_a = 0, pos_b = index_a.size();
                for (const auto& deriv : get_deriv(calc_elements[i].first, index_a))
                    deriv_a_tmp(0, pos_a++) = deriv;
                for (const auto& deriv : get_deriv(calc_elements[j].first, index_b))
                    deriv_b_tmp(pos_b++, 0) = deriv;
                covs[i][j] = covs[j][i] = prod(static_cast<Matrix>(prod(deriv_a_tmp, cov_tmp)), deriv_b_tmp)(0, 0)/count;
            }
        }
        return covs;
    }

    inline static std::tuple<std::vector<std::vector<Float64>>, Matrix, Matrix, Matrix> 
        getCovxy(CovarianceMatrix& cov_ma, const Params& params)
    {
        auto covs = getCovs(cov_ma, params);
        size_t n = covs.size();
        Matrix cov_xy(1, n-1);
        for (size_t i = 1; i < n; ++i)
            cov_xy(0, i-1) = covs[0][i];

        Matrix cov_xx(n-1, n-1);
        for (size_t i = 1; i < n; ++i)
            for (size_t j = 1; j < n; ++j)
                cov_xx(i-1, j-1) = covs[i][j];

        Matrix cov_xx_inv(cov_xx.size1(), cov_xx.size2());
        if (!invertMatrix(cov_xx, cov_xx_inv))
            throw Exception("InvertMatrix failed. some variables in the table are perfectly collinear.", ErrorCodes::BAD_ARGUMENTS);
        return std::make_tuple(covs, cov_xy, cov_xx, cov_xx_inv);
    }

    inline static std::pair<Float64, Float64> getMeanAndVar(CovarianceMatrix& cov_ma, const Params& params)
    {
        auto means = cov_ma.getMeans();
        auto [covs, cov_xy, cov_xx, cov_xx_inv] = getCovxy(cov_ma, params);
        Matrix theta = prod(cov_xy, cov_xx_inv);
        FunctionParser y_parser;
        if (!y_parser.parse(params.g))
            throw Exception("params of g is not valid", ErrorCodes::BAD_ARGUMENTS);
        Float64 result_mean = y_parser.getExpressionResult(means);
        Float64 result_var = covs[0][0] + prod(static_cast<Matrix>(prod(theta, cov_xx)), tMatrix(theta))(0, 0)
                             - 2*(static_cast<Matrix>(prod(theta, tMatrix(cov_xy)))(0, 0));
        return std::make_pair(result_mean, result_var);
    }

    static inline Matrix getCupedMean(CovarianceMatrix& cov_ma, const Params& params)
    {
        if (params.cuped.empty())
            return Matrix(0, 0);
        auto means = cov_ma.getMeans();
        LOG(means.begin(), means.end(), "means");
        std::vector<String> cuped_elements;
        boost::split(cuped_elements, params.cuped, boost::is_any_of("+"));
        Matrix result(cuped_elements.size(), 1);
        for (size_t i = 0; i < cuped_elements.size(); ++i)
        {
            LOG(cuped_elements[i], "cuped_elements[i]");
            FunctionParser parser;
            if (!parser.parse(cuped_elements[i]))
                throw Exception("params of cuped is not valid", ErrorCodes::BAD_ARGUMENTS);
            result(i, 0) = parser.getExpressionResult(means);
        }
        return result;
    }

    static inline std::string getTtestResult(Float64 estimate, Float64 stderr_var, Alternative alternative,
        Float64 alpha, UInt64 count, bool only_data = false, std::optional<std::pair<Float64, Float64>> means = std::nullopt)
    {
        Float64 t_stat = estimate / stderr_var;
        Float64 p_value = 0;
        Float64 lower = 0;
        Float64 upper = 0;
        boost::math::normal normal_dist(0, 1);
        
        String prefix;
        if (isnan(t_stat) || isinf(t_stat))
        {
            prefix = "Warning: Data are essentially constant (standard error is zero), making t-statistic undefined. Check for variability in your data.\n\n";
            if (fabs(estimate) < 1e-20)
                t_stat = 0;
            else
                t_stat = std::numeric_limits<Float64>::infinity() * (estimate > 0 ? 1 : -1);
        }


        if (isnan(t_stat))
            p_value = t_stat;
        else if (isinf(t_stat))
            p_value = 0;
        else if (alternative == Alternative::TwoSided)
            p_value = 2 * (1 - cdf(normal_dist, std::abs(t_stat)));
        else if (alternative == Alternative::Less)
            p_value = cdf(normal_dist, t_stat);
        else if (alternative == Alternative::Greater)
            p_value = 1 - cdf(normal_dist, t_stat);

        if (alpha > 0)
        {
            boost::math::students_t_distribution<> dist(count - 1);
            Float64 t_quantile = 0;
            if (alternative == Alternative::TwoSided)
            {
                t_quantile = quantile(dist, 1 - alpha / 2);
                lower = estimate - t_quantile * stderr_var;
                upper = estimate + t_quantile * stderr_var;
            }
            else if (alternative == Alternative::Less)
            {
                t_quantile = quantile(dist, 1 - alpha);
                lower = -std::numeric_limits<Float64>::infinity();
                upper = estimate + t_quantile * stderr_var;
            }
            else if (alternative == Alternative::Greater)
            {
                t_quantile = quantile(dist, 1 - alpha);
                lower = estimate - t_quantile * stderr_var;
                upper = std::numeric_limits<Float64>::infinity();
            }
        }

        std::string result;
        if (!only_data) 
        {
            if (means.has_value())
                result += to_string_with_precision("mean0") + to_string_with_precision("mean1");
            result += to_string_with_precision("estimate") + to_string_with_precision("stderr") 
                  + to_string_with_precision("t-statistic") + to_string_with_precision("p-value") 
                  + to_string_with_precision("lower") + to_string_with_precision("upper") + "\n";
        }

        if (means.has_value())
            result += to_string_with_precision(means->first) + to_string_with_precision(means->second);
        result += to_string_with_precision(estimate) + to_string_with_precision(stderr_var)
               + to_string_with_precision(t_stat) + to_string_with_precision(p_value)
               + to_string_with_precision(lower) + to_string_with_precision(upper) + "\n";
        return prefix + result;
    }

    static inline std::string apply(CovarianceMatrix& cov_ma, std::map<UInt64, CovarianceMatrix>&, const Params& params, std::map<UInt64, std::map<UInt64, UInt64>>&)
    {
        auto [mean, var] = getMeanAndVar(cov_ma, params);
        Float64 estimate = mean - params.mu;
        Float64 stderr_var = sqrt(var);
        return "\n" + getTtestResult(estimate, stderr_var, params.alternative, params.alpha, cov_ma.getCount());
    }
};

struct Ttest2Samp : public Ttest1Samp
{
    static constexpr auto name = "Ttest_2samp";

    static inline std::tuple<Float64, Float64, Float64, Float64> 
        getEstimateAndStderr(std::vector<CovarianceMatrix> & single_cov_ma, const Params & params, 
                             const CovarianceMatrix & theta_cov = CovarianceMatrix())
    {
        auto cov_ma = single_cov_ma[0];
        cov_ma.merge(single_cov_ma[1]);
        if (!theta_cov.isEmpty())
            cov_ma = theta_cov;
        Matrix theta;
        {
            auto [covs, cov_xy, cov_xx, cov_xx_inv] = getCovxy(cov_ma, params);
            for (size_t i = 0; i < cov_xx_inv.size1(); i++)
                for (size_t j = 0; j < cov_xx_inv.size2(); j++)
                    if (isnan(cov_xx_inv(i, j)))
                        cov_xx_inv(i, j) = 0;
            theta = prod(cov_xy, cov_xx_inv);
        }
        auto cuped_mean = getCupedMean(cov_ma, params);
        std::vector<Float64> means(2), vars(2);
        if (single_cov_ma.size() == 2)
        {
            for (size_t index = 0; index < 2; index++)
            {
                if (params.cuped.empty())
                    std::tie(means[index], vars[index]) = getMeanAndVar(single_cov_ma[index], params);
                else
                {
                    auto [covs, cov_xy, cov_xx, cov_xx_inv] = getCovxy(single_cov_ma[index], params);
                    Float64 var = covs[0][0] + prod(static_cast<Matrix>(prod(theta, cov_xx)), tMatrix(theta))(0, 0)
                                  - 2*prod(theta, tMatrix(cov_xy))(0, 0);
                    vars[index] = var;
                    FunctionParser y_parser;
                    if (!y_parser.parse(params.g))
                        throw Exception("params of g is not valid", ErrorCodes::BAD_ARGUMENTS);
                    LOG(params.g, "params.g");
                    means[index] = y_parser.getExpressionResult(single_cov_ma[index].getMeans())
                                   - prod(theta, getCupedMean(single_cov_ma[index], params) - cuped_mean)(0, 0);
                    LOG(y_parser.getExpressionResult(single_cov_ma[index].getMeans()), "y_parser.getExpressionResult(single_cov_ma[index].getMeans())");
                    LOG(theta, "theta");
                    LOG(getCupedMean(single_cov_ma[index], params), "getCupedMean(single_cov_ma[index], params)");
                    LOG(cuped_mean, "cuped_mean");
                    LOG(means[index], "means[index]");
                }
            }
        }
        return {means[0], means[1], vars[0], vars[1]};
    }

    static inline String getTtest2SampResult(std::vector<CovarianceMatrix> & single_cov_ma, 
                                             const Params & params, bool only_data = false)
    {
        if (single_cov_ma.size() != 2)
            throw Exception("Ttest_2samp only support two samples", ErrorCodes::BAD_ARGUMENTS);
        if (single_cov_ma[0].isEmpty() || single_cov_ma[1].isEmpty())
            return "error: at least 2 groups are required for 2-sample t-test, please check the argument of index";
        Float64 estimate;
        Float64 stderr_var;
        Float64 mean0, mean1;
        Float64 vars0, vars1;
        std::tie(mean0, mean1, vars0, vars1) = getEstimateAndStderr(single_cov_ma, params);
        estimate = mean1 - mean0;
        stderr_var = sqrt(vars0 + vars1);
        return getTtestResult(estimate, stderr_var, params.alternative, params.alpha, single_cov_ma[0].getCount() + single_cov_ma[1].getCount(), only_data, std::make_pair(mean0, mean1));
    }

    static inline String getTtest2SamppseResult(CovarianceMatrix & covs, std::map<UInt64, CovarianceMatrix>& index2covs, 
                                                 const Params& params, std::map<UInt64, std::map<UInt64, UInt64>>& pse2covs)
    {
        UInt64 total_count = covs.getCount();

        Float64 estimate = 0;
        Float64 stderr_var = 0;
        Float64 mean0 = 0;
        Float64 mean1 = 0;
        bool only_one_treatment = false;
        bool only_one_sample = false;

        for (auto & [pse, group2covs] : pse2covs)
        {
            if (group2covs.size() > 2)
                throw Exception("Ttest_2samp only support two samples", ErrorCodes::BAD_ARGUMENTS);
            if (group2covs.size() == 1)
            {
                only_one_treatment = true;
                continue;
            }
            UInt64 pse_count = 0;
            std::vector<CovarianceMatrix> single_cov_ma;
            for (auto & [group, cov_index] : group2covs)
            {
                if (!index2covs.contains(cov_index))
                    throw Exception("Some covariance matrix is missing.", ErrorCodes::BAD_ARGUMENTS);
                auto && cov = index2covs[cov_index];
                pse_count += cov.getCount();
                single_cov_ma.push_back(cov);
            }
            if (single_cov_ma[0].isEmpty() || single_cov_ma[1].isEmpty())
                return "At least 2 groups are required for 2-sample t-test, please check the argument of index";
            if (single_cov_ma[0].getCount() == 1 || single_cov_ma[1].getCount() == 1)
            {
                only_one_sample = true;
                continue;
            }
            Float64 pse_mean0, pse_mean1;
            Float64 pse_vars0, pse_vars1;
            std::tie(pse_mean0, pse_mean1, pse_vars0, pse_vars1) = getEstimateAndStderr(single_cov_ma, params);
            Float64 ratio = static_cast<Float64>(pse_count) / total_count;
            mean0 += pse_mean0 * ratio;
            mean1 += pse_mean1 * ratio;
            stderr_var += (pse_vars0 + pse_vars1) * ratio * ratio;
        }
        estimate = (mean1 - mean0);
        stderr_var = sqrt(stderr_var);
        String prefix;
        if (only_one_treatment)
            prefix = "Warning: Variance cannot be computed as one of the groups has only a single sample.\n";
        if (only_one_sample)
            prefix += "Warning: Cannot perform t-test as it contains only one group.\n";
        if (!prefix.empty())
            prefix += "\n";
        return prefix + getTtestResult(estimate, stderr_var, params.alternative, params.alpha, total_count, false, std::make_pair(mean0, mean1));
    }

    static inline String apply(CovarianceMatrix & covs, std::map<UInt64, CovarianceMatrix>& index2covs, const Params& params, std::map<UInt64, std::map<UInt64, UInt64>>& pse2covs)
    {
        if (params.pse != 0)
            return "\n" + getTtest2SamppseResult(covs, index2covs, params, pse2covs);
        std::vector<CovarianceMatrix> single_cov_ma;
        single_cov_ma.push_back(index2covs[0]);
        single_cov_ma.push_back(index2covs[1]);
        return "\n" + getTtest2SampResult(single_cov_ma, params);
    }
};


struct Ttests2Samp : Ttest2Samp
{
    static constexpr auto name = "Ttests_2samp";

    static inline String apply(CovarianceMatrix &, std::map<UInt64, CovarianceMatrix>& index2covs, const Params& params, std::map<UInt64, std::map<UInt64, UInt64>>&)
    {
        String result = "\n" + to_string_with_precision("control") + to_string_with_precision("treatment")
                        + to_string_with_precision("estimate") + to_string_with_precision("stderr")
                        + to_string_with_precision("t-statistic") + to_string_with_precision("p-value")
                        + to_string_with_precision("lower") + to_string_with_precision("upper") + "\n";

        for (auto it = index2covs.begin(); it != index2covs.end(); ++it) {
            for (auto it2 = std::next(it); it2 != index2covs.end(); ++it2) {
                std::vector<CovarianceMatrix> single_cov_ma{it->second, it2->second};
                result += to_string_with_precision(it->first) + to_string_with_precision(it2->first)
                        + getTtest2SampResult(single_cov_ma, params, true) + "\n";
                if (params.alternative != Alternative::TwoSided) {
                    std::swap(single_cov_ma[0], single_cov_ma[1]);
                    result += to_string_with_precision(it2->first) + to_string_with_precision(it->first)
                            + getTtest2SampResult(single_cov_ma, params, true) + "\n";
                }
            }
        }
        result.pop_back();
        return result;
    }
};

template <typename Op, bool use_index>
class AggregateFunctionTtestSamp final:
    public IAggregateFunctionDataHelper<TtestSampData<Op, use_index>, AggregateFunctionTtestSamp<Op, use_index>>
{
private:
    using Data = TtestSampData<Op, use_index>;
    size_t arguments_num;
    typename Op::Params ttest_params;

public:
    explicit AggregateFunctionTtestSamp(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<TtestSampData<Op, use_index>, AggregateFunctionTtestSamp<Op, use_index>> ({arguments}, {})
    {
        arguments_num = arguments.size();
        ttest_params.g = params[0].get<String>();
        ttest_params.arguments_num = arguments.size();

        if (params.size() > 1)
        {
            const auto & param = params[1].get<String>();
            if (param == "two-sided")
                ttest_params.alternative = Op::Alternative::TwoSided;
            else if (param == "less")
                ttest_params.alternative = Op::Alternative::Less;
            else if (param == "greater")
                ttest_params.alternative = Op::Alternative::Greater;
            else
                throw Exception("Unknown alternative parameter in aggregate function " + getName() +
                        ". It must be one of: 'two-sided', 'less', 'greater'", ErrorCodes::BAD_ARGUMENTS);
        }

        size_t index = 2;

        if (index < params.size() && Op::name == "Ttest_1samp") {
            if (params[index].getType() == Field::Types::UInt64)
              ttest_params.mu = params[index].get<UInt64>();
            else if (params[index].getType() == Field::Types::Float64)
              ttest_params.mu = params[index].get<Float64>();
            index++;
        }

        for (; index < params.size(); ++index) 
        {
            if (params[index].getType() == Field::Types::String)
                ttest_params.cuped = params[index].get<String>();
            else if (params[index].getType() == Field::Types::UInt64)
                ttest_params.pse = params[index].get<UInt64>();
            else if (params[index].getType() == Field::Types::Int64)
                ttest_params.pse = params[index].get<Int64>();
            else  
                ttest_params.alpha = params[index].get<Float64>();
        }


        auto & cuped = ttest_params.cuped;
        if (!cuped.empty())
        {
            if (!(cuped.size() >= 2 && std::tolower(cuped[0]) == 'x'  && cuped[1] == '=') || cuped.size() < 2)
                throw Exception("Cuped params is not valid", ErrorCodes::BAD_ARGUMENTS);
            cuped = cuped.substr(2);
        }

        FunctionParser partial_derivatives;
        String total_g = ttest_params.g;
        if (!ttest_params.cuped.empty())
            total_g += " + " + ttest_params.cuped;
        if (arguments.size() <= static_cast<size_t>(use_index) + ttest_params.pse)
            throw Exception("params of pse is not valid", ErrorCodes::BAD_ARGUMENTS);
        if (!partial_derivatives.parse(total_g, arguments_num - static_cast<size_t>(use_index) - ttest_params.pse))
            throw Exception("params of g is not valid", ErrorCodes::BAD_ARGUMENTS);
    }

    String getName() const override
    {
        return Op::name;
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeString>();
    }

    void create(AggregateDataPtr __restrict place) const override /// NOLINT
    {
        new (place) Data(arguments_num, ttest_params.pse);
    }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
         this->data(place).add(columns, row_num);
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> ) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> , Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        auto result = this->data(place).publish(ttest_params);
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }
};


template <typename Key, typename HashFunction = MurmurHash3<Int32>>
class XexptTtest2Samp
{
public:
    XexptTtest2Samp() = default;

    explicit XexptTtest2Samp(const size_t & arguments_num) 
        : calc_col_num(arguments_num - 2), hash(std::make_shared<HashFunction>(0)) 
    {
        if (arguments_num < 4)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Requires at least 4 arguments [numerator, denominator, uin, groupname]");
    }

    void add(const IColumn ** column, size_t row_num)
    {
        Int32 uin = column[calc_col_num]->getInt(row_num);
        UInt32 uin_ha = (*hash)(uin) / bucket_divisor;
        Key index;
        if constexpr (std::is_same_v<Key, String>)
            index = column[calc_col_num + 1]->getDataAt(row_num).toString();
        else
            index = column[calc_col_num + 1]->getInt(row_num);
        if (!index2group.count(index))
        {
            index2group[index] = GroupData(calc_col_num);
            if (index2group.size() > 2)
                throw Exception("The number of group is not equal to two, please check the input data.", ErrorCodes::BAD_ARGUMENTS);
        }

        auto & group = index2group[index];
        group.count++;
        for (size_t i = 0; i < calc_col_num; ++i)
            group.col_data[i][uin_ha] += column[i]->getFloat64(row_num);
    }

    void merge(const XexptTtest2Samp & source)
    {
        for (const auto & [index, group] : source.index2group)
        {
            if (!index2group.count(index))
                index2group[index] = GroupData(calc_col_num);
            auto & this_group = index2group[index];
            this_group.count += group.count;
            if (this_group.col_data.size() < group.col_data.size())
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: group size not equal");
            for (size_t i = 0; i < group.col_data.size(); ++i)
            {
                if (this_group.col_data[i].size() < group.col_data[i].size())
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: group size not equal");
                for (size_t j = 0; j < group.col_data[i].size(); ++j)
                    this_group.col_data[i][j] += group.col_data[i][j];
            }
        }
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(index2group.size(), buf);
        for (const auto & [index, group] : index2group)
        {
            if constexpr (std::is_same_v<String, Key>)
                writeStringBinary(index, buf);
            else
                writeVarInt(index, buf);
            writeVarUInt(group.count, buf);
            writeVarUInt(group.col_data.size(), buf);
            for (size_t i = 0; i < group.col_data.size(); ++i)
            {
                writeVarUInt(group.col_data[i].size(), buf);
                for (size_t j = 0; j < group.col_data[i].size(); j++)
                    writeFloatBinary(group.col_data[i][j], buf);
            }
        }
    }

    void deserialize(ReadBuffer & buf)
    {
        size_t group_size = 0;
        readVarUInt(group_size, buf);
        for (size_t i = 0; i < group_size; ++i)
        {
            Key index;
            if constexpr (std::is_same_v<String, Key>)
                readStringBinary(index, buf);
            else
                readVarInt(index, buf);
            if (!index2group.count(index))
                index2group[index] = GroupData(calc_col_num);
            auto & group = index2group[index];
            readVarUInt(group.count, buf);
            size_t col_size;
            readVarUInt(col_size, buf);
            group.col_data.resize(col_size);
            for (size_t col_i = 0; col_i < col_size; ++col_i)
            {
                size_t bucket_size;
                readVarUInt(bucket_size, buf);
                auto & bucket_data = group.col_data[col_i];
                bucket_data.resize_fill(bucket_size, 0);
                for (size_t bucket_i = 0; bucket_i < bucket_size; ++bucket_i)
                {
                    Float64 sum;
                    readFloatBinary(sum, buf);
                    bucket_data[bucket_i] += sum;
                }
            }
        }
    }

    String publish(const Ttests2Samp::Params& param)
    {
        if (index2group.size() != 2)
            return "Xexpt_Ttest_2samp need 2 group";

        PaddedPODArray<String> groupnames;
        PaddedPODArray<Float64> numerators, denominators, counts, std_samp, numerators_pre, denominators_pre, means, vars, means_avg, std_avg, vars_avg;
        TtestSampData<Ttest2Samp, true> ttest_2samp(calc_col_num + 1, 0); // add index
        TtestSampData<Ttest2Samp, true> ttest_2samp_sum(calc_col_num + 1, 0); // add index
        size_t index_int = 0;

        std::vector<PaddedPODArray<Float64>> args(bucket_num);
        
        std::vector<UInt64> count(2);
        for (const auto & [index, group] : index2group)
        {
            count[index_int] += group.count;
            index_int++;
        }
        index_int = 0;
        LOG(count[0], "count[0]");
        LOG(count[1], "count[1]");

        for (const auto & [index, group] : index2group)
        {
            if (group.col_data.size() != calc_col_num || group.col_data[0].size() != bucket_num) 
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: col_data is incomplete");

            if constexpr (std::is_same_v<String, Key>)
                groupnames.push_back(index);
            else 
                groupnames.push_back(std::to_string(index));
            std::vector<PaddedPODArray<Float64>> bucket_datas(bucket_num);

            for (size_t i = 0; i < group.col_data.size(); i++)
                for (size_t j = 0; j < group.col_data[i].size(); j++)
                {
                    bucket_datas[j].push_back(group.col_data[i][j] / (count[index_int] / 128));
                    if (i < args[j].size())
                        args[j][i] += group.col_data[i][j] / (count[index_int] / 128);
                    else
                        args[j].push_back(group.col_data[i][j] / (count[index_int] / 128));
                }

            /*
            for (auto & bucket : bucket_datas)
                for (auto & value : bucket)
                    value /= 1. * (count[0] + count[1]) / 256;

            for (auto & arg_i : args)
                for (auto & value : arg_i)
                    value /= 1. * (count[0] + count[1]) / 256;
                    */

            for (auto & bucket : bucket_datas)
            {
                bucket.push_back(index_int);
                String bucket_str;
                ttest_2samp.addWithPODArray(bucket);
                bucket[1] = param.ratios[index_int] / bucket_num;
                if (!param.cuped.empty() && 3 < bucket.size() && param.cuped == "x3/x4") {
                    bucket[3] = param.ratios[index_int] / bucket_num;
                    LOG("ERROR!");
                }
                ttest_2samp_sum.addWithPODArray(bucket);
            }
            Float64 numerator_sum = std::accumulate(group.col_data[0].begin(), group.col_data[0].end(), 0.0);
            numerators.push_back(numerator_sum);
            Float64 denominator_sum = std::accumulate(group.col_data[1].begin(), group.col_data[1].end(), 0.0);
            denominators.push_back(denominator_sum);
            if (group.col_data.size() >= 4)
            {
                Float64 numerator_pre_sum = std::accumulate(group.col_data[2].begin(), group.col_data[2].end(), 0.0);
                numerators_pre.push_back(numerator_pre_sum);
                Float64 denominator_pre_sum = std::accumulate(group.col_data[3].begin(), group.col_data[3].end(), 0.0);
                denominators_pre.push_back(denominator_pre_sum);
            }
            counts.push_back(group.count);
            index_int++;
        }

        CovarianceMatrix theta_cov(calc_col_num);
        CovarianceMatrix theta_cov_sum(calc_col_num);
        for (size_t i = 0; i < bucket_num; ++i)
        {
            theta_cov.add(args[i]);
            args[i][1] = (param.ratios[0] + param.ratios[1]) / bucket_num;
            if (!param.cuped.empty() && 3 < args[i].size())
                args[i][3] = (param.ratios[0] + param.ratios[1]) / bucket_num;
            theta_cov_sum.add(args[i]);
        }
        
        FunctionParser partial_derivatives;
        String total_g = param.g;
        if (!param.cuped.empty())
            total_g += " + " + param.cuped;
        if (!partial_derivatives.parse(total_g, calc_col_num))
            throw Exception("params of g is not valid", ErrorCodes::BAD_ARGUMENTS);

        std::vector<CovarianceMatrix> single_cov_ma;
        single_cov_ma.push_back(ttest_2samp.index2covs[0]);
        single_cov_ma.push_back(ttest_2samp.index2covs[1]);

        means_avg.resize(2);
        vars_avg.resize(2);
        std::tie(means_avg[0], means_avg[1], vars_avg[0], vars_avg[1]) = Ttest2Samp::getEstimateAndStderr(single_cov_ma, param, theta_cov);
        LOG(means_avg[0], "means_avg[0]");
        LOG(means_avg[1], "means_avg[1]");


        if (param.metric_type == Ttest1Samp::MetricType::Avg)
        {
            means.insert(means.end(), means_avg.begin(), means_avg.end());
            vars.insert(vars.end(), vars_avg.begin(), vars_avg.end());
        }

        if (param.metric_type == Ttest1Samp::MetricType::Sum)
        {
            std::vector<CovarianceMatrix> single_cov_ma_sum;
            single_cov_ma_sum.push_back(ttest_2samp_sum.index2covs[0]);
            single_cov_ma_sum.push_back(ttest_2samp_sum.index2covs[1]);
            means.resize(2);
            vars.resize(2);
            std::tie(means[0], means[1], vars[0], vars[1]) = Ttest2Samp::getEstimateAndStderr(single_cov_ma_sum, param, theta_cov_sum);
        }

        std::vector<Float64> std_samp_avg;
        if (param.cuped.empty())
        {
            std_samp_avg.push_back(sqrt(DeltaMethod::apply(ttest_2samp.index2covs[0], partial_derivatives)) * sqrt(denominators[0]));
            std_samp_avg.push_back(sqrt(DeltaMethod::apply(ttest_2samp.index2covs[1], partial_derivatives)) * sqrt(denominators[1]));
            if (param.metric_type == Ttest1Samp::MetricType::Sum)
            {
                std_samp.push_back(sqrt(DeltaMethod::apply(ttest_2samp_sum.index2covs[0], partial_derivatives)) * sqrt(param.ratios[0]));
                std_samp.push_back(sqrt(DeltaMethod::apply(ttest_2samp_sum.index2covs[1], partial_derivatives)) * sqrt(param.ratios[1]));
            }
        }
        else 
        {
            std_samp_avg.push_back(sqrt(vars_avg[0] * denominators[0]));
            std_samp_avg.push_back(sqrt(vars_avg[1] * denominators[1]));
            if (param.metric_type == Ttest1Samp::MetricType::Sum)
            {
                std_samp.push_back(sqrt(vars[0] * param.ratios[0]));
                std_samp.push_back(sqrt(vars[1] * param.ratios[1]));
            }
        }

        if (param.metric_type == Ttest1Samp::MetricType::Avg)
            std_samp.insert(std_samp.end(), std_samp_avg.begin(), std_samp_avg.end());
        Float64 estimate = means[1] - means[0];
        Float64 stderr_var = sqrt(vars[0] + vars[1]);
        Float64 diff_relative = estimate / means[0];
        Float64 t_stat = estimate / stderr_var;
        Float64 p_value = 0;
        Float64 lower = 0;
        Float64 upper = 0;
        boost::math::normal normal_dist(0, 1);
        String prefix;

        if (isnan(t_stat) || isinf(t_stat))
        {
            prefix = "Warning: Data are essentially constant (standard error is zero), making t-statistic undefined. Check for variability in your data.\n\n";
            if (fabs(estimate) < 1e-20)
                t_stat = 0;
            else
                t_stat = std::numeric_limits<Float64>::infinity() * (estimate > 0 ? 1 : -1);
        }

        if (isnan(t_stat))
            p_value = t_stat;
        else if (isinf(t_stat))
            p_value = 0;
        else
            p_value = 2 * (1 - cdf(normal_dist, std::abs(t_stat)));
        if (param.alpha > 0)
        {
            boost::math::students_t_distribution<> dist(counts[0] + counts[1]);
            Float64 t_quantile = 0;
            t_quantile = quantile(dist, param.alpha / 2);
            lower = estimate - fabs(t_quantile * stderr_var);
            upper = estimate + fabs(t_quantile * stderr_var);
        }

        Float64 lower_relative = lower / means[0];
        Float64 upper_relative = upper / means[0];
        Float64 mde = param.mde;
        Float64 alpha = param.alpha;
        Float64 power = 1 - cdf(normal_dist, quantile(normal_dist, 1 - alpha / 2) 
                      - fabs(means[0] * param.mde) / stderr_var) + cdf(normal_dist, quantile(normal_dist, alpha / 2) 
                      - fabs(means[0] * param.mde) / stderr_var); 

        Float64 result_mde = (quantile(normal_dist, 1 - alpha / 2)+quantile(normal_dist, param.power))*stderr_var/means[0];
        Float64 std_ratio = std_samp_avg[0] / std_samp_avg[1];
        Float64 cnt_ratio = denominators[0] / denominators[1];
        Float64 alpha_power = quantile(normal_dist, 1 - alpha / 2) - quantile(normal_dist, 1 - param.power);
        Float64 recommend_samples = ((std_ratio * std_ratio + cnt_ratio) / cnt_ratio) 
                                     * pow(alpha_power, 2) * pow(std_samp_avg[1] / means_avg[0], 2) / pow(mde, 2);

        String title;
        String group0;
        String group1;

        auto add_result3 = [&title, &group0, &group1](const String & title_, const String & group0_, const String & group1_)
        {
            title += to_string_with_precision<false>(title_);
            group0 += to_string_with_precision<false>(group0_);
            group1 += to_string_with_precision<false>(group1_);
            size_t max_len = std::max({title.size(), group0.size(), group1.size()});
            title += String(max_len - title.size(), ' ');
            group0 += String(max_len - group0.size(), ' ');
            group1 += String(max_len - group1.size(), ' ');
        };

        if constexpr (std::is_same_v<String, Key>)
            add_result3("groupname", index2group.begin()->first, (++index2group.begin())->first);
        else 
            add_result3("groupname", to_string_with_precision<false>(index2group.begin()->first), to_string_with_precision<false>((++index2group.begin())->first));
        if (param.metric_type == Ttest1Samp::MetricType::Avg)
            add_result3("denominator", to_string_with_precision<false, true>(denominators[0]), to_string_with_precision<false, true>(denominators[1]));
        else
            add_result3("ratio", to_string_with_precision<false>(param.ratios[0], 12, 0), to_string_with_precision<false>(param.ratios[1], 12, 0));

        add_result3("numerator", to_string_with_precision<false, true>(numerators[0]), to_string_with_precision<false, true>(numerators[1]));

        if (!denominators_pre.empty())
        {
            if (param.metric_type == Ttest1Samp::MetricType::Avg)
                add_result3("denominator_pre", to_string_with_precision<false, true>(denominators_pre[0]), to_string_with_precision<false, true>(denominators_pre[1]));
            add_result3("numerator_pre", to_string_with_precision<false, true>(numerators_pre[0]), to_string_with_precision<false, true>(numerators_pre[1]));
        }

        if (param.metric_type == Ttest1Samp::MetricType::Avg)
        {
            add_result3("mean", to_string_with_precision<false>(means[0]), to_string_with_precision<false>(means[1]));
            add_result3("std_samp", to_string_with_precision<false>(std_samp[0]), to_string_with_precision<false>(std_samp[1]));
        }

        String ci_prefix = std::to_string((1 - param.alpha) * 100);
        while (!ci_prefix.empty() && ci_prefix.back() == '0')
            ci_prefix.pop_back();
        if (!ci_prefix.empty() && ci_prefix.back() == '.')
            ci_prefix.pop_back();

        String res = '\n' + title + '\n' + group0 + '\n' + group1 + '\n' + '\n';
        
        title = "";
        String group;

        auto add_result2 = [&title, &group](const String & title_, const String & group_)
        {
            title += to_string_with_precision<false>(title_);
            group += to_string_with_precision<false>(group_);
            size_t max_len = std::max({title.size(), group.size()});
            title += String(max_len - title.size(), ' ');
            group += String(max_len - group.size(), ' ');
        };

        add_result2("diff_relative", std::to_string(diff_relative * 100) + "%");
        add_result2(ci_prefix + "%_relative_CI", "[" + std::to_string(lower_relative * 100) + "%," + std::to_string(upper_relative * 100) + "%]");
        add_result2("p-value", to_string_with_precision<false>(p_value));
        add_result2("t-statistic", to_string_with_precision<false>(t_stat));

        if (param.metric_type == Ttest1Samp::MetricType::Avg)
        {
            add_result2("diff", to_string_with_precision<false>(estimate));
            add_result2(ci_prefix + "%_CI", "[" + std::to_string(lower) + "," + std::to_string(upper) + "]");
        }
        add_result2("power(MDE=" + toString(param.mde) + ")", to_string_with_precision<false>(power));
        add_result2("recommend_samples", to_string_with_precision<false>(static_cast<UInt64>(std::floor(recommend_samples + 0.5))));
        add_result2("MDE(power=" + toString(param.power) + ")", to_string_with_precision<false>(result_mde));
        res += title + '\n' + group + '\n';
        return prefix + res;
    }

private:

    static constexpr UInt16 bucket_num = 128;

    static constexpr UInt32 bucket_divisor = (1 << (32 - 7));

    const size_t calc_col_num = 0;

    struct GroupData
    {
        GroupData() = default;
        explicit GroupData(const size_t & calc_col_num_)
        {
            count = 0;
            col_data.resize(calc_col_num_);
            for (auto & data : col_data)
                data.resize_fill(bucket_num, 0);
        }

        using Buckets = PaddedPODArray<Float64>;
        std::vector<Buckets> col_data;
        UInt64 count = 0;
    };

    std::map<Key, GroupData> index2group;

    const std::shared_ptr<HashBase> hash;
};


template <typename Key>
class AggregateFunctionXexptTtest2Samp final:
    public IAggregateFunctionDataHelper<XexptTtest2Samp<Key>, AggregateFunctionXexptTtest2Samp<Key>>
{
private:
    using Data = XexptTtest2Samp<Key>;
    size_t arguments_num;
    Ttest2Samp::Params param;

public:
    explicit AggregateFunctionXexptTtest2Samp(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<XexptTtest2Samp<Key>, AggregateFunctionXexptTtest2Samp> ({arguments}, {params})
    {
        arguments_num = arguments.size();
        param.g = "x1/x2";
        param.arguments_num = arguments.size() - 1; // - uin_col

        if (!params.empty())
            param.alpha = params[0].get<Float64>();
        if (params.size() > 1)
            param.mde = params[1].get<Float64>();
        if (params.size() > 2)
            param.power = params[2].get<Float64>();

        size_t param_index = 3;
        if (params.size() > 3) 
        {
            String metric_type = params[3].get<String>();
            if (Poco::toLower(metric_type) == "avg")
            {
                param.metric_type = Ttest2Samp::MetricType::Avg;
                param_index++;
            }
            else if (Poco::toLower(metric_type) == "sum")
            {
                param.metric_type = Ttest2Samp::MetricType::Sum;
                param_index++;
                param.ratios.resize(2);
                if (param_index < params.size() && params[param_index].getType() != Field::Types::String)
                {
                    if (params[param_index].getType() == Field::Types::Float64)
                        param.ratios[0] = params[param_index].get<Float64>();
                    else
                        param.ratios[0] = params[param_index].get<UInt64>();
                    param_index++;
                    if (param_index < params.size())
                    {
                        if (params[param_index].getType() == Field::Types::Float64)
                            param.ratios[1] = params[param_index].get<Float64>();
                        else
                            param.ratios[1] = params[param_index].get<UInt64>();
                        param_index++;
                    }
                    else 
                        throw Exception("ratio is not valid, must have two values", ErrorCodes::BAD_ARGUMENTS);
                }


            }
            else if (std::tolower(metric_type[0]) != 'x')
                throw Exception("metric_type is not valid, 'avg' or 'sum' is supported", ErrorCodes::BAD_ARGUMENTS);
        }
        if (params.size() > param_index)
            param.cuped = params[param_index].get<String>();

        auto & cuped = param.cuped;
        if (!cuped.empty())
        {
            if (cuped.size() < 2 || std::tolower(cuped[0]) != 'x'  || cuped[1] != '=')
                throw Exception("Cuped params is not valid", ErrorCodes::BAD_ARGUMENTS);
            cuped = cuped.substr(2);
        }
        FunctionParser partial_derivatives;
        String total_g = param.g;
        if (!param.cuped.empty())
            total_g += " + " + param.cuped;
        if (!partial_derivatives.parse(total_g, arguments_num - 2))
            throw Exception("params of g is not valid", ErrorCodes::BAD_ARGUMENTS);
    }

    String getName() const override
    {
        return "Xexpt_Ttest_2samp";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeString>();
    }

    void create(AggregateDataPtr __restrict place) const override /// NOLINT
    {
        new (place) Data(arguments_num);
    }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        this->data(place).add(columns, row_num);
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> ) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t>, Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        auto result = this->data(place).publish(param);
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }
};

}

