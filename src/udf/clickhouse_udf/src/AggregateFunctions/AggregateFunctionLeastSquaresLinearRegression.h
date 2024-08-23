#pragma once
#include <format>
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <AggregateFunctions/StatCommon.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnArray.h>
#include <Core/Field.h>
#include <base/types.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesNumber.h>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/algorithm/string.hpp>
#include <Common/typeid_cast.h>
#include <DataTypes/IDataType.h>
#include <Columns/ColumnsNumber.h>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

namespace ublas = boost::numeric::ublas;
using Matrix = ublas::matrix<Float64>;

template <typename T>
class LeastSquaresParams
{
public:
    struct Params
    {
        T standard_error;
        T r_squared, adjusted_r_squared;
        T f_statistic, f_value;
        std::vector<T> coef, std, t_value, p_value;
        UInt64 n, k, degrees_of_freedom;
        UInt64 arg_num;
        bool use_bias;
        T sigma2;
        Matrix xx_1;
        std::vector<size_t> nan_indexs;
    };

    LeastSquaresParams() = delete;

    LeastSquaresParams(const size_t arg_num_, const size_t use_bias_, const UInt64 count_,
        Matrix && dot_product_matrix_, Matrix && covariance_matrix_, Matrix && xx_inverse_, Matrix && xx_weighted_)
        : arg_num(arg_num_), use_bias(use_bias_), count(count_),
        data(dot_product_matrix_), covar_ma(covariance_matrix_), xx_inverse(xx_inverse_), xx_weighted(xx_weighted_)
    {
        parser();
    }

    void parser()
    {
        params.arg_num = arg_num;
        params.use_bias = use_bias;
        Matrix & matrix = data;
        size_t size_xx = arg_num - 1 + use_bias;
        Matrix xx(size_xx, size_xx);
        if (size_xx >= matrix.size1() || size_xx >= matrix.size2())
            throw Exception("LeastSquaresParams: matrix size error", ErrorCodes::BAD_ARGUMENTS);
        for (size_t i = 0; i < size_xx; ++i)
            for (size_t j = 0; j < size_xx; ++j)
                xx(i, j) = matrix(i+1, j+1);

        Matrix xx_1(size_xx, size_xx);
        std::vector<size_t> nan_indexs;
        if (!invertMatrix(xx, xx_1, nan_indexs))
            throw Exception("InvertMatrix failed. some variables in the table are perfectly collinear.", ErrorCodes::BAD_ARGUMENTS);
        params.nan_indexs = nan_indexs;
        params.xx_1 = xx_1;

        Matrix xy(size_xx, 1);
        for (size_t i = 0; i < size_xx; ++i)
            xy(i, 0) = matrix(i+1, 0);

        for (const auto & nan_index : nan_indexs) {
          for (size_t i = 0; i < xx_1.size1(); i++) {
            xx_1(i, nan_index) = xx_1(nan_index, i) = 0;
            xx(nan_index, i) = xx(i, nan_index) = 0;
            xy(nan_index, 0) = 0;
          }
        }

        Matrix coef = prod(xx_1, xy);
        Matrix coef_t(1, coef.size1());
        for (size_t i = 0; i < coef.size1(); i++)
        {
            coef_t(0, i) = coef(i, 0);
            params.coef.emplace_back(coef(i, 0));
        }

        UInt64& n = params.n;
        UInt64& k = params.k;
        UInt64& df = params.degrees_of_freedom;
        n = count;
        k = arg_num - 1 - nan_indexs.size();
        df = n - k - 1;

        Float64 yy = matrix(0, 0);
        Float64 sigma = (yy - 2 * prod(coef_t, xy)(0, 0) + prod(static_cast<Matrix>(prod(coef_t, xx)), coef)(0, 0)) / df;
        params.standard_error = sqrt(sigma);
        params.sigma2 = sigma;

        Matrix & var_matrix = covar_ma;
        Matrix var_x(size_xx, size_xx);
        for (size_t i = 0; i < size_xx - use_bias; ++i)
            for (size_t j = 0; j < size_xx - use_bias; ++j)
                var_x(i, j) = var_matrix(i+1, j+1);

        if (use_bias)
        {
            for (size_t i = 0; i < size_xx; i++)
               var_x(i, size_xx - 1) = var_x(size_xx - 1, i) = 0;
        }

        Matrix var_y(1, 1);
        var_y(0, 0) = var_matrix(0, 0);
        Matrix var_predict_y = prod(coef_t, static_cast<Matrix>(prod(var_x, coef)));

        Matrix diag_xx_1(1, xx_1.size1());
        for (size_t i = 0; i < xx_1.size1(); ++i)
            diag_xx_1(0, i) = xx_1(i, i);
        Matrix std(1, xx.size1());
        if (xx_inverse.size1() == 0)
            std = diag_xx_1 * sigma;
        else
        {
            if (!(xx_inverse.size1() == xx_inverse.size2() && xx_inverse.size1() == xx.size1() &&
                  xx_weighted.size1() == xx_weighted.size2() && xx_weighted.size1() == xx.size1()))
                throw Exception("xx_inverse size error", ErrorCodes::BAD_ARGUMENTS);
            Matrix tmp = prod(static_cast<Matrix>(prod(xx_inverse, xx_weighted)), xx_inverse);
            for (size_t i = 0; i < xx.size1(); ++i)
                std(0, i) = tmp(i, i);
        }


        for (size_t i = 0; i < std.size2(); ++i)
        {
            std(0, i) = sqrt(std(0, i));
            params.std.emplace_back(std(0, i));
        }

        Matrix t_stat(1, size_xx);
        for (size_t i = 0; i < coef.size1(); ++i)
            t_stat(0, i) = coef(0, i) / std(0, i);

        Matrix p_value(1, size_xx);
        auto student = boost::math::students_t_distribution<Float64>(df);
        for (size_t i = 0; i < size_xx; ++i)
        {
            p_value(0, i) = isnan(t_stat(0, i)) ? t_stat(0, i) :
                            isinf(t_stat(0, i)) ? 0 : 2 * cdf(complement(student, std::abs(t_stat(0, i))));
            params.t_value.emplace_back(t_stat(0, i));
            params.p_value.emplace_back(p_value(0, i));
        }

        Float64 r2 = var_predict_y(0, 0) / var_y(0, 0);
        Float64 adjusted_r2 = 1 - (1 - r2) * (n - 1) / df;
        params.r_squared = r2;
        params.adjusted_r_squared = adjusted_r2;

        Float64 sse = var_y(0, 0) - var_predict_y(0, 0);
        Float64 f_statistic = (var_predict_y(0, 0) / k) / (sse / df);
        Float64 f_value = 1;
        if (f_statistic > 0)
        f_value = isnan(f_statistic) ? f_statistic :
                  isinf(f_statistic) ? 0 : cdf(complement(boost::math::fisher_f(k, df), f_statistic));

        params.f_statistic = f_statistic;
        params.f_value = f_value;

        for (const auto & nan_index : nan_indexs) {
           params.coef[nan_index] = std::numeric_limits<Float64>::quiet_NaN();
           params.std[nan_index] = std::numeric_limits<Float64>::quiet_NaN();
           params.t_value[nan_index] = std::numeric_limits<Float64>::quiet_NaN();
           params.p_value[nan_index] = std::numeric_limits<Float64>::quiet_NaN();
        }
    }

    const Params & getParams() const
    {
        return params;
    }

private:
    size_t arg_num;
    size_t use_bias;
    UInt64 count;
    Matrix data;
    Matrix covar_ma;
    Matrix xx_inverse;
    Matrix xx_weighted;
    Params params;
};

template <typename T, typename Op, bool use_weights = false>
struct LeastSquaresLinearRegressionData
{
public:
    using DotProductMatrix = ColMatrix<DotProductData<T>, true, use_weights>;
    using CovarianceMatrix = ColMatrix<CovarianceSimpleData<T>, false, use_weights>;

    LeastSquaresLinearRegressionData() = default;

    explicit LeastSquaresLinearRegressionData(size_t arg_num_, bool use_bias_,
        const Matrix & xx_inverse_, const Matrix & xx_weighted_) 
        : arg_num(arg_num_), use_bias(static_cast<size_t>(use_bias_)), data(arg_num),
        covar_ma(arg_num), xx_inverse(xx_inverse_), xx_weighted(xx_weighted_) {}

    void add(const IColumn ** column, size_t row_num)
    {
        data.add(column, row_num);
        covar_ma.add(column, row_num);
    }

    void merge(const LeastSquaresLinearRegressionData & source)
    {
        data.merge(source.data);
        covar_ma.merge(source.covar_ma);
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(arg_num, buf);
        writeVarUInt(use_bias, buf);
        data.serialize(buf);
        covar_ma.serialize(buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        readVarUInt(arg_num, buf);
        readVarUInt(use_bias, buf);
        data.deserialize(buf);
        covar_ma.deserialize(buf);
    }

    String publish(const std::vector<String> & argument_names = {})
    {
        Matrix data_matrix = data.getMatrix();
        Matrix covar_matrix = covar_ma.getMatrix();
        UInt64 count = data.getCount();
        LeastSquaresParams<Float64> params(arg_num-use_weights, use_bias, count, std::move(data_matrix),
            std::move(covar_matrix), Matrix{xx_inverse}, Matrix{xx_weighted});
        return Op::apply(params.getParams(), argument_names);
    }

    void predict(
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit) const
    {
        Matrix data_matrix = data.getMatrix();
        Matrix covar_matrix = covar_ma.getMatrix();
        UInt64 count = data.getCount();
        LeastSquaresParams<Float64> params(arg_num-use_weights, use_bias, count, std::move(data_matrix),
            std::move(covar_matrix), Matrix{xx_inverse}, Matrix{xx_weighted});
        Op::predict(to, arguments, offset, limit, params);
    }


private:
    size_t arg_num;
    size_t use_bias;
    DotProductMatrix data;
    CovarianceMatrix covar_ma;
    Matrix xx_inverse;
    Matrix xx_weighted;
};

struct AggregateFunctionOls
{
    static constexpr auto name = "Ols";

    static String apply(const LeastSquaresParams<Float64>::Params & params, const std::vector<String> & argument_names)
    {
            
        String result;
        result += std::format("\nCall:\n  lm( formula = {} ~", argument_names.empty() ? "y" : argument_names[0]);
        for (size_t i = 1; i < params.arg_num; ++i)
        {
            String argument_name = "x" + std::to_string(i);
            if (i < argument_names.size())
                argument_name = argument_names[i];
            result += " " + argument_name + " +";
        }

        result.pop_back();
        result += ")\n\n";
        result += "Coefficients:\n";
        result += to_string_with_precision(".", 16) + to_string_with_precision("Estimate") 
               + to_string_with_precision("Std. Error") + to_string_with_precision("t value") 
               + to_string_with_precision("Pr(>|t|)") + "\n";
        if (params.use_bias)
            result += to_string_with_precision("(Intercept)", 16) + to_string_with_precision(params.coef.back())
              + to_string_with_precision(params.std.back()) + to_string_with_precision(params.t_value.back()) 
              + to_string_with_precision(params.p_value.back()) + "\n";
        for (size_t i = 0; i < params.arg_num - 1; ++i)
        {
            String argument_name = "x" + std::to_string(i + 1);
            if (i + 1 < argument_names.size())
                argument_name = argument_names[i + 1];
            result += to_string_with_precision(argument_name, 16) 
                   + to_string_with_precision(params.coef[i]) + to_string_with_precision(params.std[i]) 
                   + to_string_with_precision(params.t_value[i]) + to_string_with_precision(params.p_value[i]) + "\n";
        }
        result += "\nResidual standard error: " + std::to_string(params.standard_error) 
               + " on " + std::to_string(params.degrees_of_freedom) + " degrees of freedom\n";
        result += "Multiple R-squared: " + std::to_string(params.r_squared) 
               + ", Adjusted R-squared: " + std::to_string(params.adjusted_r_squared) + "\n";
        result += "F-statistic: " + std::to_string(params.f_statistic) + " on " + std::to_string(params.k) + " and "
               + std::to_string(params.degrees_of_freedom) + " DF,  p-value: " + std::to_string(params.f_value) + "\n";
        return result;
    }

    static DataTypePtr getReturnTypeToPredict()
    {
        return std::make_shared<DataTypeFloat64>();
    }

    static void predict(
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit,
        LeastSquaresParams<Float64>& params)
    {
        size_t rows_num = arguments.front().column->size();
        if (params.getParams().arg_num != arguments.size())
            throw Exception("Number of arguments is not equal to the number of model", 
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        if (offset > rows_num || offset + limit > rows_num)
            throw Exception("Invalid offset and limit for predict. "
                            "Block has " + toString(rows_num) + " rows, but offset is " + toString(offset) +
                            " and limit is " + toString(limit), ErrorCodes::BAD_ARGUMENTS);

        auto coef = params.getParams().coef;
        for (auto & coef_item : coef)
            if (std::isnan(coef_item))
                coef_item = 0;
        if (offset > rows_num || offset + limit > rows_num)
            throw Exception("Invalid offset and limit for predict. "
                            "Block has " + toString(rows_num) + " rows, but offset is " + toString(offset) +
                            " and limit is " + toString(limit), ErrorCodes::BAD_ARGUMENTS);

        std::vector<Float64> results(limit, params.getParams().use_bias ? coef.back() : 0);
        auto * column = typeid_cast<ColumnFloat64 *>(&to);
        if (!column)
            throw Exception("Cast of column of predictions is incorrect. \
                getReturnTypeToPredict must return same value as it is casted to", ErrorCodes::BAD_ARGUMENTS);
        auto & container = column->getData();
        for (size_t i = 1; i < arguments.size(); ++i)
        {
            const ColumnWithTypeAndName & cur_col = arguments[i];

            if (!isNativeNumber(cur_col.type))
                throw Exception("Prediction arguments must have numeric type", ErrorCodes::BAD_ARGUMENTS);

            const auto & features_column = cur_col.column;
            for (size_t row_num = 0; row_num < limit; ++row_num)
                results[row_num] += features_column->getFloat64(offset + row_num) * coef[i - 1];
        }
        container.reserve(container.size() + limit);
        for (size_t row_num = 0; row_num < limit; ++row_num)
            container.emplace_back(results[row_num]);
    }
};

struct AggregateFunctionOlsInterval : AggregateFunctionOls
{
    static constexpr auto name = "OlsInterval";

    static DataTypePtr getReturnTypeToPredict()
    {
        return std::make_shared<DataTypeArray>(std::make_shared<DataTypeFloat64>());
    }

    static void predict(
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit,
        LeastSquaresParams<Float64>& params)
    {
        size_t rows_num = arguments.front().column->size();
        if (params.getParams().arg_num + 2 < arguments.size())
            throw Exception("Number of arguments is not equal to the number of model", 
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        if (offset > rows_num || offset + limit > rows_num)
            throw Exception("Invalid offset and limit for predict. "
                            "Block has " + toString(rows_num) + " rows, but offset is " + toString(offset) +
                            " and limit is " + toString(limit), ErrorCodes::BAD_ARGUMENTS);
        auto coef = params.getParams().coef;
        for (auto & coef_item : coef)
            if (std::isnan(coef_item))
                coef_item = 0;
        auto use_bias = params.getParams().use_bias;
        auto arg_num = params.getParams().arg_num;

        std::vector<Float64> results(limit, params.getParams().use_bias ? coef.back() : 0);

        ColumnArray * column = typeid_cast<ColumnArray *>(&to);
        if (!column)
            throw Exception("Cast of column of predictions is incorrect. \
                getReturnTypeToPredict must return same value as it is casted to", ErrorCodes::BAD_ARGUMENTS);
        for (size_t row_num = 0; row_num < limit; ++row_num)
        {
            const ColumnWithTypeAndName & first_col = arguments[1];
            if (!isString(first_col.type))
                throw Exception("Prediction arguments must have string type", ErrorCodes::BAD_ARGUMENTS);

            auto tmp = first_col.column->getDataAt(offset + row_num);
            std::string interval = tmp.toString();
            Float64 level = 0.95;
            if (arguments.size() - 2 == arg_num)
                level = arguments[2].column->getFloat64(offset + row_num);

            Float64 result = use_bias ? coef.back() : 0;

            size_t x_len = arguments.size() - 2 - (arguments.size() - 2 == arg_num);
            Matrix x = Matrix(1, x_len + use_bias);
            if (use_bias) x(0, x_len) = 1;

            int pos = 0;
            for (size_t i = arguments.size() - x_len; i < arguments.size(); ++i)
            {
                const ColumnWithTypeAndName & cur_col = arguments[i];

                if (!isNativeNumber(cur_col.type))
                    throw Exception("Prediction arguments must have numeric type", ErrorCodes::BAD_ARGUMENTS);

                const auto & features_column = cur_col.column;
                result += features_column->getFloat64(offset + row_num) * coef[pos];
                x(0, pos++) = features_column->getFloat64(offset + row_num);
            }

            const auto & xx_1 = params.getParams().xx_1;
            auto tmp_ma = prod(x, xx_1);

            Float64 se = 0;
            for (size_t i = 0; i < tmp_ma.size2(); ++i)
                se += (isnan(tmp_ma(0, i)) ? 0 : tmp_ma(0, i)) * x(0, i) * params.getParams().sigma2;
            se = sqrt(se);

            auto df = params.getParams().degrees_of_freedom;
            // use df and level to get qt value with boost
            auto  qt = boost::math::quantile(boost::math::students_t(df), 1 - (1 - level) / 2);

            Float64 lower, upper;
            if (interval == "confidence")
            {
                lower = result - qt * se;
                upper = result + qt * se;
            }
            else if (interval == "prediction")
            {
                lower = result - sqrt(params.getParams().sigma2 + se * se) * qt;
                upper = result + sqrt(params.getParams().sigma2 + se * se) * qt;
            }
            else
                throw Exception("Invalid interval type", ErrorCodes::BAD_ARGUMENTS);

            column->getOffsets().push_back(column->getOffsets().back() + 3);
            column->getData().insert(result);
            column->getData().insert(lower);
            column->getData().insert(upper);
        }
    }
};

struct AggregateFunctionWls : public AggregateFunctionOls
{
    static constexpr auto name = "Wls";
};

template <typename T, typename Op, bool use_weights = false>
class AggregateFunctionLeastSquaresLinearRegression final:
    public IAggregateFunctionDataHelper<LeastSquaresLinearRegressionData<T, Op, use_weights>,
           AggregateFunctionLeastSquaresLinearRegression<T, Op, use_weights>>
{
private:
    using Data = LeastSquaresLinearRegressionData<T, Op, use_weights>;
    size_t arguments_num;
    bool add_constant = true;
    Matrix xx_inverse, xx_weighted;
    std::vector<String> argument_names;
public:
    explicit AggregateFunctionLeastSquaresLinearRegression(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<LeastSquaresLinearRegressionData<T, Op, use_weights>, 
        AggregateFunctionLeastSquaresLinearRegression<T, Op, use_weights>> ({arguments}, {})
    {
        
        size_t param_index = 0;
        arguments_num = arguments.size();
        if (param_index < params.size() && params[param_index].getType() == Field::Types::String)
        {
            String argument_name = params[param_index].get<String>();
            boost::split(argument_names, argument_name, [](char c){return c == ',';});
            for (auto & name : argument_names)
                boost::trim(name);
            param_index++;
        }

        if (param_index < params.size())
        {
            add_constant = params[param_index].get<bool>();
            param_index++;
        }

        auto initailize_matrix = [](Matrix & matrix, const String & str)
        {
            // unparse str format with 2d-matrix like '[[1,2],[3,4]]' to matrix
            if (str[0] != '[' || str.back() != ']')
                throw Exception("Invalid matrix format", ErrorCodes::BAD_ARGUMENTS);
                
            std::vector<std::vector<Float64>> tmp_ma;
            size_t s = 1;
            while (s < str.size())
            {
                if (str[s] == '[')
                {
                    tmp_ma.emplace_back();
                    s++;
                }
                size_t e = s;
                while (e < str.size() && str[e] != ',' && str[e] != ']')
                    e++;
                tmp_ma.back().push_back(std::stod(str.substr(s, e - s)));
                s = e + 1 + (str[e] == ']');
            }
            if (tmp_ma.empty())
                throw Exception("Invalid matrix format", ErrorCodes::BAD_ARGUMENTS);

            matrix.resize(tmp_ma.size(), tmp_ma[0].size());
            for (size_t i = 0; i < tmp_ma.size(); ++i)
            {
                if (i && tmp_ma[i].size() != tmp_ma[i-1].size())
                    throw Exception("Invalid matrix format", ErrorCodes::BAD_ARGUMENTS);
                for (size_t j = 0; j < tmp_ma[0].size(); ++j)
                    matrix(i, j) = tmp_ma[i][j];
            }
        };

        param_index++;
        if (param_index + 1 < params.size())
        {
            if (params[param_index].getType() == Field::Types::String && params[param_index + 1].getType() == Field::Types::String)
            {
                initailize_matrix(xx_inverse, params[param_index].get<String>());
                initailize_matrix(xx_weighted, params[param_index + 1].get<String>());
            }
            else
                throw Exception("Invalid type of input matrix", ErrorCodes::BAD_ARGUMENTS);
        }
    }

    DataTypePtr getReturnTypeToPredict() const override
    {
        return Op::getReturnTypeToPredict();
    }

    void predictValues(
        ConstAggregateDataPtr __restrict place,
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit,
        ContextPtr /*context*/) const override
    {
        if (arguments.size() < arguments_num - use_weights)
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                  "Predict got incorrect number of arguments. Got: {}. Required: {}",
                  arguments.size(), arguments_num - use_weights);
        this->data(place).predict(to, arguments, offset, limit);
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

    void create(AggregateDataPtr __restrict place) const override // NOLINT
    {
        new (place) Data(arguments_num, add_constant, xx_inverse, xx_weighted);
    }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
         this->data(place).add(columns, row_num);
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t>) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t>, Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        auto result = this->data(place).publish(argument_names);
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }
};

}
