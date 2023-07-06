#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionLeastSquaresLinearRegression.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/Helpers.h>
#include <base/types.h>

namespace DB
{
struct Settings;

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

namespace
{

template <typename T, typename Op, bool use_weights = false>
AggregateFunctionPtr createAggregateFunctionLeastSquaresLinearRegression(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    if (argument_types.size() <= 1)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, 
            "Aggregate function {} requires at least two arguments", name);

    for (const auto & argument_type : argument_types)
    {
        if (!isNumber(argument_type))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of argument of aggregate function {}", argument_type->getName(), name);
    }

    if (!(parameters.empty() || (!parameters.empty() && parameters[0].getType() == Field::Types::Bool)))
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, 
            "Aggregate function {} requires no parameters or a single boolean parameter", name);

    auto res = std::make_shared<AggregateFunctionLeastSquaresLinearRegression<T, Op, use_weights>>(argument_types, parameters);
    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function {}\
            , must be Native Ints, Native UInts or Floats", name);
    return res;
}
}

void registerAggregateFunctionLeastSquaresLinearRegression(AggregateFunctionFactory & factory)
{
    factory.registerFunction("Ols", createAggregateFunctionLeastSquaresLinearRegression<Float64, AggregateFunctionOls>);
    factory.registerFunction("OlsInterval", 
                            createAggregateFunctionLeastSquaresLinearRegression<Float64, AggregateFunctionOlsInterval>);
    factory.registerFunction("Wls", 
                            createAggregateFunctionLeastSquaresLinearRegression<Float64, AggregateFunctionWls, true>);
}

}
