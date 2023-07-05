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
        throw Exception("Aggregate function " + name + " requires at least two arguments", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    for (const auto & argument_type : argument_types)
    {
        if (!isNumber(argument_type))
            throw Exception("Illegal type " + argument_type->getName() + " of argument of aggregate function " + name,
                            ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    }

    if (!(parameters.empty() || (parameters.size() >= 1 && parameters[0].getType() == Field::Types::Bool)))
        throw Exception("Aggregate function " + name + " requires no parameters or a single boolean parameter",
                        ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    auto res = std::make_shared<AggregateFunctionLeastSquaresLinearRegression<T, Op, use_weights>>(argument_types, parameters);
    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}
}

void registerAggregateFunctionLeastSquaresLinearRegression(AggregateFunctionFactory & factory)
{
    factory.registerFunction("Ols", createAggregateFunctionLeastSquaresLinearRegression<Float64, AggregateFunctionOls>);
    factory.registerFunction("OlsInterval", createAggregateFunctionLeastSquaresLinearRegression<Float64, AggregateFunctionOlsInterval>);
    factory.registerFunction("Wls", createAggregateFunctionLeastSquaresLinearRegression<Float64, AggregateFunctionWls, true>);
}

}
