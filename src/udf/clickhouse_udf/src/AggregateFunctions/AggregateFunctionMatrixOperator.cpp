#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionMatrixOperator.h>
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

template <typename AggregateFunc>
[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionMatrixOperator(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    for (size_t i = 0; i < argument_types.size(); ++i)
        if (!isNumber(argument_types[i]))
            throw Exception("Illegal type " + argument_types[i]->getName() + " of argument of aggregate function " 
                + name, ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

    for (const auto & parameter : parameters)
        if (parameter.getType() != Field::Types::Bool)
            throw Exception("Aggregate function " + name + " requires parameter: [Bool], [Bool],",
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    auto res = std::make_shared<AggregateFunc>(argument_types, parameters);
    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}

}

void registerAggregateFunctionMatrixOperator([[maybe_unused]] AggregateFunctionFactory & factory)
{
#if ENABLE_ALL_IN_SQL
    factory.registerFunction("MatrixMultiplication", 
        createAggregateFunctionMatrixOperator<AggregateFunctionMatrixOperator<AggregateFunctionMatrixMulData<Float64>>>);
#endif
}

}
