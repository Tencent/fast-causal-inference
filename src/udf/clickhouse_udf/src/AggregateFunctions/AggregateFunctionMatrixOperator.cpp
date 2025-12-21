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
    const std::string &, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    for (size_t i = 0; i < argument_types.size(); ++i)
        if (!isNumber(argument_types[i]))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Illegal type {} of argument of aggregate function ", argument_types[i]->getName());

    for (const auto & parameter : parameters)
        if (parameter.getType() != Field::Types::Bool)
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Aggregate function requires parameter: [Bool], [Bool],");

    auto res = std::make_shared<AggregateFunc>(argument_types, parameters);
    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function, must be Native Ints, Native UInts or Floats");
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
