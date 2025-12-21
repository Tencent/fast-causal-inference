#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionPearsonr.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/Helpers.h>
#include <Common/Exception.h>
#include <DataTypes/IDataType.h>

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

[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionPearsonr(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    if (argument_types.size() != 2)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Aggregate function {} requires 2 arguments", name);

    for (const auto & type : argument_types)
    {
        if (!isNativeNumber(type))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                "Illegal type {} of argument of aggregate function {}, must be Native Ints, Native UInts or Floats", type->getName(), name);
    }

    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionPearsonr>(argument_types, parameters);

    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function {}, must be Native Ints, Native UInts or Floats", name);
    return res;
}

}

void registerAggregateFunctionPearsonr([[maybe_unused]] AggregateFunctionFactory & factory)
{
#if ENABLE_ALL_IN_SQL
    factory.registerFunction("Pearsonr", createAggregateFunctionPearsonr);
#endif
}

}

