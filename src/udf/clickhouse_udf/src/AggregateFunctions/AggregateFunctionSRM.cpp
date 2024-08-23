#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionSRM.h>
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

[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionSRM(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    if (argument_types.size() != 3)
        throw Exception("Aggregate function " + name + " requires 3 arguments", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    if (!isNativeNumber(argument_types[0]) || 
        (!isString(argument_types[1]) && !isInteger(argument_types[1])) || 
        !isArray(argument_types[2]))
        throw Exception("Aggregate function " + name + " requires arguments of types: Number, Integer or String, Array",
                        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

    AggregateFunctionPtr res;
    if (isString(argument_types[1]))
        res = std::make_shared<AggregateFunctionSRM<String>>(argument_types, parameters);
    else
        res = std::make_shared<AggregateFunctionSRM<Int64>>(argument_types, parameters);

    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}

}

void registerAggregateFunctionSRM([[maybe_unused]] AggregateFunctionFactory & factory)
{
#if ENABLE_ALL_IN_SQL
    factory.registerFunction("SRM", createAggregateFunctionSRM);
#endif
}

}
