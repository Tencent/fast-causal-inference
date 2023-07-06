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

AggregateFunctionPtr createAggregateFunctionSRM(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    if (argument_types.size() != 3)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Aggregate function {} requires 3 arguments", name);

    if (!isNativeNumber(argument_types[0]) || 
        (!isString(argument_types[1]) && !isInteger(argument_types[1])) || 
        !isArray(argument_types[2]))
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Aggregate function {} requires arguments of types: Number, Integer or String, Array", name);

    AggregateFunctionPtr res;
    if (isString(argument_types[1]))
        res = std::make_shared<AggregateFunctionSRM<String>>(argument_types, parameters);
    else
        res = std::make_shared<AggregateFunctionSRM<Int64>>(argument_types, parameters);

    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, 
        "Illegal types arguments of aggregate function {},\
            must be Native Ints, Native UInts or Floats", name
        );
    return res;
}

}

void registerAggregateFunctionSRM(AggregateFunctionFactory & factory)
{
    factory.registerFunction("SRM", createAggregateFunctionSRM);
}

}
