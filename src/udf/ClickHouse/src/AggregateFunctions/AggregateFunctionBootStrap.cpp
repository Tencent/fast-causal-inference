#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionBootStrap.h>
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

AggregateFunctionPtr createAggregateFunctionDistributedNodeRowNumber(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    assertArityAtMost<1>(name, argument_types);
    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionDistributedNodeRowNumber>(argument_types, parameters);

    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function {}, must be Native Ints, Native UInts or Floats", name);
    return res;
}

AggregateFunctionPtr createAggregateFunctionBootStrap(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionBootStrap>(argument_types, parameters);

    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function {}, must be Native Ints, Native UInts or Floats", name);
    return res;
}

}

void registerAggregateFunctionBootStrap(AggregateFunctionFactory & factory)
{
    factory.registerFunction("DistributedNodeRowNumber", createAggregateFunctionDistributedNodeRowNumber);
    factory.registerFunction("BootStrap", createAggregateFunctionBootStrap);
}

}
