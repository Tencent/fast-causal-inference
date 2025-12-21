#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionMatchingInfo.h>
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

AggregateFunctionPtr createAggregateFunctionMatchingInfo(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionMatchingInfo>(argument_types, parameters);
    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function {}, must be Native Ints, Native UInts or Floats", name);
    if (argument_types.size() < 3)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
        "Illegal number of arguments of aggregate function {}, must be at least 3", name);
    for (size_t i = 0; i < 3; ++i)
        if (!isNativeNumber(argument_types[i]))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
            "Illegal type {} of argument of aggregate function {}, must be Native Ints, Native UInts or Floats", argument_types[i]->getName(), name);
    return res;
}

}

void registerAggregateFunctionMatchingInfo(AggregateFunctionFactory & factory)
{
    factory.registerFunction("CaliperMatchingInfo", createAggregateFunctionMatchingInfo);
}

}
