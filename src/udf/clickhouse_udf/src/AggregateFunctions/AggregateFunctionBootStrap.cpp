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

[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionDistributedNodeRowNumber(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionDistributedNodeRowNumber>(argument_types, parameters);

    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}

template <typename Func>
[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionBootStrap(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    res = std::make_shared<Func>(argument_types, parameters);

    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}

template <typename T, typename Op>
[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionBootStrapOls(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionBootStrapOls<T, Op>>(argument_types, parameters);
    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function {}, must be Native Ints, Native UInts or Floats", name);
    return res;
}

}

void registerAggregateFunctionBootStrap([[maybe_unused]] AggregateFunctionFactory & factory)
{
#if ENABLE_ALL_IN_SQL
    factory.registerFunction("DistributedNodeRowNumber", createAggregateFunctionDistributedNodeRowNumber);
    factory.registerFunction("BootStrap", createAggregateFunctionBootStrap<AggregateFunctionBootStrap>);
    factory.registerFunction("BootStrapMulti", createAggregateFunctionBootStrap<AggregateFunctionBootStrapMulti>);
    factory.registerFunction("BootStrapOls", createAggregateFunctionBootStrapOls<Float64, AggregateFunctionOls>);
#endif
}

}
