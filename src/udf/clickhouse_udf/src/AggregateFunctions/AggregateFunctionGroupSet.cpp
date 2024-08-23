#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionGroupSet.h>
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

[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionGroupSet(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    for (const auto & arg : argument_types)
        if (!isNativeNumber(arg) && !isString(arg))
            throw Exception(
            "Illegal types of arguments of aggregate function " + name
                + ", must be Native Ints, Native UInts or Floats",
            ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

    if (argument_types.size() < 3)
        throw Exception(
        "Incorrect number of arguments for aggregate function " + name
            + ", should be at least 3 : Y, treutment, group",
        ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionGroupSet>(argument_types, parameters);

    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}

}

void registerAggregateFunctionGroupSet([[maybe_unused]] AggregateFunctionFactory & factory)
{ 
#if ENABLE_ALL_IN_SQL
    factory.registerFunction("GroupSet", createAggregateFunctionGroupSet);
#endif
}

}
