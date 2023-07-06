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

AggregateFunctionPtr createAggregateFunctionGroupSet(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    for (const auto & arg : argument_types)
        if (!isNativeNumber(arg))
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
            "Illegal types of arguments of aggregate function {}\
                , must be Native Ints, Native UInts or Floats", name);

    if (argument_types.size() < 3)
        throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
        "Incorrect number of arguments for aggregate function {}\
            , should be at least 3 : Y, treutment, group", name);

    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionGroupSet>(argument_types, parameters);

    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function {}\
            , must be Native Ints, Native UInts or Floats", name);
    return res;
}

}

void registerAggregateFunctionGroupSet(AggregateFunctionFactory & factory)
{
    factory.registerFunction("GroupSet", createAggregateFunctionGroupSet);
}

}
