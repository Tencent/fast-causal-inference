#include <AggregateFunctions/AggregateFunctionCausalForest.h>
#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/FactoryHelpers.h>
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/Helpers.h>
#include <Common/Exception.h>
#include <DataTypes/IDataType.h>

namespace DB
{
struct Settings;
namespace
{

AggregateFunctionPtr createAggregateFunctionCausalForest(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionCausalForest>(argument_types, parameters);

    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, 
        "Illegal types arguments of aggregate function {},\
            must be Native Ints, Native UInts or Floats", name
        );
    return res;
}

AggregateFunctionPtr createAggregateFunctionCausalForestPredict(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    res = std::make_shared<AggregateFunctionCausalForestPredict>(argument_types, parameters);

    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, 
        "Illegal types arguments of aggregate function {},\
            must be Native Ints, Native UInts or Floats", name
        );
    return res;
}

}

void registerAggregateFunctionCausalForest(AggregateFunctionFactory & factory)
{
    factory.registerFunction("CausalForest", createAggregateFunctionCausalForest);
    factory.registerFunction("CausalForestPredict", createAggregateFunctionCausalForestPredict);
}

}
