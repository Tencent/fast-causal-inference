#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionPermutation.h>
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


template <typename Func>
[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionPermutation(
    const std::string &, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    res = std::make_shared<Func>(argument_types, parameters);

    if (!res)
        throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
        "Illegal types arguments of aggregate function, must be Native Ints, Native UInts or Floats");
    return res;
}

}

void registerAggregateFunctionPermutation([[maybe_unused]] AggregateFunctionFactory & factory)
{
#if ENABLE_ALL_IN_SQL
    factory.registerFunction("Permutation", createAggregateFunctionPermutation<AggregateFunctionPermutation>);
    factory.registerFunction("PermutationMulti", createAggregateFunctionPermutation<AggregateFunctionPermutationMulti>);
#endif
}

}
