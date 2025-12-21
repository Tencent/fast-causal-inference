#include <array>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/IFunction.h>
#include "Common/Exception.h"
#include <Common/typeid_cast.h>
#include <DataTypes/IDataType.h>
#include <base/types.h>
#include <IO/WriteHelpers.h>
#include <base/range.h>
#include <constants.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <Poco/JSON/Parser.h>
#include <Poco/JSON/Object.h>
#include <Poco/JSON/Array.h>

namespace DB
{
namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int ILLEGAL_COLUMN;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

namespace
{

class CumuLativeDistributionFunction : public IFunction
{
public:
    static constexpr auto name = "cdf";

    static FunctionPtr create(ContextPtr) { return std::make_shared<CumuLativeDistributionFunction>(); }

    bool isVariadic() const override { return true; }

    std::string getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 0; }
    bool useDefaultImplementationForConstants() const override { return true; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (arguments.size() != 1 && arguments.size() != 2)
            throw Exception(
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                "Function {} requires exactly one or two arguments, but {} were provided",
                getName(), arguments.size());

        for (const auto & arg : arguments)
        {
            if (!isNativeNumber(arg))
                throw Exception(
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "All arguments for function {} must be number",
                    getName());
        }

        return std::make_shared<DataTypeFloat64>();
    }


    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {

        auto dst = ColumnVector<Float64>::create();
        auto & dst_data = dst->getData();
        dst_data.resize(input_rows_count);

        for (size_t row = 0; row < input_rows_count; ++row)
        {
            const Float64 score = arguments[0].column->getFloat64(row);
            Int64 degrees_of_freedom = 120;
            if (arguments.size() == 2)
            {
                degrees_of_freedom = arguments[1].column->getInt(row);
                if (degrees_of_freedom <= 0)
                    throw Exception(
                        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                        "Degrees of freedom must be a positive integer, but got {}",
                        degrees_of_freedom);
            }

            dst_data[row] = boost::math::cdf(
                boost::math::students_t(degrees_of_freedom),
                score);
        }

        return dst;
    }
};

}

REGISTER_FUNCTION(CumuLativeDistributionFunction)
{
    factory.registerFunction<CumuLativeDistributionFunction>();
}

}
