#include <array>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnsNumber.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/IFunction.h>
#include <Common/typeid_cast.h>
#include <DataTypes/IDataType.h>
#include <base/types.h>
#include <IO/WriteHelpers.h>
#include <base/range.h>
#include <constants.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
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

class FunctionCalperMatching : public IFunction
{
public:
    static constexpr auto name = "CaliperMatching";

    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionCalperMatching>(); }

    bool isVariadic() const override { return true; }

    std::string getName() const override { return name; }

    size_t getNumberOfArguments() const override { return 0; }
    bool useDefaultImplementationForConstants() const override { return true; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo & /*arguments*/) const override { return true; }

    DataTypePtr getReturnTypeImpl(const DataTypes & arguments) const override
    {
        if (arguments.size() < 4)
            throw Exception(
                ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                "Function {} requires at least 3 arguments but {} provided",
                getName(),
                arguments.size());
        for (size_t i = 1 ; i < 4; i++)
            if (!isNativeNumber(arguments[i]))
                throw Exception(
                    ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT,
                    "Argument {} for function {} must be number",
                    i,
                    getName());
        return std::make_shared<DataTypeInt64>();
    }

    mutable std::unique_ptr<std::unordered_map<Int64, std::unordered_map<UInt64, std::array<std::atomic<Int64>, 6> >>> indexs = nullptr;
    mutable std::mutex mtx; 

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t input_rows_count) const override
    {

        if (indexs == nullptr)
        {
            const ColumnConst * col_str = checkAndGetColumn<ColumnConst>(arguments[0].column.get());
            if (!col_str)
                throw Exception(ErrorCodes::ILLEGAL_COLUMN, "First argument for function {} must be constant string", getName());
            String json = col_str->getDataAt(0).toString();

            std::unique_lock<std::mutex> lock(mtx);
            if (indexs == nullptr)
                indexs = std::make_unique<std::unordered_map<Int64, std::unordered_map<UInt64, std::array<std::atomic<Int64>, 6> >>>();

            Poco::JSON::Parser parser;
            Poco::Dynamic::Var result = parser.parse(json);
            Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();

            for (const auto & item : *object)
            {
                if (getNodeKey() == item.first)
                {
                    for (const auto & node : *item.second.extract<Poco::JSON::Object::Ptr>())
                    {
                        Int64 score = std::stoll(node.first);
                        for (const auto & edge : *node.second.extract<Poco::JSON::Object::Ptr>())
                        {
                            UInt64 group_id = std::stoull(edge.first);
                            Poco::JSON::Object::Ptr edge_obj = edge.second.extract<Poco::JSON::Object::Ptr>();
                            Int64 start_0 = edge_obj->getValue<Int64>("start_0");
                            Int64 count_0 = edge_obj->getValue<Int64>("count_0");
                            Int64 start_1 = edge_obj->getValue<Int64>("start_1");
                            Int64 count_1 = edge_obj->getValue<Int64>("count_1");
                            Int64 origin_start_index = edge_obj->getValue<Int64>("origin_start_index");
                            Int64 origin_end_index = edge_obj->getValue<Int64>("origin_end_index");
                            (*indexs)[score][group_id][0] = start_0;
                            (*indexs)[score][group_id][1] = count_0;
                            (*indexs)[score][group_id][2] = start_1;
                            (*indexs)[score][group_id][3] = count_1;
                            (*indexs)[score][group_id][4] = origin_start_index;
                            (*indexs)[score][group_id][5] = origin_end_index;
                        }
                    }
                }
            }
        }


        auto dst = ColumnVector<Int64>::create();
        auto & dst_data = dst->getData();
        dst_data.resize(input_rows_count);

        for (size_t row = 0; row < input_rows_count; ++row)
        {
            const UInt64 treatment = arguments[1].column->getUInt(row);
            if (treatment != 0 && treatment != 1)
                throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Second argument for function {} must be 0 or 1", getName());

            const Float64 distance = arguments[2].column->getFloat64(row);

            const Float64 step = arguments[3].column->getFloat64(row);

            const Int64 score = static_cast<Int64>(std::floor(distance / step));

            UInt64 group_hash = 0;
            for (size_t i = 4; i < arguments.size(); ++i)
                group_hash ^= std::hash<String>()(arguments[i].column->getDataAt(row).toString());

            const size_t index = treatment == 0 ? 1 : 3;


            if (indexs == nullptr) 
                throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Indexs is nullptr");

            if (indexs->find(score) == indexs->end() || indexs->at(score).find(group_hash) == indexs->at(score).end())
                throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Indexs not found");

            Int64 count = (*indexs)[score][group_hash][index].fetch_sub(1);

            const Int64 origin_start_index = (*indexs)[score][group_hash][4];
            const Int64 origin_end_index = (*indexs)[score][group_hash][5];
            const Int64 len = origin_end_index - origin_start_index;

            if (count > 0)
            {
                Int64 value = (*indexs)[score][group_hash][index - 1] + count;
                if (value >= origin_end_index && len > 0) {
                    value = (value - origin_start_index) % len + origin_start_index;
                }

                dst_data[row] = value * (treatment == 0 ? 1 : -1);
            }
            else 
                dst_data[row] = 0;
        }

        return dst;
    }
};

}

REGISTER_FUNCTION(CaliperMatching)
{
    factory.registerFunction<FunctionCalperMatching>();
}

}
