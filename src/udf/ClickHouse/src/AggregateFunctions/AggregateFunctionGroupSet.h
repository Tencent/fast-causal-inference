#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnTuple.h>
#include <DataTypes/DataTypeTuple.h>
#include <DataTypes/DataTypeArray.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <Common/Exception.h>
#include <Common/PODArray_fwd.h>
#include <DataTypes/DataTypesNumber.h>
#include <IO/VarInt.h>
#include <base/types.h>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

struct GroupData
{
    GroupData() 
    {
        treatment.resize_fill(2, TreatmentData());
    }

    void add(UInt8 treatment_id, Float64 value)
    {
        if (treatment_id > 1)
            return ;
        treatment[treatment_id].cnt += 1;
        treatment[treatment_id].sum += value;
        treatment[treatment_id].sum2 += value * value;
    }

    void merge(const GroupData & source)
    {
        for (size_t i = 0; i < 2; ++i)
        {
            treatment[i].cnt += source.treatment[i].cnt;
            treatment[i].sum += source.treatment[i].sum;
            treatment[i].sum2 += source.treatment[i].sum2;
        }
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(treatment[0].cnt, buf);
        writeVarUInt(treatment[1].cnt, buf);
        writeFloatBinary(treatment[0].sum, buf);
        writeFloatBinary(treatment[1].sum, buf);
        writeFloatBinary(treatment[0].sum2, buf);
        writeFloatBinary(treatment[1].sum2, buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        readVarUInt(treatment[0].cnt, buf);
        readVarUInt(treatment[1].cnt, buf);
        readFloatBinary(treatment[0].sum, buf);
        readFloatBinary(treatment[1].sum, buf);
        readFloatBinary(treatment[0].sum2, buf);
        readFloatBinary(treatment[1].sum2, buf);
    }

    struct TreatmentData
    {
        UInt64 cnt = 0;
        Float64 sum = 0;
        Float64 sum2 = 0;
    };

    PaddedPODArray<TreatmentData> treatment;
};

struct GroupSetData
{
    GroupSetData() = default;

    explicit GroupSetData(const UInt64 & arguments_num_, const PaddedPODArray<String> & col_name_)
        : arguments_num(arguments_num_), col_name(col_name_.begin(), col_name_.end())
    {
        group_set.resize(arguments_num - 2);
    }

    void add(const IColumn ** column, size_t row_num)
    {
        Float64 y = column[0]->getFloat64(row_num);
        UInt8 treatment_id = column[1]->getUInt(row_num);
        for (size_t i = 2; i < arguments_num; ++i)
        {
            Int64 group_id = column[i]->getInt(row_num);
            group_set[i - 2][group_id].add(treatment_id, y);
        }
    }

    void merge(const GroupSetData & source)
    {
        for (size_t i = 0; i < source.group_set.size(); ++i)
            for (const auto & group : source.group_set[i])
                group_set[i][group.first].merge(group.second);
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(arguments_num, buf);
        if (arguments_num - 2 != group_set.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical Error: wrong group_set size");
        for (size_t i = 0; i < arguments_num - 2; ++i)
        {
            writeVarUInt(group_set[i].size(), buf);
            for (const auto & [group_key, group_data] : group_set[i])
            {
                writeIntBinary(group_key, buf);
                group_data.serialize(buf);
            }
        }
    }

    void deserialize(ReadBuffer & buf)
    {
        readVarUInt(arguments_num, buf);
        group_set.resize(arguments_num - 2);
        for (size_t i = 0; i < arguments_num - 2; ++i)
        {
            size_t group_set_size = 0;
            readVarUInt(group_set_size, buf);
            for (size_t j = 0; j < group_set_size; ++j)
            {
                Int64 group_key = 0;
                readIntBinary(group_key, buf);
                group_set[i][group_key].deserialize(buf);
            }
        }
    }

    void publish(IColumn & to)
    {
        if (col_name.size() != group_set.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical Error: col_name: {} not equal to group_set: {}", col_name.size(), group_set.size());

        auto & to_array = assert_cast<ColumnArray &>(to);
        ColumnArray::Offsets & offsets_to = to_array.getOffsets();
        auto & to_tuple = assert_cast<ColumnTuple &>(to_array.getData());

        ColumnString & to_name = typeid_cast<ColumnString &>(to_tuple.getColumn(0));
        ColumnString::Chars & to_name_chars = to_name.getChars();
        ColumnString::Offsets & to_name_offsets = to_name.getOffsets();
        ColumnString::Offset current_strings_offset = 0;

        auto & to_treatment = assert_cast<ColumnVector<UInt8> &>(to_tuple.getColumn(1));
        auto & to_group_key = assert_cast<ColumnVector<Int64> &>(to_tuple.getColumn(2));
        auto & to_cnt = assert_cast<ColumnVector<UInt64> &>(to_tuple.getColumn(3));
        auto & to_sum = assert_cast<ColumnVector<Float64> &>(to_tuple.getColumn(4));
        auto & to_sum2 = assert_cast<ColumnVector<Float64> &>(to_tuple.getColumn(5));

        UInt64 total_size = 0;
        for (size_t i = 0; i < group_set.size(); ++i)
        {
            String name = col_name[i];
            total_size += group_set[i].size() * 2;
            if (total_size > 100000)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Too many groups, the number of group must <= 100000");
            for (const auto & [group_key, group_data] : group_set[i])
            {
                if (group_data.treatment.size() != 2)
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical Error: size of treatment != 2");
                for (size_t j = 0; j < 2; ++j)
                {
                    to_name_chars.resize(to_name_chars.size() + name.size() + 1);
                    memcpy(&to_name_chars[current_strings_offset], name.c_str(), name.size());
                    to_name_chars[current_strings_offset + name.size()] = 0;
                    current_strings_offset += name.size() + 1;
                    to_name_offsets.push_back(current_strings_offset);

                    const auto & treat_data = group_data.treatment[j];
                    auto cnt = treat_data.cnt;
                    auto sum = treat_data.sum;
                    auto sum2 = treat_data.sum2;
                    to_treatment.insertValue(j);
                    to_group_key.insertValue(group_key);
                    to_cnt.insertValue(cnt);
                    to_sum.insertValue(sum);
                    to_sum2.insertValue(sum2);
                }
            }
        }
        offsets_to.push_back(offsets_to.back() + total_size);
    }

    UInt64 arguments_num;
    PaddedPODArray<String> col_name;
    std::vector<std::unordered_map<Int64, GroupData>> group_set;

};

class AggregateFunctionGroupSet final:
    public IAggregateFunctionDataHelper<GroupSetData, AggregateFunctionGroupSet>
{
private:
    using Data = GroupSetData;
    size_t arguments_num;
    PaddedPODArray<String> col_name;

public:
    explicit AggregateFunctionGroupSet(const DataTypes & arguments, const Array & params)
      :IAggregateFunctionDataHelper<GroupSetData, AggregateFunctionGroupSet> ({arguments}, {params}) 
    {
        arguments_num = arguments.size();
        if (!params.empty())
        {
            for (size_t i = 0; i < params.size(); ++i)
                if (params[i].getType() != Field::Types::String)
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Illegal type {} of argument #{} of function {}", params[i].getTypeName(), i + 1, getName());
                else
                    col_name.push_back(params[i].get<String>());
        }
        if (col_name.empty())
            for (size_t i = 0; i < arguments_num - 2; ++i)
                col_name.push_back("col" + std::to_string(i + 1));

        if (col_name.size() != arguments_num - 2)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Number of column names {} doesn't match number of arguments {} of function {}", col_name.size(), arguments_num - 2, getName());
    }

    String getName() const override
    {
        return "GroupSet";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        DataTypes types
        {
            std::make_shared<DataTypeString>(),
            std::make_shared<DataTypeUInt8>(),
            std::make_shared<DataTypeInt64>(),
            std::make_shared<DataTypeUInt64>(),
            std::make_shared<DataTypeFloat64>(),
            std::make_shared<DataTypeFloat64>()
        };

        Strings names
        {
            "name",
            "treatment",
            "group_key",
            "cnt",
            "sum",
            "sum2"
        };

        DataTypePtr nest_type = std::make_shared<DataTypeTuple>(std::move(types), std::move(names));
        return std::make_shared<DataTypeArray>(nest_type);
    }

    void create(AggregateDataPtr __restrict place) const override // NOLINT
    {
        new (place) Data(arguments_num, col_name);
    }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
         this->data(place).add(columns, row_num);
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> ) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> , Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        this->data(place).publish(to);
    }
};

}
