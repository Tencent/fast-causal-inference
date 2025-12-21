#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/StatCommon.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <AggregateFunctions/AggregateFunctionCausalForestTrainer.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnString.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesNumber.h>
#include <IO/ReadHelpers.h>
#include <IO/VarInt.h>
#include <IO/ReadBufferFromString.h>
#include <Common/PODArray_fwd.h>
#include <fstream>

namespace DB
{
struct Settings;

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

struct CausalForestStructData
{
    CausalForestStructData() = default;

    explicit CausalForestStructData(String context)
    {
        ReadBufferFromString rb{context};
        trainer.deserialize(rb);
    }
   
    void add(const IColumn ** col, size_t row_num)
    {
        trainer.add(col, row_num);
    }

    void merge(const CausalForestStructData & other)
    { 
        trainer.merge(other.trainer);
    }

    void serialize(WriteBuffer & other) const
    {
        trainer.serialize(other);
    }

    void deserialize(ReadBuffer & other) // NOLINT
    {
        trainer.deserialize(other);      
    }

    void insertResultInto(IColumn & to) const
    {
        auto result = trainer.getForestStruct();
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }

    ForestTrainer trainer;
};

class AggregateFunctionCausalForestStruct final:
    public IAggregateFunctionDataHelper<CausalForestStructData, AggregateFunctionCausalForestStruct>
{
private:
    using Data = CausalForestStructData;
    String context;

public:
    explicit AggregateFunctionCausalForestStruct(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<CausalForestStructData, AggregateFunctionCausalForestStruct> ({arguments}, {}, getReturnType()) 
    {
        if (params.size() != 1)
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Aggregate function CausalForest requires params.");
        context = params[0].get<String>();
    }

    String getName() const override
    {
        return "CausalForestStruct";
    }

    bool allocatesMemoryInArena() const override { return false; }

    static DataTypePtr getReturnType()
    {
        return std::make_shared<DataTypeString>();
    }

    void create(AggregateDataPtr __restrict place) const override // NOLINT
    {
        new (place) Data(context);
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
        this->data(place).insertResultInto(to);
    }
};


}


