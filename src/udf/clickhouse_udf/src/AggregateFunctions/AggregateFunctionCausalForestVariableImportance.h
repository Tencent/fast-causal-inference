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

struct CausalForestVariableImportanceData
{
    CausalForestVariableImportanceData() = default;

    explicit CausalForestVariableImportanceData(String context)
    {
        ReadBufferFromString rb{context};
        trainer.deserialize(rb);
    }
   
    void add(const IColumn ** col, size_t row_num)
    {
        trainer.add(col, row_num);
    }

    void merge(const CausalForestVariableImportanceData & other)
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
        auto variable_importances = trainer.getVariableImportance();
        auto & data_to = assert_cast<ColumnInt32 &>(assert_cast<ColumnArray &>(assert_cast<ColumnArray &>(to).getData()).getData()).getData();
        auto & root_offsets_to = assert_cast<ColumnArray &>(to).getOffsets();
        auto & nested_offsets_to = assert_cast<ColumnArray &>(assert_cast<ColumnArray &>(to).getData()).getOffsets();
        for (auto & variable_importance_each_depth : variable_importances)
        {
            for (auto & variable_importance : variable_importance_each_depth)
            {
                data_to.push_back(variable_importance);
            }
            nested_offsets_to.push_back(nested_offsets_to.back() + variable_importance_each_depth.size());
        }
        root_offsets_to.push_back(root_offsets_to.back() + nested_offsets_to.size());
    }

    ForestTrainer trainer;
};

class AggregateFunctionCausalForestVariableImportance final:
    public IAggregateFunctionDataHelper<CausalForestVariableImportanceData, AggregateFunctionCausalForestVariableImportance>
{
private:
    using Data = CausalForestVariableImportanceData;
    String context;

public:
    explicit AggregateFunctionCausalForestVariableImportance(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<CausalForestVariableImportanceData, AggregateFunctionCausalForestVariableImportance> ({arguments}, {}, getReturnType()) 
    {
        if (params.size() != 1)
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Aggregate function CausalForest requires params.");
        context = params[0].get<String>();
    }

    String getName() const override
    {
        return "CausalForestVariableImportance";
    }

    bool allocatesMemoryInArena() const override { return false; }

    static DataTypePtr getReturnType()
    {
        return std::make_shared<DataTypeArray>(std::make_shared<DataTypeArray>(std::make_shared<DataTypeInt32>()));
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


