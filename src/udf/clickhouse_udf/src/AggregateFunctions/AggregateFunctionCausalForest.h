#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/StatCommon.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <Columns/ColumnArray.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnString.h>
#include <Core/Field.h>
#include <base/types.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypeString.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <pqxx/params.hxx>
#include <regex>
#include <AggregateFunctions/AggregateFunctionCausalForestTrainer.h>
#include <IO/VarInt.h>
#include <IO/WriteBuffer.h>
#include <IO/WriteBufferFromString.h>
#include <IO/ReadBufferFromString.h>
#include <Common/PODArray_fwd.h>

namespace DB
{
struct Settings;

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

struct CausalForestData
{
    CausalForestData() = default;

    explicit CausalForestData(ForestTrainer trainer_)
        :trainer(trainer_)
    {
    }
   
    void add(const IColumn ** col, size_t row_num)
    {
        trainer.add(col, row_num);
    }

    void merge(const CausalForestData & other)
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

    String publish()
    {
        trainer.afterTrain();
        WriteBufferFromOwnString wb;
        trainer.serialize(wb);
        return wb.str();
    }

    ForestTrainer trainer;
};

class AggregateFunctionCausalForest final:
    public IAggregateFunctionDataHelper<CausalForestData, AggregateFunctionCausalForest>
{
private:
    using Data = CausalForestData;
    ForestTrainer trainer;

public:
    explicit AggregateFunctionCausalForest(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<CausalForestData, AggregateFunctionCausalForest> ({arguments}, {}) 
    {
        if (params.empty())
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Aggregate function CausalForest requires params.");
        String context = params[0].get<String>();
        trainer = ForestTrainer(context, arguments.size());
    }

    String getName() const override
    {
        return "CausalForest";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeString>();
    }

    void create(AggregateDataPtr __restrict place) const override // NOLINT
    {
        new (place) Data(trainer);
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
        auto result = this->data(place).publish();
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }
};

struct CausalForestPredictData
{
    CausalForestPredictData() = default;

    explicit CausalForestPredictData(const std::vector<String> & contexts, size_t arguments_size)
    {
        for (const auto & context : contexts)
        {
            trainers.emplace_back(context, arguments_size, true);
            tree_count += trainers.back().getNumTrees();
        }
        for (auto & trainer : trainers)
            trainer.initHonesty();
    }
   
    void add(const IColumn ** col, size_t row_num)
    {
        for (auto & trainer : trainers)
            trainer.add(col, row_num);
    }

    void merge(const CausalForestPredictData & other)
    {
        for (size_t i = 0; i < trainers.size(); ++i)
            trainers[i].merge(other.trainers[i]);
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(tree_count, buf);
        writeVarUInt(trainers.size(), buf);
        for (const auto & trainer : trainers)
            trainer.serialize(buf);
    }

    void deserialize(ReadBuffer & buf) // NOLINT
    {
        readVarUInt(tree_count, buf);
        size_t trainer_size;
        readVarUInt(trainer_size, buf);
        trainers.resize(trainer_size);
        for (auto & trainer : trainers)
            trainer.deserialize(buf);
    }

    String publish()
    {
        if (trainers.empty())
            return "trainers is empty";
        auto & trainer = trainers[0];
        trainer.afterTrain();
        WriteBufferFromOwnString wb;
        trainer.serialize(wb);
        return wb.str();
    }

    void predict(
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit) const
    {
        size_t rows_num = arguments.front().column->size();

        if (offset > rows_num || offset + limit > rows_num)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Invalid offset and limit for predict. "
                            "Block has {} rows, but offset is {}  and limit is {}",
                            toString(rows_num), toString(offset), toString(limit));

        if (offset > rows_num || offset + limit > rows_num)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Invalid offset and limit for predict. "
                            "Block has {} rows, but offset is {}  and limit is {}",
                            toString(rows_num), toString(offset), toString(limit));

        auto * column = typeid_cast<ColumnFloat64 *>(&to);
        if (!column)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Cast of column of predictions is incorrect. \
                getReturnTypeToPredict must return same value as it is casted to");
        auto & container = column->getData();

        container.reserve(container.size() + limit);
        for (size_t row_num = 0; row_num < limit; ++row_num)
        {
            PaddedPODArray<Float64> average(7, 0);
            for (const auto & trainer : trainers)
                trainer.predict(arguments, offset + row_num, average);

            for (size_t i = 0; i < average.size(); ++i)
                average[i] /= tree_count;


            double instrument_effect_numerator = average[OUTCOME_INSTRUMENT] * average[WEIGHT]
            - average[OUTCOME] * average[INSTRUMENT];
            double first_stage_numerator = average[TREATMENT_INSTRUMENT] * average[WEIGHT]
              - average[TREATMENT] * average[INSTRUMENT];
            container.emplace_back(instrument_effect_numerator / first_stage_numerator);
        }
    }

    std::vector<ForestTrainer> trainers;
    UInt16 tree_count = 0;
    bool is_honesty = false;
    static const std::size_t OUTCOME = 0;
    static const std::size_t TREATMENT = 1;
    static const std::size_t INSTRUMENT = 2;
    static const std::size_t OUTCOME_INSTRUMENT = 3;
    static const std::size_t TREATMENT_INSTRUMENT = 4;
    static const std::size_t INSTRUMENT_INSTRUMENT = 5;
    static const std::size_t WEIGHT = 6;
    static const std::size_t NUM_TYPES = 7;

};


class AggregateFunctionCausalForestPredict final:
    public IAggregateFunctionDataHelper<CausalForestPredictData, AggregateFunctionCausalForestPredict>
{
private:
    using Data = CausalForestPredictData;
    std::vector<String> contexts;
    size_t arguments_size;

public:
    explicit AggregateFunctionCausalForestPredict(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<CausalForestPredictData, AggregateFunctionCausalForestPredict> ({arguments}, {}) 
    {
        for (size_t i = 0; i < params.size(); ++i)
        {
            if (params[i].getType() != Field::Types::String)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function CausalForestPredict requires params to be String.");
            String context = params[i].get<String>();
            contexts.push_back(context);
        }
        arguments_size = arguments.size();
    }

    String getName() const override
    {
        return "CausalForestPredict";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnTypeToPredict() const override
    {
        return std::make_shared<DataTypeFloat64>();
    }

    void predictValues(
        ConstAggregateDataPtr __restrict place,
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit,
        ContextPtr /*context*/) const override
    {
        this->data(place).predict(to, arguments, offset, limit);
    }

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeString>();
    }

    void create(AggregateDataPtr __restrict place) const override // NOLINT
    {
        new (place) Data(contexts, arguments_size);
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
        auto result = this->data(place).publish();
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }
};

}
