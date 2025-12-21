#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnArray.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <DataTypes/DataTypeTuple.h>
#include <Common/Exception.h>
#include <Common/PODArray_fwd.h>
#include "Moments.h"
#include <IO/VarInt.h>
#include <base/types.h>
#include <AggregateFunctions/Moments.h>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

struct PearsonrData
{
    PearsonrData() = default;

    void add(const IColumn ** column, size_t row_num)
    {
        moments_corr.add(
            column[0]->getFloat64(row_num),
            column[1]->getFloat64(row_num));
    }

    void merge(const PearsonrData & source)
    {
        moments_corr.merge(source.moments_corr);
    }

    void serialize(WriteBuffer & buf) const
    {
        moments_corr.write(buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        moments_corr.read(buf);
    }

    std::pair<Float64, Float64> publish() const
    {
        Float64 r = moments_corr.get();
        Int64 n = static_cast<Int64>(moments_corr.m0);
        Int64 df = n - 2;
        if (df < 1)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Pearsonr requires at least 3 rows");
        /*
         *  t_stat = r * (df ** 0.5) / ((1 - r ** 2) ** 0.5)
    print(t_stat)
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    */
        Float64 t_stat = r * std::sqrt(df) / std::sqrt(1 - r * r);
        boost::math::students_t dist(df);
        Float64 p_value = 2 * (1 - boost::math::cdf(dist, std::abs(t_stat)));
        return {r, p_value};
    }

    CorrMoments<Float64> moments_corr;
};

class AggregateFunctionPearsonr final:
    public IAggregateFunctionDataHelper<PearsonrData, AggregateFunctionPearsonr>
{
private:
    using Data = PearsonrData;

public:
    explicit AggregateFunctionPearsonr(const DataTypes & arguments, const Array &)
        :IAggregateFunctionDataHelper<PearsonrData, AggregateFunctionPearsonr> ({arguments}, {}, getReturnType()) {}

    String getName() const override
    {
        return "Pearsonr";
    }

    bool allocatesMemoryInArena() const override { return false; }

    static DataTypePtr getReturnType()
    {
        DataTypes types
        {
            std::make_shared<DataTypeNumber<Float64>>(),
            std::make_shared<DataTypeNumber<Float64>>(),
        };

        Strings names
        {
            "u_statistic",
            "p_value"
        };

        return std::make_shared<DataTypeTuple>(
            std::move(types),
            std::move(names)
        );
    }

    void create(AggregateDataPtr __restrict place) const override
    {
        new (place) Data();
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
        auto & column = assert_cast<ColumnTuple &>(to);
        auto & u_statistic_column = assert_cast<ColumnVector<Float64> &>(column.getColumn(0));
        auto & p_value_column = assert_cast<ColumnVector<Float64> &>(column.getColumn(1));
        u_statistic_column.getData().push_back(result.first);
        p_value_column.getData().push_back(result.second);
    }
};

}

