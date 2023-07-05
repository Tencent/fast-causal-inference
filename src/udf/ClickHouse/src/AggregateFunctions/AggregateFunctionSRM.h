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
#include <Common/Exception.h>
#include <Common/PODArray_fwd.h>
#include <IO/VarInt.h>
#include <base/types.h>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

template <typename Key>
struct SRMData
{
    SRMData() = default;

    void add(const IColumn ** column, size_t row_num)
    {
        if constexpr (std::is_same_v<Key, String>)
        {
            Key index = column[1]->getDataAt(row_num).toString();
            group2sum[index] += column[0]->getFloat64(row_num);
        }
        else
            group2sum[column[1]->getInt(row_num)] += column[0]->getFloat64(row_num);
        const ColumnArray * column_arr = checkAndGetColumn<ColumnArray>(*(column[2]));
        if (!column_arr)
            throw Exception("Illegal column " + column[2]->getName() + " of argument of aggregate function SRM",
                            ErrorCodes::BAD_ARGUMENTS);

        const ColumnArray::Offsets & offsets = column_arr->getOffsets();
        if (offsets.size() <= row_num)
            throw Exception("Array offsets error", ErrorCodes::BAD_ARGUMENTS);
        size_t start = offsets[row_num - 1];
        if (ratios.empty())
        {
            const auto & data = column_arr->getData();
            for (size_t pos = start; pos < offsets[row_num]; pos++)
                    ratios.push_back(data.getFloat64(pos));
        }
    }

    void merge(const SRMData & source)
    {
        for (const auto & [index, sum] : source.group2sum)
            group2sum[index] += sum;
        if (group2sum.size() > 1000)
            throw Exception("Too many groups", ErrorCodes::BAD_ARGUMENTS);
        if (ratios.empty())
            ratios.insert(ratios.begin(), source.ratios.begin(), source.ratios.end());
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(group2sum.size(), buf);
        for (const auto & [index, sum] : group2sum)
        {
            if constexpr (std::is_same_v<String, Key>)
                writeStringBinary(index, buf);
            else
                writeVarInt(index, buf);
            writeFloatBinary(sum, buf);
        }
        writeVarUInt(ratios.size(), buf);
        for (const auto & ratio : ratios)
            writeFloatBinary(ratio, buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        size_t size = 0;
        readVarUInt(size, buf);
        for (size_t i = 0; i < size; ++i)
        {
            Key index;
            Float64 sum = 0;
            if constexpr (std::is_same_v<String, Key>)
                readStringBinary(index, buf);
            else
                readVarInt(index, buf);
            readFloatBinary(sum, buf);
            group2sum[index] = sum;
        }
        readVarUInt(size, buf);
        ratios.resize(size);
        for (size_t i = 0; i < size; i++)
            readFloatBinary(ratios[i], buf);
    }

    String publish()
    {
        if (group2sum.size() != ratios.size())
            return "the number of group must equal to the number of ratios";
        if (group2sum.empty())
            return "the table is empty";
        PaddedPODArray<Float64> f_obs, f_exp;
        Float64 f_obs_sum = 0;
        for (const auto & [group, ob] : group2sum)
        {
            f_obs.emplace_back(ob);
            f_obs_sum += ob;
        }
        Float64 ratio_sum = std::accumulate(ratios.begin(), ratios.end(), 0.);
        if (fabs(ratio_sum) <= 1e-6)
            return "sum of ratio must not equal to zero!";
        for (auto & ratio : ratios)
        {
            Float64 exp = ratio / ratio_sum * f_obs_sum;
            if (exp <= 1e-6)
                return "f_exp should not contain zeros or negative.";
            f_exp.emplace_back(exp);
        }
        Float64 p_value = 0;
        Float64 chisquare = 0;
        for (size_t i = 0; i < f_obs.size(); i++)
            chisquare += (f_obs[i] - f_exp[i]) * (f_obs[i] - f_exp[i]) / f_exp[i];
        if (chisquare <= 1e-6)
            return "chisquare should not equal to zero!";
        Float64 dof = f_obs.size() - 0 - 1;
        p_value = 1 - boost::math::cdf(boost::math::chi_squared{dof}, chisquare);
        String result;
        result = to_string_with_precision("groupname") + to_string_with_precision("f_obs") + to_string_with_precision("ratio") + to_string_with_precision("chisquare") + to_string_with_precision("p-value") + "\n";
        size_t pos = 0;
        for (const auto & [group, ob] : group2sum)
        {
            result += "  " + to_string_with_precision(group) + to_string_with_precision(ob) + to_string_with_precision(ratios[pos]);
            if (!pos)
                result += to_string_with_precision(chisquare) + to_string_with_precision(p_value);
            result += "\n";
            pos++;
        }
        return result;
    }

    std::map<Key, Float64> group2sum;
    PaddedPODArray<Float64> ratios;
};

template <typename Key>
class AggregateFunctionSRM final:
    public IAggregateFunctionDataHelper<SRMData<Key>, AggregateFunctionSRM<Key>>
{
private:
    using Data = SRMData<Key>;
    size_t arguments_num;

public:
    explicit AggregateFunctionSRM(const DataTypes & arguments, const Array &)
        :IAggregateFunctionDataHelper<SRMData<Key>, AggregateFunctionSRM<Key>> ({arguments}, {}) {}

    String getName() const override
    {
        return "SRM";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeString>();
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
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }
};

}
