#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <AggregateFunctions/StatCommon.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <DataTypes/DataTypeArray.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnArray.h>
#include <Core/Field.h>
#include <base/types.h>
#include <Common/PODArray_fwd.h>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

namespace ublas = boost::numeric::ublas;
using Matrix = ublas::matrix<Float64>;

template <typename T = Float64>
struct AggregateFunctionMatrixMulData
{
public:
    using DotProductMatrix = ColMatrix<DotProductData<T>, false, false>;

    AggregateFunctionMatrixMulData() = default;

    explicit AggregateFunctionMatrixMulData(size_t arg_num_, bool use_weights_) :
        arg_num(arg_num_ - static_cast<size_t>(use_weights_)), use_weights(use_weights_), data(arg_num) {}

    void add(const IColumn ** column, size_t row_num)
    {
        PaddedPODArray<Float64> arguments;
        for (size_t i = 0; i < arg_num; ++i)
            arguments.push_back(use_weights ? column[i]->getFloat64(row_num) * column[arg_num]->getFloat64(row_num) :
                                                                               column[i]->getFloat64(row_num));
        data.add(arguments);
    }

    void merge(const AggregateFunctionMatrixMulData & source)
    {
        data.merge(source.data);
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(arg_num, buf);
        writeVarUInt(use_weights, buf);
        data.serialize(buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        readVarUInt(arg_num, buf);
        readVarUInt(use_weights, buf);
        data.deserialize(buf);
    }

    void publish(bool to_inverse, IColumn & to) 
    {
        Matrix data_matrix = data.getMatrix();
        if (to_inverse && !invertMatrix(data_matrix, data_matrix)) 
          throw Exception("InvertMatrix failed. some variables in the table are perfectly collinear.", ErrorCodes::BAD_ARGUMENTS);

        auto & data_to = assert_cast<ColumnFloat64 &>(assert_cast<ColumnArray &>(
              assert_cast<ColumnArray &>(to).getData()).getData()
              ).getData();
        auto & root_offsets_to = assert_cast<ColumnArray &>(to).getOffsets();
        auto & nested_offsets_to = assert_cast<ColumnArray &>(assert_cast<ColumnArray &>(to).getData()).getOffsets();
        for (size_t i = 0; i < data_matrix.size1(); ++i)
        {
            for (size_t j = 0; j < data_matrix.size2(); ++j)
                data_to.push_back(data_matrix(i, j));
            nested_offsets_to.push_back(nested_offsets_to.back() + data_matrix.size2());
        }
        root_offsets_to.push_back(root_offsets_to.back() + data_matrix.size1());
    }

private:
    size_t arg_num;
    size_t use_weights;
    DotProductMatrix data;
};

template <typename AggregateFunctionMatrixOperatorData>
class AggregateFunctionMatrixOperator final:
    public IAggregateFunctionDataHelper<AggregateFunctionMatrixOperatorData, 
        AggregateFunctionMatrixOperator<AggregateFunctionMatrixOperatorData>>
{
private:
    using Data = AggregateFunctionMatrixOperatorData;
    size_t arguments_num;
    bool to_inverse = false;
    bool use_weights = false;
public:
    explicit AggregateFunctionMatrixOperator(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<AggregateFunctionMatrixOperatorData, 
        AggregateFunctionMatrixOperator<AggregateFunctionMatrixOperatorData>> ({arguments}, {})
    {
        arguments_num = arguments.size();
        if (!params.empty())
            to_inverse = params[0].get<bool>();
        if (params.size() > 1)
            use_weights = params[1].get<bool>();
    }

    String getName() const override
    {
        return "MatrixMultiplication";
    }

    bool allocatesMemoryInArena() const override { return false; } 

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeArray>(std::make_shared<DataTypeArray>(std::make_shared<DataTypeFloat64>()));
    }

    void create(AggregateDataPtr __restrict place) const override 
    {
        new (place) Data(arguments_num, use_weights);
    }

    void add(AggregateDataPtr __restrict place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        this->data(place).add(columns, row_num);
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t>) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t>, Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        this->data(place).publish(to_inverse, to);
    }
};

}
