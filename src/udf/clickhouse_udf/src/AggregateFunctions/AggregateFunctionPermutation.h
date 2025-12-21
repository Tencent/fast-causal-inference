#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <AggregateFunctions/parseAggregateFunctionParameters.h>
#include <AggregateFunctions/AggregateFunctionLeastSquaresLinearRegression.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <DataTypes/DataTypeArray.h>
#include <DataTypes/DataTypeTuple.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnTuple.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnArray.h>
#include <Common/Exception.h>
#include <Common/PODArray_fwd.h>
#include "Parsers/Lexer.h"
#include <Core/ColumnsWithTypeAndName.h>
#include <DataTypes/IDataType.h>
#include <IO/VarInt.h>
#include <base/types.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <Poco/Net/DNS.h>
#include <mutex>
#include <random>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int ILLEGAL_COLUMN;
}


class Permutation
{
public:
    Permutation() = default;

    Permutation(const size_t & permutation_num, const AggregateFunctionPtr & agg_func_) : 
        agg_func(agg_func_), places(permutation_num)
    {
        for (auto & place : places)
        {
            place = arena->alignedAlloc(agg_func->sizeOfData(), agg_func->alignOfData());
            agg_func->create(place);
        }
    }

    void add(const IColumn ** column, size_t row_num, size_t place_index, Arena *)
    {
        if (place_index >= places.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: place_index overflow");
        agg_func->add(places[place_index], column, row_num, arena.get());
    }

    void merge(const Permutation & rhs, Arena *)
    {
        if (places.size() != rhs.places.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: size of places not equal");
        for (size_t i = 0; i < places.size(); ++i)
            agg_func->merge(places[i], rhs.places[i], arena.get());
    }

    void serialize(WriteBuffer & buf) const
    {
        for (const auto & place : places)
            agg_func->serialize(place, buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        for (auto & place : places)
            agg_func->deserialize(place, buf);
    }

    template <typename T>
    void publish(IColumn & to , Arena *, IColumn & nested_to) // NOLINT
    {
        for (const auto & place : places)
            agg_func->insertResultInto(place, nested_to, arena.get());

        if constexpr (std::is_same_v<T, String>)
        {
            const ColumnString * col_str = checkAndGetColumn<ColumnString>(nested_to);
            if (!col_str) 
                throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cast to ColumnString fail");

            ColumnArray & arr_to = assert_cast<ColumnArray &>(to);
            ColumnString & res_strings = typeid_cast<ColumnString &>(arr_to.getData());
            ColumnArray::Offsets & res_offsets = arr_to.getOffsets();
            ColumnString::Chars & res_strings_chars = res_strings.getChars();
            ColumnString::Offsets & res_strings_offsets = res_strings.getOffsets();

            ColumnString::Offset current_strings_offset = 0;
            for (size_t i = 0; i < nested_to.size(); ++i)
            {
                String res_str = col_str->getDataAt(i).toString();
                res_strings_chars.resize(res_strings_chars.size() + res_str.size() + 1);
                memcpy(&res_strings_chars[current_strings_offset], res_str.c_str(), res_str.size());
                res_strings_chars[current_strings_offset + res_str.size()] = 0;
                current_strings_offset += res_str.size() + 1;
                res_strings_offsets.push_back(current_strings_offset);
            }
            res_offsets.push_back(res_offsets.back() + res_strings_offsets.size());
        }
        else if constexpr (std::is_same_v<T, Tuple>)
        {
            const ColumnTuple * col_tuple = checkAndGetColumn<ColumnTuple>(nested_to);
            if (!col_tuple) 
                throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Cast to ColumnTuple fail");

            ColumnArray & arr_to = assert_cast<ColumnArray &>(to);
            ColumnTuple & res_tuple = typeid_cast<ColumnTuple &>(arr_to.getData());

            for (size_t i = 0; i < nested_to.size(); ++i)
            {
                  Field row_value;
                  nested_to.get(i, row_value);
                  res_tuple.insert(row_value);
            }
            ColumnArray::Offsets & res_offsets = arr_to.getOffsets();
            res_offsets.push_back(res_offsets.back() + nested_to.size());
        }
        else
        {
            ColumnArray & arr_to = assert_cast<ColumnArray &>(to);
            ColumnArray::Offsets & offsets_to = arr_to.getOffsets();
            typename ColumnVector<T>::Container & data_to = assert_cast<ColumnVector<T> &>(arr_to.getData()).getData();
            size_t old_size = data_to.size();
            data_to.resize(old_size + nested_to.size());

            for (size_t i = 0; i < nested_to.size(); ++i)
                data_to[old_size + i] = static_cast<T>(nested_to[i].get<T>());
            offsets_to.push_back(offsets_to.back() + nested_to.size());
        }
    }

    void destroy()
    {
        for (const auto & place : places)
            agg_func->destroy(place);
    }

private:
    const AggregateFunctionPtr agg_func = nullptr;
    PaddedPODArray<AggregateDataPtr> places;
    std::unique_ptr<Arena> arena = std::make_unique<Arena>();

};

class AggregateFunctionPermutation final :
    public IAggregateFunctionDataHelper<Permutation, AggregateFunctionPermutation>
{
private:
    AggregateFunctionPtr aggregate_function;
    size_t permutation_num = 1;
    DataTypes forward_argument_types;
    Float64 mde = 0.0;
    UInt8 mde_type = 1;

public:
    explicit AggregateFunctionPermutation(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<Permutation, AggregateFunctionPermutation> ({arguments}, {params}, getReturnType(arguments, params)) 
    {
        if (params.size() < 2)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} requires at least 2 paramers", getName());

        String aggregate_function_name_with_params;
        String aggregate_function_name;
        aggregate_function_name_with_params = params[0].get<String>();
        for (auto & ch : aggregate_function_name_with_params)
            if (ch == '\"')
                ch = '\'';

        // if not suffix of 'Xexpt_Ttest_2samp', 'Ttest_2samp', 'mannWhitneyUTest', throw exception
        if (!boost::algorithm::contains(aggregate_function_name_with_params, "Xexpt_Ttest_2samp") &&
            !boost::algorithm::contains(aggregate_function_name_with_params, "Ttest_2samp") &&
            !boost::algorithm::contains(aggregate_function_name_with_params, "mannWhitneyUTest"))
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid, only support Ttest_2samp, mannWhitneyUTest, Xexpt_Ttest_2samp");

        boost::algorithm::trim(aggregate_function_name_with_params);
        if (boost::algorithm::contains(aggregate_function_name_with_params, "State"))
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Params of aggregate_name is not support for funcState,\
                please use PermutationState");

        Array params_row;
        getAggregateFunctionNameAndParametersArray(aggregate_function_name_with_params,
                                                   aggregate_function_name, params_row, "function " + getName(), nullptr);
        AggregateFunctionProperties properties;
        forward_argument_types = argument_types;
        forward_argument_types.push_back(std::make_shared<DataTypeUInt8>());

        aggregate_function = AggregateFunctionFactory::instance().get(aggregate_function_name, forward_argument_types,
                                                                      params_row, properties);
        if (!aggregate_function)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid");

        permutation_num = params[1].get<UInt64>();
        if (permutation_num == 0 || permutation_num > 100000)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Permutation num must in [1, 100000]");

        if (params.size() > 2)
        {
            mde = params[2].get<Float64>();
            if (mde < 0.0)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "MDE must be non-negative");
        }

        if (params.size() > 3)
        {
            mde_type = params[3].get<UInt8>();
            if (mde_type != 0 && mde_type != 1)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "MDE type must be 0 or 1");
        }
    }

    void create(AggregateDataPtr __restrict place) const override
    {
        new (place) Data(permutation_num, aggregate_function);
    }

    static MutableColumnPtr getTreatmentColumn(size_t length)
    {
        MutableColumnPtr treatment = DataTypeUInt8().createColumn();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 1);
        for (size_t i = 0; i < length; ++i)
            treatment->insert(dis(gen));
        return treatment;
    }

    void addBatchSinglePlace( /// NOLINT
        size_t row_begin,
        size_t row_end,
        AggregateDataPtr __restrict place,
        const IColumn ** columns,
        Arena * arena,
        ssize_t if_argument_pos = -1) const override
    {
        // clone the columns
        MutableColumns cloned_columns;
        std::vector<const IColumn *> add_columns(forward_argument_types.size(), nullptr);

        for (size_t i = 0; i < argument_types.size(); ++i)
        {
            cloned_columns.push_back(columns[i]->cloneResized(row_end));
            add_columns[i] = cloned_columns.back().get();
        }
        const IColumn * first_column = columns[0];

        for (size_t j = 0; j < permutation_num; j++) 
        {
            // 对于每个 permutation_num, 都要重新生成一个 treatment
            auto treatment = getTreatmentColumn(row_end);
            add_columns.back() = treatment.get();

            MutableColumnPtr mde_column = DataTypeFloat64().createColumn();
            for (size_t i = 0; i < row_end; i++)
            {
                if (mde_type == 0)
                {
                    mde_column->insert(first_column->getFloat64(i) + mde * treatment->getUInt(i));
                }
                else
                {
                    mde_column->insert(first_column->getFloat64(i) * (1 + mde * treatment->getUInt(i)));
                }
            }
            add_columns[0] = mde_column.get();

            if (if_argument_pos >= 0)
            {
                const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    if (flags[i]) 
                        this->data(place).add(add_columns.data(), i, j, arena);
                }
            }
            else
            {
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    this->data(place).add(add_columns.data(), i, j, arena);
                }
            }
        }
    }

    String getName() const override
    {
        return "Permutation";
    }

    bool allocatesMemoryInArena() const override { return false; }

    static DataTypePtr getReturnType(const DataTypes & argument_type, const Array & params)
    {
        if (params.size() < 2)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function requires at least 2 paramers");

        String aggregate_function_name_with_params;
        String aggregate_function_name;
        aggregate_function_name_with_params = params[0].get<String>();
        for (auto & ch : aggregate_function_name_with_params)
            if (ch == '\"')
                ch = '\'';

        // if not suffix of 'Xexpt_Ttest_2samp', 'Ttest_2samp', 'mannWhitneyUTest', throw exception
        if (!boost::algorithm::contains(aggregate_function_name_with_params, "Xexpt_Ttest_2samp") &&
            !boost::algorithm::contains(aggregate_function_name_with_params, "Ttest_2samp") &&
            !boost::algorithm::contains(aggregate_function_name_with_params, "mannWhitneyUTest"))
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid, only support Ttest_2samp, mannWhitneyUTest, Xexpt_Ttest_2samp");

        boost::algorithm::trim(aggregate_function_name_with_params);
        if (boost::algorithm::contains(aggregate_function_name_with_params, "State"))
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Params of aggregate_name is not support for funcState,\
                please use PermutationState");

        Array params_row;
        getAggregateFunctionNameAndParametersArray(aggregate_function_name_with_params,
                                                   aggregate_function_name, params_row, "function ", nullptr);
        AggregateFunctionProperties properties;

        DataTypes forward_argument_type = argument_type;
        forward_argument_type.push_back(std::make_shared<DataTypeUInt8>());

        auto agg_func = AggregateFunctionFactory::instance().get(aggregate_function_name, forward_argument_type,
                                                                      params_row, properties);

        return std::make_shared<DataTypeArray>(agg_func->getResultType());
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena * arena) const override
    {
        MutableColumnPtr nested_col_ptr = aggregate_function->getResultType()->createColumn();
        IColumn & nested_col = *nested_col_ptr;
        auto nested_type = aggregate_function->getResultType();
#define PUBLISH(type) \
        if (WhichDataType(nested_type).is##type()) \
            this->data(place).publish<type>(to, arena, nested_col);
        FOR_BASIC_NUMERIC_TYPES(PUBLISH)
        PUBLISH(String)
        PUBLISH(Tuple)
#undef PUBLISH
    }

    void add(AggregateDataPtr __restrict, const IColumn **, size_t, Arena *) const override 
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: Permutation aggregate function does not support add");
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena * arena) const override
    {
        this->data(place).merge(this->data(rhs), arena);
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> ) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> , Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void destroy(AggregateDataPtr __restrict place) const noexcept override
    {
        this->data(place).destroy();
    }

    bool hasTrivialDestructor() const override
    {
        return true;
    }

};


struct PermutationMultiData
{

    PermutationMultiData() = default;

    PermutationMultiData(const size_t & bs_num_, const std::vector<AggregateFunctionPtr> & agg_funcs_, const std::vector<UInt32> & argument_nums_)
        : agg_funcs(agg_funcs_)
    {
        for (const auto & agg_func : agg_funcs_)
            datas.emplace_back(bs_num_, agg_func);
        argument_positions.push_back(0);
        for (size_t i = 1; i < argument_nums_.size(); ++i)
            argument_positions.push_back(argument_positions.back() + argument_nums_[i - 1]);
    }

    void add(const IColumn ** column, size_t row_num, size_t place_index, Arena *)
    {
        for (size_t i = 0; i < datas.size(); ++i)
            datas[i].add(column + argument_positions[i], row_num, place_index, nullptr);
    }

    void merge(const PermutationMultiData & rhs, Arena *)
    {
        for (size_t i = 0; i < datas.size(); ++i)
            datas[i].merge(rhs.datas[i], nullptr);
    }

    void serialize(WriteBuffer & buf) const
    {
        for (const auto & data : datas)
            data.serialize(buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        for (auto & data : datas)
            data.deserialize(buf);
    }

    void publish(IColumn & to)
    {
        auto & column_tuple = assert_cast<ColumnTuple &>(to);
        for (size_t i = 0; i < datas.size(); i++)
        {
            auto & data = datas[i];
            IColumn & col = column_tuple.getColumn(i);
            //
            MutableColumnPtr nested_col_ptr = agg_funcs[i]->getResultType()->createColumn();
            IColumn & nested_col = *nested_col_ptr;
            auto nested_type = agg_funcs[i]->getResultType();
#define PUBLISH(type) \
        if (WhichDataType(nested_type).is##type()) \
            data.publish<type>(col, nullptr, nested_col);
        FOR_BASIC_NUMERIC_TYPES(PUBLISH)
        PUBLISH(String)
        PUBLISH(Tuple)
#undef PUBLISH
        }
    }

    void destroy()
    {
        for (auto & data : datas)
            data.destroy();
    }

    std::vector<Permutation> datas;
    std::vector<UInt32> argument_positions;
    const std::vector<AggregateFunctionPtr> agg_funcs = {};
};



class AggregateFunctionPermutationMulti final :
    public IAggregateFunctionDataHelper<PermutationMultiData, AggregateFunctionPermutationMulti>
{
private:
    //AggregateFunctionPtr aggregate_function;
    size_t permutation_num = 1;
    //DataTypes forward_argument_types;
    Float64 mde = 0.0;
    UInt8 mde_type = 1;

    std::vector<AggregateFunctionPtr> aggregate_functions;
    std::vector<UInt32> argument_nums;
    UInt32 argument_num_count = 0;

public:
    explicit AggregateFunctionPermutationMulti(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<PermutationMultiData, AggregateFunctionPermutationMulti> ({arguments}, {params}, getReturnType(arguments, params)) 
    {
        if (params.size() < 2)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} requires at least 2 paramers", getName());

        if (params[0].getType() != Field::Types::String)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} first param must be String", getName());

        std::vector<String> aggregate_names;
        boost::split(aggregate_names, params[0].get<String>(), boost::is_any_of(";"));
        for (auto & aggregate_name : aggregate_names) 
        {
            // if not suffix of 'Xexpt_Ttest_2samp', 'Ttest_2samp', 'mannWhitneyUTest', throw exception
            if (!boost::algorithm::contains(aggregate_name, "Xexpt_Ttest_2samp") &&
                !boost::algorithm::contains(aggregate_name, "Ttest_2samp") &&
                !boost::algorithm::contains(aggregate_name, "mannWhitneyUTest"))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid, only support Ttest_2samp, mannWhitneyUTest, Xexpt_Ttest_2samp");

            boost::algorithm::trim(aggregate_name);
            if (boost::algorithm::contains(aggregate_name, "State"))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Params of aggregate_name is not support for funcState,\
                    please use PermutationState");

            boost::algorithm::trim(aggregate_name);
            std::replace(aggregate_name.begin(), aggregate_name.end(), '\"', '\'');


            std::vector<String> aggregate_name_and_num;
            boost::split(aggregate_name_and_num, aggregate_name, boost::is_any_of(":"));
            if (aggregate_name_and_num.size() != 2)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} first param error, need pair of aggregate function name and number of argument", getName());
            boost::algorithm::trim(aggregate_name_and_num[0]);
            boost::algorithm::trim(aggregate_name_and_num[1]);
            String aggregate_function_name_with_params = aggregate_name_and_num[0];
            UInt32 argument_num = std::stoi(aggregate_name_and_num[1]);

            Array params_row;
            String aggregate_function_name;
            getAggregateFunctionNameAndParametersArray(aggregate_function_name_with_params,
                                                      aggregate_function_name, params_row, "function " + getName(), nullptr);
            AggregateFunctionProperties properties;

            DataTypes each_argument_types;

            for (size_t i = 0; i < argument_num; ++i)
            {
                if (argument_num_count + i >= arguments.size())
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "The number of arguments in the aggregate function is invalid");
                each_argument_types.push_back(arguments[argument_num_count + i]);
            }
            argument_num_count += argument_num;
            argument_nums.push_back(argument_num);
            //each_argument_types.push_back(std::make_shared<DataTypeUInt8>());

            AggregateFunctionPtr aggregate_function = AggregateFunctionFactory::instance().get(aggregate_function_name, each_argument_types,
                                                                          params_row, properties);
            if (!aggregate_function)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function {} is not found", aggregate_function_name);
            aggregate_functions.push_back(aggregate_function);
            if (!aggregate_function)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid");
        }


        permutation_num = params[1].get<UInt64>();
        if (permutation_num == 0 || permutation_num > 100000)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Permutation num must in [1, 100000]");

        if (params.size() > 2)
        {
            mde = params[2].get<Float64>();
            if (mde < 0.0)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "MDE must be non-negative");
        }

        if (params.size() > 3)
        {
            mde_type = params[3].get<UInt8>();
            if (mde_type != 0 && mde_type != 1)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "MDE type must be 0 or 1");
        }

        if (argument_num_count != arguments.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The number of arguments in the aggregate function is invalid");
    }

    void create(AggregateDataPtr __restrict place) const override
    {
        new (place) Data(permutation_num, aggregate_functions, argument_nums);
    }

    static MutableColumnPtr getTreatmentColumn(size_t length)
    {
        MutableColumnPtr treatment = DataTypeUInt8().createColumn();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 1);
        for (size_t i = 0; i < length; ++i)
            treatment->insert(dis(gen));
        return treatment;
    }

    void addBatchSinglePlace( /// NOLINT
        size_t row_begin,
        size_t row_end,
        AggregateDataPtr __restrict place,
        const IColumn ** columns,
        Arena * arena,
        ssize_t if_argument_pos = -1) const override
    {
        // clone the columns
        MutableColumns cloned_columns;
        std::vector<const IColumn *> add_columns(argument_num_count, nullptr);

        for (size_t i = 0; i < argument_num_count; ++i)
        {
            cloned_columns.push_back(columns[i]->cloneResized(row_end));
            add_columns[i] = cloned_columns.back().get();
        }


        for (size_t j = 0; j < permutation_num; j++) 
        {
            // 对于每个 permutation_num, 都要重新生成一个 treatment
            UInt64 argument_count = 0;

            std::vector<MutableColumnPtr> treatment_columns;
            for (const auto & argument_num : argument_nums)
            {
                treatment_columns.emplace_back(getTreatmentColumn(row_end));
                argument_count += argument_num;
                if (argument_count > add_columns.size())
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: argument_count overflow");
                add_columns[argument_count - 1] = treatment_columns.back().get();
            }

            if (if_argument_pos >= 0)
            {
                const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    if (flags[i]) 
                        this->data(place).add(add_columns.data(), i, j, arena);
                }
            }
            else
            {
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    this->data(place).add(add_columns.data(), i, j, arena);
                }
            }
        }
    }

    String getName() const override
    {
        return "PermutationMulti";
    }

    bool allocatesMemoryInArena() const override { return false; }

    static DataTypePtr getReturnType(const DataTypes & arguments, const Array & params)
    {
        if (params.size() < 2)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function requires at least 2 paramers");

        if (params[0].getType() != Field::Types::String)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} first param must be String");

        std::vector<String> aggregate_names;
        boost::split(aggregate_names, params[0].get<String>(), boost::is_any_of(";"));
        std::vector<AggregateFunctionPtr> agg_funcs;

        for (auto & aggregate_name : aggregate_names) 
        {
            // if not suffix of 'Xexpt_Ttest_2samp', 'Ttest_2samp', 'mannWhitneyUTest', throw exception
            if (!boost::algorithm::contains(aggregate_name, "Xexpt_Ttest_2samp") &&
                !boost::algorithm::contains(aggregate_name, "Ttest_2samp") &&
                !boost::algorithm::contains(aggregate_name, "mannWhitneyUTest"))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid, only support Ttest_2samp, mannWhitneyUTest, Xexpt_Ttest_2samp");

            boost::algorithm::trim(aggregate_name);
            if (boost::algorithm::contains(aggregate_name, "State"))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Params of aggregate_name is not support for funcState,\
                    please use PermutationState");

            boost::algorithm::trim(aggregate_name);
            std::replace(aggregate_name.begin(), aggregate_name.end(), '\"', '\'');


            std::vector<String> aggregate_name_and_num;
            boost::split(aggregate_name_and_num, aggregate_name, boost::is_any_of(":"));
            if (aggregate_name_and_num.size() != 2)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} first param error, need pair of aggregate function name and number of argument");
            boost::algorithm::trim(aggregate_name_and_num[0]);
            boost::algorithm::trim(aggregate_name_and_num[1]);
            String aggregate_function_name_with_params = aggregate_name_and_num[0];
            UInt32 argument_num = std::stoi(aggregate_name_and_num[1]);

            Array params_row;
            String aggregate_function_name;
            getAggregateFunctionNameAndParametersArray(aggregate_function_name_with_params,
                                                      aggregate_function_name, params_row, "function ", nullptr);
            AggregateFunctionProperties properties;

            DataTypes each_argument_types;

            UInt32 argument_cnt = 0;
            std::vector<UInt32> arg_nums;
            for (size_t i = 0; i < argument_num; ++i)
            {
                if (argument_cnt + i >= arguments.size())
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "The number of arguments in the aggregate function is invalid");
                each_argument_types.push_back(arguments[argument_cnt + i]);
            }
            argument_cnt += argument_num;
            arg_nums.push_back(argument_num);
            //each_argument_types.push_back(std::make_shared<DataTypeUInt8>());

            AggregateFunctionPtr aggregate_function = AggregateFunctionFactory::instance().get(aggregate_function_name, each_argument_types,
                                                                          params_row, properties);
            if (!aggregate_function)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function {} is not found", aggregate_function_name);
            agg_funcs.push_back(aggregate_function);
            if (!aggregate_function)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid");
        }

        DataTypes types;
        for (const auto & aggregate_function : agg_funcs)
            types.push_back(std::make_shared<DataTypeArray>(aggregate_function->getResultType()));
        return std::make_shared<DataTypeTuple>(types);
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        this->data(place).publish(to);
    }

    void add(AggregateDataPtr __restrict, const IColumn **, size_t, Arena *) const override 
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: PermutationMulti aggregate function does not support add");
    }

    void merge(AggregateDataPtr __restrict place, ConstAggregateDataPtr rhs, Arena * arena) const override
    {
        this->data(place).merge(this->data(rhs), arena);
    }

    void serialize(ConstAggregateDataPtr __restrict place, WriteBuffer & buf, std::optional<size_t> ) const override
    {
        this->data(place).serialize(buf);
    }

    void deserialize(AggregateDataPtr __restrict place, ReadBuffer & buf, std::optional<size_t> , Arena *) const override
    {
        this->data(place).deserialize(buf);
    }

    void destroy(AggregateDataPtr __restrict place) const noexcept override
    {
        this->data(place).destroy();
    }

    bool hasTrivialDestructor() const override
    {
        return true;
    }

};

}
