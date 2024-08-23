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
#include <Core/ColumnsWithTypeAndName.h>
#include <DataTypes/IDataType.h>
#include <IO/VarInt.h>
#include <base/types.h>
#include <Poco/JSON/Parser.h>
#include <Poco/JSON/Object.h>
#include <Poco/JSON/Array.h>
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

struct DistributedNodeRowNumberData
{
    DistributedNodeRowNumberData() : node_key(getNodeKey()) 
    {
        if (!key2cnt.contains(node_key))
            key2cnt[node_key] = 0;
    }

    void add(const IColumn **, size_t)
    {
        key2cnt[node_key]++;
    }

    void merge(const DistributedNodeRowNumberData & source)
    {
        for (const auto & [key, cnt] : source.key2cnt)
            key2cnt[key] += cnt;
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(key2cnt.size(), buf);
        for (const auto & [key, cnt] : key2cnt)
        {
            writeStringBinary(key, buf);
            writeVarUInt(cnt, buf);
        }
    }

    void deserialize(ReadBuffer & buf)
    {
        size_t size = 0;
        readVarUInt(size, buf);
        for (size_t i = 0; i < size; ++i)
        {
            String key;
            UInt64 cnt = 0;
            readStringBinary(key, buf);
            readVarUInt(cnt, buf);
            key2cnt[key] = cnt;
        }
    }

    String publish(const size_t & seed)
    {
        Poco::JSON::Object json_obj;
        for (auto & [key, cnt] : key2cnt)
        {
            json_obj.set(key, cnt);
        }
        json_obj.set("random_seed", seed);
        std::ostringstream json_stream;
        json_obj.stringify(json_stream);
        return json_stream.str();
    }

    static void parse(const String & json, std::unordered_map<String, UInt64> & key2cnt, size_t & seed)
    {
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);
        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        seed = object->get("random_seed");
        for (const auto & item : *object)
        {
            String key = item.first;
            String value = item.second.convert<std::string>();
            if (key != "random_seed")
                key2cnt[key] = std::stoull(value);
        }
    }

    const String node_key;
    std::unordered_map<String, UInt64> key2cnt;
};


class AggregateFunctionDistributedNodeRowNumber final:
    public IAggregateFunctionDataHelper<DistributedNodeRowNumberData, AggregateFunctionDistributedNodeRowNumber>
{
private:
    using Data = DistributedNodeRowNumberData;
    size_t seed;


public:
    explicit AggregateFunctionDistributedNodeRowNumber(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<DistributedNodeRowNumberData, AggregateFunctionDistributedNodeRowNumber> 
        ({arguments}, {params}) 
    {
        if (!params.empty())
            seed = params[0].get<UInt64>();
        else
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        if (params.size() > 1)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "This aggregate function neeed only one parameters [seed]");
    }

    String getName() const override
    {
        return "DistributedNodeRowNumber";
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
        auto result = this->data(place).publish(seed);
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }
};

class BootStrapBinomialDistribution
{
public:
    BootStrapBinomialDistribution() = default;

    BootStrapBinomialDistribution(size_t total_row_num_, size_t rest_row_num_, size_t seed_) 
        : random_get(seed_), total_row_num(total_row_num_), sample_num(rest_row_num_) {}

    size_t operator()(const size_t & row_num)
    {
        if (row_num > total_row_num)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "row number error");
        size_t res = std::binomial_distribution<size_t>(sample_num, 1. * row_num / total_row_num)(random_get);
        sample_num -= res;
        total_row_num -= row_num;
        return res;
    }

protected:
    std::default_random_engine random_get;
    size_t total_row_num;
    size_t sample_num;
};

class BootStrapBinomialDistributionWithLock : public BootStrapBinomialDistribution
{
public:
    BootStrapBinomialDistributionWithLock() = default;

    BootStrapBinomialDistributionWithLock(size_t total_row_num_, size_t rest_row_num_, size_t seed_) 
        : BootStrapBinomialDistribution(total_row_num_, rest_row_num_, seed_) {} 

    size_t operator()(const size_t & row_num)
    {
        std::unique_lock<std::mutex> lock(mtx);
        return BootStrapBinomialDistribution::operator()(row_num);
    }

    BootStrapBinomialDistributionWithLock & operator=(BootStrapBinomialDistributionWithLock && other) noexcept
    {
        if (this != &other) 
        {
            total_row_num = std::move(other.total_row_num);
            sample_num = std::move(other.sample_num);
        }
        return *this;
    }
  
private:
    std::mutex mtx;
};

class BootStrapData
{
public:
    BootStrapData() = default;

    BootStrapData(const size_t & bs_num_, const AggregateFunctionPtr & agg_func_) : 
        agg_func(agg_func_), places(bs_num_)
    {
        for (auto & place : places)
        {
            place = arena->alignedAlloc(agg_func->sizeOfData(), agg_func->alignOfData());
            agg_func->create(place);
        }
    }

    void add(const IColumn ** column, size_t row_num, size_t place_index, size_t row_sample_num, Arena *)
    {
        if (place_index >= places.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Logical error: place_index overflow");
        for (size_t i = 0; i < row_sample_num; ++i)
            agg_func->add(places[place_index], column, row_num, arena.get());
    }

    void merge(const BootStrapData & rhs, Arena *)
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
    void publish(IColumn & to , Arena *, IColumn & nested_to)
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
                data_to[old_size + i] = nested_to[i].get<T>();
            offsets_to.push_back(offsets_to.back() + nested_to.size());
        }
    }

    void predict(
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit,
        ContextPtr context) const
    {
        if (arguments.empty())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The arguments of predict should not be empty");
        const auto & bs_column = arguments.back().column;
        UInt64 bs_index = bs_column->getUInt(offset);
        for (size_t row_num = 1; row_num < limit; ++row_num)
            if (bs_index != bs_column->getUInt(row_num + offset))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The predict index is not same");
        if (bs_index >= places.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Predict index overflow");
        ColumnsWithTypeAndName new_arguments;
        for (size_t i = 0; i < arguments.size() - 1; i++)
            new_arguments.push_back(arguments[i]);
        agg_func->predictValues(places[bs_index], to, new_arguments, offset, limit, context);
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

class AggregateFunctionBootStrap final :
    public IAggregateFunctionDataHelper<BootStrapData, AggregateFunctionBootStrap>
{
private:
    AggregateFunctionPtr aggregate_function;
    size_t seed = 0;
    size_t bs_num = 1;
    size_t sample_num = 0;
    size_t total_row_number_this_node = 0;
    mutable PaddedPODArray<BootStrapBinomialDistributionWithLock> batch_schedulers;

public:
    explicit AggregateFunctionBootStrap(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<BootStrapData, AggregateFunctionBootStrap> ({arguments}, {params}) 
    {
        if (params.size() != 4)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function " + getName() + " requires 4 paramers");
        String aggregate_function_name_with_params;
        String aggregate_function_name;
        aggregate_function_name_with_params = params[0].get<String>();
        for (auto & ch : aggregate_function_name_with_params)
            if (ch == '\"')
                ch = '\'';

        boost::algorithm::trim(aggregate_function_name_with_params);
        if (boost::algorithm::contains(aggregate_function_name_with_params, "State"))
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Params of aggregate_name is not support for funcState,\
                please use BootStrapState");

        Array params_row;
        getAggregateFunctionNameAndParametersArray(aggregate_function_name_with_params,
                                                   aggregate_function_name, params_row, "function " + getName(), nullptr);
        AggregateFunctionProperties properties;
        aggregate_function = AggregateFunctionFactory::instance().get(aggregate_function_name, argument_types,
                                                                      params_row, properties);
        if (!aggregate_function)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid");

        bs_num = params[2].get<UInt64>();
        if (bs_num == 0 || bs_num > 10000)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "BootStrap num must in [1, 10000]");

        String json;
        json = params[3].get<String>();
        std::unordered_map<String, UInt64> key2cnt;
        DistributedNodeRowNumberData::parse(json, key2cnt, seed);

        String node_key = getNodeKey();

        PaddedPODArray<std::pair<String, UInt64>> row_numbers;
        UInt64 total_row_number = 0;

        for (const auto & [key, value] : key2cnt)
        {
            if (key != "random_seed")
            {
                row_numbers.emplace_back(std::make_pair(key, value));
                total_row_number += value;
            }
        }

        if (params.size() > 1)
        {
            if (params[1].getType() == Field::Types::Which::Float64)
            {
                auto ratio = params[1].get<Float64>();
                if (ratio < 0. || ratio > 1.)
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "The ratio must in [0, 1]");
                sample_num = ratio * total_row_number;
            }
            else
                sample_num = params[1].get<UInt64>();
        }

        if (sample_num > 10000000000)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "sample_num must less than 10000000000");

        std::sort(row_numbers.begin(), row_numbers.end());
        BootStrapBinomialDistribution bs_distribu(total_row_number, sample_num, seed);
        bool have_node = false;
        for (const auto & [k, v] : row_numbers)
        {
            auto sample_num_each_node = bs_distribu(v);
            if (k == node_key)
            {
                have_node = true;
                sample_num = sample_num_each_node;
                total_row_number_this_node = v;
            }
        }
        if (!have_node)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "This node key is not input");

        for (size_t i = 0; i < bs_num; ++i)
            batch_schedulers.emplace_back(total_row_number_this_node, sample_num, seed + i + 1);

    }

    void create(AggregateDataPtr __restrict place) const override
    {
        new (place) Data(bs_num, aggregate_function);
    }

    void addBatchSinglePlace( /// NOLINT
        size_t row_begin,
        size_t row_end,
        AggregateDataPtr __restrict place,
        const IColumn ** columns,
        Arena * arena,
        ssize_t if_argument_pos = -1) const override
    {
        size_t batch_size = row_end - row_begin;
        for (size_t j = 0; j < bs_num; j++)
        {
            auto batch_sample_num = batch_schedulers[j](batch_size);

            BootStrapBinomialDistribution row_scheduler(batch_size, batch_sample_num, seed + j + 1);
            PaddedPODArray<UInt32> uniform_count;
            bool use_uniform = false;
            if (batch_sample_num <= 100000)
            {
                use_uniform = true;
                std::default_random_engine generator(seed + j + 1);
                std::uniform_int_distribution<UInt32> distribution(0, batch_size - 1);
                uniform_count.resize_fill(batch_size, 0);
                for (size_t i = 0; i < batch_sample_num; ++i)
                    uniform_count[distribution(generator)]++;
            }

            if (if_argument_pos >= 0)
            {
                const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    if (flags[i])
                    {
                        UInt64 row_sample_num;
                        if (use_uniform)
                            row_sample_num = uniform_count[i - row_begin];
                        else
                            row_sample_num = row_scheduler(1);
                        this->data(place).add(columns, i, j, row_sample_num, arena);
                    }
                }
            }
            else
            {
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    UInt64 row_sample_num;
                    if (use_uniform)
                        row_sample_num = uniform_count[i - row_begin];
                    else
                        row_sample_num = row_scheduler(1);
                    this->data(place).add(columns, i, j, row_sample_num, arena);
                }
            }
        }
    }

    String getName() const override
    {
        return "BootStrap";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeArray>(aggregate_function->getReturnType());
    }

    DataTypePtr getReturnTypeToPredict() const override
    {
        return aggregate_function->getReturnTypeToPredict();
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena * arena) const override
    {
        MutableColumnPtr nested_col_ptr = aggregate_function->getReturnType()->createColumn();
        IColumn & nested_col = *nested_col_ptr;
        auto nested_type = aggregate_function->getReturnType();
#define PUBLISH(type) \
        if (WhichDataType(nested_type).is##type()) \
            this->data(place).publish<type>(to, arena, nested_col);
        FOR_BASIC_NUMERIC_TYPES(PUBLISH)
        PUBLISH(String)
        PUBLISH(Tuple)
#undef PUBLISH
    }

    void predictValues(
        ConstAggregateDataPtr __restrict place,
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit,
        ContextPtr context) const override
    {
        this->data(place).predict(to, arguments, offset, limit, context);
    }

    void add(AggregateDataPtr __restrict, const IColumn **, size_t, Arena *) const override {}

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

};

template <typename T, typename Op>
class AggregateFunctionBootStrapOls :
    public IAggregateFunctionDataHelper<LeastSquaresLinearRegressionData<T, Op>, AggregateFunctionBootStrapOls<T, Op>>
{
private:
    using Data = LeastSquaresLinearRegressionData<T, Op>;
    size_t seed = 0;
    size_t sample_num = 0;
    size_t total_row_number_this_node = 0;
    const size_t bs_num = 1;
    mutable PaddedPODArray<BootStrapBinomialDistributionWithLock> batch_schedulers;

    size_t arguments_num;
    bool add_constant = true;

public:
    explicit AggregateFunctionBootStrapOls(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<LeastSquaresLinearRegressionData<T, Op>, AggregateFunctionBootStrapOls<T, Op>> ({arguments}, {params}) 
    {
        if (params.size() != 4)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} requires 4 paramers", getName());
        arguments_num = arguments.size();

        String json;
        json = params[3].get<String>();
        std::unordered_map<String, UInt64> key2cnt;
        DistributedNodeRowNumberData::parse(json, key2cnt, seed);

        String node_key = getNodeKey();

        PaddedPODArray<std::pair<String, UInt64>> row_numbers;
        UInt64 total_row_number = 0;

        for (const auto & [key, value] : key2cnt)
        {
            if (key != "random_seed")
            {
                row_numbers.emplace_back(std::make_pair(key, value));
                total_row_number += value;
            }
        }

        if (params.size() > 1)
        {
            if (params[1].getType() == Field::Types::Which::Float64)
            {
                auto ratio = params[1].get<Float64>();
                if (ratio < 0. || ratio > 1.)
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "The ratio must in [0, 1]");
                sample_num = static_cast<UInt64>(ratio * total_row_number);
            }
            else
                sample_num = params[1].get<UInt64>();
        }
        if (sample_num > 10000000000)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "sample_num must less than 10000000000");

        std::sort(row_numbers.begin(), row_numbers.end());
        BootStrapBinomialDistribution bs_distribu(total_row_number, sample_num, seed);
        bool have_node = false;
        for (const auto & [k, v] : row_numbers)
        {
            auto sample_num_each_node = bs_distribu(v);
            if (k == node_key)
            {
                have_node = true;
                sample_num = sample_num_each_node;
                total_row_number_this_node = v;
            }
        }
        if (!have_node)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "This node key is not input");

        for (size_t i = 0; i < bs_num; ++i)
            batch_schedulers.emplace_back(total_row_number_this_node, sample_num, seed + i + 1);

    }

    void create(AggregateDataPtr __restrict place) const override // NOLINT
    {
        new (place) Data(arguments_num, add_constant, Matrix(), Matrix());
    }

    void addBatchSinglePlace( /// NOLINT
        size_t row_begin,
        size_t row_end,
        AggregateDataPtr __restrict place,
        const IColumn ** columns,
        Arena * arena,
        ssize_t if_argument_pos = -1) const override
    {
        UInt32 batch_size = static_cast<UInt32>(row_end - row_begin);
        for (size_t j = 0; j < bs_num; j++)
        {
            auto batch_sample_num = batch_schedulers[j](batch_size);
            BootStrapBinomialDistribution row_scheduler(batch_size, batch_sample_num, static_cast<UInt32>(seed + j + 1));
            PaddedPODArray<UInt32> uniform_count;
            bool use_uniform = false;
            if (batch_sample_num <= 100000)
            {
                use_uniform = true;
                std::default_random_engine generator(static_cast<UInt32>(seed + j + 1));
                std::uniform_int_distribution<UInt32> distribution(0, batch_size - 1);
                uniform_count.resize_fill(batch_size, 0);
                for (size_t i = 0; i < batch_sample_num; ++i)
                    uniform_count[distribution(generator)]++;
            }

            if (if_argument_pos >= 0)
            {
                const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    if (flags[i])
                    {
                        UInt64 row_sample_num;
                        if (use_uniform)
                            row_sample_num = uniform_count[i - row_begin];
                        else
                            row_sample_num = row_scheduler(1);
                        for (size_t repeat = 0; repeat < row_sample_num; ++repeat)
                            add(place, columns, i, arena);
                    }
                }
            }
            else
            {
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    UInt64 row_sample_num;
                    if (use_uniform)
                        row_sample_num = uniform_count[i - row_begin];
                    else
                        row_sample_num = row_scheduler(1);
                    for (size_t repeat = 0; repeat < row_sample_num; ++repeat)
                        add(place, columns, i, arena);
                }
            }
        }
    }

    String getName() const override
    {
        return "BootStrapOls";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeString>();
    }

    DataTypePtr getReturnTypeToPredict() const override
    {
        return Op::getReturnTypeToPredict();
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        auto result = this->data(place).publish();
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }

    void predictValues(
        ConstAggregateDataPtr __restrict place,
        IColumn & to,
        const ColumnsWithTypeAndName & arguments,
        size_t offset,
        size_t limit,
        ContextPtr) const override
    {
        if (arguments.size() < arguments_num)
            throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH,
                  "Predict got incorrect number of arguments. Got: {}. Required: {}",
                  arguments.size(), arguments_num);
        this->data(place).predict(to, arguments, offset, limit);
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
};


struct BootStrapMultiData
{
    BootStrapMultiData() = default;

    BootStrapMultiData(const size_t & bs_num_, const std::vector<AggregateFunctionPtr> & agg_funcs_, const std::vector<UInt32> & argument_nums_)
        : agg_funcs(agg_funcs_)
    {
        for (const auto & agg_func : agg_funcs_)
            datas.emplace_back(bs_num_, agg_func);
        argument_positions.push_back(0);
        for (size_t i = 1; i < argument_nums_.size(); ++i)
            argument_positions.push_back(argument_positions.back() + argument_nums_[i - 1]);
    }

    void add(const IColumn ** column, size_t row_num, size_t place_index, size_t row_sample_num, Arena *)
    {
        for (size_t i = 0; i < datas.size(); ++i)
            datas[i].add(column + argument_positions[i], row_num, place_index, row_sample_num, nullptr);
    }

    void merge(const BootStrapMultiData & rhs, Arena *)
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
            MutableColumnPtr nested_col_ptr = agg_funcs[i]->getReturnType()->createColumn();
            IColumn & nested_col = *nested_col_ptr;
            auto nested_type = agg_funcs[i]->getReturnType();
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
       
    std::vector<BootStrapData> datas;
    std::vector<UInt32> argument_positions;
    const std::vector<AggregateFunctionPtr> agg_funcs = {};
};

class AggregateFunctionBootStrapMulti final :
    public IAggregateFunctionDataHelper<BootStrapMultiData, AggregateFunctionBootStrapMulti>
{
private:
    std::vector<AggregateFunctionPtr> aggregate_functions;
    std::vector<UInt32> argument_nums;
    size_t seed = 0;
    size_t bs_num = 1;
    size_t sample_num = 0;
    size_t total_row_number_this_node = 0;
    mutable PaddedPODArray<BootStrapBinomialDistributionWithLock> batch_schedulers;

public:
    explicit AggregateFunctionBootStrapMulti(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<BootStrapMultiData, AggregateFunctionBootStrapMulti> ({arguments}, {params}) 
    {
        if (params.size() != 4)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} requires 4 paramers", getName());

        if (params[0].getType() != Field::Types::String)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} first param must be String", getName());

        std::vector<String> aggregate_names;
        boost::split(aggregate_names, params[0].get<String>(), boost::is_any_of(";"));


        UInt32 argument_num_count = 0;
        for (auto & aggregate_name : aggregate_names)
        {
            boost::algorithm::trim(aggregate_name);
            std::replace(aggregate_name.begin(), aggregate_name.end(), '\"', '\'');
            if (boost::algorithm::contains(aggregate_name, "State"))
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Params of aggregate_name is not support for funcState,\
                    please use BootStrapState");
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

            AggregateFunctionPtr aggregate_function = AggregateFunctionFactory::instance().get(aggregate_function_name, each_argument_types,
                                                                          params_row, properties);
            if (!aggregate_function)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function {} is not found", aggregate_function_name);
            aggregate_functions.push_back(aggregate_function);
            if (!aggregate_function)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "The aggregate function is invalid");
        }
        if (argument_num_count != arguments.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The number of arguments in the aggregate function is invalid");

        bs_num = params[2].get<UInt64>();
        if (bs_num == 0 || bs_num > 10000)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "BootStrap num must in [1, 10000]");

        String json;
        json = params[3].get<String>();
        std::unordered_map<String, UInt64> key2cnt;
        DistributedNodeRowNumberData::parse(json, key2cnt, seed);

        String node_key = getNodeKey();

        PaddedPODArray<std::pair<String, UInt64>> row_numbers;
        UInt64 total_row_number = 0;

        for (const auto & [key, value] : key2cnt)
        {
            if (key != "random_seed")
            {
                row_numbers.emplace_back(std::make_pair(key, value));
                total_row_number += value;
            }
        }


        if (params.size() > 1)
        {
            if (params[1].getType() == Field::Types::Which::Float64)
            {
                auto ratio = params[1].get<Float64>();
                if (ratio < 0. || ratio > 1.)
                    throw Exception(ErrorCodes::BAD_ARGUMENTS, "The ratio must in [0, 1]");
                sample_num = ratio * total_row_number;
            }
            else
                sample_num = params[1].get<UInt64>();
        }
        if (sample_num > 10000000000)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "sample_num must less than 10000000000");

        std::sort(row_numbers.begin(), row_numbers.end());
        BootStrapBinomialDistribution bs_distribu(total_row_number, sample_num, seed);
        bool have_node = false;
        for (const auto & [k, v] : row_numbers)
        {
            auto sample_num_each_node = bs_distribu(v);
            if (k == node_key)
            {
                have_node = true;
                sample_num = sample_num_each_node;
                total_row_number_this_node = v;
            }
        }
        if (!have_node)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "This node key is not input");

        for (size_t i = 0; i < bs_num; ++i)
            batch_schedulers.emplace_back(total_row_number_this_node, sample_num, seed + i + 1);

    }

    void create(AggregateDataPtr __restrict place) const override
    {
        new (place) Data(bs_num, aggregate_functions, argument_nums);
    }

    void addBatchSinglePlace( /// NOLINT
        size_t row_begin,
        size_t row_end,
        AggregateDataPtr __restrict place,
        const IColumn ** columns,
        Arena * arena,
        ssize_t if_argument_pos = -1) const override
    {
        size_t batch_size = row_end - row_begin;
        for (size_t j = 0; j < bs_num; j++)
        {
            auto batch_sample_num = batch_schedulers[j](batch_size);
            BootStrapBinomialDistribution row_scheduler(batch_size, batch_sample_num, seed + j + 1);
            PaddedPODArray<UInt32> uniform_count;
            bool use_uniform = false;
            if (batch_sample_num <= 100000)
            {
                use_uniform = true;
                std::default_random_engine generator(seed + j + 1);
                std::uniform_int_distribution<UInt32> distribution(0, batch_size - 1);
                uniform_count.resize_fill(batch_size, 0);
                for (size_t i = 0; i < batch_sample_num; ++i)
                    uniform_count[distribution(generator)]++;
            }

            if (if_argument_pos >= 0)
            {
                const auto & flags = assert_cast<const ColumnUInt8 &>(*columns[if_argument_pos]).getData();
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    if (flags[i])
                    {
                        UInt64 row_sample_num;
                        if (use_uniform)
                            row_sample_num = uniform_count[i - row_begin];
                        else
                            row_sample_num = row_scheduler(1);
                        this->data(place).add(columns, i, j, row_sample_num, arena);
                    }
                }
            }
            else
            {
                for (size_t i = row_begin; i < row_end; ++i)
                {
                    UInt64 row_sample_num;
                    if (use_uniform)
                        row_sample_num = uniform_count[i - row_begin];
                    else
                        row_sample_num = row_scheduler(1);
                    this->data(place).add(columns, i, j, row_sample_num, arena);
                }
            }
        }
    }

    String getName() const override
    {
        return "BootStrapMulti";
    }

    bool allocatesMemoryInArena() const override { return false; }

    DataTypePtr getReturnType() const override
    {
        DataTypes types;
        for (const auto & aggregate_function : aggregate_functions)
            types.push_back(std::make_shared<DataTypeArray>(aggregate_function->getReturnType()));
        return std::make_shared<DataTypeTuple>(types);
    }

    DataTypePtr getReturnTypeToPredict() const override
    {
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "BootStrapMulti does not support predict");
    }

    void insertResultInto(AggregateDataPtr __restrict place, IColumn & to, Arena *) const override
    {
        this->data(place).publish(to);
    }

    void add(AggregateDataPtr __restrict, const IColumn **, size_t, Arena *) const override {}

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

};


}
