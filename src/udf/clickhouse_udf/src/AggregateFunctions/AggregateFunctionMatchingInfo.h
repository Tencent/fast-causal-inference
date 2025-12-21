#pragma once
#include <AggregateFunctions/IAggregateFunction.h>
#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionCausalInference.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <Columns/IColumn.h>
#include <Columns/ColumnVector.h>
#include <Columns/ColumnString.h>
#include <Common/Exception.h>
#include <Common/PODArray_fwd.h>
#include <Core/ColumnsWithTypeAndName.h>
#include <DataTypes/IDataType.h>
#include <IO/VarInt.h>
#include <base/types.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <random>
#include <unordered_map>
#include <Poco/JSON/Parser.h>
#include <Poco/JSON/Object.h>
#include <Poco/JSON/Array.h>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int ILLEGAL_COLUMN;
}

struct CaliperMatchingInfo
{
    CaliperMatchingInfo() : node_key(getNodeKey()), argument_num(0) {}

    explicit CaliperMatchingInfo(const size_t argument_num_) : node_key(getNodeKey()), argument_num(argument_num_) {}

    void add(const IColumn ** column, size_t row_num)
    {
        UInt64 treatment = column[0]->getUInt(row_num);
        Float64 distance = column[1]->getFloat64(row_num);
        Float64 step = column[2]->getFloat64(row_num);
        if (treatment != 0 && treatment != 1)
            throw Exception(ErrorCodes::ILLEGAL_COLUMN, "treatment must be 0 or 1");
        Int64 score = static_cast<Int64>(std::floor(distance / step));
        UInt64 group_hash = 0;
        for (size_t i = 3; i < argument_num; ++i)
            group_hash ^= std::hash<String>()(column[i]->getDataAt(row_num).toString());
        if (treatment == 0)
            matching_info[node_key][score][group_hash].first += 1;
        else
            matching_info[node_key][score][group_hash].second += 1;
        if (matching_info[node_key].size() > 100000)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Too many score, more than 100000");
    }

    void merge(const CaliperMatchingInfo & source)
    {
        for (const auto & [key , score_data] : source.matching_info)
            for (const auto & [score, node_data] : score_data)
                for (const auto & [group_hash, cnt] : node_data)
                {
                    matching_info[key][score][group_hash].first += cnt.first;
                    matching_info[key][score][group_hash].second += cnt.second;
                }
        for (const auto & [key, score_data] : matching_info)
            if (score_data.size() > 100000)
                throw Exception(ErrorCodes::BAD_ARGUMENTS, "Too many score, more than 100000");
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(matching_info.size(), buf);
        for (const auto & [key, score_data] : matching_info)
        {
            writeStringBinary(key, buf);
            writeVarUInt(score_data.size(), buf);
            for (const auto & [score, node_data] : score_data)
            {
                writeVarUInt(score, buf);
                writeVarUInt(node_data.size(), buf);
                for (const auto & [group_hash, cnt] : node_data)
                {
                    writeVarUInt(group_hash, buf);
                    writeVarUInt(cnt.first, buf);
                    writeVarUInt(cnt.second, buf);
                }
            }
        }
    }

    void deserialize(ReadBuffer & buf)
    {
        size_t size = 0;
        readVarUInt(size, buf);
        for (size_t i = 0; i < size; ++i)
        {
            String key;
            readStringBinary(key, buf);
            size_t score_data_size = 0;
            readVarUInt(score_data_size, buf);
            for (size_t j = 0; j < score_data_size; ++j)
            {
                Int64 score = 0;
                readVarUInt(score, buf);
                size_t node_data_size = 0;
                readVarUInt(node_data_size, buf);
                for (size_t k = 0; k < node_data_size; ++k)
                {
                    UInt64 group_hash = 0;
                    readVarUInt(group_hash, buf);
                    UInt64 treatment_0_cnt = 0;
                    readVarUInt(treatment_0_cnt, buf);
                    UInt64 treatment_1_cnt = 0;
                    readVarUInt(treatment_1_cnt, buf);
                    matching_info[key][score][group_hash].first += treatment_0_cnt;
                    matching_info[key][score][group_hash].second += treatment_1_cnt;
                }
            }
        }
    }

    String publish(size_t k)
    {
        std::unordered_map<Int64, std::unordered_map<UInt64, std::array<UInt64, 4>>> matching_info_total;
        for (const auto & [key, score_data] : matching_info)
        {
            for (const auto & [score, node_data] : score_data)
            {
                for (const auto & [group_hash, cnt] : node_data)
                {
                    const auto & [treatment_0_cnt, treatment_1_cnt] = cnt;
                    matching_info_total[score][group_hash][0] += treatment_0_cnt;
                    matching_info_total[score][group_hash][1] += treatment_1_cnt;
                }
            }
        }
        UInt64 total_index = 1;
        for (auto & [score, node_data] : matching_info_total)
            for (auto & [group_hash, cnt] : node_data)
            {
                const UInt64 & treatment_cnt = std::min(cnt[0], cnt[1]);
                // treatment = 0 is left treatment
                cnt[0] = treatment_cnt;
                cnt[1] = std::min(k * treatment_cnt, cnt[1]);

                cnt[2] = total_index;
                total_index += treatment_cnt;
                cnt[3] = total_index;
            }

        // transform matching_info to json by poco
        Poco::JSON::Object json_obj;
        for (const auto & [key, score_data] : matching_info)
        {
            Poco::JSON::Object matching_info_obj;
            for (const auto & [score, node_data] : score_data)
            {
                Poco::JSON::Object score_obj;
                for (const auto & [group_hash, cnt] : node_data)
                {
                    Poco::JSON::Object group_obj;
                    // [start_index, end_index)
                    const auto start_index = matching_info_total[score][group_hash][2]; 
                    const auto end_index = matching_info_total[score][group_hash][3];
                    const auto len = end_index - start_index;
                    const auto [treatment_0_cnt, treatment_1_cnt] = cnt; // 机器上实际的数量

                    const auto rest_0_cnt = matching_info_total[score][group_hash][0]; // 剩余需要匹配的数量
                    const auto count_0 = std::min(treatment_0_cnt, rest_0_cnt);
                    const auto start_0 = start_index + rest_0_cnt - count_0;
                    group_obj.set("start_0", start_0);
                    group_obj.set("count_0", count_0);
                    matching_info_total[score][group_hash][0] -= count_0;

                    const auto rest_1_cnt = matching_info_total[score][group_hash][1];
                    const auto count_1 = std::min(treatment_1_cnt, rest_1_cnt);
                    auto start_1 = start_index + rest_1_cnt - count_1;

                    if (start_1 >= end_index && len > 0)
                        start_1 = (start_1 - start_index) % len + start_index;
                        
                    group_obj.set("start_1", start_1);
                    group_obj.set("count_1", count_1);
                    matching_info_total[score][group_hash][1] -= count_1;

                    group_obj.set("origin_start_index", start_index);
                    group_obj.set("origin_end_index", end_index);
                    score_obj.set(std::to_string(group_hash), group_obj);
                }
                matching_info_obj.set(std::to_string(score), score_obj);
            }
            json_obj.set(key, matching_info_obj);
        }

        std::ostringstream json_stream;
        json_obj.stringify(json_stream);
        return json_stream.str();
    }

    static String getName() { return "CaliperMatchingInfo"; }

private:

    const String node_key;
    const size_t argument_num;
    std::unordered_map<String, std::unordered_map<Int64, std::unordered_map<UInt64, std::pair<UInt64, UInt64>>>> matching_info;
};


class AggregateFunctionMatchingInfo final:
    public IAggregateFunctionDataHelper<CaliperMatchingInfo, AggregateFunctionMatchingInfo>
{
private:
    using Data = CaliperMatchingInfo;
    const size_t argument_num = 0;
    size_t k = 1;

public:
    explicit AggregateFunctionMatchingInfo(const DataTypes & arguments, const Array & params)
        :IAggregateFunctionDataHelper<CaliperMatchingInfo, AggregateFunctionMatchingInfo> 
        ({arguments}, {params}, getReturnType()), argument_num(arguments.size())
    {
        if (!params.empty())
            k = params[0].get<size_t>();
    }

    String getName() const override
    {
        return Data::getName();
    }

    bool allocatesMemoryInArena() const override { return false; }

    static DataTypePtr getReturnType()
    {
        return std::make_shared<DataTypeString>();
    }

    void create(AggregateDataPtr __restrict place) const override
    {
        new (place) Data(argument_num);
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
        auto result = this->data(place).publish(k);
        assert_cast<ColumnString &>(to).insertData(result.data(), result.size());
    }
};

}
