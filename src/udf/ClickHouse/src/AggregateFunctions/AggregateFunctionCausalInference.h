#pragma  once
#include <Common/assert_cast.h>
#include <base/types.h>
#include <DataTypes/DataTypesDecimal.h>
#include <DataTypes/DataTypeNullable.h>
#include <IO/ReadHelpers.h>
#include <IO/WriteHelpers.h>
#include <DataTypes/DataTypeString.h>
#include <boost/math/tools/polynomial.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/asio.hpp>
#include <Common/PODArray.h>
#include <random>
#include <functional>

namespace DB
{

struct Settings;
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int NOT_IMPLEMENTED;
}

template <int N = 12, typename T>
std::string to_string_with_precision(const T& value) {
    std::ostringstream ss;
    if (std::is_floating_point<T>::value) {
        ss << std::fixed << std::setprecision(6) << value;
        if (ss.str().size() > N - 1) {
            ss.str("");
            ss << std::scientific << std::setprecision(3) << value;
        }
    } else
        ss << value;
    std::string str = ss.str();
    str.resize(N, ' ');
    str.back() = ' ';
    return str;
}

namespace ublas = boost::numeric::ublas;
using Matrix = ublas::matrix<Float64>;

// LU decomposition to invert a matrix
template<class T>
bool invertMatrix (const ublas::matrix<T>& input, ublas::matrix<T>& inverse) 
{
    try 
    {
        using pmatrix = ublas::permutation_matrix<std::size_t>;
        ublas::matrix<T> l_ma(input);
        pmatrix pm(l_ma.size1());
        auto res = lu_factorize(l_ma,pm);
        if(res != 0) return false;
        inverse.assign(ublas::identity_matrix<T>(l_ma.size1()));
        lu_substitute(l_ma, pm, inverse);
        return true;
    } 
    catch (...) 
    {
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "InvertMatrix failed. some variables in the table are perfectly collinear.");
    }
}

template <class T>
Matrix tMatrix(const ublas::matrix<T>& m)
{
    Matrix result(m.size2(), m.size1());
    for (size_t i = 0; i < m.size1(); ++i)
        for (size_t j = 0; j < m.size2(); ++j)
            result(j, i) = m(i, j);
    return result;
}

template <typename T>
class DotProductData
{
public:
    void add(const IColumn & column_left, const IColumn & column_right, size_t row_num)
    {
        T valx = column_left.getFloat64(row_num);
        T valy = column_right.getFloat64(row_num);
        sumxy += valx * valy;
        count++;
    }

    void add(const T& valx, const IColumn & column_right, size_t row_num)
    {
        T valy = column_right.getFloat64(row_num);
        sumxy += valx * valy;
        count++;
    }

    void add(const T& valx, const T& valy)
    {
        sumxy += valx * valy;
        count++;
    }

    void merge(const DotProductData & source)
    {
        sumxy += source.sumxy;
        count += source.count;
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(count, buf);
        writeBinary(sumxy, buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        readVarUInt(count, buf);
        readBinary(sumxy, buf);
    }

    T getResult() const 
    { 
        return sumxy; 
    }

    UInt64 getCount() const 
    {
        return count; 
    }

private:
    T sumxy = 0;
    UInt64 count = 0;
};

template <typename T>
class CovarianceSimpleData 
{
public:
    void add(const IColumn & column_left, const IColumn & column_right, size_t row_num)
    {
        T valx = column_left.getFloat64(row_num);
        T valy = column_right.getFloat64(row_num);
        sumx += valx;
        sumy += valy;
        sumxy += valx * valy;
        count++;
    }

    void add(const T& valx, const T& valy)
    {
        sumx += valx;
        sumy += valy;
        sumxy += valx * valy;
        count++;
    }

    void merge(const CovarianceSimpleData & source)
    {
        sumx += source.sumx;
        sumy += source.sumy;
        sumxy += source.sumxy;
        count += source.count;
    }

    void serialize(WriteBuffer & buf) const
    {
        writeVarUInt(count, buf);
        writeBinary(sumx, buf);
        writeBinary(sumy, buf);
        writeBinary(sumxy, buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        readVarUInt(count, buf);
        readBinary(sumx, buf);
        readBinary(sumy, buf);
        readBinary(sumxy, buf);
    }

    Float64 publish() const
    {
        if (count < 2)
            return std::numeric_limits<Float64>::infinity();
        else 
        {
          Float64 meanx = sumx / count;
          Float64 meany = sumy / count;
          Float64 cov = (sumxy - sumx * meany - sumy * meanx + meanx * meany * count) / (count - 1);
          return cov;
        }
    }

    Float64 getResult() const
    {
        if (count < 2)
            return std::numeric_limits<Float64>::infinity();
        else 
        {
          Float64 meanx = sumx / count;
          Float64 meany = sumy / count;
          Float64 cov = (sumxy - sumx * meany - sumy * meanx + meanx * meany * count) / (count - 1);
          return cov;
        }
    }

    Float64 getYMean() const
    {
        return sumy / count;
    }

    Float64 getXMean() const
    {
        return sumx / count;
    }

    UInt64 getCount() const
    {
        return count;
    }

    void setCount(const UInt64 & count_)
    {
        count = count_;
    }

private:
    T sumxy = 0;
    T sumx = 0;
    T sumy = 0;
    UInt64 count = 0;
};

template <typename Op, bool use_bias, bool use_weights = false>
class ColMatrix 
{
    using Data = std::vector<std::vector<Op>>;
public:
    ColMatrix() = default;

    explicit ColMatrix(size_t col_size1_, size_t col_size2_) : col_size1(col_size1_), col_size2(col_size2_)
    {
        if constexpr (use_weights)
        {
            col_size1 -= 1;
            col_size2 -= 1;
        }
        data.resize(col_size1 + use_bias);
        for (auto& row : data)
            row.resize(col_size2 + use_bias);
    }

    explicit ColMatrix(size_t col_size1_) 
        : ColMatrix(col_size1_, col_size1_)
    {
        single_col = true;
    }
    
    void add(const IColumn ** columns_left, const IColumn ** columns_right, size_t row_num)
    { 
        for (size_t i = 0; i < col_size1; ++i)
        {
            for (size_t j = single_col ? i : 0; j < col_size2; ++j)
            {
                if constexpr (use_weights)
                {
                    auto weight_left = columns_left[col_size1]->getFloat64(row_num);
                    auto weight_right = columns_right[col_size2]->getFloat64(row_num);
                    data[i][j].add(columns_left[i]->getFloat64(row_num)*sqrt(weight_left),
                                      columns_right[j]->getFloat64(row_num)*sqrt(weight_right));
                    if (weight_left < 0 || weight_right < 0)
                        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Weights must be non-negative");
                }
                else 
                    data[i][j].add(columns_left[i]->getFloat64(row_num), columns_right[j]->getFloat64(row_num));
            }
        }
        if (likely(use_bias))
        {
            for (size_t j = 0; j < col_size2; ++j)
            {
                if constexpr (use_weights)
                    data[col_size1][j].add(sqrt(columns_left[col_size1]->getFloat64(row_num)),
                        columns_right[j]->getFloat64(row_num)*sqrt(columns_right[col_size2]->getFloat64(row_num)));
                else 
                    data[col_size1][j].add(1, columns_right[j]->getFloat64(row_num));
            }
            for (size_t i = 0; i < col_size1; ++i)
            {
                if constexpr (use_weights)
                    data[i][col_size2].add(columns_left[i]->getFloat64(row_num)*sqrt(columns_left[col_size1]->getFloat64(row_num)),
                        sqrt(columns_right[col_size2]->getFloat64(row_num)));
                else 
                    data[i][col_size2].add(columns_left[i]->getFloat64(row_num), 1);
            }
            if constexpr (use_weights)
                data[col_size1][col_size2].add(sqrt(columns_left[col_size1]->getFloat64(row_num)),
                    sqrt(columns_right[col_size2]->getFloat64(row_num)));
            else 
                data[col_size1][col_size2].add(1, 1);
        }
    }

    void add(const PaddedPODArray<Float64>& arguments)
    {
        if (col_size1 > arguments.size() || col_size2 > arguments.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Number of arguments overflow");
        for (size_t i = 0; i < col_size1; ++i)
            for (size_t j = single_col ? i : 0; j < col_size2; ++j)
                data[i][j].add(arguments[i], arguments[j]);
    }

    void add(const IColumn ** columns, size_t row_num)
    {
        add(columns, columns, row_num);
    }

    void merge(const ColMatrix & source)
    {
        if (data.size() != source.data.size() || (data.size() && data[0].size() != source.data[0].size()))
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Cannot merge matrices with different sizes");

        for (size_t i = 0; i < data.size(); ++i)
            for (size_t j = single_col ? i : 0; j < data[0].size(); ++j)
                data[i][j].merge(source.data[i][j]);
    }

    void serialize(WriteBuffer & buf) const
    {
        for (auto & row : data)
            for (auto & col : row)
                col.serialize(buf);
    }

    void deserialize(ReadBuffer & buf)
    {
        for (auto & row : data)
            for (auto & col : row)
                col.deserialize(buf);
    }

    Matrix getMatrix() const
    {
        if (data.size() == 0)
            return Matrix(0, 0);
        Matrix matrix(data.size(), data[0].size());

        for (size_t i = 0; i < data.size(); ++i)
            for (size_t j = 0; j < data[0].size(); ++j)
                matrix(i, j) = data[i][j].getResult();

        if (single_col)
        {
            for (size_t i = 0; i < data.size(); ++i)
                for (size_t j = i + 1; j < data[0].size(); ++j)
                    matrix(j, i) = matrix(i, j);
        }
        return matrix;
    }

    Matrix getSubMatrix(std::vector<size_t> & index) const 
    {
        Matrix matrix(index.size(), index.size());
        if (*std::max_element(index.begin(), index.end()) >= data.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Index is out of bounds");

        for (size_t i = 0; i < index.size(); ++i)
            for (size_t j = 0; j < index.size(); ++j)
                matrix(i, j) = data[index[i]][index[j]].getResult();

        if (single_col)
        {
            for (size_t i = 0; i < index.size(); ++i)
                for (size_t j = i + 1; j < index.size(); ++j)
                    matrix(j, i) = matrix(i, j);
        }
        return matrix;
    }

    std::vector<Float64> getMeans() const
    {
        if (data.size() == 0) return {};
        std::vector<Float64> means(data[0].size());
        for (size_t j = 0; j < data[0].size(); ++j)
            means[j] = data[0][j].getYMean();
        return means;
    }

    std::vector<Float64> getSubMeans(std::vector<size_t> & index) const
    {
        std::vector<Float64> means(index.size());
        if (*std::max_element(index.begin(), index.end()) >= data.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Index is out of bounds");
        for (size_t j = 0; j < index.size(); ++j)
            means[j] = data[0][index[j]].getYMean();
        return means;
    }

    UInt64 getCount() const 
    {
        if (data.empty())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Data is empty when get count");
        return data[0][0].getCount();
    }

    void setCount(const UInt64 & count)
    {
        for (auto & col1 : data)
            for(auto & col2 : col1)
                col2.setCount(count);
    }

    bool isEmpty() const
    {
        return data.size() == 0;
    }

private:
    Data data;
    size_t col_size1, col_size2;
    bool single_col = false;
};

class HashBase : private boost::noncopyable
{
public:
    explicit HashBase(const UInt64 & seed_ = 0) : seed(seed_) {}

    virtual ~HashBase() = default;

    virtual UInt32 operator()(const Int32 &) const
    {
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Hash function for 32bit not implemented.");
    }

    virtual UInt32 operator()(const Int64 &) const
    {
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Hash function for 64bit not implemented.");
    }

private:
    [[maybe_unused]] const uint64_t seed = 0;
};

template <typename T>
class MurmurHash3;

template <>
class MurmurHash3<Int32> : public HashBase
{
public:
    using hash_type = Int32;
    explicit MurmurHash3(const UInt32 & seed_ = 0) : seed(seed_) {}

    // use murmurhash x86 32bit
    std::make_unsigned_t<hash_type> operator()(const hash_type & key) const override
    {
        size_t len = sizeof(key);
        const uint8_t *data = reinterpret_cast<const uint8_t *>(&key);
        const int nblocks = sizeof(key) / 4;
        int i;
        uint32_t h1 = seed;
        uint32_t c1 = 0xcc9e2d51;
        uint32_t c2 = 0x1b873593;
        const uint32_t *blocks = reinterpret_cast<const uint32_t *>(data + nblocks * 4);
        for (i = -nblocks; i; i++) 
        {
            uint32_t k1 = blocks[i];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            h1 = rotl32(h1, 13);
            h1 = h1 * 5 + 0xe6546b64;
        }
        const uint8_t *tail = static_cast<const uint8_t *>(data + nblocks * 4);
        uint32_t k1 = 0;
        switch (len & 3) 
        {
            case 3:
              k1 ^= tail[2] << 16;
              [[fallthrough]];
            case 2: 
              k1 ^= tail[1] << 8;
              [[fallthrough]];
            case 1:
              k1 ^= tail[0];
              k1 *= c1;
              k1 = rotl32(k1, 15);
              k1 *= c2;
              h1 ^= k1;
        }
        h1 ^= len;
        h1 ^= h1 >> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >> 16;
        return h1;
    }

private:
    static uint32_t rotl32(uint32_t x, int8_t r) { return (x << r) | (x >> (32 - r)); }

    uint32_t seed;
};

}
