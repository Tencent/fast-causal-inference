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
#include <cmath>
#include <Common/Exception.h>
#include <Common/RadixSort.h>
#include <Core/AccurateComparison.h>
#include <IO/WriteBuffer.h>
#include <IO/ReadBuffer.h>
#include <IO/VarInt.h>
#include <Poco/Net/DNS.h>

namespace DB
{

namespace ublas = boost::numeric::ublas;
using Matrix = ublas::matrix<Float64>;

struct Settings;
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int NOT_IMPLEMENTED;
}

template <bool use_host_name = true>
String getNodeKey()
{
    try
    {
        if constexpr (use_host_name)
        {
            static String hostname = Poco::Net::DNS::hostName();
            std::replace(hostname.begin(), hostname.end(), '.', '_');
            return hostname;
        }
        else
        {
            static String hostname = std::to_string(time(nullptr));
            std::replace(hostname.begin(), hostname.end(), '.', '_');
            return hostname;
        }
    } 
    catch (...)
    {
        return "unknown";
    }
}

template <bool Scientific = true, bool remove_fractional_zero = false, typename T>
String to_string_with_precision(const T& value, UInt32 len = 12, UInt32 precision = 6)
{
    std::ostringstream ss;
    if constexpr (std::is_floating_point<T>::value)
    {
        if (precision > 0)
            ss << std::fixed << std::setprecision(precision);
        ss << value;
        String temp = ss.str();
        if (Scientific && ss.str().size() > len)
        {
            ss.str("");
            ss << std::scientific << std::setprecision(4) << value;
        } 
        else if (remove_fractional_zero)
        {
            while (temp.size() > 1 && temp.back() == '0')
                temp.pop_back();
            if (temp.back() == '.')
                temp.pop_back();
            ss.str("");
            ss << temp;
        }
    } 
    else
        ss << value;
    std::string str = ss.str();
    if (str.size() < len)
        str.resize(len, ' ');
    if (!str.empty() && str.back() != ' ')
        str.push_back(' ');
    return str;
}

// Function to calculate dot product

template<typename T>
double dotProduct(const std::vector<T> & lhs, const std::vector<T> & rhs) {
    T sum = 0;
    for (size_t i = 0; i < lhs.size(); i++)
        sum += lhs[i] * rhs[i];
    return sum;
}

// Function to subtract vectors
template<typename T>
std::vector<double> subtract(const std::vector<T> & lhs, const std::vector<T> & rhs) {
  std::vector<T> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); i++)
        result[i] = lhs[i] - rhs[i];
    return result;
}

// Function to multiply vector by scalar
template<typename T>
std::vector<double> multiply(const std::vector<T> & lhs, T rhs) {
  std::vector<T> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); i++)
        result[i] = lhs[i] * rhs;
    return result;
}

// Function to calculate norm of vec vector
template<typename T>
double norm(const std::vector<T> & vec) {
    return sqrt(dotProduct(vec, vec));
}

// Function to normalize vec vector
template<typename T>
std::vector<double> normalize(const std::vector<T> & vec) {
    return multiply(vec, 1.0 / norm(vec));
}

// Function to perform QR decomposition
template<typename T>
std::vector<size_t> qrDecomposition(ublas::matrix<T> ma) {
    std::vector<std::vector<T>> q, r, a;
    a.resize(ma.size1(), std::vector<T>(ma.size2()));
    for (size_t i = 0; i < ma.size1(); i++)
        for (size_t j = 0; j < ma.size2(); j++)
            a[i][j] = ma(i, j);

    q = a;
    r = std::vector<std::vector<T>>(a.size(), std::vector<T>(a.size(), 0));

    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < i; j++) {
            r[j][i] = dotProduct(q[j], a[i]);
            q[i] = subtract(q[i], multiply(q[j], r[j][i]));
        }
        r[i][i] = norm(q[i]);
        q[i] = normalize(q[i]);
    }

    std::vector<size_t> nan_index;
    for (size_t i = 0; i < r.size(); i++)
      if (fabs(r[i][i]) < 1e-6)
        nan_index.push_back(i);
    reverse(nan_index.begin(), nan_index.end());
    return nan_index;
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

// LU decomposition to invert a matrix
template<class T>
bool invertMatrix(const ublas::matrix<T>& input, ublas::matrix<T>& inverse, std::vector<size_t> & nan_index) 
{
    nan_index.clear();
    std::vector<size_t> index;
    index.resize(input.size1());
    std::iota(index.begin(), index.end(), 0);
    using pmatrix = ublas::permutation_matrix<std::size_t>;

    Matrix m = input;
    while (m.size1() > 0) {
      ublas::matrix<T> l_ma(m);
      pmatrix pm(l_ma.size1());
      auto res = lu_factorize(l_ma,pm);
      if (!res) break;

      auto collinear_indexs = qrDecomposition(m);
      if (collinear_indexs.empty()) break;
      for (const auto & collinear_index : collinear_indexs) {
        nan_index.push_back(index[collinear_index]);
        index.erase(index.begin() + collinear_index);
      }

      ublas::matrix<T> new_input(index.size(), index.size());
      for (size_t i = 0; i < index.size(); i++)
        for (size_t j = 0; j < index.size(); j++)
          new_input(i, j) = input(index[i], index[j]);
      m = new_input;
    }

    try 
    {
        ublas::matrix<T> l_ma(m);
        pmatrix pm(l_ma.size1());
        auto res = lu_factorize(l_ma,pm);
        if(res != 0) return false;
        inverse = Matrix(l_ma.size1(), l_ma.size2());
        inverse.assign(ublas::identity_matrix<T>(l_ma.size1()));
        lu_substitute(l_ma, pm, inverse);
        Matrix inverse_new(input.size1(), input.size2(), std::numeric_limits<T>::quiet_NaN());
        if (input.size1() != nan_index.size() + inverse.size1()) 
          throw Exception("InvertMatrix failed, size error", ErrorCodes::BAD_ARGUMENTS);

        size_t num_i = 0;
        for (size_t i = 0; i < input.size1(); i++) {
          if (std::find(nan_index.begin(), nan_index.end(), i) != nan_index.end()) continue;
          size_t num_j = 0;
          for (size_t j = 0; j < input.size2(); j++) {
            if (std::find(nan_index.begin(), nan_index.end(), j) != nan_index.end()) continue;
            inverse_new(i, j) = inverse(num_i, num_j);
            num_j++;
          }
          num_i++;
        }
        inverse = inverse_new;
        
        return true;
    } 
    catch (...) 
    {
        throw Exception("InvertMatrix failed. some variables in the table are perfectly collinear.", ErrorCodes::BAD_ARGUMENTS);
    }
}

template<class T>
bool invertMatrix(const ublas::matrix<T>& input, ublas::matrix<T>& inverse) {
  std::vector<size_t> nan_index;
  return invertMatrix(input, inverse, nan_index);
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
                        throw Exception("Weights must be non-negative", ErrorCodes::BAD_ARGUMENTS);
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
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Cannot merge matrices with different sizes {} {}", data.size(), source.data.size());

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
            throw Exception("Index is out of bounds", ErrorCodes::BAD_ARGUMENTS);

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
            throw Exception("Index is out of bounds", ErrorCodes::BAD_ARGUMENTS);
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
        throw Exception("Hash function for 32bit not implemented.", ErrorCodes::NOT_IMPLEMENTED);
    }

    virtual UInt32 operator()(const Int64 &) const
    {
        throw Exception("Hash function for 64bit not implemented.", ErrorCodes::NOT_IMPLEMENTED);
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
        const int nblocks = len / 4;
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



namespace ErrorCodes
{
    extern const int CANNOT_PARSE_INPUT_ASSERTION_FAILED;
    extern const int DECIMAL_OVERFLOW;
    extern const int TOO_LARGE_ARRAY_SIZE;
}
template <typename T>
class CausalQuantileTDigest
{
    using Value = Float32;
    using Count = Float32;
    using BetterFloat = Float64; // For intermediate results and sum(Count). Must have better precision, than Count

    /** The centroid stores the weight of points around their mean value
      */
    struct Centroid
    {
        Value mean;
        Count count;

        Centroid() = default;

        explicit Centroid(Value mean_, Count count_)
            : mean(mean_)
            , count(count_)
        {}

        bool operator<(const Centroid & other) const
        {
            return mean < other.mean;
        }
    };


    /** :param epsilon: value \delta from the article - error in the range
      *                    quantile 0.5 (default is 0.01, i.e. 1%)
      *                    if you change epsilon, you must also change max_centroids
      * :param max_centroids: depends on epsilon, the better accuracy, the more centroids you need
      *                       to describe data with this accuracy. Read article before changing.
      * :param max_unmerged: when accumulating count of new points beyond this
      *                      value centroid compression is triggered
      *                      (default is 2048, the higher the value - the
      *                      more memory is required, but amortization of execution time increases)
      *                      Change freely anytime.
      */
    struct Params
    {
        Value epsilon = static_cast<Value>(0.01);
        size_t max_centroids = 2048;
        size_t max_unmerged = 2048;
    };

    /** max_centroids_deserialize should be >= all max_centroids ever used in production.
     *  This is security parameter, preventing allocation of too much centroids in deserialize, so can be relatively large.
     */
    static constexpr size_t max_centroids_deserialize = 65536;

    Params params{};

    static constexpr size_t bytes_in_arena = 128 - sizeof(PODArray<Centroid>) - sizeof(BetterFloat) - sizeof(size_t); // If alignment is imperfect, sizeof(TDigest) will be more than naively expected
    using Centroids = std::vector<Centroid>;

    Centroids centroids;
    Float64 count = 0;
    size_t unmerged = 0;

    /** Linear interpolation at the point x on the line (x1, y1)..(x2, y2)
      */
    static Value interpolate(Value x, Value x1, Value y1, Value x2, Value y2)
    {
        /// Symmetric interpolation for better results with infinities.
        double k = (x - x1) / (x2 - x1);
        return static_cast<Float32>((1 - k) * y1 + k * y2);
    }

    struct RadixSortTraits
    {
        using Element = Centroid;
        using Result = Element;
        using Key = Value;
        using CountType = UInt32;
        using KeyBits = UInt32;

        static constexpr size_t PART_SIZE_BITS = 8;

        using Transform = RadixSortFloatTransform<KeyBits>;
        using Allocator = RadixSortAllocator;

        /// The function to get the key from an array element.
        static Key & extractKey(Element & elem) { return elem.mean; }
        static Result & extractResult(Element & elem) { return elem; }
    };

    /** Adds a centroid `c` to the digest
     * centroid must be valid, validity is checked in add(), deserialize() and is maintained by compress()
      */
    void addCentroid(const Centroid & c)
    {
        centroids.push_back(c);
        count += c.count;
        ++unmerged;
        if (unmerged > params.max_unmerged)
            compress();
    }

    inline bool canBeMerged(const BetterFloat & l_mean, const Value & r_mean)
    {
        return l_mean == r_mean || (!std::isinf(l_mean) && !std::isinf(r_mean));
    }

    void compressBrute()
    {
        if (centroids.size() <= params.max_centroids)
            return;
        const size_t batch_size = (centroids.size() + params.max_centroids - 1) / params.max_centroids; // at least 2

        auto l = centroids.begin();
        auto r = std::next(l);
        BetterFloat sum = 0;
        BetterFloat l_mean = static_cast<Float32>(l->mean); // We have high-precision temporaries for numeric stability
        BetterFloat l_count = static_cast<Float32>(l->count);
        size_t batch_pos = 0;

        for (; r != centroids.end(); ++r)
        {
            if (batch_pos < batch_size - 1)
            {
                /// The left column "eats" the right. Middle of the batch
                l_count += r->count;
                if (r->mean != l_mean) /// Handling infinities of the same sign well.
                {
                    l_mean += r->count * (r->mean - l_mean) / l_count; // Symmetric algo (M1*C1 + M2*C2)/(C1+C2) is numerically better, but slower
                }
                l->mean = static_cast<Float32>(l_mean);
                l->count = static_cast<Float32>(l_count);
                batch_pos += 1;
            }
            else
            {
                // End of the batch, start the next one
                if (!std::isnan(l->mean)) /// Skip writing batch result if we compressed something to nan.
                {
                    sum += l->count; // Not l_count, otherwise actual sum of elements will be different
                    ++l;
                }

                /// We skip all the values "eaten" earlier.
                *l = *r;
                l_mean = l->mean;
                l_count = l->count;
                batch_pos = 0;
            }
        }

        if (!std::isnan(l->mean))
        {
            count = sum + l_count; // Update count, it might be different due to += inaccuracy
            centroids.resize(l - centroids.begin() + 1);
        }
        else /// Skip writing last batch if (super unlikely) it's nan.
        {
            count = sum;
            centroids.resize(l - centroids.begin());
        }
        // Here centroids.size() <= params.max_centroids
    }

public:
    CausalQuantileTDigest() = default;

    CausalQuantileTDigest(size_t max_centroids_, size_t max_unmerged_, Value epsilon_)
    {
        params.max_centroids = max_centroids_;
        params.max_unmerged = max_unmerged_;
        params.epsilon = epsilon_;
    }

    /** Performs compression of accumulated centroids
      * When merging, the invariant is retained to the maximum size of each
      * centroid that does not exceed `4 q (1 - q) \ delta N`.
      */
    void compress()
    {
        if (unmerged > 0 || centroids.size() > params.max_centroids)
        {
            // unmerged > 0 implies centroids.size() > 0, hence *l is valid below
            RadixSort<RadixSortTraits>::executeLSD(centroids.data(), centroids.size());

            /// A pair of consecutive bars of the histogram.
            auto l = centroids.begin();
            auto r = std::next(l);

            const BetterFloat count_epsilon_4 = count * params.epsilon * 4; // Compiler is unable to do this optimization
            BetterFloat sum = 0;
            BetterFloat l_mean = l->mean; // We have high-precision temporaries for numeric stability
            BetterFloat l_count = l->count;
            while (r != centroids.end())
            {
                /// N.B. We cannot merge all the same values into single centroids because this will lead to
                /// unbalanced compression and wrong results.
                /// For more information see: https://arxiv.org/abs/1902.04023

                /// The ratio of the part of the histogram to l, including the half l to the entire histogram. That is, what level quantile in position l.
                BetterFloat ql = (sum + l_count * 0.5) / count;
                BetterFloat err = ql * (1 - ql);

                /// The ratio of the portion of the histogram to l, including l and half r to the entire histogram. That is, what level is the quantile in position r.
                BetterFloat qr = (sum + l_count + r->count * 0.5) / count;
                BetterFloat err2 = qr * (1 - qr);

                if (err > err2)
                    err = err2;

                BetterFloat k = count_epsilon_4 * err;

                /** The ratio of the weight of the glued column pair to all values is not greater,
                  *  than epsilon multiply by a certain quadratic coefficient, which in the median is 1 (4 * 1/2 * 1/2),
                  *  and at the edges decreases and is approximately equal to the distance to the edge * 4.
                  */

                if (l_count + r->count <= k && canBeMerged(l_mean, r->mean))
                {
                    // it is possible to merge left and right
                    /// The left column "eats" the right.
                    l_count += r->count;
                    if (r->mean != l_mean) /// Handling infinities of the same sign well.
                    {
                        l_mean += r->count * (r->mean - l_mean) / l_count; // Symmetric algo (M1*C1 + M2*C2)/(C1+C2) is numerically better, but slower
                    }
                    l->mean = static_cast<Float32>(l_mean);
                    l->count = static_cast<Float32>(l_count);
                }
                else
                {
                    // not enough capacity, check the next pair
                    sum += l->count; // Not l_count, otherwise actual sum of elements will be different
                    ++l;

                    /// We skip all the values "eaten" earlier.
                    if (l != r)
                        *l = *r;
                    l_mean = l->mean;
                    l_count = l->count;
                }
                ++r;
            }
            count = sum + l_count; // Update count, it might be different due to += inaccuracy

            /// At the end of the loop, all values to the right of l were "eaten".
            centroids.resize(l - centroids.begin() + 1);
            unmerged = 0;
        }

        // Ensures centroids.size() < max_centroids, independent of unprovable floating point blackbox above
        compressBrute();
    }

    /** Adds to the digest a change in `x` with a weight of `cnt` (default 1)
      */
    void add(T x, UInt64 cnt = 1)
    {
        auto vx = static_cast<Value>(x);
        if (cnt == 0 || std::isnan(vx))
            return; // Count 0 breaks compress() assumptions, Nan breaks sort(). We treat them as no sample.
        addCentroid(Centroid{vx, static_cast<Count>(cnt)});
    }

    void merge(const CausalQuantileTDigest & other)
    {
        for (const auto & c : other.centroids)
            addCentroid(c);
    }

    void serialize(WriteBuffer & buf)
    {
        writeVarUInt(params.max_centroids, buf);
        writeVarUInt(params.max_unmerged, buf);
        writeBinary(params.epsilon, buf);
        compress();
        writeVarUInt(centroids.size(), buf);
        buf.write(reinterpret_cast<const char *>(centroids.data()), centroids.size() * sizeof(centroids[0]));
    }

    void deserialize(ReadBuffer & buf)
    {
        readVarUInt(params.max_centroids, buf);
        readVarUInt(params.max_unmerged, buf);
        readBinary(params.epsilon, buf);
        size_t size = 0;
        readVarUInt(size, buf);

        if (size > max_centroids_deserialize)
            throw Exception(ErrorCodes::TOO_LARGE_ARRAY_SIZE, "Too large t-digest centroids size");

        count = 0;
        unmerged = 0;

        centroids.resize(size);
        // From now, TDigest will be in invalid state if exception is thrown.
        buf.readStrict(reinterpret_cast<char *>(centroids.data()), size * sizeof(centroids[0]));

        for (const auto & c : centroids)
        {
            if (c.count <= 0 || std::isnan(c.count)) // invalid count breaks compress()
                throw Exception(ErrorCodes::CANNOT_PARSE_INPUT_ASSERTION_FAILED, "Invalid centroid ");
            if (!std::isnan(c.mean))
            {
                count += c.count;
            }
        }

        auto it = std::remove_if(centroids.begin(), centroids.end(), [](Centroid & c) { return std::isnan(c.mean); });
        centroids.erase(it, centroids.end());

        compress(); // Allows reading/writing TDigests with different epsilon/max_centroids params
    }

    /** Calculates the quantile q [0, 1] based on the digest.
      * For an empty digest returns NaN.
      */
    template <typename ResultType>
    ResultType getImpl(Float64 level)
    {
        if (centroids.empty())
            return std::is_floating_point_v<ResultType> ? std::numeric_limits<ResultType>::quiet_NaN() : 0;

        compress();

        if (centroids.size() == 1)
            return checkOverflow<ResultType>(centroids.front().mean);

        Float64 x = level * count;
        Float64 prev_x = 0;
        Count sum = 0;
        Value prev_mean = centroids.front().mean;
        Count prev_count = centroids.front().count;

        for (const auto & c : centroids)
        {
            Float64 current_x = sum + c.count * 0.5;

            if (current_x >= x)
            {
                /// Special handling of singletons.
                Float64 left = prev_x + 0.5 * (prev_count == 1);
                Float64 right = current_x - 0.5 * (c.count == 1);

                if (x <= left)
                    return checkOverflow<ResultType>(prev_mean);
                else if (x >= right)
                    return checkOverflow<ResultType>(c.mean);
                else
                    return checkOverflow<ResultType>(interpolate(x, left, prev_mean, right, c.mean));
            }

            sum += c.count;
            prev_mean = c.mean;
            prev_count = c.count;
            prev_x = current_x;
        }

        return checkOverflow<ResultType>(centroids.back().mean);
    }

    std::vector<Float64> getQuantiles(const size_t quantiles_count = 100)
    {
        std::vector<Float64> levels(quantiles_count);
        for (size_t i = 0; i < quantiles_count; ++i)
            levels[i] = i / (1. * quantiles_count);
        std::vector<size_t> levels_permutation(quantiles_count);
        for (size_t i = 0; i < quantiles_count; ++i)
            levels_permutation[i] = i;
        std::vector<Float64> result(quantiles_count);
        getManyImpl(levels.data(), levels_permutation.data(), quantiles_count, result.data());
        // unique result
        result.erase(std::unique(result.begin(), result.end()), result.end());
        //result.resize(1);
        return result;
    }


    /** Get multiple quantiles (`size` parts).
      * levels - an array of levels of the desired quantiles. They are in a random order.
      * levels_permutation - array-permutation levels. The i-th position will be the index of the i-th ascending level in the `levels` array.
      * result - the array where the results are added, in order of `levels`,
      */
    template <typename ResultType>
    void getManyImpl(const Float64 * levels, const size_t * levels_permutation, size_t size, ResultType * result)
    {
        if (centroids.empty())
        {
            for (size_t result_num = 0; result_num < size; ++result_num)
                result[result_num] = std::is_floating_point_v<ResultType> ? NAN : 0;
            return;
        }

        compress();

        if (centroids.size() == 1)
        {
            for (size_t result_num = 0; result_num < size; ++result_num)
                result[result_num] = centroids.front().mean;
            return;
        }

        Float64 x = levels[levels_permutation[0]] * count;
        Float64 prev_x = 0;
        Count sum = 0;
        Value prev_mean = centroids.front().mean;
        Count prev_count = centroids.front().count;

        size_t result_num = 0;
        for (const auto & c : centroids)
        {
            Float64 current_x = sum + c.count * 0.5;

            if (current_x >= x)
            {
                /// Special handling of singletons.
                Float64 left = prev_x + 0.5 * (prev_count == 1);
                Float64 right = current_x - 0.5 * (c.count == 1);

                while (current_x >= x)
                {
                    if (x <= left)
                        result[levels_permutation[result_num]] = prev_mean;
                    else if (x >= right)
                        result[levels_permutation[result_num]] = c.mean;
                    else
                        result[levels_permutation[result_num]] = static_cast<Float32>(interpolate(static_cast<Float32>(x), static_cast<Float32>(left), static_cast<Float32>(prev_mean), static_cast<Float32>(right), static_cast<Float32>(c.mean)));

                    ++result_num;
                    if (result_num >= size)
                        return;

                    x = levels[levels_permutation[result_num]] * count;
                }
            }

            sum += c.count;
            prev_mean = c.mean;
            prev_count = c.count;
            prev_x = current_x;
        }

        auto rest_of_results = centroids.back().mean;
        for (; result_num < size; ++result_num)
            result[levels_permutation[result_num]] = rest_of_results;
    }

    T get(Float64 level)
    {
        return getImpl<T>(level);
    }

    Float32 getFloat(Float64 level)
    {
        return getImpl<Float32>(level);
    }

    void getMany(const Float64 * levels, const size_t * indices, size_t size, T * result)
    {
        getManyImpl(levels, indices, size, result);
    }

    void getManyFloat(const Float64 * levels, const size_t * indices, size_t size, Float32 * result)
    {
        getManyImpl(levels, indices, size, result);
    }

private:
    template <typename ResultType>
    static ResultType checkOverflow(Value val)
    {
        ResultType result;
        if (accurate::convertNumeric(val, result))
            return result;
        throw DB::Exception(ErrorCodes::DECIMAL_OVERFLOW, "Numeric overflow");
    }
};

}
