#include <AggregateFunctions/AggregateFunctionFactory.h>
#include <AggregateFunctions/AggregateFunctionDeltamethod.h>
#include <AggregateFunctions/FactoryHelpers.h>

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
}

namespace DB
{

bool FunctionParser::parse(std::string_view expr, int arg_num)
{
    rpn_expr.clear();
    std::set<std::string> xargs;
    std::vector<std::string> op_stack, tokens;
    std::string token;
    for (const auto& c : expr)
    {
        if (isOperator(c) || std::isspace(c))
        {
            if (!token.empty())
            {
                tokens.emplace_back(token);
                token.clear();
            }
            if (c == '-' && (tokens.empty() || isOperator(tokens.back())))
                tokens.emplace_back("0");
            if (isOperator(c))
                tokens.emplace_back(std::string(1, c));
        }
        else
            token += c;
    }
    if (!token.empty())
        tokens.emplace_back(token);
    for (const auto & t : tokens)
    {
        if (!isIdentifier(t) && !isOperator(t) && !isNumbers(t))
            return false;
        if (isIdentifier(t))
            xargs.insert(t);
        if (!isOperator(t) || t == "(")
            rpn_expr.emplace_back(t);
        else if (t == ")")
        {
            while (!op_stack.empty() && op_stack.back() != "(")
            {
                rpn_expr.emplace_back(op_stack.back());
                op_stack.pop_back();
            }
            if (op_stack.empty() || op_stack.back() != "(")
                return false;
            op_stack.pop_back();
        }
        else
        {
            while (!op_stack.empty() && getOpRank(op_stack.back()[0]) >= getOpRank(t[0]))
            {
                rpn_expr.emplace_back(op_stack.back());
                op_stack.pop_back();
            }
            op_stack.emplace_back(t);
        }
    }
    while (!op_stack.empty())
    {
        rpn_expr.emplace_back(op_stack.back());
        op_stack.pop_back();
    }
    // check if the data matches the function
    if (arg_num != -1)
    {
        if (xargs.size() != static_cast<size_t>(arg_num))
            return false;
        for (int i = 1; i <= arg_num; ++i)
        {
            std::string arg = "x" + std::to_string(i);
            if (xargs.find(arg) == xargs.end())
                return false;
        }
    }
    // check the validity of the expression
    int simulation = 0;
    for (const auto & t : rpn_expr)
    {
        if (t.size() == 1 && isOperator(t)) simulation--;
        else simulation++;
        if (simulation < 1) return false;
    }
    return (simulation == 1);
}

std::vector<Float64> FunctionParser::getPartialDeriv(const std::vector<Float64>& means) const
{
    std::vector<Float64> results(means.size());
    std::map<std::string, Float64> var_map;
    for (size_t i = 0; i < means.size(); ++i)
        var_map["x" + std::to_string(i+1)] = means[i];
    for (const auto & t : rpn_expr)
        if (isNumbers(t)) var_map[t] = std::stof(t);
    int div_num = std::count(rpn_expr.begin(), rpn_expr.end(), "/");

    auto is_zero_poly = [](const polynomial & poly) -> bool
    {
        Float64 sum = 0.0;
        for (size_t i = 0; i < poly.size(); i++)
            sum += poly[i];
        return fabs(sum) <= 1e-30;
    };

    for (size_t i = 0; i < means.size(); ++i)
    {
        var_map.erase("x" + std::to_string(i+1));
        std::vector<polynomial> calc_stack;
        for (const auto & t : rpn_expr)
        {
            if (!isOperator(t))
            {
                std::vector<Float64> poly_index(div_num+1, 0);
                if (var_map.count(t))
                    poly_index.back() = var_map[t];
                else if (t == "x" + std::to_string(i+1))
                    poly_index.emplace_back(1);
                calc_stack.emplace_back(polynomial(poly_index));
            }
            else
            {
                if (calc_stack.size() < 2)
                    throw Exception("Invalid expression of g", ErrorCodes::BAD_ARGUMENTS);
                auto left = calc_stack.back();
                calc_stack.pop_back();
                auto right = calc_stack.back();
                calc_stack.pop_back();
                if (t == "+") calc_stack.emplace_back(left + right);
                if (t == "-") calc_stack.emplace_back(right - left);
                if (t == "*") calc_stack.emplace_back((left*right) >> div_num);
                if (t == "/") 
                {
                    if (is_zero_poly(left))
                        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Division by zero, please check the data.");
                    calc_stack.emplace_back((right<<div_num) / left);
                }
            }
        }
        if (calc_stack.size() != 1)
            throw Exception("Invalid expression of g", ErrorCodes::BAD_ARGUMENTS);
        auto poly = calc_stack.back();
        Float64 result = 0;
        for (size_t j = 0; j < poly.size(); ++j) 
        {
            if (fabs(poly[j]) > 1e-30)
                result += poly[j] * (static_cast<Float64>(j)-div_num) * (j == static_cast<size_t>(div_num) ? 0 : pow(means[i], static_cast<Int32>(j)-div_num-1));
        }
        results[i] = result;
        var_map["x" + std::to_string(i+1)] = means[i];
    }
    return results;
}

Float64 FunctionParser::getExpressionResult(const std::vector<Float64>& x) const
{
    std::map<std::string, Float64> var_map;
    for (size_t i = 0; i < x.size(); ++i)
        var_map["x" + std::to_string(i+1)] = x[i];
    std::vector<Float64> stack;
    for (const auto & t : rpn_expr)
    {
        if (!isOperator(t))
            stack.emplace_back(isNumbers(t) ? std::stof(t) : var_map[t]);
        else
        {
            if (stack.size() < 2)
                throw Exception("Invalid expression of g", ErrorCodes::BAD_ARGUMENTS);
            auto left = stack.back();
            stack.pop_back();
            auto right = stack.back();
            stack.pop_back();
            if (t == "+") stack.emplace_back(left + right);
            if (t == "-") stack.emplace_back(right - left);
            if (t == "*") stack.emplace_back(left * right);
            if (t == "/") stack.emplace_back(right / left);
        }
    }
    if (stack.size() != 1)
        throw Exception("Invalid expression of g", ErrorCodes::BAD_ARGUMENTS);
    return stack.back();
}

struct Settings;

namespace
{

[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionDeltaMethod(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    if (argument_types.empty())
        throw Exception("Aggregate function " + name + " requires at least one argument", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    for (const auto & argument_type : argument_types)
    {
        if (!isNumber(argument_type))
            throw Exception("Illegal type " + argument_type->getName() + " of argument of aggregate function " + name,
                            ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    }
    AggregateFunctionPtr res = std::make_shared<AggregateFunctionDeltaMethod>(argument_types, parameters);
    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}

template <typename Op, bool use_index = false>
[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionTtestSamp(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    if (argument_types.empty())
        throw Exception("Aggregate function " + name + " requires at least one argument",
            ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
    std::vector<Field::Types::Which> require_types = {Field::Types::String, Field::Types::String, Field::Types::Float64};
    if constexpr (use_index)
        require_types.erase(require_types.end() - 2);

    if (use_index && !isInteger(argument_types.back()))
        throw Exception("Illegal type " + argument_types.back()->getName() + " of index argument of aggregate function,\
            It must be Integer as [0/1]" + name, ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

    if (parameters.empty())
        throw Exception("Aggregate function " + name + " requires at least one parameter, String, [String]," + (!use_index ? " [Float64]," : "") + " [String]", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);
    for (size_t i = 0; i < parameters.size() && i < require_types.size(); ++i)
        if (require_types[i] == Field::Types::String && parameters[i].getType() != require_types[i])
            throw Exception("Aggregate function " + name + " requires parameter: String, [String]," + (!use_index ? " [Float64]," : "") + " [String]", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

    AggregateFunctionPtr res = std::make_shared<AggregateFunctionTtestSamp<Op, use_index>>(argument_types, parameters);
    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}

[[maybe_unused]] AggregateFunctionPtr createAggregateFunctionXexptTtest2Samp(
    const std::string & name, const DataTypes & argument_types, const Array & parameters, const Settings *)
{
    AggregateFunctionPtr res;
    if (argument_types.size() < 4)
        throw Exception(ErrorCodes::BAD_ARGUMENTS, "Aggregate function {} requires at least 4 arguments \
            [numerator, denominator, uin, groupname]", name);
    if (isString(argument_types.back()))
        res = std::make_shared<AggregateFunctionXexptTtest2Samp<String>>(argument_types, parameters);
    else
        res = std::make_shared<AggregateFunctionXexptTtest2Samp<Int64>>(argument_types, parameters);
    if (!res)
        throw Exception(
        "Illegal types arguments of aggregate function " + name
            + ", must be Native Ints, Native UInts or Floats",
        ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);
    return res;
}
}

void registerAggregateFunctionDeltaMethod([[maybe_unused]] AggregateFunctionFactory & factory)
{
#if ENABLE_ALL_IN_SQL
    factory.registerFunction("Deltamethod", createAggregateFunctionDeltaMethod);
    factory.registerFunction("Ttest_1samp", createAggregateFunctionTtestSamp<Ttest1Samp>);
    factory.registerFunction("Ttest_2samp", createAggregateFunctionTtestSamp<Ttest2Samp, true>);
    factory.registerFunction("Ttests_2samp", createAggregateFunctionTtestSamp<Ttests2Samp, true>);
    factory.registerFunction("Xexpt_Ttest_2samp", createAggregateFunctionXexptTtest2Samp);
#endif
}

}
