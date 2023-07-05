#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <map>
#include <boost/algorithm/string.hpp>


using namespace std;
using String = string;

namespace mmexpt
{
namespace sqlparser
{

enum class ParamType {
  kBasic,
  kString,
};

struct retContext {
  String func;
  String with;
  String end;
  String temp;
};

class RuleBase
{
public:
  RuleBase(size_t index_) : index(index_) {}

  virtual ~RuleBase() = default;

  virtual void operator()(vector<String> & params, vector<String> & argums) = 0;

  virtual String getName() const = 0;

protected:
  size_t index;
};

class FunctionParserRule : public RuleBase
{
public:
  explicit FunctionParserRule(size_t index_) : RuleBase(index_) {}

  ~FunctionParserRule() override = default;

  void operator()(vector<String> & params, vector<String> & argums) override;

  String getName() const override { return "FunctionParserRule"; }
};

class MoveParamToArgumRule : public RuleBase
{
public:
  explicit MoveParamToArgumRule(size_t index_) : RuleBase(index_) {}

  ~MoveParamToArgumRule() override = default;

  void operator()(vector<String> & params, vector<String> & argums) override {
    if (index >= params.size()) return ;
    argums.push_back(params[index]);
    params.erase(params.begin() + index);
  }

  String getName() const override { return "MoveParamToArgumRule"; }
};

class EraseParam : public RuleBase
{
public:
  explicit EraseParam(size_t index_) : RuleBase(index_) {}

  ~EraseParam() override = default;

  void operator()(vector<String> & params, vector<String> & argums) override {
    if (index >= params.size()) return ;
    params.erase(params.begin() + index);
  }

  String getName() const override { return "EraseParam"; }
};

class ParserFunctionWithSeparators : public RuleBase
{
public:
  explicit ParserFunctionWithSeparators(size_t index_, std::vector<char> separators_) :
    RuleBase(index_), separators(std::move(separators_)) {}

  ~ParserFunctionWithSeparators() override = default;

  void operator()(vector<String> & params, vector<String> & argums) override {
    if (index >= params.size()) return ;
    auto & func = params[index];
    // split params by any separator use boost
    std::vector<String> split_func;
    boost::split(split_func, func, boost::is_any_of(separators));
    argums.insert(argums.end(), split_func.begin(), split_func.end());
  }

  String getName() const override { return "ParserFunctionWithSeparators"; }

  std::vector<char> separators;
};

class SqlForwardBase
{
public:
  SqlForwardBase() = default;

  virtual ~SqlForwardBase() = default;

  virtual String getName() = 0;

  virtual retContext operator()(const String & parameters, const String & arguments) = 0;

  virtual retContext getForwardResult() {
    retContext ret;
    for (size_t i = 0; i < params.size(); i++) {
      if (i < param_types.size()) {
        if (param_types[i] == ParamType::kString)
          params[i] = "'" + params[i] + "'";
      }
    }
    String parameters = boost::algorithm::join(params, ",");
    String arguments = boost::algorithm::join(argums, ",");
    String forward_sql = getName() + (parameters.empty() ? "" : "(" + parameters + ")" ) + "(" + arguments + ")";
    forward_sql[0] = toupper(forward_sql[0]);
    ret.func = forward_sql;
    return ret;
  }

  void parserParamsAndArguments(const String & parameters, const String & arguments) {
    printf("parserParamsAndArguments called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
    auto parser = [](const String & str, vector<String> & vec) {
      vec.clear();
      String tmp;
      size_t cnt = 0;
      for (auto & ch : str) {
        if (ch == ',' && !cnt) {
          vec.push_back(tmp);
          tmp.clear();
        } else {
          if (ch == '(') {
            cnt++;
          } else if (ch == ')') {
            cnt--;
          }
          tmp.push_back(ch);
        }
      }
      if (!tmp.empty()) {
        vec.push_back(tmp);
        tmp.clear();
      }
    };

    parser(parameters, params);
    parser(arguments, argums);
    // remove \' or '" in params
    for (auto & param : params) {
      boost::trim(param);
      if (param[0] == '\'' || param[0] == '"') {
        param = param.substr(1, param.size() - 2);
      }
    }
  }

  String toString() {
    String result = "params: ";
    for (auto & param : params)
      result += param + " ";
    result += "argums: ";
    for (auto & argum : argums)
      result += argum + " ";
    return result;
  }

  void registerRulesAndParamTypes(std::vector<std::shared_ptr<RuleBase>> && rules_, std::vector<ParamType> && param_types_) {
    rules = std::move(rules_);
    param_types = std::move(param_types_);
  }

  virtual std::vector<std::shared_ptr<RuleBase>> getRules() = 0;

  virtual std::vector<ParamType> getParamTypes() = 0;

protected:
  std::vector<String> params;
  std::vector<String> argums;
  std::vector<std::shared_ptr<RuleBase>> rules;
  std::vector<ParamType> param_types;
};

class DeltaMethodForward : public SqlForwardBase
{
public:
  DeltaMethodForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~DeltaMethodForward() override = default;

  String getName() override { return "deltamethod"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    return std::vector<std::shared_ptr<RuleBase>>{
      std::make_shared<FunctionParserRule>(0)
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kString,
      ParamType::kBasic
    };
  }

};

class Ttest_1SampForward : public SqlForwardBase
{
public:
  Ttest_1SampForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~Ttest_1SampForward() override = default;

  String getName() override { return "ttest_1samp"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    return std::vector<std::shared_ptr<RuleBase>>{
      std::make_shared<FunctionParserRule>(0),
      std::make_shared<FunctionParserRule>(3)
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kString,
      ParamType::kString,
      ParamType::kBasic,
      ParamType::kString,
    };
  }

};

class Ttest_2SampForward : public SqlForwardBase
{
public:
  Ttest_2SampForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~Ttest_2SampForward() override = default;

  String getName() override { return "ttest_2samp"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    return std::vector<std::shared_ptr<RuleBase>>{
      std::make_shared<FunctionParserRule>(0),
      std::make_shared<FunctionParserRule>(3),
      std::make_shared<MoveParamToArgumRule>(2)
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kString,
      ParamType::kString,
      ParamType::kString,
    };
  }
};

class Ttests_2SampForward : public SqlForwardBase
{
public:
  Ttests_2SampForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~Ttests_2SampForward() override = default;

  String getName() override { return "ttests_2samp"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    return std::vector<std::shared_ptr<RuleBase>>{
      std::make_shared<FunctionParserRule>(0),
      std::make_shared<FunctionParserRule>(4),
      std::make_shared<MoveParamToArgumRule>(2)
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kString,
      ParamType::kString,
      ParamType::kBasic,
      ParamType::kString,
    };
  }
};

class OlsForward : public SqlForwardBase
{
public:
  OlsForward(bool interval_ = false) : interval(interval_) {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~OlsForward() override = default;

  String getName() override { return "ols"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    std::vector<char> operators{'+', '~', '='};
    return std::vector<std::shared_ptr<RuleBase>>{
      std::make_shared<ParserFunctionWithSeparators>(0, std::move(operators)),
      std::make_shared<EraseParam>(0)
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kBasic,
    };
  }
  bool interval = false;
};

class WlsForward : public SqlForwardBase
{
public:
  WlsForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~WlsForward() override = default;

  String getName() override { return "wls"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    std::vector<char> operators{'+', '~', '='};
    return std::vector<std::shared_ptr<RuleBase>>{
      std::make_shared<ParserFunctionWithSeparators>(0, std::move(operators)),
      std::make_shared<MoveParamToArgumRule>(1),
      std::make_shared<EraseParam>(1),
      std::make_shared<EraseParam>(0)
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kBasic,
    };
  }
};

class IVregressionForward : public SqlForwardBase
{
public:
  IVregressionForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~IVregressionForward() override = default;

  String getName() override { return "IVregression"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    return std::vector<std::shared_ptr<RuleBase>>{
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kBasic,
      ParamType::kBasic,
      ParamType::kBasic
    };
  }

  static constexpr const char * with_template = "\
    ( \
        SELECT OlsState(true)(@D1, @IV1@X) \
        FROM @TBL \
    ) AS model1, \
    ( \
        SELECT OlsState(true)(@D2, @IV2@X) \
        FROM @TBL \
    ) AS model2, \
    ( \
        SELECT OlsState(true)(@Y, evalMLMethod(model1, @IV1@X), evalMLMethod(model2, @IV2@X)@X) \
        FROM @TBL \
    ) AS model_final, \
    ( \
      select  \
      MatrixMultiplication(true, false)(1, evalMLMethod(model1, @IV1@X), evalMLMethod(model2, @IV2@X)@X) from @TBL \
    ) as xx_inverse, \
    ( \
      select  \
      MatrixMultiplication(false, true)(1, evalMLMethod(model1, @IV1@X), evalMLMethod(model2, @IV2@X)@X, ABS(@Y - evalMLMethod(model_final, @D1, @D2@X))) from @TBL \
    )  as xx_weighted,";

   //static constexpr const char * func_template = "Ols(true, false, xx_inverse, xx_weighted)(@Y,evalMLMethod(model1, @IV1@X), evalMLMethod(model2, @IV2@X)@X)";
   static constexpr const char * func_template = "Ols(true, false)(@Y,evalMLMethod(model1, @IV1@X), evalMLMethod(model2, @IV2@X)@X)";

};

class ExactMatchingForward : public SqlForwardBase
{
public:
  ExactMatchingForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~ExactMatchingForward() override = default;

  String getName() override { return "exactMatching"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    return std::vector<std::shared_ptr<RuleBase>>{
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kBasic,
      ParamType::kBasic,
      ParamType::kBasic
    };
  }

  static constexpr const char * with_template = "\
  tmp as (\
      select \
          @PHSelectItems\
          row_number() OVER () AS uin_tmp,\
          @PH1 as t, @PHends\
      from \
          @TBL\
  ),\
  tmp_T AS\
  (\
      SELECT\
          toUInt32(uin_tmp) as uin_tmp,t,@PHLabels,\
          row_number() OVER (PARTITION BY @PHLabels ORDER BY rand() asc) AS rn\
      FROM tmp\
      where t = 1\
  ),tmp_C AS\
  (\
      SELECT\
          toUInt32(uin_tmp) as uin_tmp,t,@PHLabels,\
          row_number() OVER (PARTITION BY @PHLabels ORDER BY rand() asc) AS rn\
      FROM tmp\
      where t = -1\
  )";

   static constexpr const char * func_template = "coalesce(b.t*b.index, 0) ";

   static constexpr const char * end_template = " \
  as a left join\
  (\
      select \
          a.t as t,\
          a.uin_tmp as uin_tmp,\
          @PHLabelA, \
          row_number() over () as index\
      from \
      tmp_T as a \
      inner join \
      tmp_C as b on @PHLabelEqual and a.rn = b.rn \
      union all \
      select \
          b.t as t,\
          b.uin_tmp as uin_tmp,\
          @PHLabelB, \
          row_number() over () as index\
      from \
      tmp_T as a \
      inner join \
      tmp_C as b on @PHLabelEqual and a.rn = b.rn \
  )b on a.uin_tmp = b.uin_tmp";

};

class CaliperMatchingForward : public SqlForwardBase
{
public:
  CaliperMatchingForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~CaliperMatchingForward() override = default;

  String getName() override { return "caliperMatching"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    return std::vector<std::shared_ptr<RuleBase>>{
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kBasic,
      ParamType::kBasic,
      ParamType::kBasic
    };
  }

  static constexpr const char * with_template = "\
    tmp AS\
    (\
        SELECT\
            @PHSelectItems\
            row_number() OVER () AS uin_tmp,\
            toInt8(v * @PHSplictNum) + 1 AS score_bin,\
            @PHt as t,\
            @PHv as v\
        FROM @TBL\
    ),\
    tmp_T AS\
    (\
        SELECT\
            toUInt32(uin_tmp) as uin_tmp,\
            t,\
            score_bin,\
            v,\
            row_number() OVER (PARTITION BY toInt8(v * @PHSplictNum) + 1 ORDER BY v ASC) AS rn\
        FROM tmp\
        WHERE t = 1\
    ),\
    tmp_C AS\
    (\
        SELECT\
            toUInt32(uin_tmp) as uin_tmp,\
            t,\
            score_bin,\
            v,\
            row_number() OVER (PARTITION BY toInt8(v * @PHSplictNum) + 1 ORDER BY v ASC) AS rn\
        FROM tmp\
        WHERE t = -1\
    )";

   static constexpr const char * func_template = "coalesce(b.t*b.index, 0) ";

   static constexpr const char * end_template = " \
   AS a LEFT JOIN\
  (\
      SELECT\
          a.t AS t,\
          a.uin_tmp AS uin_tmp,\
          a.v AS v,\
          a.score_bin AS score_bin,\
          row_number() OVER () AS index\
      FROM tmp_T AS a\
      INNER JOIN tmp_C AS b ON (a.score_bin = b.score_bin) AND (a.rn = b.rn)\
      UNION ALL\
      SELECT\
          b.t AS t,\
          b.uin_tmp AS uin_tmp,\
          b.v AS v,\
          b.score_bin AS score_bin,\
          row_number() OVER () AS index\
      FROM tmp_T AS a\
      INNER JOIN tmp_C AS b ON (a.score_bin = b.score_bin) AND (a.rn = b.rn)\
  ) AS b ON a.uin_tmp = b.uin_tmp";


};

class PredictForward : public SqlForwardBase
{
public:
  PredictForward() {
    registerRulesAndParamTypes(getRules(), getParamTypes());
  }

  ~PredictForward() override = default;

  String getName() override { return "predict"; }

  retContext operator()(const String & parameters, const String & arguments) override;

  std::vector<std::shared_ptr<RuleBase>> getRules() override {
    return std::vector<std::shared_ptr<RuleBase>>{
    };
  }

  std::vector<ParamType> getParamTypes() override {
    return std::vector<ParamType>{
      ParamType::kBasic,
      ParamType::kBasic,
      ParamType::kBasic
    };
  }

  static constexpr const char * with_template = "(\
     SELECT\
         @PHfunc\
     FROM\
         @TBL\
    ) AS model";

  static constexpr const char * func_template = " evalMLMethod(model@PHInterval@PHargs) ";

  static constexpr const char * func_interval_template = " interval_tmp[1] as fit, interval_tmp[2] as lower, interval_tmp[3] as upper from ( select evalMLMethod(model@PHInterval@PHargs) as interval_tmp ";

};

class SqlParser
{
public:
  SqlParser();

  ~SqlParser() {
    for (auto& forward : forwards) {
      delete forward.second;
    }
  }

  void operator()(std::string& sql);

  void registerForward(SqlForwardBase* forward) {
    forwards[forward->getName()] = forward;
  }

  std::unordered_map<String, SqlForwardBase*> forwards;
};

}
}
