#include "sql_parser.h"
#include <cctype>

namespace mmexpt
{
namespace sqlparser
{

void find_parenthesis(size_t & left, size_t & right, const String & sql) {
  int left_count = 1;
  right = left + 1;
  while (left_count != 0 && right < sql.size()) {
    if (sql[right] == '(') {
      left_count++;
    } else if (sql[right] == ')') {
      left_count--;
    }
    right++;
  }
  right--;
  if (left_count != 0 || right >= sql.size() || sql[right] != ')') {
    right = std::string::npos;
  }
  if (left == std::string::npos || right == std::string::npos || left > right)
    throw std::runtime_error("find ( or ) error");
};

String getOneIdentifier(const String & sql, size_t & pos) {
  String ret;
  while (pos < sql.size() && sql[pos] == ' ') pos++;
  while (pos < sql.size() && sql[pos] != ' ') {
    ret += sql[pos];
    pos++;
  }
  return ret;
}

retContext DeltaMethodForward::operator()(const String & parameters, const String & arguments) {
  printf("DeltaMethodForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (!arguments.empty())
    return getForwardResult();
  for (auto & rule : rules)
    (*rule)(params, argums);
  return getForwardResult();
}

retContext Ttest_1SampForward::operator()(const String & parameters, const String & arguments) {
  printf("Ttest_1SampForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (!arguments.empty())
    return getForwardResult();
  for (auto & rule : rules)
    (*rule)(params, argums);
  if (params.size() == 4) params[3] = "X=" + params[3];
  return getForwardResult();
}

retContext Ttest_2SampForward::operator()(const String & parameters, const String & arguments) {
  printf("Ttest_2SampForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (!arguments.empty())
    return getForwardResult();
  for (auto & rule : rules)
    (*rule)(params, argums);
  if (params.size() == 3) params[2] = "X=" + params[2];
  return getForwardResult();
}

retContext Ttests_2SampForward::operator()(const String & parameters, const String & arguments) {
  printf("Ttests_2SampForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (!arguments.empty())
    return getForwardResult();
  for (auto & rule : rules)
    (*rule)(params, argums);
  if (params.size() == 4) params[3] = "X=" + params[3];
  return getForwardResult();
}

retContext OlsForward::operator()(const String & parameters, const String & arguments) {
  printf("OlsForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (!arguments.empty())
    return getForwardResult();
  for (auto & rule : rules)
    (*rule)(params, argums);

  /*
  if (interval) {
    if (params.size() == 0) params.push_back("TRUE");
    if (params.size() == 1) params.push_back("TRUE");
  }
  */

  auto res = getForwardResult();
  std::string temp = "";
  for (size_t i = 1; i < argums.size(); i++) {
    temp += "," + argums[i];
  }
  res.temp = temp;
  return res;
}

retContext WlsForward::operator()(const String & parameters, const String & arguments) {
  printf("WlsForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (!arguments.empty())
    return getForwardResult();
  for (auto & rule : rules)
    (*rule)(params, argums);
  auto res = getForwardResult();
  std::string temp = "";
  for (size_t i = 1; i < argums.size() - 1; i++) {
    temp += "," + argums[i];
  }
  res.temp = temp;
  return res;
}

retContext IVregressionForward::operator()(const String & parameters, const String & arguments) {
  printf("IVregressionForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (params.size() != 1)
    throw std::runtime_error("IVregressionForward: only one parameter is allowed");

  auto & param = params[0];
  printf("param: %s\n", param.c_str());
  // parse : Y ~ (D1 ~ IV1) + (D2 ~ IV2) + ...
  std::map<String, String> iv_mp;

  auto parse_iv = [](const String & str, String & d, String & iv) {
    size_t pos = str.find("~");
    if (pos == String::npos)
      throw std::runtime_error("IVregressionForward: invalid parameter");
    d = str.substr(0, pos);
    iv = str.substr(pos + 1);
  };

  size_t pos = 0, start = 0, end;
  end = param.find('~', start);
  if (end == String::npos)
    throw std::runtime_error("IVregressionForward: invalid parameter");
  iv_mp["@Y"] = param.substr(start, end - start);
  pos = end + 1;

  start = param.find('(', pos);
  find_parenthesis(start, end, param);

  parse_iv(param.substr(start + 1, end - start - 1), iv_mp["@D1"], iv_mp["@IV1"]);
  pos = end + 1;
  start = param.find('(', pos);
  find_parenthesis(start, end, param);
  parse_iv(param.substr(start + 1, end - start - 1), iv_mp["@D2"], iv_mp["@IV2"]);

  pos = end + 1;
  pos = param.find('+', pos);

  if (pos != String::npos) {
    pos += 1;
    String X = param.substr(pos);
    boost::replace_all(X, "+", ",");
    iv_mp["@X"] = ", "+X;
  }
  else iv_mp["@X"] = "";


  retContext ret;
  ret.func = func_template;
  ret.with = with_template;
  for (auto & kv : iv_mp) {
    auto & k = kv.first;
    auto & v = kv.second;
    boost::replace_all(ret.func, k, v);
    boost::replace_all(ret.with, k, v);
  }
  return ret;
}

retContext ExactMatchingForward::operator()(const String & parameters, const String & arguments) {
  printf("ExactMatchingForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (params.size() <= 1)
    throw std::runtime_error("ExactMatchingForward: only one parameter is allowed");

  String treatment = params[0];
  vector<String> labels;
  for (size_t i = 1; i < params.size(); i++) {
    labels.push_back(params[i]);
  }
  map<String, String> mp;
  mp["@PH1"] = treatment;
  String PHLabels = "";
  for (int i = 1; i <= labels.size(); i++) {
    PHLabels += "l" + std::to_string(i) + ", "[i == labels.size()];
  }
  mp["@PHLabels"] = PHLabels;

  String PHends = "";
  for (int i = 1; i <= labels.size(); i++) {
    PHends += labels[i-1] + " as " + "l" + std::to_string(i) + ", "[i == labels.size()];
  }
  mp["@PHends"] = PHends;

  String PHLabelsA,PHLabelsB;
  for (int i = 1; i <= labels.size(); i++) {
    PHLabelsA += "a.l" + std::to_string(i) +  " as " + "l" + std::to_string(i) + ", "[i == labels.size()];
    PHLabelsB += "b.l" + std::to_string(i) +  " as " + "l" + std::to_string(i) + ", "[i == labels.size()];
  }

  mp["@PHLabelA"] = PHLabelsA;
  mp["@PHLabelB"] = PHLabelsB;

  String PHLabelEqual = "";
  for (int i = 1; i <= labels.size(); i++) {
    PHLabelEqual += "a.l" + std::to_string(i) + " = b.l" + std::to_string(i) + (i == labels.size() ? "" : " and ");
  }
  mp["@PHLabelEqual"] = PHLabelEqual;

  retContext ret;
  ret.with = with_template;
  ret.end = end_template;
  ret.func = func_template;

  for (auto & kv : mp) {
    auto & k = kv.first;
    auto & v = kv.second;
    boost::replace_all(ret.func, k, v);
    boost::replace_all(ret.with, k, v);
    boost::replace_all(ret.end, k, v);
  }
  return ret;
}

retContext CaliperMatchingForward::operator()(const String & parameters, const String & arguments) {
  printf("CaliperMatchingForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (params.size() <= 2)
    throw std::runtime_error("CaliperMatchingForward: only one parameter is allowed");

  String treatment = params[0];
  String value = params[1];

  map<String, String> mp;
  mp["@PHt"] = treatment;
  mp["@PHv"] = value;
  mp["@PHSplictNum"] = params.size() > 2 ? params[2] : "5";

  retContext ret;
  ret.with = with_template;
  ret.end = end_template;
  ret.func = func_template;

  for (auto & kv : mp) {
    auto & k = kv.first;
    auto & v = kv.second;
    boost::replace_all(ret.func, k, v);
    boost::replace_all(ret.with, k, v);
    boost::replace_all(ret.end, k, v);
  }
  return ret;
}

retContext PredictForward::operator()(const String & parameters, const String & arguments) {
  printf("PredictForward::operator() called with parameters: %s and arguments: %s\n", parameters.c_str(), arguments.c_str());
  parserParamsAndArguments(parameters, arguments);
  if (!arguments.empty())
    return getForwardResult();
  String Interval = "";
  bool use_interval_predict = false;
  if (params.size() > 1) {
    for (size_t i = 1; i < params.size(); i++) {
      if (params[i] == "confidence" || params[i] == "prediction") {
        Interval +=  ",\'" + params[i] + "\'";
        use_interval_predict = true;
      }
      else Interval += "," + params[i];
    }
  }

  std::string sql = params[0] + " ";
  std::unordered_map<String, SqlForwardBase*> forwards;
  forwards["ols"] = new OlsForward(Interval.size() != 0);
  forwards["wls"] = new WlsForward();
  int i = 0;
  retContext res;
  res.with = with_template;
  res.func = use_interval_predict ? func_interval_template : func_template;
  String temp;
  for (auto kv : forwards) {
    auto & name = kv.first;
    auto & forward = kv.second;
    if (sql.compare(0, name.size(), name) == 0) {
      // 找到第一个括号的内容
      size_t left = sql.find('(', i), right;
      find_parenthesis(left, right, sql);
      size_t end = right;
      // 找到下一个括号的内容
      size_t pos = right + 1;
      size_t next_left = 0, next_right = 0;
      while (pos < sql.size() && sql[pos] == ' ') {
        pos++;
      }
      if (sql[pos] == '(') {
        next_left = sql.find('(', pos);
        next_right = sql.find(')', pos);
        find_parenthesis(next_left, next_right, sql);
        end = next_right;
      }
      String parameters = sql.substr(left + 1, right - left - 1);
      String arguments = sql[pos] == '(' ? sql.substr(next_left + 1, next_right - next_left - 1) : "";
      auto replace_context = (*forward)(parameters, arguments);
      sql.erase(i, end - i + 1);
      sql.insert(i, replace_context.func);
      temp = replace_context.temp;
      break;
    }
  }
  printf("INTERVAL:%s\n", Interval.c_str());
  boost::replace_all(res.func, "@PHInterval", Interval);
  sql.insert(3, use_interval_predict? "IntervalState" : "State" );
  boost::replace_all(res.with, "@PHfunc", sql);

  if ((!use_interval_predict && !Interval.empty()) || (use_interval_predict && params.size() > 3))
    boost::replace_all(res.func, "@PHargs", "");
  else
    boost::replace_all(res.func, "@PHargs", temp);

  if (use_interval_predict) {
    res.end = ")";
  }
  return res;
}


void SqlParser::operator()(std::string& sql) {
  String with;
  String table_name;
  String sql_end;
  String PHSelectItems;

  /*
  for (auto & ch : sql)
    ch = tolower(ch);
    */

  for (size_t i = 0; i < sql.size(); i++) {
    if (sql[i] == ' ') continue;
    if (i == 0 || !isalpha(sql[i - 1])) {
      for (auto kv : forwards) {
        auto & name = kv.first;
        auto & forward = kv.second;
        if (sql.compare(i, name.size(), name) == 0) {
          // 找到第一个括号的内容
          size_t left = sql.find('(', i), right;
          find_parenthesis(left, right, sql);
          size_t end = right;
          // 找到下一个括号的内容
          size_t pos = right + 1;
          size_t next_left = 0, next_right = 0;
          while (pos < sql.size() && sql[pos] == ' ') {
            pos++;
          }
          if (sql[pos] == '(') {
            next_left = sql.find('(', pos);
            next_right = sql.find(')', pos);
            find_parenthesis(next_left, next_right, sql);
            end = next_right;
          }
          String parameters = sql.substr(left + 1, right - left - 1);
          String arguments = sql[pos] == '(' ? sql.substr(next_left + 1, next_right - next_left - 1) : "";
          auto replace_context = (*forward)(parameters, arguments);
          sql.erase(i, end - i + 1);
          sql.insert(i, replace_context.func);
          if (!replace_context.with.empty()) {
            with += replace_context.with;
          }
          if (!replace_context.end.empty()) {
            sql_end += replace_context.end;
          }
        }
      }
      if (sql.compare(i, 4, "from") == 0 || sql.compare(i, 4, "FROM") == 0) {
        i += 4;
        table_name = sql.substr(i);
      }
      if (sql.compare(i, 6, "select") == 0 || sql.compare(i, 6, "SELECT") == 0) {
        i += 6;
        auto pos = sql.find("from");
        if (pos == std::string::npos)
          pos = sql.find("FROM");
        PHSelectItems = sql.substr(i, pos - i);
      }
    }
  }

  // add with
  if (!with.empty()) {
    if (with.back() == ',')
      with.pop_back();
    with.push_back(' ');
    sql.insert(0, "with ");
    sql.insert(5, with);
  }

  if (!sql_end.empty()) {
    sql += sql_end;
  }


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

  std::vector<String> select_items;
  parser(PHSelectItems, select_items);
  PHSelectItems = "";
  printf("B%sB",table_name.c_str());

  for (int i = 0; i < select_items.size(); i++) {
    auto & item = select_items[i];
    boost::trim(item);
    if (item.compare(0, 7, "caliper") == 0 || item.compare(0, 5, "exact") == 0) {
      boost::replace_all(sql, table_name, " tmp ");
      continue;
    }
    PHSelectItems += item + ",";
  }
  boost::replace_all(sql, "@PHSelectItems", PHSelectItems);
  boost::replace_all(sql, "@TBL", table_name);

}

SqlParser::SqlParser() {
  registerForward(new DeltaMethodForward());
  registerForward(new Ttest_1SampForward());
  registerForward(new Ttest_2SampForward());
  registerForward(new Ttests_2SampForward());
  registerForward(new OlsForward());
  registerForward(new WlsForward());
  registerForward(new IVregressionForward());
  registerForward(new ExactMatchingForward());
  registerForward(new CaliperMatchingForward());
  registerForward(new PredictForward());
}
///////////// rule


void FunctionParserRule::operator()(vector<String> & params, vector<String> & argums) {
  if (index >= params.size())
    return ;
  auto & func = params[index];
  size_t pos = 0, left_count = 0;
  bool is_avg = false;
  String new_func;
  while (pos < func.size()) {
    if (func.substr(pos, 3) == "avg") {
      pos += 3;
      is_avg = true;
      if (func[pos] != '(')
        throw std::runtime_error("avg function error, cant find (, should be avg(...)");
      size_t right;
      find_parenthesis(pos, right, func);
      argums.push_back(func.substr(pos + 1, right - pos - 1));
      new_func += "x" + std::to_string(argums.size());
      pos = right + 1;
    }
    else if (!isalpha(func[pos]))
      new_func += func[pos++];
    else
      throw std::runtime_error("cant find avg");
  }
  func = new_func;
}

}
}
