// Copyright 2021-present StarRocks, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/math/distributions.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "exprs/agg/delta_method.h"
#include "exprs/helpers/expr_tree.hpp"
#include "exprs/helpers/math_helpers.hpp"
#include "glog/logging.h"

namespace starrocks {

enum class TtestAlternative : uint8_t { Unknown = 0, TwoSided, Less, Greater };

namespace ublas = boost::numeric::ublas;

class TtestCommon {
public:
    constexpr static double kDefaultAlphaValue = 0.05;
    constexpr static double kDefaultMDEValue = 0.005;
    constexpr static double kDefaultPowerValue = 0.8;
    const static std::unordered_map<std::string, TtestAlternative> str2alternative;

    static bool calc_means_and_vars(std::string const& Y_expression, std::string const& cuped_expression,
                                    int num_variables, size_t count0, size_t count1,
                                    ublas::vector<double> const& means0, ublas::vector<double> const& means1,
                                    ublas::vector<double> const& means, ublas::matrix<double> const& cov_matrix0,
                                    ublas::matrix<double> const& cov_matrix1, ublas::matrix<double> const& cov_matrix,
                                    double& mean0, double& mean1, double& var0, double& var1) {
        if (cuped_expression.empty()) {
            ExprTree<double> Y_expr_tree(Y_expression, num_variables);
            mean0 = Y_expr_tree.value(means0);
            mean1 = Y_expr_tree.value(means1);
            var0 = DeltaMethodStats::calc_delta_method(Y_expr_tree, count0, means0, cov_matrix0, false);
            var1 = DeltaMethodStats::calc_delta_method(Y_expr_tree, count1, means1, cov_matrix1, false);
        } else {
            ublas::matrix<double> cuped_means = calc_cuped_means(cuped_expression, num_variables, means);
            double var_y_tmp;
            ublas::matrix<double> theta, cov_XX_tmp, cov_XY_tmp;
            if (!calc_cuped_covs(Y_expression, cuped_expression, num_variables, count0 + count1, means, cov_matrix,
                                 var_y_tmp, cov_XY_tmp, cov_XX_tmp, theta)) {
                return false;
            }
            if (!calc_cuped_mean_and_var(Y_expression, cuped_expression, num_variables, count0, means0, cuped_means,
                                         cov_matrix0, theta, mean0, var0)) {
                return false;
            }
            if (!calc_cuped_mean_and_var(Y_expression, cuped_expression, num_variables, count1, means1, cuped_means,
                                         cov_matrix1, theta, mean1, var1)) {
                return false;
            }
        }
        return true;
    }

    static bool calc_means_and_vars(std::string const& Y_expression, std::string const& cuped_expression,
                                    int num_variables, DeltaMethodStats const& stats0, DeltaMethodStats const& stats1,
                                    DeltaMethodStats const& stats, double& mean0, double& mean1, double& var0,
                                    double& var1) {
        size_t count0 = stats0.count();
        size_t count1 = stats1.count();
        ublas::vector<double> const means0 = stats0.means();
        ublas::vector<double> const means1 = stats1.means();
        ublas::vector<double> const means = stats.means();
        ublas::matrix<double> const cov_matrix0 = stats0.cov_matrix();
        ublas::matrix<double> const cov_matrix1 = stats1.cov_matrix();
        ublas::matrix<double> const cov_matrix = stats.cov_matrix();
        return calc_means_and_vars(Y_expression, cuped_expression, num_variables, count0, count1, means0, means1, means,
                                   cov_matrix0, cov_matrix1, cov_matrix, mean0, mean1, var0, var1);
    }

    static ublas::matrix<double> calc_cuped_means(std::string const& cuped_expression, int num_variables,
                                                  ublas::vector<double> const& means) {
        std::vector<std::string> cuped_elements;
        boost::split(cuped_elements, cuped_expression, boost::is_any_of("+"));
        ublas::matrix<double> result(cuped_elements.size(), 1);
        for (size_t i = 0; i < cuped_elements.size(); ++i) {
            ExprTree<double> expr_tree(cuped_elements[i], num_variables);
            result(i, 0) = expr_tree.value(means);
        }
        return result;
    }

    static bool calc_cuped_mean_and_var(std::string const& Y_expression, std::string const& cuped_expression,
                                        int num_variables, double count, ublas::vector<double> const& means,
                                        ublas::matrix<double> const& cuped_means,
                                        ublas::matrix<double> const& cov_matrix, ublas::matrix<double> const& theta,
                                        double& cuped_mean, double& cuped_var) {
        double var_y;
        ublas::matrix<double> theta_tmp, cov_XX, cov_XY;
        if (!calc_cuped_covs(Y_expression, cuped_expression, num_variables, count, means, cov_matrix, var_y, cov_XY,
                             cov_XX, theta_tmp)) {
            return false;
        }
        cuped_var = var_y + prod(static_cast<ublas::matrix<double>>(prod(theta, cov_XX)), ublas::trans(theta))(0, 0) -
                    2 * prod(theta, ublas::trans(cov_XY))(0, 0);
        ExprTree<double> y_expr_tree(Y_expression, num_variables);
        cuped_mean = y_expr_tree.value(means) -
                     prod(theta, calc_cuped_means(cuped_expression, num_variables, means) - cuped_means)(0, 0);
        return true;
    }

    static bool calc_cuped_mean_and_var(std::string const& Y_expression, std::string const& cuped_expression,
                                        int num_variables, double count, ublas::vector<double> const& means,
                                        ublas::matrix<double> const& cov_matrix, double& cuped_mean,
                                        double& cuped_var) {
        ExprTree<double> Y_expr_tree(Y_expression, num_variables);
        cuped_mean = Y_expr_tree.value(means);
        if (!calc_cuped_var(Y_expression, cuped_expression, num_variables, count, means, cov_matrix, cuped_var)) {
            return false;
        }
        return true;
    }

    static bool calc_cuped_covs(std::string const& Y_expression, std::string const& cuped_expression, int num_variables,
                                size_t count, ublas::vector<double> const& means,
                                ublas::matrix<double> const& cov_matrix, double& var_y, ublas::matrix<double>& cov_XY,
                                ublas::matrix<double>& cov_XX, ublas::matrix<double>& theta) {
        DCHECK(!cuped_expression.empty());
        std::vector<std::pair<std::string, ExprTree<double>>> expressions;
        std::vector<std::string> cuped_elements;
        boost::split(cuped_elements, cuped_expression, boost::is_any_of("+"));
        for (std::string const& cuped_element : cuped_elements) {
            ExprTree<double> expr_tree;
            CHECK(expr_tree.init(cuped_element, num_variables));
            expressions.emplace_back(std::move(cuped_element), std::move(expr_tree));
        }

        ExprTree<double> Y_expr_tree(Y_expression, num_variables);
        uint32_t num_parts = expressions.size();
        cov_XX.resize(num_parts, num_parts);
        cov_XY.resize(1, num_parts);
        for (uint32_t part_i = 0; part_i < num_parts; ++part_i) {
            const auto& [_, Xi_expr_tree] = expressions[part_i];
            cov_XX(part_i, part_i) = DeltaMethodStats::calc_delta_method(Xi_expr_tree, count, means, cov_matrix, false);
            cov_XY(0, part_i) =
                    DeltaMethodStats::calc_delta_method_cov(Y_expr_tree, Xi_expr_tree, count, means, cov_matrix);
            for (uint32_t part_j = part_i + 1; part_j < num_parts; ++part_j) {
                const auto& [_, Xj_expr_tree] = expressions[part_j];
                cov_XX(part_j, part_i) = cov_XX(part_i, part_j) =
                        DeltaMethodStats::calc_delta_method_cov(Xi_expr_tree, Xj_expr_tree, count, means, cov_matrix);
            }
        }

        ublas::matrix<double> cov_XX_inv(num_parts, num_parts, 0);
        MathHelpers::invert_matrix(cov_XX, cov_XX_inv);
        for (size_t i = 0; i < cov_XX_inv.size1(); i++) {
            for (size_t j = 0; j < cov_XX_inv.size2(); j++) {
                if (std::isnan(cov_XX_inv(i, j))) {
                    cov_XX_inv(i, j) = 0;
                }
            }
        }
        theta = ublas::prod(cov_XY, cov_XX_inv);
        var_y = DeltaMethodStats::calc_delta_method(Y_expr_tree, count, means, cov_matrix, false);
        return true;
    }

    static bool calc_cuped_var(std::string const& Y_expression, std::string const& cuped_expression, int num_variables,
                               size_t count, ublas::vector<double> const& means,
                               ublas::matrix<double> const& cov_matrix, double& var) {
        if (cuped_expression.empty()) {
            var = DeltaMethodStats::calc_delta_method(ExprTree<double>(Y_expression, num_variables), count, means,
                                                      cov_matrix, false);
            return true;
        }

        double var_y;
        ublas::matrix<double> theta;
        ublas::matrix<double> cov_XX;
        ublas::matrix<double> cov_XY;
        if (!calc_cuped_covs(Y_expression, cuped_expression, num_variables, count, means, cov_matrix, var_y, cov_XY,
                             cov_XX, theta)) {
            return false;
        }
        var = var_y + ublas::prod(ublas::matrix<double>(ublas::prod(theta, cov_XX)), ublas::trans(theta))(0, 0) -
              2 * ublas::prod(theta, ublas::trans(cov_XY))(0, 0);
        return true;
    }

    static double calc_pvalue(double t_stat, TtestAlternative alternative) {
        boost::math::normal normal_dist(0, 1);
        double p_value = 0;
        if (std::isnan(t_stat)) {
            p_value = std::numeric_limits<double>::quiet_NaN();
        } else if (std::isinf(t_stat)) {
            p_value = 0;
        } else if (alternative == TtestAlternative::TwoSided) {
            p_value = 2 * (1 - cdf(normal_dist, std::abs(t_stat)));
        } else if (alternative == TtestAlternative::Less) {
            p_value = cdf(normal_dist, t_stat);
        } else if (alternative == TtestAlternative::Greater) {
            p_value = 1 - cdf(normal_dist, t_stat);
        } else {
            p_value = std::numeric_limits<double>::quiet_NaN();
        }
        return p_value;
    }

    static std::pair<double, double> calc_confidence_interval(double estimate, double stderr_var, size_t count,
                                                              double alpha, TtestAlternative alternative) {
        double lower = 0, upper = 0;
        if (alpha > 0) {
            boost::math::students_t_distribution<> dist(count - 1);
            double t_quantile = 0;
            if (alternative == TtestAlternative::TwoSided) {
                t_quantile = quantile(dist, 1 - alpha / 2);
                lower = estimate - t_quantile * stderr_var;
                upper = estimate + t_quantile * stderr_var;
            } else if (alternative == TtestAlternative::Less) {
                t_quantile = quantile(dist, 1 - alpha);
                lower = -std::numeric_limits<double>::infinity();
                upper = estimate + t_quantile * stderr_var;
            } else if (alternative == TtestAlternative::Greater) {
                t_quantile = quantile(dist, 1 - alpha);
                lower = estimate - t_quantile * stderr_var;
                upper = std::numeric_limits<double>::infinity();
            } else {
                lower = upper = std::numeric_limits<double>::quiet_NaN();
            }
        }
        return {lower, upper};
    }
};

const std::unordered_map<std::string, TtestAlternative> TtestCommon::str2alternative{
        {"two-sided", TtestAlternative::TwoSided},
        {"less", TtestAlternative::Less},
        {"greater", TtestAlternative::Greater},
};

} // namespace starrocks
