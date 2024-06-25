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

package com.starrocks.sql.analyzer;

import com.starrocks.analysis.BoolLiteral;
import com.starrocks.analysis.DecimalLiteral;
import com.starrocks.analysis.FloatLiteral;
import com.starrocks.analysis.FunctionCallExpr;
import com.starrocks.analysis.FunctionName;
import com.starrocks.analysis.FunctionParams;
import com.starrocks.analysis.IntLiteral;
import com.starrocks.analysis.StringLiteral;
import com.starrocks.catalog.FunctionSet;

import java.util.LinkedList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AllInSqlAnalyzer {
    public static void analyze(FunctionCallExpr functionCallExpr) {
        FunctionName fnName = functionCallExpr.getFnName();

        if (fnName.getFunction().equals(FunctionSet.DELTA_METHOD)) {
            analyzeDeltaMethodFunction(functionCallExpr);
        }

        if (fnName.getFunction().equals(FunctionSet.TTEST_1SAMP)) {
            analyzeTtest1SampFunction(functionCallExpr);
        }

        if (fnName.getFunction().equals(FunctionSet.TTEST_2SAMP)) {
            analyzeTtest2SampFunction(functionCallExpr);
        }

        if (fnName.getFunction().equals(FunctionSet.TTESTS_2SAMP)) {
            analyzeTtest2SampFunction(functionCallExpr);
        }

        if (fnName.getFunction().equals(FunctionSet.XEXPT_TTEST_2SAMP)) {
            analyzeXexptTtest2SampFunction(functionCallExpr);
        }

        if (fnName.getFunction().equals(FunctionSet.OLS_TRAIN)) {
            analyzeOlsFunction(functionCallExpr);
        }

        if (fnName.getFunction().equals(FunctionSet.OLS)) {
            analyzeOlsFunction(functionCallExpr);
        }

        if (fnName.getFunction().equals(FunctionSet.WLS_TRAIN)) {
            analyzeOlsFunction(functionCallExpr);
        }

        if (fnName.getFunction().equals(FunctionSet.WLS)) {
            analyzeOlsFunction(functionCallExpr);
        }
    }

    private static void analyzeOlsFunction(FunctionCallExpr functionCallExpr) {
        FunctionParams fnParams = functionCallExpr.getParams();

        if (fnParams.isDistinct()) {
            throw new SemanticException("ols does not support DISTINCT", functionCallExpr.getPos());
        }
    }

    private static void analyzeXexptTtest2SampFunction(FunctionCallExpr functionCallExpr) {
        FunctionParams fnParams = functionCallExpr.getParams();
        if (fnParams.isDistinct()) {
            throw new SemanticException("xexpt_ttest_2samp does not support DISTINCT", functionCallExpr.getPos());
        }

        // uin, treatment[, [numerator, denominator, cuped_data...], cuped, {alpha, mde, power}], [type, ratios]
        if (functionCallExpr.getChildren().size() >= 4) {
            int numVariables = functionCallExpr.getChild(2).getChildren().size();
            checkIsValidCupedExpression(functionCallExpr, numVariables, 3);
        }

        if (functionCallExpr.getChildren().size() >= 9) {
            if (!(functionCallExpr.getChild(7) instanceof StringLiteral)) {
                throw new SemanticException(
                        "7-th arg should be a string literal in ['avg', 'sum'], but get : "
                                + functionCallExpr.getChild(7).toString());
            }
            String type = ((StringLiteral) functionCallExpr.getChild(7)).getStringValue();
            if (!(type.equals("avg") || type.equals("sum"))) {
                throw new SemanticException("2-nd arg should be a string literal in ['avg', 'sum'], but get : " + type);
            }
        }

        if (functionCallExpr.getChildren().size() >= 9) {
            if (functionCallExpr.getChild(8).getChildren().size() != 2) {
                throw new SemanticException(
                        "8-th arg should be a two ratios, but get :"
                                + functionCallExpr.getChild(8).toString());
            }
        }
    }

    private static void analyzeDeltaMethodFunction(FunctionCallExpr functionCallExpr) {
        FunctionParams fnParams = functionCallExpr.getParams();

        if (!(functionCallExpr.getChild(0) instanceof StringLiteral)) {
            throw new SemanticException("first arg should be a string literal, but get : "
                    + functionCallExpr.getChild(0).toString());
        }
        if (!(functionCallExpr.getChild(1) instanceof BoolLiteral)) {
            throw new SemanticException("second arg should be a boolean literal, but get : "
                    + functionCallExpr.getChild(1).toString());
        }
        if (fnParams.isDistinct()) {
            throw new SemanticException("delta_method does not support DISTINCT", functionCallExpr.getPos());
        }
        int numVariables = functionCallExpr.getChild(2).getChildren().size();
        checkIsValidDeltaMethodFunctionExpr(((StringLiteral) functionCallExpr.getChild(0)).getStringValue(),
                numVariables);
    }

    private static void analyzeTtest1SampFunction(FunctionCallExpr functionCallExpr) {
        // expression, side, mu, data, [cuped, alpha]
        FunctionParams fnParams = functionCallExpr.getParams();

        checkTtestParams(functionCallExpr);
        int numVariables = functionCallExpr.getChild(3).getChildren().size();
        if (!(functionCallExpr.getChild(2) instanceof DecimalLiteral ||
                functionCallExpr.getChild(2) instanceof IntLiteral ||
                functionCallExpr.getChild(2) instanceof FloatLiteral)) {
            throw new SemanticException(
                    "second arg should be a decimal literal, but get : " + functionCallExpr.getChild(2).toString());
        }
        checkIsValidCupedExpression(functionCallExpr, numVariables, 3);
        checkIsValidCupedExpression(functionCallExpr, numVariables, 4);
        if (fnParams.isDistinct()) {
            throw new SemanticException("ttest_1samp does not support DISTINCT", functionCallExpr.getPos());
        }
    }

    private static void analyzeTtest2SampFunction(FunctionCallExpr functionCallExpr) {
        // expression, side, treatment, data, [cuped, alpha]
        FunctionParams fnParams = functionCallExpr.getParams();

        checkTtestParams(functionCallExpr);
        int numVariables = functionCallExpr.getChild(3).getChildren().size();
        checkIsValidCupedExpression(functionCallExpr, numVariables, 4);
        checkIsValidCupedExpression(functionCallExpr, numVariables, 5);
        if (fnParams.isDistinct()) {
            throw new SemanticException("ttest_1samp does not support DISTINCT", functionCallExpr.getPos());
        }
    }

    private static void checkIsValidCupedExpression(FunctionCallExpr functionCallExpr, int numVariables, int idx) {
        if (functionCallExpr.getChildren().size() >= idx + 1) {
            if (functionCallExpr.getChild(idx) instanceof StringLiteral) {
                String cuped = ((StringLiteral) functionCallExpr.getChild(idx)).getStringValue();
                if (!cuped.startsWith("X=")) {
                    throw new SemanticException(
                            "5-th arg should be a string literal start with 'X=', but get : "
                                    + cuped);
                }
                if (cuped.length() == 2) {
                    return;
                }
                checkIsValidDeltaMethodFunctionExpr(cuped.substring(2), numVariables);
            }
        }
    }

    private static void checkTtestParams(FunctionCallExpr functionCallExpr) {
        if (!(functionCallExpr.getChild(0) instanceof StringLiteral)) {
            throw new SemanticException("1-st arg should be a string literal, but get : "
                    + functionCallExpr.getChild(0).toString());
        }
        if (!(functionCallExpr.getChild(1) instanceof StringLiteral)) {
            throw new SemanticException(
                    "2-nd arg should be a string literal in ['two-sided', 'greater', 'less'], but get : "
                            + functionCallExpr.getChild(1).toString());
        }
        String alternative = ((StringLiteral) functionCallExpr.getChild(1)).getStringValue();
        if (!(alternative.equals("two-sided") || alternative.equals("greater") || alternative.equals("less"))) {
            throw new SemanticException(
                    "2-nd arg should be a string literal in ['two-sided', 'greater', 'less'], but get : "
                            + alternative);
        }
        int numVariables = functionCallExpr.getChild(3).getChildren().size();
        checkIsValidDeltaMethodFunctionExpr(((StringLiteral) functionCallExpr.getChild(0)).getStringValue(),
                numVariables);
    }

    private static void checkIsValidDeltaMethodFunctionExpr(String expr, int numVariables) {
        Pattern pattern = Pattern.compile("[^0-9.+\\-*/()x\\s]+|(x{2,})");
        Matcher matcher = pattern.matcher(expr);

        if (matcher.find()) {
            throw new SemanticException("Find invalid symbol `%s` at position %s of `%s`", matcher.group(),
                    matcher.start() + 1, expr);
        }

        expr = expr.replaceAll("\\s*", "");

        checkIsVariableNames(expr, numVariables);
        checkIsValidBrackets(expr);
        checkIsValidOperators(expr);
    }

    private static void checkIsVariableNames(String expr, int numVariables) {
        String pattern = "x\\d*";
        Pattern compiledPattern = Pattern.compile(pattern);
        Matcher matcher = compiledPattern.matcher(expr);
        while (matcher.find()) {
            int pos = matcher.start();
            String variableName = matcher.group();
            if (pos > 0 && !"+-*/(".contains("" + expr.charAt(pos - 1))) {
                throw new SemanticException("Find invalid symbol `%s` at position %s of `%s`",
                        expr.charAt(pos - 1) + variableName, pos, expr);
            }
            if (variableName.equals("x")) {
                throw new SemanticException(
                        "Variable name should be `x1` to `x%s`, but found `%s` at position %s of `%s`",
                        numVariables, variableName, pos + 1, expr);
            }
            int variableIdx = Integer.parseInt(variableName.substring(1));
            if (!(variableIdx >= 1 && variableIdx <= numVariables)) {
                throw new SemanticException(
                        "Variable name should be `x1` to `x%s`, but found `%s` at position %s of `%s`",
                        numVariables, variableName, pos + 1, expr);
            }
        }
    }

    private static void checkIsValidBrackets(String expr) {
        LinkedList<Integer> bracketStack = new LinkedList<>();
        byte[] bytes = expr.getBytes();
        for (int i = 0; i < bytes.length; ++i) {
            byte ch = bytes[i];
            if (ch == '(') {
                if (i > 0 && bytes[i - 1] == ')') {
                    throw new SemanticException(
                            "Cannot find a operator between right bracket and left bracket at position %s of `%s`",
                            i + 1, expr);
                }
                bracketStack.add(i);
            }
            if (ch == ')') {
                if (i > 0 && bytes[i - 1] == '(') {
                    throw new SemanticException(
                            "Cannot find a objects between left bracket and right bracket at position %s of `%s`",
                            i + 1, expr);
                }
                if (bracketStack.isEmpty()) {
                    throw new SemanticException(
                            "Cannot find a left bracket to match with the right bracket at position %s of `%s`", i + 1,
                            expr);
                }
                bracketStack.removeLast();
            }
        }
        if (!bracketStack.isEmpty()) {
            throw new SemanticException(
                    "Cannot find a right bracket to match with the left bracket at position %s of `%s`",
                    bracketStack.getFirst(), expr);
        }
    }

    private static void checkIsValidOperators(String expr) {
        String pattern = "[+\\-*/(][*/)]";
        Pattern compiledPattern = Pattern.compile(pattern);
        Matcher matcher = compiledPattern.matcher(expr);
        if (matcher.find()) {
            int pos = matcher.start();
            String ops = matcher.group();
            throw new SemanticException("Find invalid operator usage `%s` at position %s of `%s`", ops, pos + 1, expr);
        }

        pattern = "[+\\-*/][+-]";
        compiledPattern = Pattern.compile(pattern);
        matcher = compiledPattern.matcher(expr);
        if (matcher.find()) {
            int pos = matcher.start();
            String ops = matcher.group();
            throw new SemanticException(
                    "Find weird operators `%s` at position %s of `%s`, " +
                            "please wrap the signs and the objects they modify with brackets if possible.",
                    ops, pos + 1, expr);
        }

        if (expr.charAt(0) == '*' || expr.charAt(0) == '/') {
            throw new SemanticException("Find invalid operator usage `%s` at position %s of `%s`", expr.charAt(0), 1,
                    expr);
        }

        if ("+-*/".contains(expr.substring(expr.length() - 1))) {
            throw new SemanticException("Find invalid operator usage `%s` at position %s of `%s`", expr.charAt(0), 1,
                    expr);
        }
    }
}
