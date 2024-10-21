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

#include "exprs/helpers/expr_tree.hpp"

#include <gtest/gtest.h>

namespace starrocks {

class ExprTreeTest : public testing::Test {
public:
    ExprTreeTest() = default;

    void SetUp() override {}

    void TearDown() override {}

private:
};

TEST_F(ExprTreeTest, ValueTest) {
    ExprTree<double> expr_tree;
    ASSERT_TRUE(expr_tree.init("-x/(y*3)", {{"x", 0}, {"y", 1}}));
    ASSERT_EQ(expr_tree.value({3, 2}), -0.5);
}

TEST_F(ExprTreeTest, PDValueTest) {
    ExprTree<double> expr_tree;
    ASSERT_TRUE(expr_tree.init("-x/(y*3)", {{"x", 0}, {"y", 1}}));
    ASSERT_EQ(expr_tree.pdvalue({3, 2})[1], 0.25);
}

} // namespace starrocks
