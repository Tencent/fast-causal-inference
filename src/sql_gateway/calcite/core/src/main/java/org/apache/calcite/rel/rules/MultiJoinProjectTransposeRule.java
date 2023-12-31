/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.calcite.rel.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptRuleOperand;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.tools.RelBuilderFactory;

import org.immutables.value.Value;

/**
 * MultiJoinProjectTransposeRule implements the rule for pulling
 * {@link org.apache.calcite.rel.logical.LogicalProject}s that are on top of a
 * {@link MultiJoin} and beneath a
 * {@link org.apache.calcite.rel.logical.LogicalJoin} so the
 * {@link org.apache.calcite.rel.logical.LogicalProject} appears above the
 * {@link org.apache.calcite.rel.logical.LogicalJoin}.
 *
 * <p>In the process of doing
 * so, also save away information about the respective fields that are
 * referenced in the expressions in the
 * {@link org.apache.calcite.rel.logical.LogicalProject} we're pulling up, as
 * well as the join condition, in the resultant {@link MultiJoin}s
 *
 * <p>For example, if we have the following sub-query:
 *
 * <blockquote><pre>{@code
 *   (select X.x1, Y.y1
 *    from X, Y
 *    where X.x2 = Y.y2 and X.x3 = 1 and Y.y3 = 2)}</pre></blockquote>
 *
 * <p>The {@link MultiJoin} associated with (X, Y) associates x1 with X and
 * y1 with Y. Although x3 and y3 need to be read due to the filters, they are
 * not required after the row scan has completed and therefore are not saved.
 * The join fields, x2 and y2, are also tracked separately.
 *
 * <p>Note that by only pulling up projects that are on top of
 * {@link MultiJoin}s, we preserve projections on top of row scans.
 *
 * <p>See the superclass for details on restrictions regarding which
 * {@link org.apache.calcite.rel.logical.LogicalProject}s cannot be pulled.
 *
 * @see CoreRules#MULTI_JOIN_BOTH_PROJECT
 * @see CoreRules#MULTI_JOIN_LEFT_PROJECT
 * @see CoreRules#MULTI_JOIN_RIGHT_PROJECT
 */
@Value.Enclosing
public class MultiJoinProjectTransposeRule extends JoinProjectTransposeRule {

  /** Creates a MultiJoinProjectTransposeRule. */
  protected MultiJoinProjectTransposeRule(Config config) {
    super(config);
  }

  @Deprecated // to be removed before 2.0
  public MultiJoinProjectTransposeRule(
      RelOptRuleOperand operand,
      String description) {
    this(ImmutableMultiJoinProjectTransposeRule.Config.of().withDescription(description)
        .withOperandSupplier(b -> b.exactly(operand)));
  }

  @Deprecated // to be removed before 2.0
  public MultiJoinProjectTransposeRule(
      RelOptRuleOperand operand,
      RelBuilderFactory relBuilderFactory,
      String description) {
    this(ImmutableMultiJoinProjectTransposeRule.Config.of().withDescription(description)
        .withRelBuilderFactory(relBuilderFactory)
        .withOperandSupplier(b -> b.exactly(operand)));
  }

  //~ Methods ----------------------------------------------------------------

  @Override protected boolean hasLeftChild(RelOptRuleCall call) {
    return call.rels.length != 4;
  }

  @Override protected boolean hasRightChild(RelOptRuleCall call) {
    return call.rels.length > 3;
  }

  @Override protected Project getRightChild(RelOptRuleCall call) {
    if (call.rels.length == 4) {
      return call.rel(2);
    } else {
      return call.rel(3);
    }
  }

  @Override protected RelNode getProjectChild(
      RelOptRuleCall call,
      Project project,
      boolean leftChild) {
    // locate the appropriate MultiJoin based on which rule was fired
    // and which projection we're dealing with
    MultiJoin multiJoin;
    if (leftChild) {
      multiJoin = call.rel(2);
    } else if (call.rels.length == 4) {
      multiJoin = call.rel(3);
    } else {
      multiJoin = call.rel(4);
    }

    // create a new MultiJoin that reflects the columns in the projection
    // above the MultiJoin
    return RelOptUtil.projectMultiJoin(multiJoin, project);
  }

  /** Rule configuration. */
  @Value.Immutable
  @SuppressWarnings("immutables:subtype")
  public interface Config extends JoinProjectTransposeRule.Config {
    Config BOTH_PROJECT = ImmutableMultiJoinProjectTransposeRule.Config.of()
        .withOperandSupplier(b0 ->
            b0.operand(LogicalJoin.class).inputs(
                b1 -> b1.operand(LogicalProject.class).oneInput(b2 ->
                    b2.operand(MultiJoin.class).anyInputs()),
                b3 -> b3.operand(LogicalProject.class).oneInput(b4 ->
                    b4.operand(MultiJoin.class).anyInputs())))
        .withDescription(
            "MultiJoinProjectTransposeRule: with two LogicalProject children");

    Config LEFT_PROJECT = ImmutableMultiJoinProjectTransposeRule.Config.of()
        .withOperandSupplier(b0 ->
            b0.operand(LogicalJoin.class).inputs(b1 ->
                b1.operand(LogicalProject.class).oneInput(b2 ->
                    b2.operand(MultiJoin.class).anyInputs())))
        .withDescription(
            "MultiJoinProjectTransposeRule: with LogicalProject on left");

    Config RIGHT_PROJECT = ImmutableMultiJoinProjectTransposeRule.Config.of()
        .withOperandSupplier(b0 ->
            b0.operand(LogicalJoin.class).inputs(
                b1 -> b1.operand(RelNode.class).anyInputs(),
                b2 -> b2.operand(LogicalProject.class).oneInput(b3 ->
                    b3.operand(MultiJoin.class).anyInputs())))
        .withDescription(
            "MultiJoinProjectTransposeRule: with LogicalProject on right");


    @Override default MultiJoinProjectTransposeRule toRule() {
      return new MultiJoinProjectTransposeRule(this);
    }
  }
}
