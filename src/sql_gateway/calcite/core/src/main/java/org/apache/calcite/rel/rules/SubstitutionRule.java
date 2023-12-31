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

/**
 * A rule that implements this interface indicates that the new RelNode
 * is typically better than the old one. All the substitution rules will
 * be executed first until they are done. The execution order of
 * substitution rules depends on the match order.
 */
public interface SubstitutionRule extends TransformationRule {

  /**
   * Whether the planner should automatically prune old node when
   * there is at least 1 equivalent rel generated by the rule.
   *
   * <p>Default is false, the user needs to prune the old node
   * manually in the rule.
   */
  default boolean autoPruneOld() {
    return false;
  }
}
