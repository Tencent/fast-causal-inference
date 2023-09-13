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
package org.apache.calcite.sql.olap;

public class SqlForwardUtil {
  public static String exchangeFunc(String func, int offset) {
    int max_index = 0;
    for (int i = 1; ; i++) {
      if (!func.contains("x" + Integer.toString(i))) {
        max_index = i - 1;
        break;
      }
    }
    for (int i = max_index; i >= 0; i--) {
      func = func.replaceAll("x" + Integer.toString(i), "x" + Integer.toString(i + offset));
    }
    return func;
  }

  public static String exchangIdentity(String str) {
    str = str.trim();
    // remove the first and last char if it is ' or " or `
    if (str.length() > 1) {
      if (str.charAt(0) == str.charAt(str.length() - 1) && (
          str.charAt(0) == '\"' ||
          str.charAt(0) == '\'')) {
        str = str.substring(1, str.length() - 1);
      }
    }
    return str.replaceAll("`", "");
  }
}
