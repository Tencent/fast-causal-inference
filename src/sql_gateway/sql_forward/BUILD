cc_library(
    name = "sqlforward",
    srcs = [
        "sql_parser.cc",
    ],
    hdrs = [
        "sql_parser.h",
    ],
    includes = ['.'],
    deps = [
        "//mm3rd/boost-1.66:boost_math_c99",
        "//mm3rd/boost-1.66:boost_math_c99f",
        "//mm3rd/boost-1.66:boost_math_c99l",
        "//mm3rd/boost-1.66:boost_math_tr1",
        "//mm3rd/boost-1.66:boost_math_tr1f",
        "//mm3rd/boost-1.66:boost_math_tr1l",
    ],
    copts = [
        "-std=c++11",
        "-Wno-float-equal",
        #"-DDEBUG"
    ],
    linkopts = [
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "unit_test",
    srcs = [
        "unit_test.cc",
    ],
    includes = ['.'],
    deps = [
      "//mmbd/mmexptimpl/mmexptgate/sqlforward:sqlforward",
      "//mm3rd/gtest:gtest",
    ],
    copts = [
        "-std=c++11",
        "-Wno-float-equal",
    ],
    linkopts = [
    ],
    visibility = ["//visibility:public"],
)
