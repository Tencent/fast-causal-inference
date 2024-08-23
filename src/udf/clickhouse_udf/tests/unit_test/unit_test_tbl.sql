DROP TABLE IF EXISTS causal_inference_test;

CREATE TABLE causal_inference_test
(
    `treatment` UInt8,
    `numerator` Float64,
    `denominator` UInt8,
    `numerator_pre` Int64,
    `denominator_pre` UInt8,
    `Y` Float64,
    `X1` Int32,
    `X2` Int32,
    `X3` Int32,
    `X3_string` String,
    `X7_needcut` Int64,
    `X8_needcut` Int64,
    `weight` Float64,
    `distance` Float64
)
ENGINE = MergeTree()
ORDER by numerator
