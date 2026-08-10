from __future__ import annotations

import pytest

from kernelvision.benchmarking.ncu import parse_ncu_csv, parse_ncu_rules


def test_parse_ncu_csv_skips_profiler_preamble() -> None:
    output = """==PROF== Connected to process 42
"ID","Section Name","Metric Name","Metric Unit","Metric Value"
"1","Launch Statistics","Block Size","thread","256"
"1","Launch Statistics","Registers Per Thread","register/thread","12"
"""

    rows = parse_ncu_csv(output)

    assert rows == [
        {
            "ID": "1",
            "Section Name": "Launch Statistics",
            "Metric Name": "Block Size",
            "Metric Unit": "thread",
            "Metric Value": "256",
        },
        {
            "ID": "1",
            "Section Name": "Launch Statistics",
            "Metric Name": "Registers Per Thread",
            "Metric Unit": "register/thread",
            "Metric Value": "12",
        },
    ]


def test_parse_ncu_csv_requires_metric_table() -> None:
    with pytest.raises(ValueError, match="did not contain a metric table"):
        parse_ncu_csv("==ERROR== no profiler table")


def test_parse_ncu_rules_keeps_profiler_guidance_separate() -> None:
    output = '''"ID","Section Name","Metric Name","Metric Unit","Metric Value","Rule Name","Rule Type","Rule Description","Estimated Speedup Type","Estimated Speedup"
"1","Memory Workload Analysis","L2 Hit Rate","%","99.76","","","","",""
"1","MemoryWorkloadAnalysis_Tables","","","","MemoryCacheAccessPattern","OPT","Only 10.7 of 32 bytes are utilized.","global","34.6"
'''

    rules = parse_ncu_rules(output)

    assert len(rules) == 1
    assert rules[0]["Rule Name"] == "MemoryCacheAccessPattern"
    assert rules[0]["Rule Type"] == "OPT"
    assert rules[0]["Estimated Speedup"] == "34.6"
    assert "10.7 of 32 bytes" in rules[0]["Rule Description"]


def test_parse_ncu_rules_allows_no_recommendations() -> None:
    output = '''"ID","Section Name","Metric Name","Metric Unit","Metric Value","Rule Name"
"1","Launch Statistics","Block Size","thread","256",""
'''

    assert parse_ncu_rules(output) == []
