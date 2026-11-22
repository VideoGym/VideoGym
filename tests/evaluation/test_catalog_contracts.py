from evaluation.catalog.contracts import build_base_catalog


def test_build_base_catalog_resolves_all_families_groups_and_evaluation_tasks():
    catalog = build_base_catalog()
    assert catalog["totals"]["tasks"] == 246
    assert catalog["totals"]["families"] == 11
    assert len(catalog["evaluation"]["tasks"]) == 365
    assert [g["label"] for g in catalog["task_groups"]] == [
        "Single-Step Group",
        "Multi-Step Group",
    ]
