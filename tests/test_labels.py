from zd_mast.labels import map_core_antibiotic


def test_core_antibiotic_mapping_is_exact() -> None:
    assert map_core_antibiotic("头孢他啶") == "ceftazidime"
    assert map_core_antibiotic("ceftazidime") == "ceftazidime"
    assert map_core_antibiotic("头孢他啶/阿维巴坦") == ""
    assert map_core_antibiotic("ceftazidime-avibactam") == ""


def test_known_cefepime_brand_alias_is_preserved() -> None:
    assert map_core_antibiotic("头孢吡肟") == "cefepime"
    assert map_core_antibiotic("头孢吡肟(马斯平）") == "cefepime"
