import pytest

from app.services import rag_service


def _point(knowledge_id, title, content, source_id):
    return {
        "knowledge_id": knowledge_id,
        "book_id": "H7A",
        "book_title": "义务教育教科书·中国历史 七年级上册",
        "unit_id": "H7A-U03",
        "unit_title": "秦汉时期",
        "lesson_id": "H7A-L09",
        "lesson_number": 9,
        "lesson_title": "秦统一中国",
        "section_title": "签核知识点",
        "title": title,
        "content": content,
        "summary": content,
        "keywords": [title],
        "people": [],
        "qualifiers": [],
        "certainty": "fragment_linked_repaired",
        "source_refs": [
            {
                "source_id": source_id,
                "fragment_id": source_id,
                "excerpt": content,
                "textbook_pages": "52",
                "pdf_pages": "57",
                "verification_status": "pdf_text_layer_lesson_page_content_match",
            }
        ],
    }


@pytest.fixture
def signed_qin_points():
    points = [
        _point(
            "L09-Q005",
            "秦灭六国的战略与过程",
            "秦王嬴政积极筹划统一大计，秦国采取各个击破的战略方针，先后攻灭六国。",
            "F-Q005",
        ),
        _point(
            "L09-Q009",
            "秦朝中央集权制度的建立",
            "秦朝创立了以皇帝为中心的中央集权制度。",
            "F-Q009",
        ),
        _point(
            "L09-Q016",
            "秦朝统一文字",
            "秦始皇命令李斯等人制定小篆，作为通用文字颁行全国。",
            "F-Q016",
        ),
        _point(
            "L09-Q017",
            "秦朝统一货币",
            "秦始皇以圆形方孔半两钱作为统一货币，在全国流通。",
            "F-Q017",
        ),
        _point(
            "L09-Q018",
            "秦朝统一度量衡",
            "朝廷以秦制为基础统一度量衡，促进经济发展。",
            "F-Q018",
        ),
        _point(
            "L09-Q019",
            "秦朝统一车轨与道路",
            "秦始皇下令统一车轨和道路宽度，并修筑贯通全国的道路。",
            "F-Q019",
        ),
        _point(
            "L09-Q021",
            "秦朝统一岭南及东南沿海地区",
            "秦始皇统一六国后，派兵统一岭南及东南沿海地区，设置郡县。",
            "F-Q021",
        ),
        _point(
            "L09-Q022",
            "蒙恬北击匈奴",
            "秦始皇派大将蒙恬北击匈奴，夺回被匈奴占领的地区。",
            "F-Q022",
        ),
    ]
    points.extend(
        [
            _point(
                "L10-Q001",
                "秦始皇统治的历史作用与暴政",
                "秦始皇完成统一，也实行了严酷统治。",
                "F-L10-Q001",
            ),
            _point(
                "L09-Q007",
                "秦朝疆域与都城",
                "秦朝定都咸阳，疆域辽阔。",
                "F-L09-Q007",
            ),
            _point(
                "L08-Q009",
                "扁鹊的医学成就",
                "扁鹊总结诊断方法，后人称为四诊法。",
                "F-L08-Q009",
            ),
            _point(
                "TEST-INVENTION",
                "古代技术发明",
                "教材介绍了古代技术发明。",
                "F-INVENTION",
            ),
            _point(
                "TEST-CONSTRUCTION",
                "道路修建",
                "古代劳动人民修建道路。",
                "F-CONSTRUCTION",
            ),
            _point(
                "TEST-COMPUTER",
                "现代技术辅助研究",
                "研究人员使用电子计算机辅助复原历史图景。",
                "F-COMPUTER",
            ),
        ]
    )
    return points


@pytest.mark.parametrize(
    ("question", "expected_id"),
    [
        ("秦灭六国采取了什么战略？", "L09-Q005"),
        ("秦朝建立了什么中央集权制度？", "L09-Q009"),
        ("秦始皇如何统一文字？", "L09-Q016"),
        ("秦始皇如何统一货币？", "L09-Q017"),
        ("秦朝如何统一度量衡？", "L09-Q018"),
        ("秦始皇如何统一车轨和道路？", "L09-Q019"),
        ("秦始皇统一六国后如何经营岭南？", "L09-Q021"),
        ("蒙恬北击匈奴有什么历史内容？", "L09-Q022"),
    ],
)
def test_signed_points_are_retrieved_by_specific_question(
    monkeypatch, signed_qin_points, question, expected_id
):
    monkeypatch.setattr(rag_service, "load_history_knowledge", lambda: signed_qin_points)

    results = rag_service._retrieve_keyword_history_knowledge(question)

    assert results[0]["knowledge_id"] == expected_id
    assert results[0]["source_refs"][0]["source_id"] == f"F-{expected_id[4:]}"


def test_historical_person_alias_matches_signed_content(monkeypatch, signed_qin_points):
    monkeypatch.setattr(rag_service, "load_history_knowledge", lambda: signed_qin_points)

    results = rag_service._retrieve_keyword_history_knowledge("嬴政如何统一文字？")

    assert results[0]["knowledge_id"] == "L09-Q016"


@pytest.mark.parametrize(
    "question",
    [
        "秦朝有哪些著名陶瓷器？",
        "火星上的秦朝车站由谁修建？",
        "秦始皇发明了量子计算机吗？",
    ],
)
def test_common_or_fictitious_terms_do_not_create_false_recall(
    monkeypatch, signed_qin_points, question
):
    monkeypatch.setattr(rag_service, "load_history_knowledge", lambda: signed_qin_points)

    assert rag_service._retrieve_keyword_history_knowledge(question) == []


def test_original_json_and_signed_points_remain_searchable_together(
    monkeypatch, signed_qin_points
):
    original = _point(
        "H7A-L01-010",
        "北京人用火",
        "北京人会使用火，还会长时间保存火种。",
        "H7A-P006-B",
    )
    monkeypatch.setattr(
        rag_service,
        "load_history_knowledge",
        lambda: [original, *signed_qin_points],
    )

    unit1 = rag_service._retrieve_keyword_history_knowledge("北京人用火有什么作用？")
    signed = rag_service._retrieve_keyword_history_knowledge("秦灭六国采取了什么战略？")

    assert unit1[0]["knowledge_id"] == "H7A-L01-010"
    assert signed[0]["knowledge_id"] == "L09-Q005"


def _signed_spring_autumn_points():
    return [
        _point(
            "L05-Q001",
            "春秋时期诸侯国设置县、郡加强地方控制",
            "一些诸侯国为了加强对地方的控制，设置县、郡，委派官员直接管理。",
            "H7A-L05-CAND-001-SRC-01",
        ),
        _point(
            "L05-Q002",
            '周王在名义上仍是天下"共主"',
            "周王在名义上仍是天下‘共主’，但缺乏经济、军事实力，权威明显削弱，渐渐失去对各地诸侯的控制。",
            "H7A-L05-CAND-002-SRC-01",
        ),
        _point(
            "L05-Q005",
            "春秋时期诸侯国林立",
            "春秋时期，诸侯国林立，其中比较强大的有十几个。",
            "H7A-L05-CAND-005-SRC-01",
        ),
        _point(
            "L05-Q006",
            "春秋诸侯争霸的影响",
            "春秋时期的诸侯争霸，导致许多中小诸侯国覆灭。",
            "H7A-L05-CAND-006-SRC-01",
        ),
    ]


def test_k05_01_keyword_retrieves_both_required_signed_facts(monkeypatch):
    points = _signed_spring_autumn_points()
    monkeypatch.setattr(rag_service, "load_history_knowledge", lambda: points)

    results = rag_service._retrieve_keyword_history_knowledge(
        '判断并说明理由：“春秋时期，周天子仍牢牢控制诸侯，诸侯争霸并未出现。”'
    )

    result_ids = [item["knowledge_id"] for item in results]
    assert "L05-Q002" in result_ids
    assert "L05-Q006" in result_ids


def test_zhou_king_title_alias_retrieves_zhou_royal_house_evidence(monkeypatch):
    points = _signed_spring_autumn_points()
    monkeypatch.setattr(rag_service, "load_history_knowledge", lambda: points)

    results = rag_service._retrieve_keyword_history_knowledge(
        "春秋时期周天子还能牢牢控制诸侯吗？"
    )

    assert results[0]["knowledge_id"] == "L05-Q002"
