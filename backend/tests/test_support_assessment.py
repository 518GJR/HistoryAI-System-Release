import pytest

from app.services import rag_service
from app.services.knowledge_service import assemble_textbook_answer, build_knowledge_outline
from app.services.support_assessment import (
    _evidence_supports_subject_aspect,
    analyze_question_requirements,
    assess_support,
    evidence_covers_assessment,
    supporting_answer_items,
)


def _textbook(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_MODE", "textbook")
    monkeypatch.setenv("RETRIEVAL_MODE", "keyword")
    return rag_service.load_textbook_knowledge_from_json()


def test_linked_excerpt_stays_complete_for_experience_answer_and_source_metadata():
    excerpt = (
        "老子和《道德经》老子姓李，名耳，字聃（dn），春秋后期楚国人。"
        "他做过周朝的史官，掌管王室的典籍，学识渊博。"
        "老子思想的核心是‘道’。"
    )
    point = {
        "knowledge_id": "L07-Q001",
        "title": "老子的生平及其主要经历",
        "summary": "老子姓李，名耳，字聃，春秋后期楚国人。",
        "keywords": ["老子"],
        "people": [],
    }
    candidate = {
        **point,
        "knowledge_ids": ["L07-Q001"],
        "retrieval_mode": "keyword",
        "retrieval_unit": "knowledge_point",
        "source_refs": [
            {
                "source_id": "H7A-L07-P039-PDF-FF73662D1D1A",
                "material_type": "教材PDF片段",
                "textbook_pages": "39",
                "pdf_pages": "44",
                "excerpt": excerpt,
            }
        ],
    }
    question = "老子的生平及其主要经历是什么？"

    assessment = assess_support(question, "keyword", [candidate], [point])
    answer_items = supporting_answer_items([candidate], assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, [point]
    )
    outline = build_knowledge_outline(question, answer_items)
    source = rag_service.knowledge_item_to_source(candidate)

    assert assessment.status == "sufficient"
    assert "aspect:process" in assessment.covered_facets
    assert assessment.unsupported_facets == []
    assert mode == "controlled_evidence"
    assert "老子姓李，名耳，字聃" in answer
    assert "春秋后期楚国人" in answer
    assert "做过周朝的史官" in answer
    assert "掌管王室的典籍" in answer
    assert "学识渊博" in answer
    assert "老子思想的核心" not in answer
    assert used_evidence[0]["text"] == excerpt
    assert outline["event"] == ""
    assert outline["process"] == excerpt
    assert source["material_types"] == ["教材PDF片段"]
    assert source["textbook_pages"] == "39"
    assert source["pdf_pages"] == "44"


def test_unrelated_nearest_neighbors_are_not_treated_as_support(monkeypatch):
    points = _textbook(monkeypatch)
    point_map = {point["knowledge_id"]: point for point in points}
    candidates = []
    for knowledge_id in ("H7A-L02-013", "H7A-L03-015", "H7A-L02-010"):
        point = point_map[knowledge_id]
        candidate = dict(point)
        candidate.update(
            {
                "knowledge_ids": [knowledge_id],
                "retrieval_mode": "vector",
                "retrieval_unit": "source_excerpt",
                "score": 0.5,
            }
        )
        candidates.append(candidate)

    assessment = assess_support(
        "秦始皇统一六国采取了哪些措施？", "vector", candidates, points
    )

    assert assessment.status == "insufficient"
    assert assessment.reason_codes == ["no_question_anchor"]
    assert assessment.supporting_source_ids == []


def test_normal_fact_and_paraphrase_are_not_rejected(monkeypatch):
    points = _textbook(monkeypatch)
    questions = (
        "北京人用火有什么作用？",
        "掌握火给北京人的生存带来了哪些好处？",
        "仰韶文化分布在哪里，距今多少年？",
    )

    for question in questions:
        candidates = rag_service.retrieve_history_knowledge(question)
        assessment = assess_support(question, "keyword", candidates, points)
        assert assessment.status == "sufficient", question
        assert assessment.unsupported_facets == [], question
        assert assessment.supporting_source_ids, question


def test_comparison_requires_both_named_sides(monkeypatch):
    points = _textbook(monkeypatch)
    candidates = rag_service.retrieve_history_knowledge(
        "北京人与山顶洞人在工具制作方面有什么不同？"
    )
    only_beijing = [
        item for item in candidates if item["knowledge_id"] == "H7A-L01-009"
    ]

    assessment = assess_support(
        "北京人与山顶洞人在工具制作方面有什么不同？",
        "keyword",
        only_beijing,
        points,
    )

    assert assessment.status == "partial"
    assert "subject:山顶洞人" in assessment.unsupported_facets
    assert "comparison:all_subjects" in assessment.unsupported_facets


def test_used_evidence_must_cover_every_comparison_facet(monkeypatch):
    points = _textbook(monkeypatch)
    question = "北京人与山顶洞人在工具制作方面有什么不同？"
    candidates = rag_service.retrieve_history_knowledge(question)
    assessment = assess_support(question, "keyword", candidates, points)

    assert assessment.status == "sufficient"
    assert not evidence_covers_assessment(
        assessment,
        [{"knowledge_ids": ["H7A-L01-009"]}],
    )
    assert evidence_covers_assessment(
        assessment,
        [
            {"knowledge_ids": ["H7A-L01-009"]},
            {"knowledge_ids": ["H7A-L01-013"]},
        ],
    )


def test_answer_input_contains_only_supporting_verbatim_sources(monkeypatch):
    points = _textbook(monkeypatch)
    question = "北京人用火有什么作用？"
    candidates = rag_service.retrieve_history_knowledge(question)
    assessment = assess_support(question, "keyword", candidates, points)

    answer_items = supporting_answer_items(candidates, assessment)

    assert [item["knowledge_id"] for item in answer_items] == ["H7A-L01-010"]
    assert answer_items[0]["event"] == answer_items[0]["source_refs"][0]["excerpt"]
    assert answer_items[0]["content"] == answer_items[0]["source_refs"][0]["excerpt"]
    assert answer_items[0]["reason"] == ""
    assert answer_items[0]["impact"] == ""


def _k05_01_evidence():
    records = [
        (
            "L05-Q001",
            "春秋时期诸侯国设置县、郡加强地方控制",
            "一些诸侯国为了加强对地方的控制，设置县、郡，委派官员直接管理。",
            "H7A-L05-CAND-001-SRC-01",
        ),
        (
            "L05-Q002",
            '周王在名义上仍是天下"共主"',
            "周王在名义上仍是天下‘共主’，但缺乏经济、军事实力，权威明显削弱，渐渐失去对各地诸侯的控制。",
            "H7A-L05-CAND-002-SRC-01",
        ),
        (
            "L05-Q006",
            "春秋诸侯争霸的影响",
            "春秋时期的诸侯争霸，导致许多中小诸侯国覆灭，少数强大诸侯国的疆域不断扩展。",
            "H7A-L05-CAND-006-SRC-01",
        ),
    ]
    points = []
    candidates = []
    for knowledge_id, title, excerpt, source_id in records:
        point = {
            "knowledge_id": knowledge_id,
            "title": title,
            "summary": excerpt,
            "content": excerpt,
            "keywords": [title],
            "people": [],
        }
        candidate = {
            **point,
            "knowledge_ids": [knowledge_id],
            "retrieval_mode": "keyword",
            "retrieval_unit": "knowledge_point",
            "source_refs": [
                {
                    "source_id": source_id,
                    "fragment_id": source_id,
                    "excerpt": excerpt,
                    "textbook_pages": "28" if knowledge_id == "L05-Q002" else "29",
                    "pdf_pages": "33" if knowledge_id == "L05-Q002" else "34",
                    "verification_status": "inherited_existing_retrieval_evidence",
                }
            ],
        }
        points.append(point)
        candidates.append(candidate)
    return points, candidates


K05_01_QUESTION = '判断并说明理由：“春秋时期，周天子仍牢牢控制诸侯，诸侯争霸并未出现。”'


def test_judgment_instruction_does_not_create_result_impact_facet():
    points, _ = _k05_01_evidence()

    requirements = analyze_question_requirements(K05_01_QUESTION, points)

    assert "result_impact" not in requirements["aspects"]
    assert requirements["aspects"] == ["content"]


def test_historical_explanation_question_still_creates_result_impact_facet():
    points, _ = _k05_01_evidence()

    requirements = analyze_question_requirements("这一现象说明什么？", points)

    assert "result_impact" in requirements["aspects"]


def _signed_support_case(knowledge_id, title, excerpt, source_id):
    point = {
        "knowledge_id": knowledge_id,
        "title": title,
        "summary": excerpt,
        "content": excerpt,
        "keywords": [title],
        "people": [],
    }
    candidate = {
        **point,
        "knowledge_ids": [knowledge_id],
        "retrieval_mode": "keyword",
        "retrieval_unit": "knowledge_point",
        "source_refs": [
            {
                "source_id": source_id,
                "fragment_id": source_id,
                "excerpt": excerpt,
                "textbook_pages": "1",
                "pdf_pages": "1",
                "verification_status": "inherited_existing_retrieval_evidence",
            }
        ],
    }
    return point, candidate


def test_k06_01_parenthetical_title_still_provides_question_anchor():
    point, candidate = _signed_support_case(
        "L06-Q019",
        "战国七雄（事实补充）",
        "战国七雄分别是齐、楚、燕、韩、赵、魏、秦。",
        "H7A-L06-CAND-019-SRC-01",
    )

    assessment = assess_support(
        "战国七雄指哪七个诸侯国？", "keyword", [candidate], [point]
    )

    assert assessment.status == "sufficient"
    assert assessment.reason_codes == ["all_required_facets_covered"]
    assert assessment.supporting_knowledge_ids == ["L06-Q019"]


def test_k06_01_selects_relevant_nonfirst_fragment_from_same_knowledge_point():
    point, candidate = _signed_support_case(
        "L06-Q019",
        "战国七雄（事实补充）",
        "unused",
        "unused",
    )
    candidate["source_refs"] = [
        {
            "source_id": "H7A-L06-CAND-002-SRC-01",
            "fragment_id": "H7A-L06-CAND-002-SRC-01",
            "excerpt": "战国时期，诸侯们各自为政，发展实力，致力于拓展疆域。",
            "textbook_pages": "33",
            "pdf_pages": "38",
        },
        {
            "source_id": "H7A-L06-CAND-019-SRC-01",
            "fragment_id": "H7A-L06-CAND-019-SRC-01",
            "excerpt": "齐、楚、燕、韩、赵、魏、秦七国的势力较强，史称“战国七雄”。",
            "textbook_pages": "33",
            "pdf_pages": "38",
        },
    ]
    question = "战国七雄指哪七个诸侯国？"

    assessment = assess_support(question, "keyword", [candidate], [point])
    answer_items = supporting_answer_items([candidate], assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, [point]
    )

    assert assessment.status == "sufficient"
    assert assessment.supporting_source_ids == ["H7A-L06-CAND-019-SRC-01"]
    assert "齐、楚、燕、韩、赵、魏、秦" in answer
    assert "各自为政" not in answer
    assert mode == "controlled_evidence"
    assert [item["source_ids"] for item in used_evidence] == [
        ["H7A-L06-CAND-019-SRC-01"]
    ]


def test_k06_01_unrelated_fragment_cannot_make_content_sufficient():
    point, candidate = _signed_support_case(
        "L06-Q019",
        "战国七雄（事实补充）",
        "战国时期，诸侯们各自为政，发展实力，致力于拓展疆域。",
        "H7A-L06-CAND-002-SRC-01",
    )

    assessment = assess_support(
        "战国七雄指哪七个诸侯国？", "keyword", [candidate], [point]
    )

    assert assessment.status == "insufficient"
    assert "aspect:content" in assessment.unsupported_facets
    assert assessment.supporting_source_ids == []


def test_k08_01_core_entity_in_long_titles_provides_question_anchor():
    definition = _signed_support_case(
        "L08-Q013",
        "商朝甲骨文的产生",
        "商朝人刻写在龟甲和兽骨上的文字，被称为甲骨文。",
        "H7A-L08-CAND-013-SRC-01",
    )
    status = _signed_support_case(
        "L08-Q016",
        "甲骨文的文字结构及历史地位",
        "甲骨文已经具备汉字的基本结构，是我国已发现的古代文字中年代最早、体系较为完整的文字。",
        "H7A-L08-CAND-016-SRC-01",
    )
    points = [definition[0], status[0]]
    candidates = [definition[1], status[1]]

    assessment = assess_support(
        "甲骨文是什么？它在我国文字史上的地位如何？",
        "keyword",
        candidates,
        points,
    )

    assert assessment.status == "sufficient"
    assert "no_question_anchor" not in assessment.reason_codes
    assert "aspect:definition" in assessment.covered_facets
    assert "aspect:historical_status" in assessment.covered_facets
    assert set(assessment.supporting_knowledge_ids) == {"L08-Q013", "L08-Q016"}

    answer_items = supporting_answer_items(candidates, assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        "甲骨文是什么？它在我国文字史上的地位如何？",
        answer_items,
        assessment,
        points,
    )
    assert mode == "controlled_evidence"
    assert "龟甲和兽骨上的文字" in answer
    assert "年代最早、体系较为完整" in answer
    assert {item["knowledge_id"] for item in used_evidence} == {
        "L08-Q013",
        "L08-Q016",
    }


def test_k08_01_status_only_keeps_evidence_and_reports_missing_definition():
    definition = _signed_support_case(
        "L08-Q013",
        "商朝甲骨文的产生",
        "商朝人刻写在龟甲和兽骨上的文字，被称为甲骨文。",
        "H7A-L08-CAND-013-SRC-01",
    )
    status = _signed_support_case(
        "L08-Q016",
        "甲骨文的文字结构及历史地位",
        "甲骨文已经具备汉字的基本结构，是我国已发现的古代文字中年代最早、体系较为完整的文字。",
        "H7A-L08-CAND-016-SRC-01",
    )
    question = "甲骨文是什么？它在我国文字史上的地位如何？"

    assessment = assess_support(
        question, "vector", [status[1]], [definition[0], status[0]]
    )
    answer_items = supporting_answer_items([status[1]], assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, [definition[0], status[0]]
    )

    assert assessment.status == "partial"
    assert "aspect:historical_status" in assessment.covered_facets
    assert "aspect:definition" in assessment.unsupported_facets
    assert assessment.supporting_knowledge_ids == ["L08-Q016"]
    assert "年代最早、体系较为完整" in answer
    assert "证据不足" in answer and "定义" in answer
    assert mode == "controlled_partial"
    assert [item["knowledge_id"] for item in used_evidence] == ["L08-Q016"]


def test_k08_wrong_definition_label_does_not_adopt_same_source_fragment():
    content = _signed_support_case(
        "L08-Q015",
        "甲骨文记载的主要内容",
        "甲骨文记载的内容涉及祭祀、战争、农牧业、官制、刑法和医药等。",
        "H7A-U02-L08-P045-CAND",
    )
    content[1]["source_refs"][0]["fragment_id"] = "H7A-L08-CAND-010-SRC-01"
    content[1]["matched_subquestions"] = ["definition"]
    status = _signed_support_case(
        "L08-Q016",
        "甲骨文的文字结构及历史地位",
        "甲骨文已经具备汉字的基本结构，是我国已发现的古代文字中年代最早、体系较为完整的文字。",
        "H7A-U02-L08-P045-CAND",
    )
    status[1]["source_refs"][0]["fragment_id"] = "H7A-L08-CAND-011-SRC-01"
    status[1]["matched_subquestions"] = ["historical_status", "definition"]
    question = "甲骨文是什么？它在我国文字史上的地位如何？"
    points = [content[0], status[0]]
    candidates = [content[1], status[1]]

    assessment = assess_support(question, "vector", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)

    assert assessment.status == "partial"
    assert "aspect:definition" in assessment.unsupported_facets
    assert assessment.supporting_knowledge_ids == ["L08-Q016"]
    assert [item["knowledge_id"] for item in answer_items] == ["L08-Q016"]


def test_k13_01_compound_titles_anchor_founding_and_reign_questions():
    founding = _signed_support_case(
        "L13-Q003",
        "东汉建立与刘秀完成全国统一",
        "25年，刘秀称帝，后定都洛阳，史称东汉。",
        "H7A-L13-P073-PDF-3E49405B7CE8",
    )
    prosperity = _signed_support_case(
        "L13-Q005",
        "“光武中兴”的表现",
        "刘秀统治后期，社会出现比较安定的局面，经济得到恢复和发展，史称“光武中兴”。",
        "H7A-L13-CAND-005-SRC-01",
    )
    points = [founding[0], prosperity[0]]
    candidates = [founding[1], prosperity[1]]

    assessment = assess_support(
        "东汉建立于哪一年？建立者是谁？“光武中兴”指什么局面？",
        "keyword",
        candidates,
        points,
    )

    assert assessment.status == "sufficient"
    assert "no_question_anchor" not in assessment.reason_codes
    assert "aspect:time" in assessment.covered_facets
    assert "aspect:founding_agent" in assessment.covered_facets
    assert set(assessment.supporting_knowledge_ids) == {"L13-Q003", "L13-Q005"}

    answer_items = supporting_answer_items(candidates, assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        "东汉建立于哪一年？建立者是谁？“光武中兴”指什么局面？",
        answer_items,
        assessment,
        points,
    )
    assert "25年，刘秀称帝" in answer
    assert "经济得到恢复和发展" in answer
    assert "社会出现比较安定的局面" in answer
    assert mode == "controlled_evidence"
    assert {item["knowledge_id"] for item in used_evidence} == {
        "L13-Q003",
        "L13-Q005",
    }


def test_k13_01_prosperity_only_keeps_supported_subanswer():
    founding = _signed_support_case(
        "L13-Q003",
        "东汉建立与刘秀完成全国统一",
        "25年，刘秀称帝，后定都洛阳，史称东汉。",
        "H7A-L13-P073-PDF-3E49405B7CE8",
    )
    prosperity = _signed_support_case(
        "L13-Q005",
        "“光武中兴”的表现",
        "到光武帝统治后期，经济得到恢复和发展，社会出现比较安定的局面，史称“光武中兴”。",
        "H7A-L13-CAND-008-SRC-01",
    )
    question = "东汉建立于哪一年？建立者是谁？“光武中兴”指什么局面？"

    assessment = assess_support(
        question, "vector", [prosperity[1]], [founding[0], prosperity[0]]
    )
    answer_items = supporting_answer_items([prosperity[1]], assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, [founding[0], prosperity[0]]
    )

    assert assessment.status == "partial"
    assert "subject:光武中兴" in assessment.covered_facets
    assert "subject:东汉建立" in assessment.unsupported_facets
    assert assessment.supporting_knowledge_ids == ["L13-Q005"]
    assert "经济得到恢复和发展" in answer
    assert "证据不足" in answer and "东汉建立" in answer
    assert mode == "controlled_partial"
    assert [item["knowledge_id"] for item in used_evidence] == ["L13-Q005"]


def test_k13_western_han_founding_labels_do_not_cover_eastern_han():
    eastern = _signed_support_case(
        "L13-Q003",
        "东汉建立与刘秀完成全国统一",
        "25年，刘秀称帝，后定都洛阳，史称东汉。",
        "H7A-U03-L13-P073-CAND",
    )
    western = _signed_support_case(
        "L11-Q005",
        "西汉建立及都城",
        "公元前202年，刘邦建立汉朝，后定都长安，史称西汉。",
        "H7A-U03-L11-P063-CAND",
    )
    western[1]["matched_subquestions"] = ["time", "founding_agent"]
    prosperity = _signed_support_case(
        "L13-Q005",
        "“光武中兴”的表现",
        "到光武帝统治后期，经济得到恢复和发展，社会出现比较安定的局面，史称“光武中兴”。",
        "H7A-U03-L13-P074-CAND",
    )
    question = "东汉建立于哪一年？建立者是谁？“光武中兴”指什么局面？"
    points = [eastern[0], western[0], prosperity[0]]
    candidates = [western[1], prosperity[1]]

    assessment = assess_support(question, "vector", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)

    assert assessment.status == "partial"
    assert "subject:东汉建立" in assessment.unsupported_facets
    assert "aspect:time" in assessment.unsupported_facets
    assert "aspect:founding_agent" in assessment.unsupported_facets
    assert assessment.supporting_knowledge_ids == ["L13-Q005"]
    assert [item["knowledge_id"] for item in answer_items] == ["L13-Q005"]


def test_k13_other_eastern_han_events_do_not_cover_founding_facets():
    founding = _signed_support_case(
        "L13-Q003",
        "东汉建立与刘秀完成全国统一",
        "25年，刘秀称帝，后定都洛阳，史称东汉。",
        "H7A-U03-L13-P073-CAND",
    )
    western_regions = _signed_support_case(
        "L14-Q018",
        "东汉设置西域长史府",
        "123年，东汉改西域都护府为西域长史府，继续管理西域。",
        "H7A-U03-L14-P082-CAND",
    )
    powerful_families = _signed_support_case(
        "L13-Q011",
        "东汉豪强大族势力膨胀及其影响",
        "曾帮助刘秀建立东汉政权的豪强大族，此时势力不断壮大。",
        "H7A-U03-L13-P075-CAND",
    )
    prosperity = _signed_support_case(
        "L13-Q005",
        "“光武中兴”的表现",
        "到光武帝统治后期，经济得到恢复和发展，社会出现比较安定的局面，史称“光武中兴”。",
        "H7A-U03-L13-P074-CAND",
    )
    western_regions[1]["matched_subquestions"] = ["time"]
    powerful_families[1]["matched_subquestions"] = ["founding_agent"]
    question = "东汉建立于哪一年？建立者是谁？“光武中兴”指什么局面？"
    points = [founding[0], western_regions[0], powerful_families[0], prosperity[0]]
    candidates = [western_regions[1], powerful_families[1], prosperity[1]]

    assessment = assess_support(question, "vector", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, points
    )

    assert assessment.status == "partial"
    assert "subject:东汉建立" in assessment.unsupported_facets
    assert "aspect:time" in assessment.unsupported_facets
    assert "aspect:founding_agent" in assessment.unsupported_facets
    assert assessment.supporting_knowledge_ids == ["L13-Q005"]
    assert [item["knowledge_id"] for item in answer_items] == ["L13-Q005"]
    assert [item["knowledge_id"] for item in used_evidence] == ["L13-Q005"]
    assert "经济得到恢复和发展" in answer
    assert "证据不足" in answer
    assert "123年" not in answer
    assert "豪强大族" not in answer
    assert mode == "controlled_partial"


@pytest.mark.parametrize(
    ("excerpt", "expected"),
    [
        ("25年，刘秀称帝，后定都洛阳，史称东汉。", True),
        ("公元25年，刘秀建立东汉政权。", True),
        ("东汉于公元25年建立。", True),
        ("公元25年建立东汉政权。", True),
        ("123年，东汉改西域都护府为西域长史府。", False),
        ("184年，东汉爆发黄巾起义。", False),
        ("刘秀经过10多年征战后称帝，史称东汉。", False),
        ("公元前221年，秦统一六国。", False),
    ],
)
def test_founding_time_requires_year_to_describe_the_founding_event(
    excerpt, expected
):
    assert (
        _evidence_supports_subject_aspect("东汉建立", "time", excerpt) is expected
    )


@pytest.mark.parametrize(
    ("excerpt", "expected"),
    [
        ("刘秀建立东汉政权。", True),
        ("东汉政权由刘秀建立。", True),
        ("25年，刘秀称帝，后定都洛阳，史称东汉。", True),
        ("曾帮助刘秀建立东汉政权的豪强大族，此时势力不断壮大。", False),
        ("公元前202年，刘邦建立汉朝，后定都长安，史称西汉。", False),
    ],
)
def test_founding_agent_requires_direct_actor_event_relation(excerpt, expected):
    assert (
        _evidence_supports_subject_aspect(
            "东汉建立", "founding_agent", excerpt
        )
        is expected
    )


def test_overlapping_same_source_evidence_is_not_repeated_in_answer():
    short = _signed_support_case(
        "L13-Q005",
        "“光武中兴”的表现",
        "到光武帝统治后期，经济得到恢复和发展，社会出现比较安定的局面，史称“光武中兴”。",
        "H7A-U03-L13-P074-CAND",
    )
    long = _signed_support_case(
        "L13-Q006",
        "东汉农业生产的发展",
        "到光武帝统治后期，经济得到恢复和发展，社会出现比较安定的局面，史称“光武中兴”。东汉时期，社会经济继续发展。",
        "H7A-U03-L13-P074-CAND",
    )
    question = "“光武中兴”指什么局面？"
    points = [short[0], long[0]]
    candidates = [short[1], long[1]]

    assessment = assess_support(question, "vector", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, points
    )

    repeated = "到光武帝统治后期，经济得到恢复和发展，社会出现比较安定的局面，史称“光武中兴”。"
    assert answer.count(repeated) == 1
    assert "东汉时期，社会经济继续发展。" in answer
    assert {source_id for item in used_evidence for source_id in item["source_ids"]} == {
        "H7A-U03-L13-P074-CAND"
    }
    assert mode == "controlled_evidence"


def test_k05_01_requires_zhou_authority_and_warlord_rivalry_independently():
    points, candidates = _k05_01_evidence()
    by_id = {item["knowledge_id"]: item for item in candidates}

    zhou_only = assess_support(K05_01_QUESTION, "keyword", [by_id["L05-Q002"]], points)
    rivalry_only = assess_support(K05_01_QUESTION, "keyword", [by_id["L05-Q006"]], points)
    complete = assess_support(
        K05_01_QUESTION,
        "keyword",
        [by_id["L05-Q002"], by_id["L05-Q006"]],
        points,
    )

    assert zhou_only.status == "partial"
    assert rivalry_only.status == "partial"
    assert complete.status == "sufficient"
    assert "subject:周天子" in zhou_only.covered_facets
    assert "subject:诸侯争霸" in zhou_only.unsupported_facets
    assert "subject:诸侯争霸" in rivalry_only.covered_facets
    assert "subject:周天子" in rivalry_only.unsupported_facets
    assert set(complete.supporting_knowledge_ids) == {"L05-Q002", "L05-Q006"}


def test_k05_01_unrelated_local_control_cannot_replace_zhou_authority():
    points, candidates = _k05_01_evidence()
    by_id = {item["knowledge_id"]: item for item in candidates}

    assessment = assess_support(
        K05_01_QUESTION,
        "keyword",
        [by_id["L05-Q001"], by_id["L05-Q006"]],
        points,
    )

    assert assessment.status == "partial"
    assert "subject:周天子" in assessment.unsupported_facets
    assert "L05-Q001" not in assessment.supporting_knowledge_ids


def test_k05_01_complete_answer_uses_only_both_signed_excerpts_and_marks_false():
    points, candidates = _k05_01_evidence()
    by_id = {item["knowledge_id"]: item for item in candidates}
    selected = [by_id["L05-Q002"], by_id["L05-Q006"]]
    assessment = assess_support(K05_01_QUESTION, "keyword", selected, points)
    answer_items = supporting_answer_items(selected, assessment)

    answer, mode, used_evidence = assemble_textbook_answer(
        K05_01_QUESTION, answer_items, assessment, points
    )

    assert mode == "controlled_evidence"
    assert "该说法错误" in answer
    assert "权威明显削弱" in answer
    assert "春秋时期的诸侯争霸" in answer
    assert {item["knowledge_id"] for item in used_evidence} == {
        "L05-Q002",
        "L05-Q006",
    }
    assert {source_id for item in used_evidence for source_id in item["source_ids"]} == {
        "H7A-L05-CAND-002-SRC-01",
        "H7A-L05-CAND-006-SRC-01",
    }


def test_k05_01_partial_evidence_never_outputs_complete_judgment():
    points, candidates = _k05_01_evidence()
    rivalry = next(item for item in candidates if item["knowledge_id"] == "L05-Q006")
    assessment = assess_support(K05_01_QUESTION, "keyword", [rivalry], points)
    answer_items = supporting_answer_items([rivalry], assessment)

    answer, mode, used_evidence = assemble_textbook_answer(
        K05_01_QUESTION, answer_items, assessment, points
    )

    assert mode == "controlled_partial"
    assert "该说法错误" not in answer
    assert "证据不足" in answer
    assert {item["knowledge_id"] for item in used_evidence} == {"L05-Q006"}


def test_q006_state_wording_does_not_create_process_facet(monkeypatch):
    points = _textbook(monkeypatch)
    question = (
        "北京人的体质特征有哪些？"
        "这些特征说明他们在身体形态上处于怎样的状态？"
    )
    candidates = rag_service.retrieve_history_knowledge(question)

    assessment = assess_support(question, "keyword", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)

    assert assessment.status == "sufficient"
    assert "aspect:process" not in assessment.required_facets
    assert assessment.unsupported_facets == []
    assert [item["knowledge_id"] for item in answer_items] == ["H7A-L01-008"]


def test_q011_reports_missing_banpo_time_and_place_per_subject(monkeypatch):
    points = _textbook(monkeypatch)
    question = (
        "从地点与区域、年代、房屋类型、主要农作物和其他生产生活内容"
        "比较河姆渡人与半坡人。"
    )
    candidates = rag_service.retrieve_history_knowledge(question)

    assessment = assess_support(question, "keyword", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)
    answer_ids = {item["knowledge_id"] for item in answer_items}

    assert assessment.status == "partial"
    assert "subject:河姆渡人" in assessment.covered_facets
    assert "subject:半坡人" in assessment.covered_facets
    assert "comparison:all_subjects" in assessment.covered_facets
    assert {"H7A-L02-006", "H7A-L02-012"}.issubset(answer_ids)
    assert "subject:半坡人:aspect:time" in assessment.unsupported_facets
    assert "subject:半坡人:aspect:place" in assessment.unsupported_facets


def test_q011_becomes_sufficient_after_traceable_banpo_supplement(monkeypatch):
    points = _textbook(monkeypatch)
    point_map = {point["knowledge_id"]: point for point in points}
    question = (
        "从地点与区域、年代、房屋类型、主要农作物和其他生产生活内容"
        "比较河姆渡人与半坡人。"
    )
    candidates = rag_service.retrieve_history_knowledge(question)
    supplement = dict(point_map["H7A-L02-011"])
    supplement.update(
        {
            "knowledge_ids": ["H7A-L02-011"],
            "retrieval_mode": "keyword",
            "retrieval_unit": "knowledge_point",
            "score": 1,
        }
    )

    assessment = assess_support(
        question, "keyword", [*candidates, supplement], points
    )

    assert assessment.status == "sufficient"
    assert assessment.unsupported_facets == []
    assert "H7A-L02-011" in assessment.supporting_knowledge_ids


def test_q014_aggregates_formation_and_civilization_evidence(monkeypatch):
    points = _textbook(monkeypatch)
    question = (
        "根据教材，5000多年前早期国家是怎样形成的？"
        "人类进入文明社会的重要标志是什么？"
    )
    candidates = rag_service.retrieve_history_knowledge(question)

    assessment = assess_support(question, "keyword", candidates, points)

    assert assessment.status == "sufficient"
    assert {"H7A-L03-001", "H7A-L03-002"}.issubset(
        set(assessment.supporting_knowledge_ids)
    )
    assert "aspect:process" in assessment.covered_facets
    assert "aspect:result_impact" in assessment.covered_facets


def test_q017_uses_liangzhu_and_taosi_but_not_dawenkou(monkeypatch):
    points = _textbook(monkeypatch)
    question = (
        "从地点与年代、城市建设、墓葬和随葬品、经济或文化联系四个维度"
        "比较良渚古城与陶寺古城，并说明二者共同支持什么历史结论。"
    )
    candidates = rag_service.retrieve_history_knowledge(question)

    assessment = assess_support(question, "keyword", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)
    answer_ids = {item["knowledge_id"] for item in answer_items}

    assert assessment.status == "partial"
    assert "subject:良渚古城" in assessment.covered_facets
    assert "subject:陶寺古城" in assessment.covered_facets
    assert "comparison:all_subjects" in assessment.covered_facets
    assert "H7A-L02-015" not in assessment.supporting_knowledge_ids
    assert "H7A-L02-015" not in answer_ids
    assert {"H7A-L03-004", "H7A-L03-007"}.issubset(answer_ids)
    assert "subject:良渚古城:topic:burial" in assessment.unsupported_facets
    assert "subject:陶寺古城:topic:urban" in assessment.unsupported_facets


def test_q017_dawenkou_alone_remains_insufficient(monkeypatch):
    points = _textbook(monkeypatch)
    question = "比较良渚古城与陶寺古城的墓葬和随葬品。"
    dawenkou = [
        point for point in points if point["knowledge_id"] == "H7A-L02-015"
    ]

    assessment = assess_support(question, "keyword", dawenkou, points)

    assert assessment.status == "insufficient"
    assert assessment.supporting_knowledge_ids == []
    assert "subject:良渚古城" in assessment.unsupported_facets
    assert "subject:陶寺古城" in assessment.unsupported_facets


def test_q019_process_can_be_supported_by_verbatim_evidence(monkeypatch):
    points = _textbook(monkeypatch)
    question = (
        "相传炎帝、黄帝部落经历了哪些联盟与战争过程？"
        "这些传说怎样解释华夏族的形成及炎黄二帝的地位？"
    )
    candidates = rag_service.retrieve_history_knowledge(question)

    assessment = assess_support(question, "keyword", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)

    assert assessment.status == "sufficient"
    assert "aspect:process" in assessment.covered_facets
    assert {"H7A-L03-012", "H7A-L03-013"}.issubset(
        {item["knowledge_id"] for item in answer_items}
    )


def _xia_founding_evidence():
    excerpt = "约公元前2070年，禹建立起我国历史上第一个奴隶制王朝——夏朝。"
    point = {
        "knowledge_id": "L04-001",
        "title": "夏朝建立",
        "summary": excerpt,
        "content": excerpt,
        "keywords": ["夏朝建立"],
        "people": [],
        "qualifiers": ["约公元前2070年"],
        "source_refs": [
            {
                "source_id": "H7A-P022-A",
                "fragment_id": "H7A-P022-A",
                "knowledge_id": "L04-001",
                "textbook_pages": "22",
                "pdf_pages": "27",
                "excerpt": excerpt,
            }
        ],
    }
    candidate = {
        **point,
        "knowledge_ids": ["L04-001"],
        "retrieval_mode": "vector",
        "retrieval_unit": "source_excerpt",
        "score": 0.9,
        "event": excerpt,
    }
    return point, candidate, excerpt


def test_founding_agent_questions_match_entity_relation_and_keep_qualifier():
    point, candidate, excerpt = _xia_founding_evidence()

    for question in (
        "夏朝的建立者是谁？",
        "谁建立了夏朝？",
        "夏朝是谁建立的？",
    ):
        assessment = assess_support(question, "vector", [candidate], [point])
        answer_items = supporting_answer_items([candidate], assessment)
        answer, mode, used_evidence = assemble_textbook_answer(
            question, answer_items, assessment, [point]
        )

        assert assessment.status == "sufficient", question
        assert "no_question_anchor" not in assessment.reason_codes, question
        assert "subject:夏朝" in assessment.covered_facets, question
        assert "aspect:founding_agent" in assessment.covered_facets, question
        assert assessment.supporting_knowledge_ids == ["L04-001"], question
        assert assessment.supporting_source_ids == ["H7A-P022-A"], question
        assert mode == "controlled_evidence", question
        assert "禹" in answer and "约公元前2070年" in answer, question
        assert used_evidence and used_evidence[0]["text"] == excerpt, question
        assert used_evidence[0]["source_ids"] == ["H7A-P022-A"], question


def test_false_founding_premise_is_corrected_by_textbook_evidence():
    point, candidate, _ = _xia_founding_evidence()
    question = "夏朝是商汤建立的吗？"

    assessment = assess_support(question, "vector", [candidate], [point])
    answer_items = supporting_answer_items([candidate], assessment)
    answer, _, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, [point]
    )

    assert assessment.status == "sufficient"
    assert "aspect:founding_agent" in assessment.covered_facets
    assert "禹" in answer
    assert "商汤" not in answer
    assert used_evidence[0]["source_ids"] == ["H7A-P022-A"]


def test_dynasty_decline_evidence_does_not_support_founding_agent_question():
    founder_point, _, _ = _xia_founding_evidence()
    decline_excerpt = "夏朝后期，桀不修德行，统治残暴，引起民众反抗。"
    decline_point = {
        "knowledge_id": "L04-005",
        "title": "夏朝衰亡",
        "summary": decline_excerpt,
        "keywords": ["夏朝衰亡"],
        "people": ["桀"],
    }
    candidate = {
        **decline_point,
        "knowledge_ids": ["L04-005"],
        "source_refs": [
            {"source_id": "H7A-P023-B", "excerpt": decline_excerpt}
        ],
    }

    assessment = assess_support(
        "夏朝的建立者是谁？",
        "vector",
        [candidate],
        [founder_point, decline_point],
    )

    assert assessment.status == "insufficient"
    assert "aspect:founding_agent" in assessment.unsupported_facets
    assert assessment.supporting_knowledge_ids == []
    assert assessment.supporting_source_ids == []


def test_out_of_corpus_entity_still_has_no_safe_question_anchor():
    point, candidate, _ = _xia_founding_evidence()

    assessment = assess_support(
        "玛雅文明的建立者是谁？", "vector", [candidate], [point]
    )

    assert assessment.status == "insufficient"
    assert assessment.reason_codes == ["no_question_anchor"]
    assert assessment.supporting_knowledge_ids == []


def _zhou_ritual_evidence(excerpt):
    point = {
        "knowledge_id": "L04-011",
        "title": "西周等级与礼乐",
        "summary": excerpt,
        "content": excerpt,
        "keywords": ["西周等级与礼乐"],
        "people": [],
    }
    candidate = {
        **point,
        "knowledge_ids": ["L04-011"],
        "retrieval_mode": "vector",
        "retrieval_unit": "source_excerpt",
        "score": 0.91,
        "event": excerpt,
        "reason": "",
        "impact": "",
        "source_refs": [
            {
                "knowledge_id": "L04-011",
                "source_id": "H7A-P026-B",
                "fragment_id": "H7A-P026-B",
                "textbook_pages": "26",
                "pdf_pages": "31",
                "excerpt": excerpt,
            }
        ],
    }
    return point, candidate


def test_l04_011_purpose_excerpt_covers_reason_and_builds_grounded_outline():
    excerpt = (
        "周武王去世后，周公旦辅佐年幼的周成王。"
        "为了巩固周王室的统治，周公旦“制礼作乐”，制定了一套严密的礼乐制度，"
        "包括严格区分君臣、父子、夫妻等身份和亲疏、尊卑、贵贱等地位的礼仪，"
        "以及与这些礼仪相配套的乐舞。例如，青铜鼎的使用数量有严格的区分，"
        "天子用九鼎，诸侯用七鼎，大夫用五鼎，士用三鼎。乐舞也有不同的标准，"
        "天子用八横八纵共64人的规模，诸侯用六横六纵共36人的规模，"
        "大夫用四横四纵共16人的规模。"
    )
    point, candidate = _zhou_ritual_evidence(excerpt)
    question = "西周等级与礼乐制度为什么要制定？"

    assessment = assess_support(question, "vector", [candidate], [point])
    answer_items = supporting_answer_items([candidate], assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, [point]
    )
    outline = build_knowledge_outline(question, answer_items)

    assert assessment.status == "sufficient"
    assert "aspect:reason" in assessment.covered_facets
    assert assessment.unsupported_facets == []
    assert evidence_covers_assessment(assessment, used_evidence)
    assert "证据不足" not in answer
    assert mode == "controlled_evidence"
    assert outline["time"] == "周武王去世后"
    assert outline["people"] == ["周武王", "周公旦", "周成王"]
    assert "制礼作乐" in outline["event"]
    assert outline["event"] != excerpt
    assert outline["reason"] == "为了巩固周王室的统治"
    assert "制定了一套严密的礼乐制度" in outline["process"]
    assert "严格区分君臣、父子、夫妻等身份" in outline["process"]
    assert "与这些礼仪相配套的乐舞" in outline["process"]
    assert outline["result_impact"] == ""


def test_reason_stays_unsupported_without_reason_or_purpose_in_excerpt():
    excerpt = "周公旦“制礼作乐”，制定了一套严密的礼乐制度。"
    point, candidate = _zhou_ritual_evidence(excerpt)
    question = "西周等级与礼乐制度为什么要制定？"

    assessment = assess_support(question, "vector", [candidate], [point])
    answer_items = supporting_answer_items([candidate], assessment)
    answer, mode, _ = assemble_textbook_answer(
        question, answer_items, assessment, [point]
    )

    assert assessment.status == "partial"
    assert "aspect:reason" in assessment.unsupported_facets
    assert "证据不足：当前教材来源尚未覆盖原因或目的。" in answer
    assert mode == "controlled_partial"


def test_asking_what_something_is_called_does_not_create_reason_facet():
    excerpt = "这套制度被称为礼乐制度。"
    point, candidate = _zhou_ritual_evidence(excerpt)

    assessment = assess_support(
        "这套制度被称为什么？", "vector", [candidate], [point]
    )

    assert "aspect:reason" not in assessment.required_facets


def _zhou_qin_local_government_evidence():
    zhou_excerpt = (
        "为稳定周初的政治形势，周王室根据血缘关系远近和功劳大小，"
        "将宗亲和功臣等分封到各地，授予他们管理土地和民众的权力，"
        "建立诸侯国，加强对地方的控制，扩大统治范围。"
        "受封的诸侯承担向周王进献贡赋的义务，其军队也要服从周王调遣；"
        "同时具有一定的独立性，可以在自己的封地内进行再分封。"
    )
    qin_excerpt = (
        "在地方上，秦朝全面废除西周以来实行的分封制，"
        "在全国范围内实行郡县制，由中央直接管辖。"
    )
    points = [
        {
            "knowledge_id": "L04-010",
            "title": "西周分封制",
            "summary": zhou_excerpt,
            "content": zhou_excerpt,
            "keywords": ["西周分封制"],
            "people": [],
        },
        {
            "knowledge_id": "L09-Q013",
            "title": "秦朝推行郡县制",
            "summary": qin_excerpt,
            "content": qin_excerpt,
            "keywords": ["秦朝推行郡县制"],
            "people": [],
        },
        {
            "knowledge_id": "L09-Q014",
            "title": "秦朝郡县制的管理方式",
            "summary": qin_excerpt,
            "content": qin_excerpt,
            "keywords": ["秦朝郡县制的管理方式"],
            "people": [],
        },
    ]

    def candidate(point, source_id, excerpt, textbook_pages, pdf_pages):
        return {
            **point,
            "knowledge_ids": [point["knowledge_id"]],
            "retrieval_mode": "vector",
            "retrieval_unit": "source_excerpt",
            "score": 0.9,
            "event": excerpt,
            "reason": "",
            "impact": "",
            "source_refs": [
                {
                    "knowledge_id": point["knowledge_id"],
                    "source_id": source_id,
                    "fragment_id": source_id,
                    "textbook_pages": textbook_pages,
                    "pdf_pages": pdf_pages,
                    "excerpt": excerpt,
                }
            ],
        }

    candidates = [
        candidate(
            points[1], "H7A-L09-CAND-013-SRC-01", qin_excerpt, "54", "59"
        ),
        candidate(
            points[2], "H7A-L09-CAND-013-SRC-01", qin_excerpt, "54", "59"
        ),
        candidate(points[0], "H7A-P025-C", zhou_excerpt, "25", "30"),
    ]
    return points, candidates, zhou_excerpt, qin_excerpt


def test_zhou_qin_comparison_adopts_evidence_for_both_subjects():
    points, candidates, zhou_excerpt, qin_excerpt = (
        _zhou_qin_local_government_evidence()
    )
    question = "西周分封制和秦朝郡县制有什么区别？"

    assessment = assess_support(question, "vector", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, points
    )

    assert assessment.status == "sufficient"
    assert "comparison:all_subjects" in assessment.covered_facets
    assert set(assessment.supporting_source_ids) == {
        "H7A-P025-C",
        "H7A-L09-CAND-013-SRC-01",
    }
    assert {item["knowledge_id"] for item in answer_items} == {
        "L04-010",
        "L09-Q013",
        "L09-Q014",
    }
    assert zhou_excerpt in answer
    assert qin_excerpt in answer
    assert {source_id for item in used_evidence for source_id in item["source_ids"]} == {
        "H7A-P025-C",
        "H7A-L09-CAND-013-SRC-01",
    }
    assert "证据不足" not in answer
    assert mode == "controlled_evidence"


def test_zhou_qin_comparison_reports_missing_qin_evidence():
    points, candidates, zhou_excerpt, qin_excerpt = (
        _zhou_qin_local_government_evidence()
    )
    question = "西周分封制和秦朝郡县制有什么区别？"

    assessment = assess_support(question, "vector", [candidates[2]], points)
    answer_items = supporting_answer_items([candidates[2]], assessment)
    answer, mode, _ = assemble_textbook_answer(
        question, answer_items, assessment, points
    )

    assert assessment.status == "partial"
    assert "subject:秦朝郡县制" in assessment.unsupported_facets
    assert zhou_excerpt in answer
    assert qin_excerpt not in answer
    assert "证据不足" in answer
    assert "秦朝郡县制" in answer
    assert mode == "controlled_partial"


def test_l04_010_outline_extracts_governance_details_without_repeating_fields():
    points, candidates, _, _ = _zhou_qin_local_government_evidence()
    question = "西周分封制包括哪些内容？"
    zhou_candidate = candidates[2]

    assessment = assess_support(question, "vector", [zhou_candidate], points)
    answer_items = supporting_answer_items([zhou_candidate], assessment)
    outline = build_knowledge_outline(question, answer_items)

    assert outline["reason"] == "为稳定周初的政治形势"
    assert outline["event"].rstrip("，。；") == "建立诸侯国"
    assert "根据血缘关系远近和功劳大小" in outline["process"]
    assert "将宗亲和功臣等分封到各地" in outline["process"]
    assert "授予他们管理土地和民众的权力" in outline["process"]
    assert "承担向周王进献贡赋的义务" in outline["process"]
    assert "服从周王调遣" in outline["process"]
    assert "具有一定的独立性" in outline["process"]
    assert "一定" in outline["process"]
    assert "再分封" in outline["process"]
    assert outline["reason"] not in outline["process"]
    assert outline["event"].rstrip("，。；") not in outline["process"]


def test_qin_commandery_outline_keeps_complete_event_and_governance_content():
    points, candidates, _, qin_excerpt = _zhou_qin_local_government_evidence()
    question = "西周分封制和秦朝郡县制有什么区别？"

    assessment = assess_support(question, "vector", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)
    qin_items = [
        item
        for item in answer_items
        if item["knowledge_id"] in {"L09-Q013", "L09-Q014"}
    ]
    outline = build_knowledge_outline(question, qin_items)

    assert qin_excerpt == (
        "在地方上，秦朝全面废除西周以来实行的分封制，"
        "在全国范围内实行郡县制，由中央直接管辖。"
    )
    assert outline["time"] == ""
    assert outline["people"] == []
    assert outline["event"] == (
        "秦朝全面废除西周以来实行的分封制，"
        "在全国范围内实行郡县制，由中央直接管辖。"
    )
    assert outline["reason"] == ""
    assert outline["process"] == "在全国范围内实行郡县制，由中央直接管辖。"
    assert outline["result_impact"] == ""


def test_comparison_answer_still_uses_both_sources_after_outline_fix():
    points, candidates, zhou_excerpt, qin_excerpt = (
        _zhou_qin_local_government_evidence()
    )
    question = "西周分封制和秦朝郡县制有什么区别？"

    assessment = assess_support(question, "vector", candidates, points)
    answer_items = supporting_answer_items(candidates, assessment)
    answer, mode, used_evidence = assemble_textbook_answer(
        question, answer_items, assessment, points
    )

    assert zhou_excerpt in answer
    assert qin_excerpt in answer
    assert {source_id for item in used_evidence for source_id in item["source_ids"]} == {
        "H7A-P025-C",
        "H7A-L09-CAND-013-SRC-01",
    }
    assert mode == "controlled_evidence"
