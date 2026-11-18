"""Build structured, source-grounded textbook answers and study outlines."""

import re

from app.services.rag_service import knowledge_item_to_source
from app.services.support_assessment import (
    REASON_EVIDENCE_CUES,
    SupportAssessment,
    TOPIC_GROUPS,
    analyze_question_requirements,
)


PROCESS_CUES = (
    "开始",
    "逐步",
    "出现",
    "加剧",
    "脱离",
    "专门",
    "掌握",
    "先",
    "后来",
    "最终",
    "经过",
    "联合",
    "大战",
    "激战",
    "战败",
    "归附",
    "结成",
    "对内",
    "对外",
    "从而形成",
    "做过",
    "担任",
    "任职",
    "曾任",
    "掌管",
)
RESULT_CUES = (
    "促进",
    "推动",
    "改善",
    "奠定",
    "说明",
    "反映",
    "表明",
    "标志",
    "演化",
    "尊崇",
    "比现代人",
    "与现代人相似",
    "直立行走",
)
BIOGRAPHY_QUESTION_CUES = ("生平", "经历")
JUDGMENT_INSTRUCTION_PATTERN = re.compile(
    r"^\s*(?:请)?判断(?:并)?(?:简要)?说明(?:理由)?\s*[：:，,]?\s*"
)
BIOGRAPHY_FACT_CUES = ("姓", "名", "字", "出生", "籍贯", "时期", "人")
EVENT_CUES = (
    "制礼作乐",
    "礼乐制度",
    "建立",
    "创立",
    "制定",
    "实行",
    "改革",
    "统一",
    "战争",
    "灭亡",
)
OUTLINE_PROCESS_CUES = PROCESS_CUES + (
    "根据",
    "分封",
    "授予",
    "建立",
    "承担",
    "进献",
    "服从",
    "独立性",
    "制定",
    "实行",
    "实施",
    "颁布",
    "区分",
    "使用",
    "配套",
    "直接管辖",
)
PERSON_ACTION_CUES = (
    "去世",
    "辅佐",
    "建立",
    "创立",
    "制定",
    "实行",
    "改革",
    "统一",
    "命令",
    "率领",
    "攻灭",
)
PERSON_BLOCKED_PREFIXES = ("在", "从", "由", "为", "向", "对", "于")
PERSON_BLOCKED_SUFFIXES = (
    "朝",
    "国",
    "制度",
    "地方",
    "时期",
    "范围内",
)


def _sentences(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.findall(r"[^。！？]+[。！？]?", text)
        if value.strip()
    ]


def _clauses(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.findall(r"[^，。；！？]+[，。；！？]?", text)
        if value.strip()
    ]


def _outline_excerpt(item: dict) -> str:
    excerpts = [
        str(reference.get("excerpt") or "").strip()
        for reference in item.get("source_refs") or []
        if isinstance(reference, dict)
        and str(reference.get("excerpt") or "").strip()
    ]
    return "\n".join(dict.fromkeys(excerpts))


def _extract_time(text: str) -> str:
    match = re.search(
        r"(?:约)?公元前[^，。；！？]{1,12}年"
        r"|距今[^，。；！？]{1,12}年"
        r"|[一-鿿]{1,10}(?:去世|即位|灭亡|统一)后"
        r"|[一-鿿]{1,8}(?:早期|中期|后期|末期)",
        text,
    )
    return match.group(0) if match else ""


def _extract_people(text: str) -> list[str]:
    actions = "|".join(PERSON_ACTION_CUES)
    people = re.findall(
        rf"(?:^|[，。；、：:\s])([一-鿿·]{{1,6}})(?=(?:{actions}))",
        text,
    )
    people.extend(
        re.findall(r"辅佐(?:年幼的)?([一-鿿·]{1,6})", text)
    )
    return list(
        dict.fromkeys(
            value
            for value in people
            if value
            and len(value.replace("·", "")) <= 4
            and not value.startswith(PERSON_BLOCKED_PREFIXES)
            and not value.endswith(PERSON_BLOCKED_SUFFIXES)
        )
    )


def _extract_reason(text: str) -> str:
    purpose = re.search(
        r"((?:为了|为)(?!何|什么)[^，。；！？]{1,30})(?=[，,。；！？])",
        text,
    )
    if purpose:
        return purpose.group(1)
    for sentence in _sentences(text):
        if any(cue in sentence for cue in REASON_EVIDENCE_CUES):
            return sentence
    return ""


def _matching_sentences(text: str, cues: tuple[str, ...]) -> str:
    return "".join(
        sentence
        for sentence in _sentences(text)
        if any(cue in sentence for cue in cues)
    )


def _extract_event(text: str) -> str:
    sentences = _sentences(text)
    for sentence in sentences:
        if "废除" in sentence and any(cue in sentence for cue in ("实行", "推行")):
            return re.sub(r"^在地方上[，,]", "", sentence).strip()
    return next(
        (
            clause
            for clause in _clauses(text)
            if any(cue in clause for cue in EVENT_CUES)
        ),
        sentences[0] if sentences else "",
    )


def _matching_process_clauses(text: str, excluded: tuple[str, ...] = ()) -> str:
    excluded_values = {
        value.strip().rstrip("，。；！？") for value in excluded if value.strip()
    }
    return "".join(
        clause
        for clause in _clauses(text)
        if any(cue in clause for cue in OUTLINE_PROCESS_CUES)
        and not ("废除" in clause and "以来实行的" in clause)
        and clause.strip().rstrip("，。；！？") not in excluded_values
    )


def _grounded_outline_dimensions(topic: str, item: dict) -> dict:
    """Extract only verbatim dimensions from the selected textbook excerpt."""
    excerpt = _outline_excerpt(item)
    if not excerpt:
        return {
            "time": str(item.get("time") or ""),
            "people": list(item.get("people") or []),
            "event": str(item.get("event") or ""),
            "reason": str(item.get("reason") or ""),
            "process": str(item.get("content") or ""),
            "result_impact": str(item.get("impact") or ""),
        }

    sentences = _sentences(excerpt)
    is_biography = any(cue in topic for cue in BIOGRAPHY_QUESTION_CUES)
    event = str(item.get("event") or "").strip()
    process = str(item.get("content") or "").strip()
    if event == excerpt.strip():
        event = "" if is_biography else _extract_event(excerpt)
    if process == excerpt.strip():
        process = excerpt if is_biography else _matching_process_clauses(
            excerpt,
            (
                event,
                str(item.get("reason") or "") or _extract_reason(excerpt),
            ),
        )

    people = list(item.get("people") or [])
    people.extend(value for value in _extract_people(excerpt) if value not in people)
    return {
        "time": str(item.get("time") or "") or _extract_time(excerpt),
        "people": people,
        "event": event,
        "reason": str(item.get("reason") or "") or _extract_reason(excerpt),
        "process": process,
        "result_impact": str(item.get("impact") or "")
        or _matching_sentences(excerpt, RESULT_CUES),
    }


def build_knowledge_outline(topic: str, knowledge_items: list[dict]) -> dict | None:
    """Use the highest-scoring item as the main outline and retain related points."""
    if not knowledge_items:
        return None

    main = knowledge_items[0]
    dimensions = _grounded_outline_dimensions(topic, main)
    sources = [knowledge_item_to_source(item) for item in knowledge_items]
    related_points = [
        item.get("title", "")
        for item in knowledge_items[1:]
        if item.get("title")
    ]

    return {
        "topic": main.get("title") or topic,
        "time": dimensions["time"],
        "people": dimensions["people"],
        "event": dimensions["event"],
        "reason": dimensions["reason"],
        "process": dimensions["process"],
        "result_impact": dimensions["result_impact"],
        "related_points": related_points,
        "sources": sources,
    }


def _lists(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _source_evidence(
    question: str, knowledge_items: list[dict], textbook_points: list[dict]
) -> list[dict]:
    """Deduplicate and preserve the exact linked textbook excerpts."""
    del question, textbook_points

    evidence_by_key: dict[tuple[str, str], dict] = {}
    for item in knowledge_items:
        knowledge_ids = _lists(item.get("knowledge_ids")) or _lists(
            item.get("knowledge_id")
        )
        for reference in item.get("source_refs") or []:
            if not isinstance(reference, dict):
                continue
            source_id = str(reference.get("source_id") or "").strip()
            excerpt = str(reference.get("excerpt") or "").strip()
            if not source_id or not excerpt:
                continue
            key = (source_id, excerpt)
            existing = evidence_by_key.get(key)
            if existing is not None:
                existing["knowledge_ids"] = list(
                    dict.fromkeys(existing["knowledge_ids"] + knowledge_ids)
                )
                title = str(item.get("title") or "").strip()
                if title and title not in existing["source_titles"]:
                    existing["source_titles"].append(title)
                continue

            evidence_by_key[key] = {
                "text": excerpt,
                "full_excerpt": excerpt,
                "knowledge_id": knowledge_ids[0] if knowledge_ids else "",
                "knowledge_ids": list(knowledge_ids),
                "source_ids": [source_id],
                "source_title": str(item.get("title") or ""),
                "source_titles": [str(item.get("title") or "")]
                if item.get("title")
                else [],
                "textbook_pages": str(reference.get("textbook_pages") or ""),
                "pdf_pages": str(reference.get("pdf_pages") or ""),
                "matched_subquestions": _lists(item.get("matched_subquestions")),
            }
    evidence = list(evidence_by_key.values())
    seen_by_source: dict[str, set[str]] = {}
    retained: list[dict] = []
    for item in evidence:
        source_id = item["source_ids"][0]
        sentences = [
            value.strip()
            for value in re.findall(r"[^。！？]+[。！？]?", item["text"])
            if value.strip()
        ]
        seen = seen_by_source.setdefault(source_id, set())
        unique_sentences = [sentence for sentence in sentences if sentence not in seen]
        seen.update(sentences)
        if not unique_sentences:
            existing = next(
                value for value in retained if source_id in value["source_ids"]
            )
            existing["knowledge_ids"] = list(
                dict.fromkeys(existing["knowledge_ids"] + item["knowledge_ids"])
            )
            existing["source_titles"] = list(
                dict.fromkeys(existing["source_titles"] + item["source_titles"])
            )
            existing["matched_subquestions"] = list(
                dict.fromkeys(
                    existing["matched_subquestions"]
                    + item["matched_subquestions"]
                )
            )
            continue
        item["display_text"] = "".join(unique_sentences)
        retained.append(item)
    return retained


def _facet_subject(facet: str) -> str:
    parts = facet.split(":")
    return parts[1] if len(parts) >= 2 and parts[0] == "subject" else ""


def _facet_label(facet: str) -> str:
    parts = facet.split(":")
    if len(parts) == 2 and parts[0] == "subject":
        return f"{parts[1]}相关证据"
    if len(parts) == 4 and parts[0] == "subject":
        subject, kind, value = parts[1], parts[2], parts[3]
        labels = {
            ("aspect", "time"): "年代",
            ("aspect", "place"): "地点或区域",
            ("aspect", "reason"): "原因或背景",
            ("aspect", "process"): "过程",
            ("aspect", "result_impact"): "结论或影响",
            ("aspect", "certainty"): "史料限定",
            ("aspect", "content"): "教材内容",
            ("topic", "tools"): "工具或技术",
            ("topic", "agriculture"): "农业或农作物",
            ("topic", "housing"): "房屋",
            ("topic", "livelihood"): "其他生产生活",
            ("topic", "urban"): "城市建设",
            ("topic", "burial"): "墓葬和随葬品",
            ("topic", "economy_culture"): "经济或文化联系",
            ("topic", "fire"): "用火",
            ("topic", "water_system"): "水利系统",
        }
        return f"{subject}的{labels.get((kind, value), value)}"
    if facet.startswith("aspect:"):
        return {
            "time": "年代",
            "place": "地点或区域",
            "reason": "原因或目的",
            "process": "过程",
            "result_impact": "结论或影响",
            "certainty": "史料限定",
            "founding_agent": "建立者",
            "definition": "定义",
            "historical_status": "历史地位",
        }.get(facet.removeprefix("aspect:"), facet.removeprefix("aspect:"))
    if facet.startswith("topic:"):
        return facet.removeprefix("topic:")
    if facet == "comparison:all_subjects":
        return "双方比较证据"
    return facet


def _with_locator(evidence: dict, text: str | None = None) -> str:
    value = (
        text if text is not None else evidence.get("display_text", evidence["text"])
    ).strip()
    page = evidence.get("textbook_pages") or "未标页码"
    source_id = (evidence.get("source_ids") or [""])[0]
    return f"{value}（教材第{page}页，{source_id}）"


def _aspect_text(evidence: dict, aspect: str) -> str:
    evidence_text = evidence.get("display_text", evidence["text"])
    sentences = [
        value.strip()
        for value in re.findall(r"[^。！？]+[。！？]?", evidence_text)
        if value.strip()
    ]
    cues = PROCESS_CUES if aspect == "process" else RESULT_CUES
    selected = [sentence for sentence in sentences if any(cue in sentence for cue in cues)]
    return "".join(selected) or evidence_text


def _biography_text(evidence: dict) -> str:
    """Keep textbook-backed identity and experience sentences for biography questions."""
    sentences = [
        value.strip()
        for value in re.findall(r"[^。！？]+[。！？]?", evidence["full_excerpt"])
        if value.strip()
    ]
    selected = [
        sentence
        for sentence in sentences
        if any(cue in sentence for cue in BIOGRAPHY_FACT_CUES + PROCESS_CUES)
    ]
    return "".join(selected) or evidence["text"]


def _contradicted_negative_event(question: str, evidence: list[dict]) -> bool:
    """Detect a grounded contradiction of an explicit 'event did not occur' claim."""
    if not JUDGMENT_INSTRUCTION_PATTERN.match(question):
        return False
    anchors = []
    for clause in re.split(r"[，,。；;：:\"“”]", question):
        match = re.search(
            r"([一-鿿]{2,8}?)(?:并未|并没有|未曾|没有)出现",
            clause,
        )
        if match:
            anchors.append(match.group(1))
    return bool(anchors) and any(
        anchor in item["text"]
        for anchor in anchors
        for item in evidence
    )


def assemble_textbook_answer(
    question: str,
    knowledge_items: list[dict],
    assessment: SupportAssessment,
    textbook_points: list[dict],
) -> tuple[str, str, list[dict]]:
    """Assemble a direct answer from verbatim evidence without model generation."""
    if not knowledge_items:
        return "", "insufficient_support", []

    evidence = _source_evidence(question, knowledge_items, textbook_points)
    if not evidence:
        return "", "insufficient_support", []
    for index, item in enumerate(evidence, start=1):
        item["evidence_id"] = f"E{index}"

    requirements = analyze_question_requirements(question, textbook_points)
    answer_parts = []
    judgment_added = assessment.status == "sufficient" and _contradicted_negative_event(
        question, evidence
    )
    if judgment_added:
        answer_parts.append("判断：该说法错误。")
    if requirements["is_comparison"]:
        for subject in requirements["subjects"]:
            subject_ids = assessment.facet_knowledge_ids.get(
                f"subject:{subject}", set()
            )
            subject_evidence = [
                item
                for item in evidence
                if subject_ids & set(item.get("knowledge_ids") or [])
            ]
            if subject_evidence:
                answer_parts.append(
                    f"{subject}："
                    + "；".join(_with_locator(item) for item in subject_evidence)
                )
        shared = [
            item
            for item in evidence
            if sum(
                any(alias in item["text"] for alias in (subject, subject.removesuffix("古城")))
                for subject in requirements["subjects"]
            )
            >= 2
            and any(cue in item["text"] for cue in RESULT_CUES)
        ]
        if shared:
            answer_parts.append(
                "共同结论：" + "；".join(_with_locator(item) for item in shared)
            )
    else:
        is_biography_question = any(
            cue in question for cue in BIOGRAPHY_QUESTION_CUES
        )
        if "process" in requirements["aspects"]:
            process_ids = assessment.facet_knowledge_ids.get("aspect:process", set())
            process_evidence = [
                item
                for item in evidence
                if process_ids & set(item.get("knowledge_ids") or [])
            ]
            if process_evidence:
                answer_parts.append(
                    ("回答：" if is_biography_question else "过程：")
                    + "；".join(
                        _with_locator(
                            item,
                            _biography_text(item)
                            if is_biography_question
                            else _aspect_text(item, "process"),
                        )
                        for item in process_evidence
                    )
                )
        if "result_impact" in requirements["aspects"]:
            result_ids = assessment.facet_knowledge_ids.get(
                "aspect:result_impact", set()
            )
            result_evidence = [
                item
                for item in evidence
                if result_ids & set(item.get("knowledge_ids") or [])
            ]
            if result_evidence:
                answer_parts.append(
                    "结论或影响："
                    + "；".join(
                        _with_locator(item, _aspect_text(item, "result_impact"))
                        for item in result_evidence
                    )
                )
        if not answer_parts or (judgment_added and len(answer_parts) == 1):
            answer_parts.append(
                ("教材依据：" if judgment_added else "回答：")
                + "；".join(_with_locator(item) for item in evidence)
            )

        combined = " ".join(item["text"] for item in evidence)
        if "比现代人" in combined and "与现代人相似" in combined:
            subject = requirements["subjects"][0] if requirements["subjects"] else "该主体"
            answer_parts.append(
                f"综合来看，教材证据同时显示{subject}存在与现代人不同的体质特征，"
                "也有与现代人相似的身体特征。"
            )

    if assessment.unsupported_facets:
        missing = "、".join(_facet_label(facet) for facet in assessment.unsupported_facets)
        answer_parts.append(f"证据不足：当前教材来源尚未覆盖{missing}。")

    used_evidence = [
        {
            key: value
            for key, value in item.items()
            if key
            in {
                "evidence_id",
                "text",
                "knowledge_id",
                "knowledge_ids",
                "source_ids",
                "source_title",
                "source_titles",
                "textbook_pages",
                "pdf_pages",
            }
        }
        for item in evidence
    ]
    mode = (
        "controlled_evidence"
        if assessment.status == "sufficient"
        else "controlled_partial"
    )
    return "\n".join(answer_parts), mode, used_evidence
