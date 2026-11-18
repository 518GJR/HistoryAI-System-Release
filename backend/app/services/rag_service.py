import json
import os
import re
from pathlib import Path

from app.database import SessionLocal
from app.models import HistoryKnowledge


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "history_knowledge.json"
TEXTBOOK_DATA_FILE = PROJECT_ROOT / "data" / "textbook_history_knowledge.json"


class KnowledgeDataError(RuntimeError):
    """The selected knowledge source is missing, invalid, or unsupported."""

# 不同字段的匹配权重：标题最高，关键词次之，朝代再次
FIELD_WEIGHTS = {
    "title": 10,
    "section_title": 8,
    "lesson_title": 8,
    "keywords": 7,
    "dynasty": 5,
    "people": 4,
    "event": 3,
    "time": 2,
    "reason": 2,
    "impact": 2,
    "content": 1,
    "summary": 1,
    "qualifiers": 1,
    "source_excerpt": 1,
}

# 第一版检索最多返回 3 条，避免回答过长
TOP_K = 3

QUESTION_DIMENSION_ALIASES = {
    "工具": ("工具", "制作", "石器", "骨器", "磨光", "钻孔"),
    "农作物": ("农作物", "农业", "种植", "稻作", "粟作"),
    "房屋": ("房屋", "建筑", "居住"),
    "说明": ("说明", "反映", "表明", "证实", "意义", "结果"),
}

HISTORICAL_NAME_ALIASES = (
    ("秦始皇", "秦王嬴政", "嬴政"),
    ("周王室", "周王", "周天子"),
)

CONTROLLED_EVENT_ANCHORS = ("诸侯争霸",)
JUDGMENT_INSTRUCTION_PATTERN = re.compile(
    r"^\s*(?:请)?判断(?:并)?(?:简要)?说明(?:理由)?\s*[：:，,]?\s*"
)

QUESTION_FILLERS = (
    "有什么历史内容",
    "有哪些历史内容",
    "有什么作用",
    "有什么影响",
    "有什么意义",
    "采取了什么",
    "建立了什么",
    "分别是什么",
    "是什么",
    "有哪些",
    "有何",
    "如何",
    "怎么",
    "怎样",
    "为什么",
    "什么",
    "哪些",
    "是否",
    "是谁",
    "了吗",
    "吗",
    "呢",
)

WEAK_FUZZY_TERMS = {
    "历史", "内容", "时期", "方面", "主要", "进行", "采取", "建立",
    "国家", "全国", "地区", "以后", "之前", "以及", "什么", "哪些",
    "发明", "修建",
}

BROAD_CONTEXT_TERMS = {
    "秦朝", "秦国", "中国", "统一",
}

ASPECT_STOP_CHARACTERS = set("的了着过吗呢谁何由")


def get_knowledge_mode() -> str:
    """Return the explicit knowledge mode without caching environment state."""
    mode = os.getenv("KNOWLEDGE_MODE", "demo").strip().lower()
    if mode not in {"demo", "textbook"}:
        raise KnowledgeDataError(
            "KNOWLEDGE_MODE must be either 'demo' or 'textbook'"
        )
    return mode


def load_history_knowledge_from_json() -> list[dict]:
    """读取本地历史知识 JSON 数据。"""
    if not DATA_FILE.exists():
        return []

    with DATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data if isinstance(data, list) else []


def _required_text(value, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise KnowledgeDataError(f"Textbook knowledge field is required: {field}")
    return text


def _page_label(value, field: str) -> str:
    if isinstance(value, int):
        return str(value)
    return _required_text(value, field)


def load_textbook_knowledge_from_json() -> list[dict]:
    """Load and validate the isolated textbook dataset.

    Textbook mode deliberately raises on missing or malformed data and never
    falls back to the demo database or JSON file.
    """
    if not TEXTBOOK_DATA_FILE.exists():
        raise KnowledgeDataError(
            f"Textbook knowledge file not found: {TEXTBOOK_DATA_FILE}"
        )

    try:
        with TEXTBOOK_DATA_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeDataError("Textbook knowledge file is unreadable or invalid") from exc

    if not isinstance(payload, dict):
        raise KnowledgeDataError("Textbook knowledge root must be an object")

    book = payload.get("book")
    unit = payload.get("unit")
    lessons = payload.get("lessons")
    sources = payload.get("sources")
    points = payload.get("knowledge_points")
    if (
        not isinstance(book, dict)
        or not isinstance(unit, dict)
        or not isinstance(lessons, list)
        or not isinstance(sources, list)
        or not isinstance(points, list)
    ):
        raise KnowledgeDataError(
            "Textbook knowledge must contain book, unit, lessons, sources, and knowledge_points"
        )

    lesson_map: dict[str, dict] = {}
    for lesson in lessons:
        if not isinstance(lesson, dict):
            raise KnowledgeDataError("Every textbook lesson must be an object")
        lesson_id = _required_text(lesson.get("lesson_id"), "lessons.lesson_id")
        sections = lesson.get("sections")
        if lesson_id in lesson_map or not isinstance(sections, list):
            raise KnowledgeDataError(f"Invalid or duplicate textbook lesson: {lesson_id}")
        section_map: dict[str, str] = {}
        for section in sections:
            if not isinstance(section, dict):
                raise KnowledgeDataError(f"Invalid section in {lesson_id}")
            section_id = _required_text(section.get("section_id"), "sections.section_id")
            if section_id in section_map:
                raise KnowledgeDataError(f"Duplicate section_id: {section_id}")
            section_map[section_id] = _required_text(
                section.get("title"), "sections.title"
            )
        try:
            lesson_number = int(lesson.get("lesson_number"))
        except (TypeError, ValueError) as exc:
            raise KnowledgeDataError(
                f"Invalid lesson_number for textbook lesson: {lesson_id}"
            ) from exc
        lesson_map[lesson_id] = {
            "lesson_number": lesson_number,
            "lesson_title": _required_text(lesson.get("title"), "lessons.title"),
            "sections": section_map,
        }

    source_map: dict[str, dict] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise KnowledgeDataError("Every textbook source must be an object")
        source_id = _required_text(source.get("source_id"), "sources.source_id")
        if source_id in source_map:
            raise KnowledgeDataError(f"Duplicate textbook source_id: {source_id}")
        source_map[source_id] = {
            "source_id": source_id,
            "lesson_id": _required_text(source.get("lesson_id"), "sources.lesson_id"),
            "section_id": _required_text(source.get("section_id"), "sources.section_id"),
            "material_type": _required_text(
                source.get("material_type"), "sources.material_type"
            ),
            "textbook_pages": _page_label(
                source.get("textbook_pages"), "sources.textbook_pages"
            ),
            "pdf_pages": _page_label(source.get("pdf_pages"), "sources.pdf_pages"),
            "excerpt": _required_text(source.get("excerpt"), "sources.excerpt"),
            "locator": str(source.get("locator") or "").strip(),
        }

    book_id = _required_text(book.get("book_id"), "book.book_id")
    book_title = _required_text(book.get("title"), "book.title")
    unit_id = _required_text(unit.get("unit_id"), "unit.unit_id")
    unit_title = _required_text(unit.get("title"), "unit.title")
    seen_ids: set[str] = set()
    normalized_points: list[dict] = []
    for point in points:
        if not isinstance(point, dict):
            raise KnowledgeDataError("Every textbook knowledge point must be an object")
        knowledge_id = _required_text(point.get("knowledge_id"), "knowledge_id")
        if knowledge_id in seen_ids:
            raise KnowledgeDataError(f"Duplicate textbook knowledge_id: {knowledge_id}")
        seen_ids.add(knowledge_id)

        lesson_id = _required_text(point.get("lesson_id"), f"{knowledge_id}.lesson_id")
        section_id = _required_text(point.get("section_id"), f"{knowledge_id}.section_id")
        try:
            lesson = lesson_map[lesson_id]
            section_title = lesson["sections"][section_id]
        except KeyError as exc:
            raise KnowledgeDataError(
                f"{knowledge_id} references unknown lesson or section: {exc.args[0]}"
            ) from exc

        source_ids = point.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise KnowledgeDataError(f"{knowledge_id} must reference at least one source")
        try:
            source_refs = [source_map[_required_text(value, "source_ids")] for value in source_ids]
        except KeyError as exc:
            raise KnowledgeDataError(
                f"{knowledge_id} references unknown source_id: {exc.args[0]}"
            ) from exc

        summary = _required_text(point.get("summary"), f"{knowledge_id}.summary")
        dimensions = point.get("dimensions") or {}
        if not isinstance(dimensions, dict):
            raise KnowledgeDataError(f"{knowledge_id}.dimensions must be an object")

        normalized_points.append(
            {
                "knowledge_id": knowledge_id,
                "book_id": book_id,
                "book_title": book_title,
                "unit_id": unit_id,
                "unit_title": unit_title,
                "lesson_id": lesson_id,
                "lesson_number": lesson["lesson_number"],
                "lesson_title": lesson["lesson_title"],
                "section_title": section_title,
                "title": _required_text(point.get("title"), f"{knowledge_id}.title"),
                "dynasty": str(point.get("period") or "").strip(),
                "time": str(point.get("time") or "").strip(),
                "people": _normalize_list(point.get("people")),
                "event": str(dimensions.get("content") or summary).strip(),
                "reason": str(dimensions.get("reason") or "").strip(),
                "impact": str(
                    dimensions.get("impact")
                    or dimensions.get("significance")
                    or dimensions.get("result")
                    or ""
                ).strip(),
                "content": summary,
                "summary": summary,
                "dimensions": dimensions,
                "keywords": _normalize_keywords(point.get("keywords")),
                "material_types": _normalize_list(point.get("material_types")),
                "textbook_pages": _page_label(
                    point.get("textbook_pages"), f"{knowledge_id}.textbook_pages"
                ),
                "pdf_pages": _page_label(
                    point.get("pdf_pages"), f"{knowledge_id}.pdf_pages"
                ),
                "qualifiers": _normalize_list(point.get("qualifiers")),
                "evidence_type": str(point.get("evidence_type") or "").strip(),
                "certainty": str(point.get("certainty") or "").strip(),
                "source_refs": source_refs,
            }
        )

    if not normalized_points:
        raise KnowledgeDataError("Textbook knowledge_points cannot be empty")
    return normalized_points


def _parse_json_list(value) -> list[str]:
    """解析数据库中以 JSON 字符串存储的列表字段。"""
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]

        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]

    return []


def _knowledge_model_to_dict(knowledge: HistoryKnowledge) -> dict:
    """将数据库模型转换为当前 RAG 兼容的数据结构。"""
    return {
        "title": knowledge.title or "",
        "dynasty": knowledge.dynasty or "",
        "time": knowledge.time or "",
        "people": _parse_json_list(knowledge.people),
        "event": knowledge.event or "",
        "reason": knowledge.reason or "",
        "impact": knowledge.impact or "",
        "keywords": _parse_json_list(knowledge.keywords),
        "content": knowledge.content or "",
    }


def load_history_knowledge_from_db() -> list[dict]:
    """从 MySQL 读取历史知识数据。"""
    db = SessionLocal()
    try:
        knowledge_rows = db.query(HistoryKnowledge).all()
        return [_knowledge_model_to_dict(row) for row in knowledge_rows]
    finally:
        db.close()


def load_history_knowledge() -> list[dict]:
    """Load only the source selected by KNOWLEDGE_MODE."""
    if get_knowledge_mode() == "textbook":
        points = load_textbook_knowledge_from_json()
        if os.getenv("SIGNED_TEXTBOOK_DATABASE_URL", "").strip():
            from app.services.signed_textbook_service import load_retrievable_signed_points
            points.extend(load_retrievable_signed_points())
        return points

    # Demo mode preserves the original MySQL-first, JSON-fallback behavior.
    try:
        db_data = load_history_knowledge_from_db()
        if db_data:
            return db_data
    except Exception:
        pass

    return load_history_knowledge_from_json()


def _normalize_keywords(keywords) -> list[str]:
    """将关键词字段统一转换为字符串列表，兼容列表和逗号分隔字符串。"""
    if isinstance(keywords, (list, tuple, set)):
        return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]

    if isinstance(keywords, str):
        return [
            keyword.strip()
            for keyword in re.split(r"[,，;；、\n]+", keywords)
            if keyword.strip()
        ]

    return []


def _normalize_list(value) -> list[str]:
    """将人物等字段统一转换为字符串列表，兼容列表和字符串。"""
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def _source_excerpt_terms(item: dict) -> list[str]:
    """Return verbatim source text without changing source metadata."""
    return list(
        dict.fromkeys(
            str(reference.get("excerpt") or "").strip()
            for reference in item.get("source_refs") or []
            if isinstance(reference, dict)
            and str(reference.get("excerpt") or "").strip()
        )
    )


def _normalize_match_text(value, *, question: bool = False) -> str:
    """Normalize punctuation and explicit historical-name aliases for matching."""
    text = str(value or "")
    if question:
        text = JUDGMENT_INSTRUCTION_PATTERN.sub("", text, count=1)
    for aliases in HISTORICAL_NAME_ALIASES:
        canonical = aliases[0]
        text = re.sub(
            "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)),
            canonical,
            text,
        )
    if question:
        for filler in QUESTION_FILLERS:
            text = text.replace(filler, "")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", text).lower()


def _matching_phrases(question_text: str, candidate_text: str) -> list[str]:
    """Find longest meaningful query phrases contained in one knowledge field.

    This is intentionally query-driven: textbook paragraphs are never expanded
    into a global bag of every possible bigram. Longer phrases win, contained
    subphrases are discarded, and generic two-character words cannot establish
    an evidence anchor by themselves.
    """
    question = _normalize_match_text(question_text, question=True)
    candidate = _normalize_match_text(candidate_text)
    if len(question) < 2 or len(candidate) < 2:
        return []

    matches: list[str] = []
    for length in range(min(10, len(question)), 1, -1):
        for start in range(len(question) - length + 1):
            phrase = question[start : start + length]
            if (
                phrase in WEAK_FUZZY_TERMS
                or phrase not in candidate
                or phrase[0] in ASPECT_STOP_CHARACTERS
                or phrase[-1] in ASPECT_STOP_CHARACTERS
                or (
                    len(phrase) == 2
                    and any(character in ASPECT_STOP_CHARACTERS for character in phrase)
                )
            ):
                continue
            if any(phrase in existing for existing in matches):
                continue
            matches.append(phrase)
    return matches


def knowledge_item_to_source(item: dict) -> dict:
    """Return the stable public source snapshot for API and history storage."""
    source_refs = [
        reference
        for reference in item.get("source_refs") or []
        if isinstance(reference, dict)
    ]

    def reference_values(field: str) -> list[str]:
        return list(
            dict.fromkeys(
                str(reference.get(field) or "").strip()
                for reference in source_refs
                if str(reference.get(field) or "").strip()
            )
        )

    material_types = _normalize_list(item.get("material_types")) or reference_values(
        "material_type"
    )
    textbook_pages = str(item.get("textbook_pages") or "").strip() or ";".join(
        reference_values("textbook_pages")
    )
    pdf_pages = str(item.get("pdf_pages") or "").strip() or ";".join(
        reference_values("pdf_pages")
    )
    return {
        "title": str(item.get("title") or ""),
        "dynasty": str(item.get("dynasty") or ""),
        "score": float(item.get("score", 0) or 0),
        "retrieval_mode": str(item.get("retrieval_mode") or "keyword"),
        "retrieval_unit": str(item.get("retrieval_unit") or "knowledge_point"),
        "knowledge_id": str(item.get("knowledge_id") or ""),
        "knowledge_ids": _normalize_list(item.get("knowledge_ids")) or (
            [str(item.get("knowledge_id"))] if item.get("knowledge_id") else []
        ),
        "book_id": str(item.get("book_id") or ""),
        "book_title": str(item.get("book_title") or ""),
        "unit_id": str(item.get("unit_id") or ""),
        "unit_title": str(item.get("unit_title") or ""),
        "lesson_id": str(item.get("lesson_id") or ""),
        "lesson_number": item.get("lesson_number"),
        "lesson_title": str(item.get("lesson_title") or ""),
        "section_title": str(item.get("section_title") or ""),
        "material_types": material_types,
        "textbook_pages": textbook_pages,
        "pdf_pages": pdf_pages,
        "qualifiers": _normalize_list(item.get("qualifiers")),
        "evidence_type": str(item.get("evidence_type") or ""),
        "certainty": str(item.get("certainty") or ""),
        "matched_subquestions": _normalize_list(item.get("matched_subquestions")),
        "source_refs": source_refs,
        "supporting": bool(item.get("supporting", False)),
    }


def _build_field_terms(item: dict) -> dict[str, list[str]]:
    """按字段提取检索词，兼容旧数据和 v0.3 新数据结构。"""
    field_terms = {
        "title": _normalize_list(item.get("title")),
        "section_title": _normalize_list(item.get("section_title")),
        "lesson_title": _normalize_list(item.get("lesson_title")),
        "dynasty": _normalize_list(item.get("dynasty")),
        "time": _normalize_list(item.get("time")),
        "people": _normalize_list(item.get("people")),
        "event": _normalize_list(item.get("event")),
        "reason": _normalize_list(item.get("reason")),
        "impact": _normalize_list(item.get("impact")),
        "keywords": _normalize_keywords(item.get("keywords")),
        "content": _normalize_list(item.get("content")),
        "summary": _normalize_list(item.get("summary")),
        "qualifiers": _normalize_list(item.get("qualifiers")),
        "source_excerpt": _source_excerpt_terms(item),
    }

    return field_terms


def _score_item(
    question_text: str,
    item: dict,
    subjects: list[str] | None = None,
    aspect_matches: list[str] | None = None,
) -> tuple[int, list[str]]:
    """计算单条知识点与问题的匹配分数。"""
    score = 0
    matched_terms: list[str] = []
    exact_matches: list[str] = []
    fuzzy_matches: list[str] = []
    seen_matches: set[str] = set()
    field_terms = _build_field_terms(item)

    for field, terms in field_terms.items():
        weight = FIELD_WEIGHTS.get(field, 1)
        for term in terms:
            if term and term in question_text:
                normalized = _normalize_match_text(term)
                if normalized and normalized not in seen_matches:
                    score += weight
                    matched_terms.append(term)
                    exact_matches.append(normalized)
                    seen_matches.add(normalized)
            for phrase in _matching_phrases(question_text, term):
                if phrase in seen_matches or any(
                    phrase in existing for existing in seen_matches
                ):
                    continue
                score += weight * min(4, max(1, len(phrase) - 1))
                matched_terms.append(phrase)
                fuzzy_matches.append(phrase)
                seen_matches.add(phrase)

    entity_terms = {
        _normalize_match_text(alias)
        for aliases in HISTORICAL_NAME_ALIASES
        for alias in aliases
    }
    entity_terms.update(
        _normalize_match_text(person)
        for person in _normalize_list(item.get("people"))
        if _normalize_match_text(person)
    )
    entity_terms.update(BROAD_CONTEXT_TERMS)
    subject_terms = {
        term for term in entity_terms if term and term in _normalize_match_text(question_text)
    }
    specific_exact = [
        match
        for match in exact_matches
        if match not in subject_terms
        and match not in BROAD_CONTEXT_TERMS
        and match not in WEAK_FUZZY_TERMS
    ]
    specific_fuzzy = [
        phrase
        for phrase in fuzzy_matches
        if phrase not in subject_terms and phrase not in BROAD_CONTEXT_TERMS
    ]
    has_anchor = bool(specific_exact) or any(
        len(phrase) >= 3 for phrase in specific_fuzzy
    )
    if (
        not has_anchor
        and subjects
        and any(
            match in subject_terms for match in [*exact_matches, *fuzzy_matches]
        )
        and any(len(phrase) >= 2 for phrase in specific_fuzzy)
    ):
        has_anchor = True
    if (
        not has_anchor
        and len(subjects or []) >= 2
        and aspect_matches
        and any(
            match in subject_terms for match in [*exact_matches, *fuzzy_matches]
        )
    ):
        has_anchor = True
    if not has_anchor and exact_matches and subject_terms:
        remaining = _normalize_match_text(question_text, question=True)
        for subject in sorted(subject_terms, key=len, reverse=True):
            remaining = remaining.replace(subject, "")
        has_anchor = not remaining
    if not has_anchor:
        return 0, []
    if subjects:
        item_subject_text = _normalize_match_text(
            _item_search_text(item, include_source_refs=False)
        )
        if not any(
            _normalize_match_text(subject) in item_subject_text
            for subject in subjects
            if _normalize_match_text(subject)
        ):
            return 0, []

    # 去重后保留命中顺序，便于回答中展示匹配关键词
    unique_terms = list(dict.fromkeys(matched_terms))
    return score, unique_terms


def _item_search_text(item: dict, *, include_source_refs: bool = True) -> str:
    """Return all textbook text that may distinguish a question dimension."""
    values = []
    for field, terms in _build_field_terms(item).items():
        if field == "source_excerpt" and not include_source_refs:
            continue
        values.extend(terms)
    return " ".join(values)


def _item_subject_text(item: dict) -> str:
    """Use only point-owned names and aliases, excluding shared lesson/source text."""
    values = [str(item.get("title") or "")]
    values.extend(_normalize_list(item.get("people")))
    values.extend(_normalize_keywords(item.get("keywords")))
    return " ".join(values)


def _subject_aliases(subject: str) -> list[str]:
    """Return the explicit term and its common name stem for point association."""
    aliases = [subject]
    stem = re.sub(r"(?:古城遗址|古城|遗址|先民|居民|人)$", "", subject)
    if len(stem) >= 2 and stem not in aliases:
        aliases.append(stem)
    return aliases


def _question_subjects(question_text: str, items: list[dict]) -> list[str]:
    """Find explicit people or textbook keyword subjects named in the question."""
    candidates = []
    for item in items:
        candidates.extend(_normalize_list(item.get("people")))
        candidates.extend(
            keyword
            for keyword in _normalize_keywords(item.get("keywords"))
            if re.search(r"(?:人|文化|古城|遗址|王朝)$", keyword)
        )

    matched = {
        term
        for term in candidates
        if len(term) >= 3 and term in question_text
    }
    matched.update(
        alias
        for aliases in HISTORICAL_NAME_ALIASES
        for alias in aliases
        if alias in question_text
    )
    matched.update(
        anchor
        for anchor in CONTROLLED_EVENT_ANCHORS
        if anchor in question_text
        and any(
            anchor in str(item.get("title") or "")
            or any(anchor in keyword for keyword in _normalize_keywords(item.get("keywords")))
            for item in items
        )
    )
    # Prefer the most specific form, e.g. 北京人用火 over 北京人.
    subjects = [
        term for term in matched if not any(term != other and term in other for other in matched)
    ]
    return sorted(subjects, key=question_text.index)


def _question_aspect_terms(question_text: str, subjects: list[str]) -> list[str]:
    """Build short terms for the requested aspect after removing named subjects."""
    aspect_text = JUDGMENT_INSTRUCTION_PATTERN.sub("", question_text, count=1)
    for subject in subjects:
        aspect_text = aspect_text.replace(subject, "")
    aspect_text = re.sub(r"[^\u4e00-\u9fff]", "", aspect_text)
    ignored = {
        "什么", "怎么", "怎样", "哪些", "是否", "能不", "不能", "已经",
        "方面", "有何", "有什", "分别", "主要", "不同", "了吗", "的是",
    }
    terms = []
    for index in range(len(aspect_text) - 1):
        term = aspect_text[index : index + 2]
        if (
            term not in ignored
            and term not in WEAK_FUZZY_TERMS
            and not any(character in ASPECT_STOP_CHARACTERS for character in term)
            and term not in terms
        ):
            terms.append(term)
    for cue, aliases in QUESTION_DIMENSION_ALIASES.items():
        if cue in aspect_text:
            for alias in aliases:
                if alias not in terms:
                    terms.append(alias)
    return terms


def _question_core_text(question_text: str, subjects: list[str]) -> str:
    """Return the requested object/event after removing entity context."""
    core = _normalize_match_text(question_text, question=True)
    removable = [
        *(_normalize_match_text(subject) for subject in subjects),
        *BROAD_CONTEXT_TERMS,
    ]
    for term in sorted({term for term in removable if term}, key=len, reverse=True):
        core = core.replace(term, "")
    return core


def _item_matches_question_core(item: dict, core_text: str) -> bool:
    """Require a named-subject candidate to cover the requested object/event."""
    if not core_text:
        return True
    matches = _matching_phrases(
        core_text, _item_search_text(item, include_source_refs=False)
    )
    return any(
        len(match) >= 2
        and match not in WEAK_FUZZY_TERMS
        and match not in BROAD_CONTEXT_TERMS
        for match in matches
    )


def _aspect_matches(item: dict, aspect_terms: list[str]) -> list[str]:
    # Shared excerpts may support several sibling points. Dimension ranking must
    # stay point-owned so one shared paragraph does not make every sibling look
    # equally relevant; excerpts still participate in the base keyword score.
    searchable_text = _item_search_text(item, include_source_refs=False)
    return [term for term in aspect_terms if term in searchable_text]


def _retrieve_keyword_history_knowledge(question: str) -> list[dict]:
    """根据用户问题检索相关历史知识。

    Demo mode keeps the original database/JSON behavior. Textbook mode reads
    only the isolated textbook JSON and surfaces data errors to the API.
    """
    question_text = question.strip()
    if not question_text:
        return []

    knowledge_items = load_history_knowledge()
    subjects = _question_subjects(question_text, knowledge_items)
    aspect_terms = _question_aspect_terms(question_text, subjects)
    matched_items = []
    for item in knowledge_items:
        aspect_matches = _aspect_matches(item, aspect_terms)
        score, matched_terms = _score_item(
            question_text, item, subjects, aspect_matches
        )

        if score > 0:
            matched_item = dict(item)
            matched_item.update(
                {
                    "people": _normalize_list(item.get("people")),
                    "keywords": _normalize_keywords(item.get("keywords")),
                    "matched_terms": list(dict.fromkeys(matched_terms + aspect_matches)),
                    "score": score,
                    "aspect_score": len(aspect_matches),
                    "retrieval_mode": "keyword",
                    "retrieval_unit": "knowledge_point",
                }
            )
            matched_items.append(matched_item)

    ranked_items = sorted(
        matched_items,
        key=lambda item: (
            item["aspect_score"],
            item["score"],
            len(item["matched_terms"]),
        ),
        reverse=True,
    )

    # When a question explicitly names multiple subjects, retain the best
    # aspect-relevant item for each before filling the remaining Top K slots.
    selected = []
    if subjects:
        question_core = _question_core_text(question_text, subjects)
        for subject in subjects:
            aliases = _subject_aliases(subject)
            subject_items = [
                item
                for item in ranked_items
                if any(alias in _item_subject_text(item) for alias in aliases)
            ]
            if len(subjects) == 1:
                subject_items = [
                    item
                    for item in subject_items
                    if _item_matches_question_core(item, question_core)
                ]
            subject_item = subject_items[0] if subject_items else None
            if subject_item is not None and subject_item not in selected:
                selected.append(subject_item)
    for item in ranked_items:
        if item not in selected:
            selected.append(item)
        if len(selected) >= TOP_K:
            break
    return selected[:TOP_K]


def get_retrieval_mode() -> str:
    """Return the explicitly configured retrieval strategy."""
    mode = os.getenv("RETRIEVAL_MODE", "keyword").strip().lower()
    if mode not in {"keyword", "vector"}:
        raise KnowledgeDataError(
            "RETRIEVAL_MODE must be either 'keyword' or 'vector'"
        )
    return mode


def retrieve_history_knowledge(question: str) -> list[dict]:
    """Dispatch to the configured strategy without silent fallback."""
    mode = get_retrieval_mode()
    if mode == "keyword":
        return _retrieve_keyword_history_knowledge(question)
    if get_knowledge_mode() != "textbook":
        raise KnowledgeDataError(
            "Vector retrieval requires KNOWLEDGE_MODE=textbook"
        )

    # Imported lazily so the stable keyword baseline does not import or load
    # the embedding runtime.
    from app.services.vector_service import retrieve_textbook_vectors
    from app.services.vector_service import VectorIndexError

    try:
        return retrieve_textbook_vectors(question, top_k=TOP_K)
    except VectorIndexError as exc:
        raise KnowledgeDataError(str(exc)) from exc
