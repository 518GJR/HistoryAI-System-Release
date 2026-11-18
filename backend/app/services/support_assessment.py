"""Interpretable textbook support checks between retrieval and answering."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


METHOD = "interpretable-v2"
CALIBRATION_VERSION = "comparison-matrix-v1"


ASPECT_CUES = {
    "time": ("何时", "什么时候", "哪一年", "年代", "距今", "时间", "多少年"),
    "place": ("哪里", "哪儿", "何处", "地点", "位于", "分布"),
    "reason": ("为什么", "为何", "原因", "背景"),
    # Generic "怎样/如何" wording does not necessarily ask for a process
    # (for example, "处于怎样的状态"). Keep process detection tied to
    # explicit process language instead of the interrogative alone.
    "process": (
        "过程",
        "经历",
        "措施",
        "方法",
        "做法",
        "如何形成",
        "怎样形成",
        "怎么形成",
        "如何发展",
        "怎样发展",
        "如何演变",
        "怎样演变",
    ),
    "result_impact": (
        "作用",
        "意义",
        "影响",
        "结果",
        "好处",
        "价值",
        "说明",
        "反映",
        "表明",
        "证实",
        "解释",
        "标志",
    ),
    "certainty": ("确定", "肯定", "证实", "确认", "能否", "能不能", "是否"),
}

REASON_EVIDENCE_CUES = (
    "因为",
    "由于",
    "原因",
    "背景",
    "为了",
    "目的是",
    "目的在于",
    "旨在",
)

COMPARISON_CUES = ("比较", "相比", "不同", "区别", "异同", "分别")
CONTROLLED_HISTORICAL_ALIASES = (
    ("周王室", "周王", "周天子"),
)
CONTROLLED_EVENT_ANCHORS = ("诸侯争霸",)
JUDGMENT_INSTRUCTION_PATTERN = re.compile(
    r"^\s*(?:请)?判断(?:并)?(?:简要)?说明(?:理由)?\s*[：:，,]?\s*"
)
TOPIC_GROUPS = {
    "tools": ("工具", "制作", "技术", "石器", "骨器", "磨光", "钻孔"),
    "agriculture": ("农作物", "农业", "种植", "栽培", "水稻", "稻作", "粟", "黍"),
    "housing": ("房屋", "住房", "建筑", "居住", "干栏", "半地穴"),
    "livelihood": (
        "生产生活",
        "饲养",
        "渔猎",
        "采集",
        "纺织",
        "制衣",
        "天然漆",
        "陶器",
        "玉器",
        "骨哨",
        "骨耜",
    ),
    "urban": (
        "城市建设",
        "大型城址",
        "高等级建筑",
        "宫殿区",
        "内城",
        "外城",
        "宫城",
    ),
    "burial": ("墓葬", "随葬品", "贵族墓", "普通墓", "人殉"),
    "economy_culture": (
        "经济",
        "文化联系",
        "文化因素",
        "交流",
        "融合",
        "经济发展水平",
    ),
    "civilization_marker": (
        "重要标志",
        "私有制",
        "阶级",
        "国家的产生",
    ),
    "fire": ("用火", "取火", "火"),
    "water_system": ("水利", "水利系统"),
}
GENERIC_TERMS = {
    "什么",
    "哪些",
    "怎样",
    "怎么",
    "如何",
    "为什么",
    "是否",
    "能否",
    "能不能",
    "主要",
    "方面",
    "有什么",
    "有何",
    "不同",
    "区别",
    "比较",
    "分别",
    "说明",
    "反映",
    "表明",
    "确定",
    "已经",
    "了吗",
    "文化",
    "遗址",
    "古城",
    "城市",
    "城市建设",
    "墓葬",
    "随葬品",
    "生产生活",
    "教材",
    "重要标志",
}

FOUNDING_VERBS = ("建立", "创立", "创建")
FOUNDING_RELATION_SUFFIXES = (
    "建立",
    "创立",
    "创建",
    "形成",
    "兴起",
    "灭亡",
    "衰亡",
    "发展",
)
INDIRECT_FOUNDING_CUES = ("帮助", "协助", "支持", "扶持", "扶助", "辅佐", "拥立")
FOUNDING_DATE_PATTERN = (
    r"(?:约\s*)?(?:(?:公元前|公元)\s*)?\d{1,4}年(?![前后间来])"
    r"|(?:约\s*)?(?:(?:公元前|公元)\s*)?[一二三四五六七八九十百千〇零]{1,8}年"
    r"|距今[^，。！？；]{0,16}年"
)
FOUNDING_DATE_RE = re.compile(FOUNDING_DATE_PATTERN)


@dataclass
class SupportAssessment:
    status: str
    reason_codes: list[str]
    required_facets: list[str]
    covered_facets: list[str]
    supporting_source_ids: list[str]
    supporting_knowledge_ids: list[str]
    unsupported_facets: list[str]
    method: str = METHOD
    calibration_version: str = CALIBRATION_VERSION
    facet_knowledge_ids: dict[str, set[str]] = field(default_factory=dict, repr=False)
    facet_evidence_keys: dict[str, set[tuple[str, str]]] = field(
        default_factory=dict, repr=False
    )
    supporting_reference_keys: set[tuple[str, str, str]] = field(
        default_factory=set, repr=False
    )

    def public(self) -> dict:
        return {
            "status": self.status,
            "reason_codes": self.reason_codes,
            "required_facets": self.required_facets,
            "covered_facets": self.covered_facets,
            "supporting_source_ids": self.supporting_source_ids,
            "supporting_knowledge_ids": self.supporting_knowledge_ids,
            "unsupported_facets": self.unsupported_facets,
            "method": self.method,
            "calibration_version": self.calibration_version,
        }


def _lists(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _knowledge_ids(item: dict) -> list[str]:
    return _lists(item.get("knowledge_ids")) or _lists(item.get("knowledge_id"))


def _source_ids(item: dict) -> list[str]:
    return [
        str(reference.get("source_id") or "").strip()
        for reference in item.get("source_refs") or []
        if isinstance(reference, dict) and str(reference.get("source_id") or "").strip()
    ]


def _point_text(point: dict) -> str:
    values = [
        point.get("title"),
        point.get("summary"),
        point.get("time"),
    ]
    values.extend(_lists(point.get("people")))
    values.extend(_lists(point.get("keywords")))
    dimensions = point.get("dimensions") or {}
    if isinstance(dimensions, dict):
        values.extend(dimensions.values())
    return " ".join(str(value or "") for value in values)


def _subject_terms(point: dict) -> list[str]:
    terms = [str(point.get("title") or "")]
    terms.extend(_lists(point.get("people")))
    terms.extend(_lists(point.get("keywords")))
    topic_terms = {
        alias for aliases in TOPIC_GROUPS.values() for alias in aliases
    }
    normalized = []
    for term in terms:
        term = term.strip()
        if (
            len(term) >= 2
            and term not in GENERIC_TERMS
            and term not in topic_terms
            and term not in normalized
        ):
            normalized.append(term)
    return normalized


def _term_aliases(term: str) -> list[str]:
    aliases = [term]
    stems = [term]
    # Signed titles often add a parenthetical qualifier or join two facts in a
    # single title.  Derive only explicit, reasonably long title components;
    # this keeps an anchor tied to catalogued wording instead of enabling
    # arbitrary fuzzy matching.
    title_core = re.sub(r"[（(][^）)]*[）)]$", "", term).strip(" “”?？")
    if len(title_core) >= 3 and title_core not in stems:
        stems.append(title_core)
    for value in list(stems):
        for separator in ("的", "与"):
            component = value.split(separator, 1)[0].strip(" “”?？")
            if (
                len(component) >= 3
                and component not in GENERIC_TERMS
                and component not in stems
            ):
                stems.append(component)
    for value in list(stems):
        for normalized in (
            re.sub(r"(?<=朝)(?:推行|实行|实施)", "", value),
            re.sub(r"的(?:管理方式|管理制度|地方管理)$", "", value),
        ):
            normalized = normalized.strip()
            if len(normalized) >= 2 and normalized not in stems:
                stems.append(normalized)
    for pattern in (
        r"(?:古城遗址|古城|遗址|先民|居民|文化|人)$",
        rf"(?:{'|'.join(FOUNDING_RELATION_SUFFIXES)})$",
    ):
        for value in list(stems):
            stem = re.sub(pattern, "", value).strip()
            if len(stem) >= 2 and stem not in stems:
                stems.append(stem)
    aliases.extend(value for value in stems[1:] if value not in aliases)
    for group in CONTROLLED_HISTORICAL_ALIASES:
        if any(alias in term for alias in group):
            aliases.extend(alias for alias in group if alias not in aliases)
    aliases.extend(
        anchor
        for anchor in CONTROLLED_EVENT_ANCHORS
        if anchor in term and anchor not in aliases
    )
    return aliases


def _question_subjects(question: str, points: list[dict]) -> list[str]:
    matches = []
    for point in points:
        for term in _subject_terms(point):
            aliases = _term_aliases(term)
            direct_matches = [alias for alias in aliases if alias in question]
            if direct_matches:
                matches.append(max(direct_matches, key=len))
    matches = list(dict.fromkeys(matches))
    subjects = [
        term
        for term in matches
        if not any(term != other and term in other for other in matches)
    ]
    return sorted(subjects, key=question.index)


def _required_aspects(question: str) -> list[str]:
    # “称为什么/叫作什么” asks for a name, not a reason. Mask that phrase
    # before applying the otherwise intentionally simple reason cues.
    aspect_question = JUDGMENT_INSTRUCTION_PATTERN.sub("", question, count=1)
    reason_question = re.sub(
        r"(?:被|又被|也被)?称为什么|叫(?:作|做)?什么",
        "",
        aspect_question,
    )
    aspects = []
    for aspect, cues in ASPECT_CUES.items():
        text = reason_question if aspect == "reason" else aspect_question
        if any(cue in text for cue in cues):
            aspects.append(aspect)
    asks_for_founding_agent = (
        any(cue in aspect_question for cue in ("建立者", "创立者", "创建者"))
        or (
            "谁" in aspect_question
            and any(verb in aspect_question for verb in FOUNDING_VERBS)
        )
        or bool(
            re.search(
                r"是.{1,12}(?:建立|创立|创建)(?:的)?(?:吗|么|？|\?)",
                aspect_question,
            )
        )
    )
    if asks_for_founding_agent and "founding_agent" not in aspects:
        aspects.append("founding_agent")
    # K08 asks two independent facts. Keep these controlled entity-specific
    # facets so generic “……是什么” questions retain their existing content
    # behavior instead of being reclassified as dictionary definitions.
    if re.search(r"甲骨文(?:是|指)什么", aspect_question):
        aspects.append("definition")
    if "甲骨文" in aspect_question and any(
        cue in aspect_question for cue in ("历史地位", "文字史上的地位", "地位如何")
    ):
        aspects.append("historical_status")
    if not aspects:
        aspects.append("content")
    return aspects


def _requested_topics(question: str) -> dict[str, tuple[str, ...]]:
    return {
        f"topic:{name}": aliases
        for name, aliases in TOPIC_GROUPS.items()
        if any(alias in question for alias in aliases)
    }


def analyze_question_requirements(question: str, points: list[dict]) -> dict:
    """Return stable question facets shared by assessment and supplementation."""
    subjects = _question_subjects(question, points)
    aspects = _required_aspects(question)
    topics = _requested_topics(question)
    subject_aspects: dict[str, list[str]] = {subject: [] for subject in subjects}
    previous_subjects: list[str] = []
    for clause in re.split(r"[？?。；;！!]", question):
        clause = clause.strip()
        if not clause:
            continue
        clause_subjects = [
            subject
            for subject in subjects
            if any(alias in clause for alias in _term_aliases(subject))
        ]
        # Follow-up clauses such as “建立者是谁” and “它的地位如何” inherit
        # the nearest explicit subject; this is bounded to the current question.
        inherited_subject = False
        if not clause_subjects and previous_subjects:
            clause_subjects = previous_subjects
            inherited_subject = True
        if clause_subjects:
            previous_subjects = clause_subjects
        aspect_clause = (
            f"{' '.join(clause_subjects)} {clause}"
            if inherited_subject
            else clause
        )
        for subject in clause_subjects:
            for aspect in _required_aspects(aspect_clause):
                if aspect not in subject_aspects[subject]:
                    subject_aspects[subject].append(aspect)
    return {
        "subjects": subjects,
        "aspects": aspects,
        "topics": topics,
        "subject_aspects": subject_aspects,
        "is_comparison": (
            any(cue in question for cue in COMPARISON_CUES) and len(subjects) >= 2
        ),
    }


def _point_supports_aspect(
    point: dict, aspect: str, evidence_text: str = ""
) -> bool:
    del point  # Sufficiency must be grounded in the cited textbook excerpt.
    text = evidence_text.strip()
    if aspect == "content":
        return bool(text)
    if aspect == "time":
        return bool(re.search(r"(?:距今|公元|年前|世纪|年)", text))
    if aspect == "place":
        return any(cue in text for cue in ("位于", "分布", "发现于", "地区", "流域"))
    if aspect == "reason":
        if any(cue in text for cue in REASON_EVIDENCE_CUES):
            return True
        # Textbooks also commonly express purpose as “为巩固……，”. Keep
        # this constrained to the start of a sentence/clause so ordinary
        # wording such as “被称为……” cannot establish reason coverage.
        return bool(
            re.search(
                r"(?:^|[。！？；])\s*为(?!何|什么)[^，。！？；]{1,30}[，,]",
                text,
            )
        )
    if aspect == "process":
        return any(
            cue in evidence_text
            for cue in (
                "先",
                "后来",
                "最终",
                "经过",
                "展开",
                "结成",
                "归附",
                "逐渐",
                "从而形成",
                "做过",
                "担任",
                "任职",
                "曾任",
                "掌管",
            )
        )
    if aspect == "result_impact":
        return any(
            cue in text
            for cue in (
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
        )
    if aspect == "certainty":
        return any(cue in text for cue in ("可能", "相传", "目前已知", "之一", "约"))
    if aspect == "definition":
        return bool(
            ("龟甲" in text and "兽骨" in text and "文字" in text)
            or re.search(r"(?:称为|叫作|叫做)[“\"]?甲骨文", text)
        )
    if aspect == "historical_status":
        return any(
            cue in text
            for cue in (
                "年代最早",
                "体系较为完整",
                "汉字发展的重要阶段",
                "具备了汉字的基本结构",
                "具备汉字的基本结构",
            )
        )
    if aspect == "founding_agent":
        # Require a named actor in the cited excerpt, rather than accepting any
        # passage that merely mentions the same dynasty or the act of founding.
        return bool(
            re.search(
                r"(?:^|[，。；、：:\s])[一-鿿·]{1,8}"
                r"(?:建立|创立|创建)(?:起|了)?",
                text,
            )
            or re.search(
                r"由[一-鿿·]{1,8}(?:建立|创立|创建)",
                text,
            )
            or re.search(
                r"(?:^|[，。；、：:\s])[一-鿿·]{1,8}称帝[^。；]{0,24}史称[一-鿿]{2,8}",
                text,
            )
        )
    return False


def _founding_event_target(subject: str, aspect: str) -> str:
    """Return the polity named by a founding relation in the question subject."""
    normalized = subject.strip(" ‘'\"“”《》？?")
    relation = re.match(
        r"^(.{2,24}?)(?:的)?(?:建立|创立|创建)(?:$|与|和|、)", normalized
    )
    if relation:
        return relation.group(1).strip()
    if aspect == "founding_agent":
        return normalized
    return ""


def _direct_founding_relation_in_clause(clause: str, target: str) -> bool:
    """Recognize direct actor-to-founding relations for one target polity."""
    target_pattern = rf"{re.escape(target)}(?:政权|王朝|国家)?"
    actor_patterns = (
        re.compile(
            rf"(?:^|[，,：:\s])(?P<actor>[一-鿿·]{{1,8}})"
            rf"(?:建立|创立|创建)(?:起|了)?[^，,。！？；;]{{0,32}}"
            rf"{target_pattern}"
        ),
        re.compile(
            rf"由(?P<actor>[一-鿿·]{{1,8}})(?:建立|创立|创建)(?:起|了)?"
            rf"[^，,。！？；;]{{0,32}}{target_pattern}"
        ),
        re.compile(
            rf"{target_pattern}[^，,。！？；;]{{0,16}}由"
            rf"(?P<actor>[一-鿿·]{{1,8}})(?:建立|创立|创建)(?:起|了)?"
        ),
        re.compile(
            rf"(?:^|[，,：:\s])(?P<actor>[一-鿿·]{{1,8}})称帝"
            rf"[^。！？；;]{{0,32}}史称[“\"]?{re.escape(target)}"
        ),
    )
    for pattern in actor_patterns:
        for match in pattern.finditer(clause):
            actor = match.group("actor")
            if not any(cue in actor for cue in INDIRECT_FOUNDING_CUES):
                return True
    return False


def _founding_time_relation_in_text(text: str, target: str) -> bool:
    """Require a date and the target founding event in the same sentence."""
    target_pattern = rf"{re.escape(target)}(?:政权|王朝|国家)?"
    target_established_at_date = re.compile(
        rf"{target_pattern}(?:于|在)?[^，,。！？；;]{{0,12}}"
        rf"(?:建立|创立|创建)(?:起|了)?(?:于|在)\s*(?:{FOUNDING_DATE_PATTERN})"
    )
    date_then_target_established = re.compile(
        rf"(?:{FOUNDING_DATE_PATTERN})[^。！？；;]{{0,24}}{target_pattern}"
        rf"[^，,。！？；;]{{0,8}}(?:建立|创立|创建)(?:起|了)?(?!后|以来|之后)"
    )
    target_date_then_established = re.compile(
        rf"{target_pattern}(?:于|在)\s*(?:{FOUNDING_DATE_PATTERN})"
        rf"[^，,。！？；;]{{0,8}}(?:建立|创立|创建)(?:起|了)?(?!后|以来|之后)"
    )
    date_then_establish_target = re.compile(
        rf"(?:{FOUNDING_DATE_PATTERN})[，,\s]{{0,3}}"
        rf"(?:建立|创立|创建)(?:起|了)?[^，,。！？；;]{{0,24}}{target_pattern}"
    )
    for clause in re.split(r"[。！？；;]", text):
        clause = clause.strip()
        if not clause or not FOUNDING_DATE_RE.search(clause):
            continue
        if _direct_founding_relation_in_clause(clause, target):
            return True
        if target_established_at_date.search(clause):
            return True
        if date_then_target_established.search(clause):
            return True
        if target_date_then_established.search(clause):
            return True
        if date_then_establish_target.search(clause):
            return True
    return False


def _evidence_supports_subject_aspect(
    subject: str, aspect: str, evidence_text: str
) -> bool:
    """Validate a requested aspect against the subject-event relation in evidence."""
    text = evidence_text.strip()
    if not text or (subject and not _evidence_matches_subject(subject, text)):
        return False

    target = _founding_event_target(subject, aspect)
    if aspect == "time" and target:
        return _founding_time_relation_in_text(text, target)
    if aspect == "founding_agent" and target:
        return any(
            _direct_founding_relation_in_clause(clause.strip(), target)
            for clause in re.split(r"[。！？；;]", text)
            if clause.strip()
        )
    return _point_supports_aspect({}, aspect, text)


def _point_matches_subject(
    point: dict, subject: str, evidence_text: str = ""
) -> bool:
    subject_aliases = _term_aliases(subject)
    point_aliases = [
        alias
        for term in _subject_terms(point)
        for alias in _term_aliases(term)
    ]
    if any(
        subject_alias == point_alias
        or subject_alias in point_alias
        or point_alias in subject_alias
        for subject_alias in subject_aliases
        for point_alias in point_aliases
    ):
        return True
    return any(alias in evidence_text for alias in subject_aliases)


def _candidate_points(item: dict, point_map: dict[str, dict]) -> list[dict]:
    return [point_map[value] for value in _knowledge_ids(item) if value in point_map]


def _item_evidence_text(item: dict) -> str:
    return " ".join(
        str(reference.get("excerpt") or "")
        for reference in item.get("source_refs") or []
        if isinstance(reference, dict)
    )


def _reference_key(reference: dict) -> tuple[str, str, str]:
    source_id = str(reference.get("source_id") or "").strip()
    fragment_id = str(reference.get("fragment_id") or source_id).strip()
    excerpt = str(reference.get("excerpt") or "").strip()
    return source_id, fragment_id, excerpt


def _evidence_key(reference: dict) -> tuple[str, str]:
    source_id, _, excerpt = _reference_key(reference)
    return source_id, excerpt


def _reference_points(
    item: dict, reference: dict, point_map: dict[str, dict]
) -> list[dict]:
    reference_id = str(reference.get("knowledge_id") or "").strip()
    if reference_id in point_map:
        return [point_map[reference_id]]
    return _candidate_points(item, point_map)


def _evidence_matches_subject(subject: str, evidence_text: str) -> bool:
    text = evidence_text.strip()
    if not text:
        return False
    for expected, conflicting in (("东汉", "西汉"),):
        if expected in subject and expected not in text:
            return False
        if conflicting in subject and conflicting not in text:
            return False
    for expected, conflicting in (("东周", "西周"),):
        if expected in subject and conflicting in text and expected not in text:
            return False
        if conflicting in subject and expected in text and conflicting not in text:
            return False
    aliases = _term_aliases(subject)
    historical_prefixes = ("西周", "东周", "秦朝", "夏朝", "商朝")
    for alias in list(aliases):
        for prefix in historical_prefixes:
            if alias.startswith(prefix) and len(alias) > len(prefix) + 1:
                stripped = alias.removeprefix(prefix)
                aliases.append(stripped)
                if stripped.endswith("制") and len(stripped) >= 3:
                    aliases.append(stripped[:-1])
        for separator in ("与", "的"):
            if separator in alias:
                aliases.extend(
                    part.strip()
                    for part in alias.split(separator)
                    if len(part.strip()) >= 2
                )
        if alias.endswith("制") and len(alias) >= 3:
            aliases.append(alias[:-1])
    return any(alias in text for alias in dict.fromkeys(aliases))


def assess_support(
    question: str,
    retrieval_mode: str,
    candidates: list[dict],
    textbook_points: list[dict],
) -> SupportAssessment:
    """Assess whether linked textbook points cover the question's minimum facets."""
    del retrieval_mode  # Deliberately not used as a sufficiency shortcut.
    point_map = {
        str(point.get("knowledge_id") or ""): point
        for point in textbook_points
        if str(point.get("knowledge_id") or "")
    }
    requirements = analyze_question_requirements(question, textbook_points)
    subjects = requirements["subjects"]
    aspects = requirements["aspects"]
    topics = requirements["topics"]
    subject_aspects = requirements["subject_aspects"]
    evaluation_aspects = list(
        dict.fromkeys(
            [
                *aspects,
                *(
                    aspect
                    for bound_aspects in subject_aspects.values()
                    for aspect in bound_aspects
                ),
            ]
        )
    )
    is_comparison = requirements["is_comparison"]
    required = [f"subject:{subject}" for subject in subjects]
    if is_comparison:
        required.extend(
            f"subject:{subject}:aspect:{aspect}"
            for subject in subjects
            for aspect in aspects
        )
        required.extend(
            f"subject:{subject}:{topic}"
            for subject in subjects
            for topic in topics
        )
        required.append("comparison:all_subjects")
    else:
        required.extend(f"aspect:{aspect}" for aspect in aspects)
        required.extend(topics)

    facet_ids: dict[str, set[str]] = {facet: set() for facet in required}
    facet_evidence_keys: dict[str, set[tuple[str, str]]] = {
        facet: set() for facet in required
    }
    # Keep support at exact reference granularity. A sibling fragment attached
    # to the same point or page must not inherit facets proved by another excerpt.
    reference_support: dict[tuple[str, str, str], dict] = {}

    # Questions without a named textbook subject or a requested textbook topic
    # have no safe anchor. Reject them without inventing an impossible facet.
    has_question_anchor = bool(subjects or topics)
    if has_question_anchor:
        for item in candidates:
            for reference in item.get("source_refs") or []:
                if not isinstance(reference, dict):
                    continue
                reference_key = _reference_key(reference)
                source_id, _, evidence_text = reference_key
                if not source_id or not evidence_text:
                    continue
                evidence_key = _evidence_key(reference)
                entry = reference_support.setdefault(
                    reference_key,
                    {"source_id": source_id, "knowledge_ids": set(), "facets": set()},
                )
                for point in _reference_points(item, reference, point_map):
                    knowledge_id = str(point.get("knowledge_id") or "")
                    if not knowledge_id:
                        continue
                    matched_subjects = [
                        subject
                        for subject in subjects
                        if _evidence_matches_subject(subject, evidence_text)
                        and (
                            not is_comparison
                            or _point_matches_subject(point, subject, "")
                        )
                    ]
                    matched_topics = [
                        topic
                        for topic, aliases in topics.items()
                        if any(alias in evidence_text for alias in aliases)
                    ]
                    is_relevant = bool(matched_subjects)
                    if not subjects:
                        is_relevant = bool(matched_topics)
                    if not is_relevant:
                        continue

                    subject_aspect_matches = {
                        subject: {
                            aspect: (
                                aspect
                                in (subject_aspects.get(subject) or aspects)
                                and _evidence_supports_subject_aspect(
                                    subject, aspect, evidence_text
                                )
                            )
                            for aspect in evaluation_aspects
                        }
                        for subject in matched_subjects
                    }
                    grounded_subjects = []
                    for subject in matched_subjects:
                        bound_aspects = subject_aspects.get(subject) or aspects
                        relation_aspects = [
                            aspect
                            for aspect in bound_aspects
                            if aspect in {"time", "founding_agent"}
                            and _founding_event_target(subject, aspect)
                        ]
                        if relation_aspects and not any(
                            subject_aspect_matches[subject].get(aspect, False)
                            for aspect in relation_aspects
                        ):
                            continue
                        grounded_subjects.append(subject)
                    contributed_facets: set[str] = set()
                    for subject in grounded_subjects:
                        contributed_facets.add(f"subject:{subject}")
                    for aspect in aspects:
                        aspect_subjects = [
                            subject
                            for subject in grounded_subjects
                            if subject_aspect_matches[subject].get(aspect, False)
                        ]
                        if not aspect_subjects:
                            continue
                        if is_comparison:
                            contributed_facets.update(
                                f"subject:{subject}:aspect:{aspect}"
                                for subject in aspect_subjects
                            )
                        else:
                            contributed_facets.add(f"aspect:{aspect}")
                    for topic in matched_topics:
                        if is_comparison:
                            contributed_facets.update(
                                f"subject:{subject}:{topic}"
                                for subject in matched_subjects
                            )
                        else:
                            contributed_facets.add(topic)

                    for facet in contributed_facets:
                        if facet not in facet_ids:
                            continue
                        facet_ids[facet].add(knowledge_id)
                        facet_evidence_keys[facet].add(evidence_key)
                    if contributed_facets:
                        entry["knowledge_ids"].add(knowledge_id)
                        entry["facets"].update(contributed_facets)

    if is_comparison:
        subject_facets = [f"subject:{subject}" for subject in subjects]
        if all(facet_ids.get(facet) for facet in subject_facets):
            facet_ids["comparison:all_subjects"] = set().union(
                *(facet_ids[facet] for facet in subject_facets)
            )
            facet_evidence_keys["comparison:all_subjects"] = set().union(
                *(facet_evidence_keys[facet] for facet in subject_facets)
            )

    covered = [facet for facet in required if facet_ids.get(facet)]
    unsupported = [facet for facet in required if not facet_ids.get(facet)]
    if not candidates:
        status = "insufficient"
        reasons = ["no_candidates"]
    elif not has_question_anchor:
        status = "insufficient"
        reasons = ["no_question_anchor"]
    elif not unsupported:
        status = "sufficient"
        reasons = ["all_required_facets_covered"]
    elif covered:
        status = "partial"
        reasons = ["required_facets_missing"]
    else:
        status = "insufficient"
        reasons = ["no_supported_facets"]

    # Keep the smallest deterministic source set that still covers every
    # currently covered facet. This prevents merely related excerpts from
    # leaking into the answer input when an earlier source already supplies
    # the same evidence.
    uncovered_for_sources = set(covered) - {"comparison:all_subjects"}
    selected_references: dict[tuple[str, str, str], dict] = {}
    remaining_references = {
        key: value for key, value in reference_support.items() if value["facets"]
    }
    while uncovered_for_sources and remaining_references:
        best_key = max(
            remaining_references,
            key=lambda key: len(
                remaining_references[key]["facets"] & uncovered_for_sources
            ),
        )
        best = remaining_references.pop(best_key)
        supplied = best["facets"] & uncovered_for_sources
        if not supplied:
            break
        selected_references[best_key] = best
        uncovered_for_sources.difference_update(supplied)

    # Preserve additional, question-relevant facts from the same source after
    # the minimum covering set is known. The answer layer removes repeated
    # sentences, so a longer sibling excerpt contributes only its new facts.
    selected_source_ids = {
        entry["source_id"] for entry in selected_references.values()
    }
    requested_fact_facets = {
        facet
        for facet in covered
        if facet.startswith("aspect:")
        or facet.startswith("topic:")
        or ":aspect:" in facet
        or ":topic:" in facet
    }
    for reference_key, entry in remaining_references.items():
        if (
            entry["source_id"] in selected_source_ids
            and entry["facets"] & requested_fact_facets
        ):
            selected_references[reference_key] = entry

    supporting_ids = set().union(
        *(entry["knowledge_ids"] for entry in selected_references.values())
    ) if selected_references else set()
    supporting_source_ids = list(
        dict.fromkeys(entry["source_id"] for entry in selected_references.values())
    )
    return SupportAssessment(
        status=status,
        reason_codes=reasons,
        required_facets=required,
        covered_facets=covered,
        supporting_source_ids=supporting_source_ids,
        supporting_knowledge_ids=sorted(supporting_ids),
        unsupported_facets=unsupported,
        facet_knowledge_ids=facet_ids,
        facet_evidence_keys=facet_evidence_keys,
        supporting_reference_keys=set(selected_references),
    )


def evidence_covers_assessment(
    assessment: SupportAssessment, used_evidence: list[dict]
) -> bool:
    """Require the actually used evidence to cover every required facet."""
    if assessment.status != "sufficient" or not used_evidence:
        return False
    used_evidence_keys = {
        (str(source_id), str(evidence.get("text") or "").strip())
        for evidence in used_evidence
        for source_id in evidence.get("source_ids") or []
        if source_id and str(evidence.get("text") or "").strip()
    }
    if assessment.facet_evidence_keys and used_evidence_keys:
        return all(
            bool(used_evidence_keys & assessment.facet_evidence_keys.get(facet, set()))
            for facet in assessment.required_facets
        )
    used_ids = {
        str(knowledge_id)
        for evidence in used_evidence
        for knowledge_id in evidence.get("knowledge_ids") or []
        if knowledge_id
    }
    return all(
        bool(used_ids & assessment.facet_knowledge_ids.get(facet, set()))
        for facet in assessment.required_facets
    )


def supporting_answer_items(
    candidates: list[dict], assessment: SupportAssessment
) -> list[dict]:
    """Keep candidates intact externally while feeding only supporting excerpts inward."""
    source_ids = set(assessment.supporting_source_ids)
    knowledge_ids = set(assessment.supporting_knowledge_ids)
    reference_keys = assessment.supporting_reference_keys
    selected = []
    for item in candidates:
        references = [
            reference
            for reference in item.get("source_refs") or []
            if isinstance(reference, dict)
            and reference.get("source_id") in source_ids
            and (not reference_keys or _reference_key(reference) in reference_keys)
        ]
        if not references or not (set(_knowledge_ids(item)) & knowledge_ids):
            continue
        copy = dict(item)
        copy["source_refs"] = references
        excerpt_text = "\n".join(
            str(reference.get("excerpt") or "").strip()
            for reference in references
            if str(reference.get("excerpt") or "").strip()
        )
        copy["event"] = excerpt_text
        copy["content"] = excerpt_text
        copy["reason"] = ""
        copy["impact"] = ""
        selected.append(copy)
    return selected
