"""Structured semantic judge adapter. Tests inject a mock and never call a real API."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    model_validator,
)

from app.services.provider_config import provider_config
from .models import CriterionJudgment, MajorErrorCandidate, RubricQuestion

PROMPT_VERSION = "history-fact-judge-v3"
SYSTEM_PROMPT = """你是逐事实语义判定器，不负责算分，也不得修改权重、阈值或评分点。
只依据输入的题目、答案、原始评分点、原始重大错误规则和原始等价表达规则判断。
不得只做关键词匹配；未提及必须判 missing；错误陈述不能判正确；不确定时判 uncertain。
不得扩充输入中没有的等价表达、历史事实或重大错误规则。
同一事实出现互相矛盾陈述时设置 contradiction_detected=true。
重大错误只能输出候选 major_error_candidate=true，不能自行确认。
full、partial、incorrect 必须提供答案中的原文片段；missing 可以没有片段。
只返回严格 JSON，不返回 Markdown。必须使用且仅使用下列 JSON 字段结构：
{
  "criterion_results": [
    {
      "criterion_id": "输入中的评分点ID",
      "status": "full|partial|missing|incorrect|uncertain",
      "evidence_quotes": ["答案原文片段"],
      "rationale": "简短理由",
      "atomic_results": [
        {
          "fact_id": "输入中的原子事实ID",
          "status": "full|partial|missing|incorrect|uncertain",
          "evidence_quotes": ["答案原文片段"],
          "rationale": "简短理由",
          "contradiction_detected": false,
          "major_error_candidate": false
        }
      ],
      "contradiction_detected": false,
      "major_error_candidate": false
    }
  ],
  "directly_addresses_question": true,
  "has_verifiable_fact": true,
  "answer_level_contradiction": false,
  "qualifier_issue_detected": false,
  "major_error_candidates": [
    {
      "criterion_id": "输入中的评分点ID",
      "answer_quote": "触发候选的答案原文片段",
      "rationale": "为何只是待人工确认的重大错误候选"
    }
  ]
}
criterion_results 必须与输入评分点一一对应，不得遗漏、重复或新增ID。每项 rationale 必须非空。
若评分点为 full、partial 或 incorrect，evidence_quotes 必须至少包含一段非空的答案原文。
只有输入提供了 atomic_facts 且需要逐子项判断时才填写 atomic_results，否则返回空数组。
atomic_results 只能使用所属评分点输入的原子事实ID；原子事实级矛盾或重大错误候选必须保留对应布尔标志。
major_error_candidates 只能填写对象数组，不得填写字符串或规则文本；没有候选时返回空数组。
只有评分点输入的 major_error_rules_raw 非空时才能生成候选。模型只返回 criterion_id、answer_quote、rationale；程序将按 criterion_id 绑定权威冻结规则。
候选只触发人工复核，不代表重大错误已确认。"""


class SemanticAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_results: list[CriterionJudgment]
    directly_addresses_question: StrictBool | None
    has_verifiable_fact: StrictBool | None
    answer_level_contradiction: StrictBool = False
    qualifier_issue_detected: StrictBool = False
    major_error_candidates: list[MajorErrorCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_reasoning_fields(self):
        for result in self.criterion_results:
            if not result.rationale.strip():
                raise ValueError(f"{result.criterion_id} is missing a rationale")
            if result.status.value in {"full", "partial", "incorrect"} and not any(
                quote.strip() for quote in result.evidence_quotes
            ):
                raise ValueError(f"{result.criterion_id} is missing answer evidence")
            for atomic in result.atomic_results:
                if not atomic.rationale.strip():
                    raise ValueError(f"{atomic.fact_id} is missing a rationale")
                if atomic.status.value in {"full", "partial", "incorrect"} and not any(
                    quote.strip() for quote in atomic.evidence_quotes
                ):
                    raise ValueError(f"{atomic.fact_id} is missing answer evidence")
        by_id = {result.criterion_id: result for result in self.criterion_results}
        for candidate in self.major_error_candidates:
            result = by_id.get(candidate.criterion_id)
            if result is not None:
                result.major_error_candidate = True
        return self


class SemanticJudgeError(RuntimeError):
    pass


class SemanticJudge(Protocol):
    prompt_version: str
    config_id: str

    def judge(self, question: RubricQuestion, answer_text: str) -> tuple[SemanticAssessment, dict]: ...


def _create_client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout)


def _chat_completions_url(configured_url: str) -> str:
    base = configured_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


class OpenAICompatibleSemanticJudge:
    prompt_version = PROMPT_VERSION

    def __init__(self, config: dict | None = None, retries: int = 0):
        self.config = config or provider_config("EVALUATION") or provider_config("AI")
        if not self.config:
            raise SemanticJudgeError("No EVALUATION_* or AI_* semantic scoring provider is configured")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.retries = retries
        public = f'{self.config["provider"]}|{self.config["url"]}|{self.config["model"]}'
        self.config_id = hashlib.sha256(public.encode("utf-8")).hexdigest()[:16]

    def _payload(self, question: RubricQuestion, answer_text: str) -> dict:
        return {
            "question_id": question.question_id,
            "question": question.question_text,
            "standard_answer": question.standard_answer,
            "answer": answer_text,
            "criteria": [
                {
                    "criterion_id": criterion.criterion_id,
                    "original_text": criterion.original_criterion_text,
                    "atomic_facts": [fact.model_dump(mode="json") for fact in criterion.atomic_facts],
                    "equivalence_rules_raw": criterion.equivalence_rules_raw,
                    "major_error_rules_raw": criterion.major_error_rules,
                    "review_status": criterion.review_status,
                }
                for criterion in question.criteria
            ],
            "allowed_statuses": ["full", "partial", "missing", "incorrect", "uncertain"],
            "answer_level_rules": {
                "contradiction": "set answer_level_contradiction=true and do not force a final label",
                "qualifier": "set qualifier_issue_detected=true when required qualifiers are missing or overstated",
            },
        }

    @staticmethod
    def _is_retryable_http_error(exc: httpx.HTTPError) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            return status_code == 429 or status_code >= 500
        return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))

    def _request_completion(self, messages: list[dict[str, str]]) -> str:
        for attempt in range(self.retries + 1):
            try:
                with _create_client(self.config["timeout"]) as client:
                    response = client.post(
                        _chat_completions_url(self.config["url"]),
                        headers={"Authorization": f'Bearer {self.config["key"]}'},
                        json={
                            "model": self.config["model"], "temperature": 0,
                            "response_format": {"type": "json_object"},
                            "messages": messages,
                        },
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
            except httpx.HTTPError as exc:
                if attempt < self.retries and self._is_retryable_http_error(exc):
                    continue
                raise
        raise AssertionError("semantic request retry loop exhausted without returning or raising")

    @staticmethod
    def _schema_error_text(exc: ValidationError) -> str:
        details = exc.errors(include_url=False, include_input=False)
        return json.dumps(details, ensure_ascii=False, default=str)

    @staticmethod
    def _validate_against_rubric(
        question: RubricQuestion, assessment: SemanticAssessment
    ) -> None:
        criteria = {criterion.criterion_id: criterion for criterion in question.criteria}
        expected = set(criteria)
        actual = [result.criterion_id for result in assessment.criterion_results]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise SemanticJudgeError("Semantic judge criterion IDs do not match the rubric")
        for result in assessment.criterion_results:
            expected_facts = {
                fact.fact_id for fact in criteria[result.criterion_id].atomic_facts
            }
            actual_facts = [item.fact_id for item in result.atomic_results]
            if (
                len(actual_facts) != len(set(actual_facts))
                or not set(actual_facts).issubset(expected_facts)
            ):
                raise SemanticJudgeError(
                    f"Semantic judge atomic fact IDs do not match {result.criterion_id}"
                )
        candidate_ids = {
            candidate.criterion_id for candidate in assessment.major_error_candidates
        }
        if not candidate_ids.issubset(expected):
            raise SemanticJudgeError(
                "Semantic judge major-error candidate IDs do not match the rubric"
            )
        rule_owners: dict[str, set[str]] = {}
        for criterion in question.criteria:
            for rule in criterion.major_error_rules:
                rule_owners.setdefault(rule, set()).add(criterion.criterion_id)
        for candidate in assessment.major_error_candidates:
            frozen_rules = criteria[candidate.criterion_id].major_error_rules
            if not frozen_rules:
                raise SemanticJudgeError(
                    "Semantic judge major-error candidate has no frozen rule for "
                    f"{candidate.criterion_id}"
                )
            if (
                candidate.model_rule_text is not None
                and candidate.model_rule_text in rule_owners
                and candidate.criterion_id not in rule_owners[candidate.model_rule_text]
            ):
                raise SemanticJudgeError(
                    "Semantic judge major-error candidate conflicts with "
                    f"{candidate.criterion_id}"
                )
            candidate.bind_frozen_rules(frozen_rules)
        for result in assessment.criterion_results:
            if (
                result.major_error_candidate
                and not criteria[result.criterion_id].major_error_rules
            ):
                raise SemanticJudgeError(
                    "Semantic judge major-error candidate has no frozen rule for "
                    f"{result.criterion_id}"
                )

    def judge(self, question: RubricQuestion, answer_text: str) -> tuple[SemanticAssessment, dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(self._payload(question, answer_text), ensure_ascii=False)},
        ]
        schema_corrections = 0
        candidate_corrections = 0
        try:
            while True:
                raw_text = self._request_completion(messages)
                raw = json.loads(raw_text)
                try:
                    assessment = SemanticAssessment.model_validate(raw)
                except ValidationError as exc:
                    details_text = self._schema_error_text(exc)
                    if schema_corrections == 1:
                        raise SemanticJudgeError(
                            "Invalid semantic judge response schema: "
                            + details_text
                        ) from exc
                    schema_corrections += 1
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": (
                                "上一次输出未通过原有SemanticAssessment结构校验。"
                                "只修正JSON结构，不得改变题目、答案、评分规则、评分权重或事实判断，"
                                "不得迁移或推断非法字段。请严格按原有结构重新输出且只输出JSON。"
                                "精简校验错误："
                                + details_text
                            ),
                        },
                    ]
                    continue
                try:
                    self._validate_against_rubric(question, assessment)
                except SemanticJudgeError as exc:
                    if (
                        "major-error candidate has no frozen rule" not in str(exc)
                        or candidate_corrections == 1
                    ):
                        raise
                    candidate_corrections += 1
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": (
                                "上一次输出把没有冻结重大错误规则的评分点标记为重大错误候选。"
                                "只修正重大错误候选字段，不得改变题目、答案、评分规则、"
                                "评分权重或其他事实判断，不得迁移该候选。"
                                "请严格按原有SemanticAssessment结构重新输出且只输出JSON。"
                                "错误：" + str(exc)
                            ),
                        },
                    ]
                    continue
                break
        except SemanticJudgeError:
            raise
        except httpx.TimeoutException as exc:
            raise SemanticJudgeError(
                f"Semantic judge request timed out after {self.retries + 1} attempts: {exc}"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise SemanticJudgeError(f"Invalid semantic judge response: {exc}") from exc
        audit_raw = dict(raw)
        audit_raw["major_error_candidates"] = [
            candidate.audit_dump() for candidate in assessment.major_error_candidates
        ]
        return assessment, audit_raw
