import asyncio
import os
import uuid
import logging
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any

from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime

from openai import OpenAI

from core.database import AnalysisRequest, AnalysisStatus
from services.embedding_service import (
    StructuredEmbeddingService,
    StructuredDatasetSpec,
    ITSD_DATASET_SPEC,
    get_structured_embedding_service,
)

logger = logging.getLogger(__name__)

class RagAnalysisService:
    """RAG 파이프라인 분석 서비스"""
    
    @staticmethod
    def create_analysis_request(
        db: Session, 
        repositories: List[Dict[str, Any]], 
        include_ast: bool = True,
        include_tech_spec: bool = True,
        include_correlation: bool = True,
        group_name: Optional[str] = None,
        analysis_id: Optional[str] = None
    ) -> AnalysisRequest:
        """새로운 분석 요청을 생성합니다."""
        try:
            if not analysis_id:
                analysis_id = str(uuid.uuid4())
            
            db_analysis = AnalysisRequest(
                analysis_id=analysis_id,
                repositories=repositories,
                include_ast=include_ast,
                include_tech_spec=include_tech_spec,
                include_correlation=include_correlation,
                group_name=group_name,
                status=AnalysisStatus.PENDING,
                
            )
            
            db.add(db_analysis)
            db.commit()
            db.refresh(db_analysis)
            
            return db_analysis
        except IntegrityError:
            db.rollback()
            raise ValueError(f"Analysis with ID '{analysis_id}' already exists")
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to create analysis request: {str(e)}")
    
    @staticmethod
    def get_analysis_by_id(db: Session, analysis_id: str) -> Optional[AnalysisRequest]:
        """분석 ID로 분석 요청을 조회합니다."""
        return db.query(AnalysisRequest).filter(AnalysisRequest.analysis_id == analysis_id).first()
    
    @staticmethod
    def get_all_analyses(db: Session, limit: int = 100, offset: int = 0) -> List[AnalysisRequest]:
        """모든 분석 요청을 조회합니다."""
        return db.query(AnalysisRequest).order_by(AnalysisRequest.created_at.desc()).offset(offset).limit(limit).all()
    
    @staticmethod
    def update_analysis_status(
        db: Session, 
        analysis_id: str, 
        status: AnalysisStatus,
        error_message: Optional[str] = None
    ) -> Optional[AnalysisRequest]:
        """분석 상태를 업데이트합니다."""
        try:
            db_analysis = RagAnalysisService.get_analysis_by_id(db, analysis_id)
            if not db_analysis:
                return None
            
            db_analysis.status = status
            db_analysis.updated_at = datetime.utcnow()
            
            if status == AnalysisStatus.COMPLETED:
                db_analysis.completed_at = datetime.utcnow()
            
            if error_message:
                db_analysis.error_message = error_message
            
            db.commit()
            db.refresh(db_analysis)
            
            return db_analysis
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to update analysis status: {str(e)}")
    
    @staticmethod
    def start_analysis(db: Session, analysis_id: str) -> Optional[AnalysisRequest]:
        """분석을 시작 상태로 변경합니다."""
        return RagAnalysisService.update_analysis_status(db, analysis_id, AnalysisStatus.RUNNING)
    
    @staticmethod
    def complete_analysis(db: Session, analysis_id: str) -> Optional[AnalysisRequest]:
        """분석을 완료 상태로 변경합니다."""
        return RagAnalysisService.update_analysis_status(db, analysis_id, AnalysisStatus.COMPLETED)
    
    @staticmethod
    def fail_analysis(db: Session, analysis_id: str, error_message: str) -> Optional[AnalysisRequest]:
        """분석을 실패 상태로 변경합니다."""
        return RagAnalysisService.update_analysis_status(db, analysis_id, AnalysisStatus.FAILED, error_message)


@dataclass(frozen=True)
class AssigneeRecommendationConfig:
    dataset_display_name: str = "ITSD"
    assignee_field: str = "assignee"
    id_field: str = "request_id"
    title_field: str = "title"
    description_field: str = "content"
    system_field: str = "applied_system"
    request_type_field: str = "request_type"
    department_field: str = "assignees_department"
    default_assignee_name: str = "미지정"


class AssigneeRecommendationService:
    """Recommend assignees for new service desk requests using vector search results."""

    def __init__(
        self,
        embedding_service: StructuredEmbeddingService,
        dataset_spec: StructuredDatasetSpec = ITSD_DATASET_SPEC,
        config: Optional[AssigneeRecommendationConfig] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.dataset_spec = dataset_spec
        self.config = config or AssigneeRecommendationConfig()
        self.llm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )

    async def embed_from_path(self, file_path: str, progress_cb=None) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"지정된 파일을 찾을 수 없습니다: {file_path}")
        with open(file_path, "rb") as handle:
            content = handle.read()
        return await self.embed_from_file(content, progress_cb=progress_cb)

    async def embed_from_file(self, file_content: bytes, progress_cb=None) -> int:
        return await asyncio.to_thread(
            self.embedding_service.embed_from_excel_bytes,
            file_content,
            progress_cb,
        )

    async def recommend_assignee(
        self,
        title: str,
        description: str,
        top_k: int = 50,
        page: int = 1,
        page_size: int = 5,
        use_rrf: bool | None = None,
        w_title: float | None = None,
        w_content: float | None = None,
        rrf_k0: int | None = None,
        top_k_each: int | None = None,
    ) -> str:
        cfg = self.config

        fusion_kwargs: Dict[str, Any] = {}
        if use_rrf is not None:
            fusion_kwargs["use_rrf"] = bool(use_rrf)
        if w_title is not None:
            fusion_kwargs["w_title"] = float(w_title)
        if w_content is not None:
            fusion_kwargs["w_content"] = float(w_content)
        if rrf_k0 is not None:
            fusion_kwargs["rrf_k0"] = int(rrf_k0)
        if top_k_each is not None:
            fusion_kwargs["top_k_each"] = int(top_k_each)

        similar_requests = await self.embedding_service.search_similar_records_dual(
            title=title,
            content=description,
            k=top_k,
            **fusion_kwargs,
        )

        if not similar_requests:
            return (
                "유사한 과거 요청을 찾을 수 없어 담당자를 추천할 수 없습니다. "
                "데이터가 충분히 학습되었는지 확인해주세요."
            )

        assignee_counts = Counter()
        assignee_scores = defaultdict(float)
        assignee_systems = defaultdict(list)
        examples_by_assignee = defaultdict(list)

        for item in similar_requests:
            metadata = item.get("metadata", {}) or {}
            assignee = self._safe_str(metadata.get(cfg.assignee_field, cfg.default_assignee_name))
            score = float(item.get("rerank_score", item.get("original_score", 0.0)) or 0.0)
            assignee_counts[assignee] += 1
            assignee_scores[assignee] += score
            system = metadata.get(cfg.system_field)
            if system:
                assignee_systems[assignee].append(self._safe_str(system))
            examples_by_assignee[assignee].append(item)

        assignee_avg = {
            name: assignee_scores[name] / max(1, assignee_counts[name])
            for name in assignee_counts
        }
        ranked_assignees = sorted(
            assignee_counts.keys(),
            key=lambda name: (assignee_avg[name], assignee_counts[name]),
            reverse=True,
        )
        top_assignees = ranked_assignees[:3]

        lines: List[str] = []
        lines.append(f"### {cfg.dataset_display_name} 담당자 추천 결과")
        lines.append("")
        for index, name in enumerate(top_assignees, start=1):
            count = assignee_counts[name]
            avg_score = assignee_scores[name] / max(1, count)
            top_systems = ", ".join([system for system, _ in Counter(assignee_systems[name]).most_common(3)]) or "-"
            lines.append(
                f"**{index}. {name} (유사 사례 {count}건, 평균 유사도 {avg_score:.3f})**"
            )
            lines.append(f"- 주요 시스템 이력: {top_systems}")
            for example_index, example in enumerate(examples_by_assignee[name], start=1):
                metadata = example.get("metadata", {}) or {}
                score = float(example.get("rerank_score", example.get("original_score", 0.0)) or 0.0)
                lines.append(
                    "  - 사례 {idx}: [ID {req_id}] {title} (시스템: {system}, 유형: {req_type}, 유사도: {score:.3f})".format(
                        idx=example_index,
                        req_id=self._safe_str(metadata.get(cfg.id_field)),
                        title=self._safe_str(metadata.get(cfg.title_field)),
                        system=self._safe_str(metadata.get(cfg.system_field)),
                        req_type=self._safe_str(metadata.get(cfg.request_type_field)),
                        score=score,
                    )
                )
            lines.append("")

        # Filter examples for table (top assignees only)
        relevant_examples = [
            example
            for example in similar_requests
            if self._safe_str(example.get("metadata", {}).get(cfg.assignee_field)) in top_assignees
        ]

        table = ["### 유사 사례 상세 — 상위 3명 담당자"]
        table.append("| ID | 제목 | 시스템 | 유형 | 담당자 | 유사도 |")
        table.append("|---:|---|---|---|---|---:|")
        for example in relevant_examples:
            metadata = example.get("metadata", {}) or {}
            score = float(example.get("rerank_score", example.get("original_score", 0.0)) or 0.0)
            table.append(
                "| {request_id} | {title} | {system} | {req_type} | {assignee} | {score:.3f} |".format(
                    request_id=self._escape_table(self._safe_str(metadata.get(cfg.id_field))),
                    title=self._escape_table(self._safe_str(metadata.get(cfg.title_field))),
                    system=self._escape_table(self._safe_str(metadata.get(cfg.system_field))),
                    req_type=self._escape_table(self._safe_str(metadata.get(cfg.request_type_field))),
                    assignee=self._escape_table(self._safe_str(metadata.get(cfg.assignee_field))),
                    score=score,
                )
            )

        table_markdown = "\n".join(table)
        lines.extend(table)
        lines.append("")
        lines.append(
            "> 배정 가이드: 위 추천 순서대로 검토 후 1순위 담당자에게 배정하시고, 부재/부적합 시 다음 순위로 이관하세요."
        )

        fallback_markdown = "\n".join(lines)

        try:
            max_examples_per_assignee = int(os.getenv("LLM_MAX_EXAMPLES_PER_ASSIGNEE", "0"))
        except Exception:
            max_examples_per_assignee = 0

        candidates_payload = []
        for name in top_assignees:
            count = assignee_counts[name]
            avg_score = assignee_scores[name] / max(1, count)
            systems = [system for system, _ in Counter(assignee_systems[name]).most_common(5)]
            sorted_examples = sorted(
                examples_by_assignee[name],
                key=lambda example: float(example.get("rerank_score", example.get("original_score", 0.0)) or 0.0),
                reverse=True,
            )
            limited = (
                sorted_examples[:max_examples_per_assignee]
                if max_examples_per_assignee > 0
                else sorted_examples
            )
            payload_examples = []
            for example in limited:
                metadata = example.get("metadata", {}) or {}
                payload_examples.append(
                    {
                        "request_id": self._safe_str(metadata.get(cfg.id_field)),
                        "title": self._safe_str(metadata.get(cfg.title_field)),
                        "applied_system": self._safe_str(metadata.get(cfg.system_field)),
                        "request_type": self._safe_str(metadata.get(cfg.request_type_field)),
                        "assignee": self._safe_str(metadata.get(cfg.assignee_field)),
                        "score": float(example.get("rerank_score", example.get("original_score", 0.0)) or 0.0),
                    }
                )
            candidates_payload.append(
                {
                    "assignee": name,
                    "count": count,
                    "avg_score": round(avg_score, 4),
                    "top_systems": systems,
                    "examples": payload_examples,
                }
            )

        examples_table = []
        for example in similar_requests[:5]:
            metadata = example.get("metadata", {}) or {}
            score = float(example.get("rerank_score", example.get("original_score", 0.0)) or 0.0)
            examples_table.append(
                {
                    "request_id": self._safe_str(metadata.get(cfg.id_field)),
                    "title": self._safe_str(metadata.get(cfg.title_field)),
                    "applied_system": self._safe_str(metadata.get(cfg.system_field)),
                    "request_type": self._safe_str(metadata.get(cfg.request_type_field)),
                    "assignee": self._safe_str(metadata.get(cfg.assignee_field)),
                    "score": round(score, 4),
                }
            )

        payload = {
            "new_request": {"title": title, "description": description},
            "candidates": candidates_payload,
            "top_examples": examples_table,
        }

        system_message = (
            "You are an expert service desk assignment assistant. "
            "Summarize recommendations in Markdown for operators." 
            "Use only the provided data."
        )
        user_message = (
            f"Generate a Korean report for {cfg.dataset_display_name} with the following sections in Markdown:\n"
            "1) 제목: 'ITSD 담당자 추천 결과'\n"
            "2) 상위 3명 추천: 이름, 처리 건수, 평균 유사도, 주요 시스템 이력(최대 3개), 사례 요약(유사도 포함)\n"
            "3) 유사 사례 상세 표는 포함하지 마세요. 별도 표로 제공됩니다.\n"
            "4) 배정 가이드를 명시하세요.\n"
            "형식 규칙: 굵은 텍스트와 리스트를 활용하고 과도하게 길지 않게. 수치는 소수 3자리까지.\n"
            f"데이터(JSON):\n{payload}"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=os.getenv("ASSIGNEE_REPORT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            generated = response.choices[0].message.content or ""
            return f"{generated}\n\n{table_markdown}"
        except Exception as exc:
            logger.warning("LLM formatting failed; fallback to local markdown: %s", exc)
            return fallback_markdown

    def _safe_str(self, value: Any) -> str:
        if value is None:
            return "N/A"
        return str(value)

    def _escape_table(self, value: str) -> str:
        return value.replace("|", "／")


def get_assignee_recommendation_service(
    embedding_service: StructuredEmbeddingService = Depends(get_structured_embedding_service),
) -> AssigneeRecommendationService:
    return AssigneeRecommendationService(embedding_service=embedding_service)
