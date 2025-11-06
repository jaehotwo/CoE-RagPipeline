import os
import re
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional, Sequence, Callable, Tuple

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document

from models.schemas import AnalysisResult, RepositoryAnalysis, ASTNode
from config.settings import settings
from openai import OpenAI
from utils.token_utils import TokenUtils

logger = logging.getLogger(__name__)


_embedding_service_singleton = None
_structured_embedding_services: Dict[str, "StructuredEmbeddingService"] = {}


def _resolve_bool_env(env_key: Optional[str], default: bool) -> bool:
    if not env_key:
        return default
    value = os.getenv(env_key)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class StructuredDatasetSpec:
    """Configuration for embedding tabular datasets into the vector store."""

    name: str
    group_name: str
    id_column: str
    required_columns: Sequence[str]
    metadata_columns: Sequence[str]
    title_column: Optional[str] = None
    content_column: Optional[str] = None
    combined_template: Optional[str] = None
    title_variant_name: str = "title"
    content_variant_name: str = "content"
    combined_variant_name: str = "combined"
    variant_metadata_key: str = "record_variant"
    source_label: str = "dataset"
    include_title_env: Optional[str] = None
    include_content_env: Optional[str] = None
    include_combined_env: Optional[str] = None
    default_include_title: bool = True
    default_include_content: bool = True
    default_include_combined: bool = True
    fusion_use_rrf_env: Optional[str] = None
    fusion_w_title_env: Optional[str] = None
    fusion_w_content_env: Optional[str] = None
    fusion_rrf_k0_env: Optional[str] = None
    fusion_top_k_each_env: Optional[str] = None
    cross_encoder_model_env: Optional[str] = None
    chunk_size_candidates: Sequence[int] = field(default_factory=lambda: (256, 384, 512))
    chunk_overlap_candidates: Sequence[int] = field(default_factory=lambda: (50, 80, 100))
    target_avg_tokens: int = 350
    replace_on_id: bool = True
    default_local_path: Optional[str] = None


ITSD_DATASET_SPEC = StructuredDatasetSpec(
    name="itsd_requests",
    group_name="itsd_requests",
    id_column="request_id",
    required_columns=("request_id", "title", "content", "assignee", "applied_system"),
    metadata_columns=(
        "request_id",
        "request_group_id",
        "request_type",
        "title",
        "request_status",
        "applied_system",
        "applied_date",
        "requesters_parent_department",
        "requesters_department",
        "requester",
        "requester_employee_id",
        "assignees_parent_department",
        "assignees_department",
        "assignee",
        "assignee_employee_id",
        "registration_date",
    ),
    title_column="title",
    content_column="content",
    combined_template="요청 제목: {title}\n요청 내용: {content}",
    variant_metadata_key="itsd_field",
    source_label="itsd_xlsx",
    include_title_env="ITSD_EMBED_INCLUDE_TITLE",
    include_content_env="ITSD_EMBED_INCLUDE_CONTENT",
    include_combined_env="ITSD_EMBED_INCLUDE_COMBINED",
    default_include_combined=False,
    fusion_use_rrf_env="ITSD_FUSION_USE_RRF",
    fusion_w_title_env="ITSD_FUSION_W_TITLE",
    fusion_w_content_env="ITSD_FUSION_W_CONTENT",
    fusion_rrf_k0_env="ITSD_FUSION_RRF_K0",
    fusion_top_k_each_env="ITSD_FUSION_TOP_K_EACH",
    cross_encoder_model_env="CROSS_ENCODER_MODEL",
    target_avg_tokens=int(os.getenv("OPENAI_EMBED_TARGET_CHUNK_TOKENS", "350")),
    default_local_path="data/itsd_request_data.xlsx",
)


class StructuredEmbeddingService(EmbeddingService):
    """Embed and search structured (tabular) datasets in Chroma."""

    def __init__(
        self,
        dataset_spec: StructuredDatasetSpec,
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
    ) -> None:
        super().__init__(openai_api_key=openai_api_key, openai_api_base=openai_api_base)
        self.dataset_spec = dataset_spec
        self.max_tokens_per_request = int(os.getenv("OPENAI_EMBED_MAX_TOKENS_PER_REQUEST", "250000"))
        self.max_tokens_per_doc = int(os.getenv("OPENAI_EMBED_MAX_TOKENS_PER_DOC", "8000"))
        self.max_docs_per_batch = int(os.getenv("OPENAI_EMBED_MAX_DOCS_PER_BATCH", "128"))
        self.chroma_add_max_docs = max(1, int(os.getenv("CHROMA_ADD_MAX_DOCS", "64")))
        self.llm_client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def embed_from_excel_bytes(self, file_content: bytes, progress_cb: Optional[Callable[[float, Optional[str]], None]] = None) -> int:
        """Parse an Excel file and embed structured records defined by the dataset spec."""

        try:
            df = pd.read_excel(BytesIO(file_content), engine="openpyxl")
        except Exception as exc:  # pragma: no cover - pandas raises many subclasses
            logger.error("Failed to parse Excel into dataframe: %s", exc)
            raise ValueError("Excel 파일을 파싱할 수 없습니다. 파일이 손상되었거나 형식이 올바르지 않을 수 있습니다.") from exc

        if callable(progress_cb):
            try:
                progress_cb(2, "parsing_excel")
            except Exception:  # pragma: no cover - defensive
                pass

        missing = [col for col in self.dataset_spec.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Excel 파일에 필수 컬럼이 없습니다: {missing}")

        record_count = len(df.index)
        logger.info(
            "Embedding dataset '%s' from Excel with %s rows", self.dataset_spec.name, record_count
        )

        sample_texts: List[str] = []
        sanitized_rows: List[Dict[str, Any]] = []
        for raw_row in df.to_dict(orient="records"):
            row: Dict[str, Any] = {}
            for key, value in raw_row.items():
                if pd.isna(value):
                    continue
                row[key] = self._sanitize_text(str(value) if isinstance(value, str) else value)
            sanitized_rows.append(row)
            content_col = self.dataset_spec.content_column
            if content_col and content_col in row and isinstance(row[content_col], str):
                sample_texts.append(TokenUtils.sanitize_text_basic(row[content_col]))

        if not sample_texts:
            # Fallback: use combined template previews or titles as sampling source
            combined_samples = [
                self._format_combined(row)
                for row in sanitized_rows[: min(20, len(sanitized_rows))]
                if self._format_combined(row)
            ]
            sample_texts = [TokenUtils.sanitize_text_basic(text) for text in combined_samples if text]

        if sample_texts:
            best_cs, best_overlap = TokenUtils.choose_split_params(
                sample_texts[: min(20, len(sample_texts))],
                list(self.dataset_spec.chunk_size_candidates),
                list(self.dataset_spec.chunk_overlap_candidates),
                float(self.dataset_spec.target_avg_tokens),
            )
        else:
            best_cs, best_overlap = 1000, 200

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=best_cs,
            chunk_overlap=best_overlap,
            length_function=len,
        )

        include_title = _resolve_bool_env(
            self.dataset_spec.include_title_env,
            self.dataset_spec.default_include_title,
        )
        include_content = _resolve_bool_env(
            self.dataset_spec.include_content_env,
            self.dataset_spec.default_include_content,
        )
        include_combined = _resolve_bool_env(
            self.dataset_spec.include_combined_env,
            self.dataset_spec.default_include_combined,
        )

        texts_to_embed: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        stats: Dict[str, int] = {}

        if include_combined:
            for row in sanitized_rows:
                combined_text = self._format_combined(row)
                if not combined_text:
                    continue
                chunks = splitter.split_text(combined_text)
                total_chunks = len(chunks)
                for idx, chunk in enumerate(chunks):
                    meta = self._build_metadata(row)
                    meta[self.dataset_spec.variant_metadata_key] = self.dataset_spec.combined_variant_name
                    meta["chunk_index"] = idx
                    meta["total_chunks"] = total_chunks
                    texts_to_embed.append(chunk)
                    metadatas.append(meta)
                stats[self.dataset_spec.combined_variant_name] = stats.get(self.dataset_spec.combined_variant_name, 0) + total_chunks

        if include_title and self.dataset_spec.title_column:
            for row in sanitized_rows:
                title = str(row.get(self.dataset_spec.title_column) or "")
                if not title:
                    continue
                meta = self._build_metadata(row)
                meta[self.dataset_spec.variant_metadata_key] = self.dataset_spec.title_variant_name
                meta["chunk_index"] = 0
                meta["total_chunks"] = 1
                texts_to_embed.append(title)
                metadatas.append(meta)
                stats[self.dataset_spec.title_variant_name] = stats.get(self.dataset_spec.title_variant_name, 0) + 1

        if include_content and self.dataset_spec.content_column:
            for row in sanitized_rows:
                content = str(row.get(self.dataset_spec.content_column) or "")
                if not content:
                    continue
                sanitized = TokenUtils.sanitize_text_basic(content)
                chunks = splitter.split_text(sanitized)
                total_chunks = len(chunks)
                for idx, chunk in enumerate(chunks):
                    meta = self._build_metadata(row)
                    meta[self.dataset_spec.variant_metadata_key] = self.dataset_spec.content_variant_name
                    meta["chunk_index"] = idx
                    meta["total_chunks"] = total_chunks
                    texts_to_embed.append(chunk)
                    metadatas.append(meta)
                stats[self.dataset_spec.content_variant_name] = stats.get(self.dataset_spec.content_variant_name, 0) + total_chunks

        if callable(progress_cb):
            try:
                progress_cb(10, "preparing_documents")
            except Exception:  # pragma: no cover - defensive callback handling
                pass

        stored_ids = self.embed_and_store(
            texts_to_embed,
            metadatas,
            group_name=self.dataset_spec.group_name,
            replace_by_record_id=self.dataset_spec.replace_on_id,
            progress_cb=progress_cb,
        )

        logger.info(
            "Embedded %s documents for dataset '%s' (chunk_size=%s, overlap=%s, stats=%s)",
            len(stored_ids),
            self.dataset_spec.name,
            best_cs,
            best_overlap,
            stats,
        )
        return len(stored_ids)

    def embed_and_store(
        self,
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        group_name: str,
        replace_by_record_id: bool = False,
        progress_cb: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> List[str]:
        if len(texts) != len(metadatas):
            raise ValueError("Texts and metadatas must have the same length.")

        if replace_by_record_id:
            self._purge_existing_records(metadatas, group_name)

        documents: List[Document] = []
        for text, metadata in zip(texts, metadatas):
            safe_text = self._sanitize_text(text)
            safe_meta = self._sanitize_metadata(metadata)
            safe_meta["group_name"] = group_name
            documents.append(Document(page_content=safe_text, metadata=safe_meta))

        if not documents:
            logger.warning("No documents to store for group '%s'", group_name)
            return []

        guarded_docs: List[Document] = []
        for doc in documents:
            guarded_docs.extend(self._split_document_if_needed(doc))

        batches = self._batch_by_token_budget(guarded_docs)
        logger.info(
            "Embedding %s docs for group '%s' in %s batch(es)", len(guarded_docs), group_name, len(batches)
        )

        total_ids: List[str] = []
        processed_docs = 0
        total_docs = len(guarded_docs)

        if callable(progress_cb):
            try:
                progress_cb(15, "embedding_started")
            except Exception:  # pragma: no cover
                pass

        for batch_index, batch in enumerate(batches, start=1):
            for offset in range(0, len(batch), self.chroma_add_max_docs):
                sub_batch = batch[offset : offset + self.chroma_add_max_docs]
                explicit_ids = self._build_document_ids(sub_batch)
                if explicit_ids:
                    stored = self.vectorstore.add_documents(sub_batch, ids=explicit_ids)
                else:
                    stored = self.vectorstore.add_documents(sub_batch)
                total_ids.extend(stored)
                processed_docs += len(sub_batch)
                if callable(progress_cb) and total_docs > 0:
                    try:
                        ratio = processed_docs / total_docs
                        pct = 15 + int(ratio * 83)
                        progress_cb(min(98, max(15, pct)), f"embedding_batch_{batch_index}")
                    except Exception:
                        pass

        return total_ids

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    async def search_similar_records(
        self,
        query: str,
        k: int = 5,
        disable_rerank_env: str = "DISABLE_RERANK",
    ) -> List[Dict[str, Any]]:
        filter_metadata = {"group_name": self.dataset_spec.group_name}
        return self._search_with_optional_rerank(
            query=query,
            k=k,
            filter_metadata=filter_metadata,
            disable_rerank_env=disable_rerank_env,
        )

    async def search_similar_records_dual(
        self,
        title: str,
        content: str,
        k: int = 50,
        w_title: float = 0.4,
        w_content: float = 0.6,
        use_rrf: bool = True,
        rrf_k0: int = 60,
        top_k_each: Optional[int] = None,
        cross_encoder_top_n: int = 150,
        cross_encoder_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        spec = self.dataset_spec
        if not spec.title_column or not spec.content_column:
            # Dual search not applicable; fallback to simple search.
            query = self._format_combined({
                spec.title_column or "title": title,
                spec.content_column or "content": content,
            })
            if not query:
                query = f"{title}\n{content}"
            return await self.search_similar_records(query, k=min(50, k))

        use_rrf = self._resolve_dual_env_flag(use_rrf, spec.fusion_use_rrf_env)
        w_title = self._resolve_dual_env_float(w_title, spec.fusion_w_title_env)
        w_content = self._resolve_dual_env_float(w_content, spec.fusion_w_content_env)
        rrf_k0 = self._resolve_dual_env_int(rrf_k0, spec.fusion_rrf_k0_env)
        top_k_each = self._resolve_dual_env_int(top_k_each, spec.fusion_top_k_each_env)

        if top_k_each is None:
            top_k_each = max(k, 50)
        try:
            initial_pool_cap = int(os.getenv("INITIAL_RERANK_POOL", "0"))
        except Exception:
            initial_pool_cap = 0
        per_field_k = min(top_k_each, initial_pool_cap) if initial_pool_cap > 0 else top_k_each

        filter_title = {
            "group_name": spec.group_name,
            spec.variant_metadata_key: spec.title_variant_name,
        }
        filter_content = {
            "group_name": spec.group_name,
            spec.variant_metadata_key: spec.content_variant_name,
        }

        try:
            title_results = self.vectorstore.similarity_search_with_score(title, k=per_field_k, filter=filter_title)
            content_results = self.vectorstore.similarity_search_with_score(content, k=per_field_k, filter=filter_content)
        except Exception as exc:
            logger.exception("Dual-field similarity search failed: %s", exc)
            combined_query = self._format_combined({
                spec.title_column: title,
                spec.content_column: content,
            }) or f"{title}\n{content}"
            return await self.search_similar_records(combined_query, k=k)

        if not title_results and not content_results:
            combined_query = self._format_combined({
                spec.title_column: title,
                spec.content_column: content,
            }) or f"{title}\n{content}"
            return await self.search_similar_records(combined_query, k=k)

        metric = self._detect_metric()
        title_scores = self._scores_by_record(title_results, metric)
        content_scores = self._scores_by_record(content_results, metric)

        from services.reranking_service import rrf_fusion, CrossEncoderReranker

        fused_scores: Dict[str, float]
        if use_rrf:
            rankings: Dict[str, Dict[str, int]] = {}
            for rank, record_id in enumerate(title_scores.keys(), start=1):
                rankings.setdefault(record_id, {})[spec.title_variant_name] = rank
            for rank, record_id in enumerate(content_scores.keys(), start=1):
                rankings.setdefault(record_id, {})[spec.content_variant_name] = rank
            fused_scores = rrf_fusion(rankings, k0=rrf_k0)
        else:
            fused_scores = {}
            for record_id in set(title_scores) | set(content_scores):
                fused_scores[record_id] = (
                    w_title * title_scores.get(record_id, 0.0)
                    + w_content * content_scores.get(record_id, 0.0)
                )

        representative_docs = self._representative_docs(title_results, content_results, metric)

        ranked_ids = sorted(
            fused_scores.keys(),
            key=lambda rid: fused_scores[rid],
            reverse=True,
        )

        final_docs: List[Dict[str, Any]] = []
        reranker: Optional[CrossEncoderReranker] = None
        if cross_encoder_top_n > 0:
            model_name = self._resolve_cross_encoder_model(cross_encoder_model)
            try:
                reranker = CrossEncoderReranker(model_name=model_name)
            except Exception as exc:
                logger.warning("CrossEncoderReranker init failed: %s", exc)

        if reranker and reranker.available:
            candidates: List[Tuple[str, Dict[str, Any]]] = []
            for rid in ranked_ids[:cross_encoder_top_n]:
                doc = representative_docs.get(rid)
                if not doc:
                    continue
                candidates.append((doc["content"], doc["metadata"]))
            combined_query = self._format_combined({
                spec.title_column: title,
                spec.content_column: content,
            }) or f"{title}\n{content}"
            reranked = reranker.rerank(combined_query, candidates, top_n=k)
            seen: set[str] = set()
            for content_text, score, metadata in reranked:
                rid = str((metadata or {}).get(spec.id_column, ""))
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                final_docs.append(
                    {
                        "content": content_text,
                        "metadata": metadata,
                        "original_score": fused_scores.get(rid, 0.0),
                        "rerank_score": float(score),
                    }
                )

        if not final_docs:
            for rid in ranked_ids[: k * 2]:
                doc = representative_docs.get(rid)
                if not doc:
                    continue
                final_docs.append(
                    {
                        **doc,
                        "original_score": fused_scores.get(rid, 0.0),
                        "rerank_score": fused_scores.get(rid, 0.0),
                    }
                )

        final_docs.sort(key=lambda item: item.get("rerank_score", item.get("original_score", 0.0)), reverse=True)
        return final_docs[:k]

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def get_index_stats(self) -> Dict[str, Any]:
        spec = self.dataset_spec
        try:
            collection = getattr(self.vectorstore, "_collection", None)
            if collection is None:
                return {"error": "No collection bound"}

            def _count(where: Dict[str, Any]) -> int:
                try:
                    return int(collection.count(where=where))  # type: ignore[arg-type]
                except Exception:
                    pass
                try:
                    bulk_limit = max(1000, int(os.getenv("DATASET_COUNT_BULK_LIMIT", "200000")))
                    results = collection.get(where=where, include=["ids"], limit=bulk_limit)
                    if isinstance(results, dict):
                        return len(results.get("ids", []) or [])
                except Exception:
                    pass
                try:
                    page_size = max(1, int(os.getenv("DATASET_COUNT_PAGE_SIZE", "1000")))
                    offset = 0
                    total = 0
                    for _ in range(max(1, int(os.getenv("DATASET_COUNT_MAX_PAGES", "2000")))):
                        results = collection.get(limit=page_size, offset=offset, include=["metadatas"])
                        if not isinstance(results, dict):
                            break
                        metadatas = results.get("metadatas", []) or []
                        if not metadatas:
                            break
                        for metadata in metadatas:
                            if isinstance(metadata, dict) and all(metadata.get(k) == v for k, v in where.items()):
                                total += 1
                        if len(metadatas) < page_size:
                            break
                        offset += page_size
                    return total
                except Exception:
                    pass
                try:
                    results = collection.get(where=where, include=["ids"])
                    if isinstance(results, dict):
                        return len(results.get("ids", []) or [])
                except Exception:
                    pass
                return -1

            total_docs = _count({"group_name": spec.group_name})
            variant_counts: Dict[str, int] = {}
            for variant in {
                spec.title_variant_name,
                spec.content_variant_name,
                spec.combined_variant_name,
            }:
                variant_counts[variant] = _count(
                    {
                        "group_name": spec.group_name,
                        spec.variant_metadata_key: variant,
                    }
                )

            return {
                "group": spec.group_name,
                "counts": {
                    "total": total_docs,
                    **variant_counts,
                },
            }
        except Exception as exc:
            logger.error("Failed to fetch index stats: %s", exc)
            return {"error": str(exc)}

    def sample_documents(self, variant: str, limit: int = 3) -> Dict[str, Any]:
        spec = self.dataset_spec
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return {"error": "No collection bound"}

        where = {"group_name": spec.group_name, spec.variant_metadata_key: variant}
        items: List[Dict[str, Any]] = []
        try:
            results = collection.get(where=where, limit=max(1, min(50, limit)), include=["metadatas", "documents"])
            if isinstance(results, dict):
                for metadata, content in zip(results.get("metadatas", []) or [], results.get("documents", []) or []):
                    items.append({"metadata": metadata, "content": content})
        except Exception:
            pass
        if not items:
            try:
                results = collection.get(limit=max(1, min(200, limit * 10)), include=["metadatas", "documents"])
                if isinstance(results, dict):
                    for metadata, content in zip(results.get("metadatas", []) or [], results.get("documents", []) or []):
                        if isinstance(metadata, dict) and all(metadata.get(k) == v for k, v in where.items()):
                            items.append({"metadata": metadata, "content": content})
                            if len(items) >= limit:
                                break
            except Exception:
                pass

        return {"variant": variant, "sample_count": len(items), "items": items[:limit]}

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _sanitize_text(self, text: Any) -> str:
        if not isinstance(text, str):
            text = str(text)
        value = text.replace("\r", " ").replace("\u000d", " ")
        value = re.sub(r"_x[0-9A-Fa-f]{4}_", " ", value)
        value = re.sub(r"[\n\t]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        safe_metadata: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                safe_metadata[key] = self._sanitize_text(value)
            else:
                safe_metadata[key] = value
        return safe_metadata

    def _build_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"source": self.dataset_spec.source_label}
        for column in self.dataset_spec.metadata_columns:
            if column in row and row[column] is not None:
                value = row[column]
                metadata[column] = self._sanitize_text(value) if isinstance(value, str) else str(value)
        return metadata

    def _format_combined(self, row: Dict[str, Any]) -> str:
        template = self.dataset_spec.combined_template
        if not template:
            return ""
        title_val = row.get(self.dataset_spec.title_column or "", "")
        content_val = row.get(self.dataset_spec.content_column or "", "")
        return template.format(title=title_val, content=TokenUtils.sanitize_text_basic(str(content_val or "")))

    def _purge_existing_records(self, metadatas: Sequence[Dict[str, Any]], group_name: str) -> None:
        record_ids = sorted({
            str(meta.get(self.dataset_spec.id_column))
            for meta in metadatas
            if isinstance(meta, dict) and meta.get(self.dataset_spec.id_column) is not None
        })
        if not record_ids:
            return
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return
        for record_id in record_ids:
            try:
                collection.delete(
                    where={
                        "group_name": group_name,
                        self.dataset_spec.id_column: record_id,
                    }
                )
            except Exception as exc:
                logger.warning("Failed to delete existing docs for %s=%s: %s", self.dataset_spec.id_column, record_id, exc)

    def _split_document_if_needed(self, doc: Document) -> List[Document]:
        content = doc.page_content or ""
        tokens = self._estimate_tokens(content)
        if self.max_tokens_per_doc <= 0 or tokens <= self.max_tokens_per_doc:
            return [doc]
        target_tokens = max(500, min(2000, self.max_tokens_per_doc // 2))
        chunk_size_chars = max(500, target_tokens * 4)
        overlap_chars = max(50, int(chunk_size_chars * 0.1))
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size_chars,
                chunk_overlap=overlap_chars,
                length_function=len,
            )
            parts = splitter.split_text(content)
        except Exception:
            parts = [content[i : i + chunk_size_chars] for i in range(0, len(content), chunk_size_chars)]

        docs: List[Document] = []
        for idx, part in enumerate(parts):
            metadata = dict(doc.metadata or {})
            metadata["sub_chunk_index"] = idx
            docs.append(Document(page_content=part, metadata=metadata))
        return docs

    def _batch_by_token_budget(self, documents: Sequence[Document]) -> List[List[Document]]:
        batches: List[List[Document]] = []
        current: List[Document] = []
        current_tokens = 0
        for document in documents:
            token_count = self._estimate_tokens(document.page_content or "")
            if token_count >= self.max_tokens_per_request:
                if current:
                    batches.append(current)
                    current = []
                    current_tokens = 0
                batches.append([document])
                continue
            over_budget = current_tokens + token_count > self.max_tokens_per_request
            over_doc_limit = self.max_docs_per_batch > 0 and len(current) >= self.max_docs_per_batch
            if over_budget or over_doc_limit:
                if current:
                    batches.append(current)
                current = [document]
                current_tokens = token_count
            else:
                current.append(document)
                current_tokens += token_count
        if current:
            batches.append(current)
        return batches

    def _estimate_tokens(self, text: str) -> int:
        return TokenUtils.estimate_tokens(text or "")

    def _build_document_ids(self, documents: Sequence[Document]) -> List[str]:
        ids: List[str] = []
        for doc in documents:
            metadata = doc.metadata or {}
            record_id = metadata.get(self.dataset_spec.id_column)
            chunk_index = metadata.get("chunk_index")
            if record_id is None or chunk_index is None:
                ids = []
                break
            variant = metadata.get(self.dataset_spec.variant_metadata_key, "combined")
            sub_chunk = metadata.get("sub_chunk_index")
            if sub_chunk is not None:
                ids.append(f"{self.dataset_spec.name}:{record_id}:{variant}:{chunk_index}:{sub_chunk}")
            else:
                ids.append(f"{self.dataset_spec.name}:{record_id}:{variant}:{chunk_index}")
        return ids

    def _search_with_optional_rerank(
        self,
        query: str,
        k: int,
        filter_metadata: Dict[str, Any],
        disable_rerank_env: str,
    ) -> List[Dict[str, Any]]:
        try:
            initial_pool_cap = int(os.getenv("INITIAL_RERANK_POOL", "0"))
        except Exception:
            initial_pool_cap = 0
        initial_k = max(k * 5, k)
        if initial_pool_cap > 0:
            initial_k = min(initial_k, initial_pool_cap)

        results = self.vectorstore.similarity_search_with_score(query, k=initial_k, filter=filter_metadata)
        if not results:
            return []

        metric = self._detect_metric()
        docs = [
            {
                "index": idx,
                "content": document.page_content,
                "metadata": document.metadata,
                "original_score": self._distance_to_similarity(score, metric),
            }
            for idx, (document, score) in enumerate(results)
        ]

        if _resolve_bool_env(disable_rerank_env, False):
            docs.sort(key=lambda item: item["original_score"], reverse=True)
            return docs[:k]

        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        current_tokens = 0
        batch_budget = int(os.getenv("RERANK_BATCH_TOKEN_BUDGET", "12000"))
        for item in docs:
            payload = json.dumps({"content": item["content"]}, ensure_ascii=False)
            token_count = TokenUtils.estimate_tokens(payload)
            if token_count >= batch_budget:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([item])
                continue
            if current_tokens + token_count > batch_budget and current_batch:
                batches.append(current_batch)
                current_batch = [item]
                current_tokens = token_count
            else:
                current_batch.append(item)
                current_tokens += token_count
        if current_batch:
            batches.append(current_batch)

        reranked: List[Dict[str, Any]] = []
        for batch in batches:
            idx_map = {entry["index"]: entry for entry in batch}
            try:
                llm_response = self.llm_client.chat.completions.create(
                    model=os.getenv("RERANK_LLM_MODEL", "gpt-4o-mini"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant that reranks documents based on their relevance to a query."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Query: {query}\n\n"
                                "Documents (as JSON with 'index' and 'content'):\n"
                                f"{json.dumps([{key: value for key, value in item.items() if key in {'index', 'content'}} for item in batch], ensure_ascii=False, indent=2)}\n\n"
                                "Return a JSON array of objects with 'index' and 'rerank_score' (0-1)."
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                )
                payload = llm_response.choices[0].message.content or "[]"
                scores = json.loads(payload)
                for score in scores:
                    index = score.get("index")
                    rerank_score = float(score.get("rerank_score", 0.0) or 0.0)
                    base = idx_map.get(index)
                    if not base:
                        continue
                    combined = dict(base)
                    combined["rerank_score"] = rerank_score
                    reranked.append(combined)
            except Exception as exc:
                logger.error("Rerank batch failed; fallback to original scores: %s", exc)
                for item in batch:
                    fallback = dict(item)
                    fallback["rerank_score"] = item["original_score"]
                    reranked.append(fallback)

        reranked.sort(key=lambda item: item.get("rerank_score", item["original_score"]), reverse=True)
        return reranked[:k]

    def _scores_by_record(self, results, metric: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for document, distance in results:
            metadata = document.metadata or {}
            record_id = metadata.get(self.dataset_spec.id_column)
            if record_id is None:
                continue
            similarity = self._distance_to_similarity(distance, metric)
            scores[str(record_id)] = max(scores.get(str(record_id), 0.0), similarity)
        return scores

    def _representative_docs(self, title_results, content_results, metric: str) -> Dict[str, Dict[str, Any]]:
        representatives: Dict[str, Dict[str, Any]] = {}
        for document, distance in list(title_results) + list(content_results):
            metadata = document.metadata or {}
            record_id = metadata.get(self.dataset_spec.id_column)
            if record_id is None:
                continue
            current = representatives.get(str(record_id))
            similarity = self._distance_to_similarity(distance, metric)
            payload = {
                "content": document.page_content,
                "metadata": metadata,
                "original_score": similarity,
            }
            if current is None or similarity > current.get("original_score", 0.0):
                representatives[str(record_id)] = payload
        return representatives

    def _detect_metric(self) -> str:
        try:
            collection = getattr(self.vectorstore, "_collection", None)
            if collection is not None:
                metadata = getattr(collection, "metadata", {}) or {}
                value = metadata.get("hnsw:space")
                if isinstance(value, str):
                    return value.lower()
        except Exception:
            pass
        return "cosine"

    def _distance_to_similarity(self, score: float, metric: str) -> float:
        try:
            distance = float(score)
        except Exception:
            return 0.0
        if metric == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        if distance <= 0:
            return 1.0
        return 1.0 / (1.0 + distance)

    def _resolve_dual_env_flag(self, value: bool, env_key: Optional[str]) -> bool:
        if env_key is None:
            return value
        env = os.getenv(env_key)
        if env is None:
            return value
        return str(env).strip().lower() in {"1", "true", "yes"}

    def _resolve_dual_env_float(self, value: float, env_key: Optional[str]) -> float:
        if env_key is None:
            return value
        env = os.getenv(env_key)
        if env is None:
            return value
        try:
            return float(env)
        except Exception:
            return value

    def _resolve_dual_env_int(self, value: Optional[int], env_key: Optional[str]) -> Optional[int]:
        if env_key is None:
            return value
        env = os.getenv(env_key)
        if env is None:
            return value
        try:
            return int(env)
        except Exception:
            return value

    def _resolve_cross_encoder_model(self, explicit: Optional[str]) -> str:
        if explicit:
            return explicit
        env_key = self.dataset_spec.cross_encoder_model_env
        if env_key:
            env_value = os.getenv(env_key)
            if env_value:
                return env_value
        return os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")


class ItsdEmbeddingService(StructuredEmbeddingService):
    """Backwards compatibility shim for ITSD dataset embedding."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
    ) -> None:
        super().__init__(
            dataset_spec=ITSD_DATASET_SPEC,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
        )


def get_embedding_service() -> "EmbeddingService":
    """Process-wide singleton provider for EmbeddingService.

    Avoids reinitializing embeddings model, Chroma client, and LLM client per request.
    """
    global _embedding_service_singleton
    if _embedding_service_singleton is None:
        _embedding_service_singleton = EmbeddingService()
    return _embedding_service_singleton


def get_structured_embedding_service(
    spec: StructuredDatasetSpec = ITSD_DATASET_SPEC,
) -> "StructuredEmbeddingService":
    """Return a cached StructuredEmbeddingService for the given dataset spec."""

    service = _structured_embedding_services.get(spec.name)
    if service is None:
        service = StructuredEmbeddingService(dataset_spec=spec)
        _structured_embedding_services[spec.name] = service
    return service


def get_itsd_embedding_service() -> ItsdEmbeddingService:
    return ItsdEmbeddingService()


class EmbeddingService:
    """분석 결과를 embedding하고 Chroma에 저장하는 서비스"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 openai_api_base: Optional[str] = None):
        """
        EmbeddingService 초기화
        
        Args:
            openai_api_key: OpenAI API 키
            openai_api_base: OpenAI API 베이스 URL
        """
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.openai_api_base = openai_api_base or os.getenv("OPENAI_API_BASE")
        self.chroma_host = settings.CHROMA_HOST
        self.chroma_port = settings.CHROMA_PORT
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # OpenAI Embeddings 초기화
        embedding_kwargs = {"api_key": self.openai_api_key}
        if self.openai_api_base:
            embedding_kwargs["base_url"] = self.openai_api_base
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
        if model_name:
            embedding_kwargs["model"] = model_name

        self.embeddings = OpenAIEmbeddings(**embedding_kwargs)
        
        # 텍스트 분할기 초기화 (설정 기반)
        from config.settings import settings as _settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(getattr(_settings, "EMBEDDING_CHUNK_SIZE", 1000)),
            chunk_overlap=int(getattr(_settings, "EMBEDDING_CHUNK_OVERLAP", 200)),
            length_function=len,
        )
        
        # Chroma 벡터스토어 초기화
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        try:
            chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to ChromaDB at {self.chroma_host}:{self.chroma_port}. "
                f"Ensure the Chroma server is running and CHROMA_HOST/CHROMA_PORT are correct. Error: {e}"
            )
            raise
        
        self.vectorstore = Chroma(
            client=chroma_client,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        logger.info(f"Connected to ChromaDB server at {self.chroma_host}:{self.chroma_port}, collection: {self.collection_name}")

        # LLM 클라이언트 초기화 (리랭킹용, 필요 시에만)
        self.llm_client = None
        try:
            from config.settings import settings as _settings
            if getattr(_settings, "ENABLE_RERANKING", False):
                from openai import OpenAI  # OpenAI 임포트
                self.llm_client = OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_api_base
                )
                logger.info("LLM client initialized for reranking.")
            else:
                logger.info("LLM reranking disabled by settings (ENABLE_RERANKING=false).")
        except Exception as e:
            logger.warning(f"LLM client init skipped/failed: {e}")
    
    def process_analysis_result(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """
        분석 결과를 처리하여 embedding하고 Chroma에 저장
        
        Args:
            analysis_result: 분석 결과 객체
            
        Returns:
            처리 결과 정보
        """
        try:
            # Build documents from analysis
            documents = self._create_documents_from_analysis(analysis_result)
            # Chunk long documents and attach group_name
            chunked_documents: List[Document] = []
            for doc in documents:
                text = doc.page_content or ""
                group_name = getattr(analysis_result, 'group_name', None)
                chunks = self.text_splitter.split_text(text) if text else []
                total = max(len(chunks), 1)
                if not chunks:
                    # empty or very small doc
                    new_meta = dict(doc.metadata)
                    if group_name:
                        new_meta["group_name"] = group_name
                    new_meta["chunk_index"] = 0
                    new_meta["total_chunks"] = 1
                    chunked_documents.append(Document(page_content=text, metadata=new_meta))
                else:
                    for idx, chunk in enumerate(chunks):
                        new_meta = dict(doc.metadata)
                        if group_name:
                            new_meta["group_name"] = group_name
                        new_meta["chunk_index"] = idx
                        new_meta["total_chunks"] = total
                        chunked_documents.append(Document(page_content=chunk, metadata=new_meta))
            documents = chunked_documents
            
            if not documents:
                logger.warning(f"No documents created for analysis {analysis_result.analysis_id}")
                return {"status": "no_documents", "count": 0}
            
            # 문서들을 Chroma에 저장 (stable IDs)
            import hashlib
            ids: List[str] = []
            for d in documents:
                base = (
                    f"{analysis_result.analysis_id}|"
                    f"{d.metadata.get('document_type','')}|"
                    f"{d.metadata.get('repository_url','')}|"
                    f"{d.metadata.get('file_path','')}|"
                    f"{d.metadata.get('chunk_index',0)}"
                )
                ids.append(hashlib.sha1(base.encode('utf-8')).hexdigest())
            doc_ids = self.vectorstore.add_documents(documents, ids=ids)
            
            logger.info(f"Successfully embedded {len(documents)} documents for analysis {analysis_result.analysis_id}")
            
            return {
                "status": "success",
                "count": len(documents),
                "document_ids": doc_ids,
                "analysis_id": analysis_result.analysis_id
            }
            
        except Exception as e:
            logger.error(f"Failed to process analysis result {analysis_result.analysis_id}: {str(e)}")
            raise
    
    def embed_source_summaries(
        self, 
        summaries: Dict[str, Any], 
        analysis_id: str,
        group_name: Optional[str] = None # <-- 이 파라미터 추가
    ) -> Dict[str, Any]:
        """
        소스코드 요약 결과를 embedding하고 Chroma에 저장
        
        Args:
            summaries: 소스코드 요약 결과
            analysis_id: 분석 ID
            
        Returns:
            처리 결과 정보
        """
        try:
            if not summaries or "summaries" not in summaries:
                logger.warning(f"No summaries found for analysis {analysis_id}")
                return {"status": "no_summaries", "count": 0}
            
            documents = []
            file_summaries = summaries["summaries"]
            
            import hashlib
            ids: List[str] = []
            for file_path, summary_data in file_summaries.items():
                if not summary_data or "summary" not in summary_data:
                    continue
                
                # Chunk summary content
                summary_text = summary_data.get("summary", "")
                chunks = self.text_splitter.split_text(summary_text) if summary_text else []
                total = max(len(chunks), 1)
                if not chunks:
                    chunks = [summary_text]
                for idx, chunk in enumerate(chunks):
                    meta = {
                        "analysis_id": analysis_id,
                        "source_type": "source_summary",
                        "file_path": file_path,
                        "file_name": summary_data.get("file_name", ""),
                        "language": summary_data.get("language", "Unknown"),
                        "file_size": summary_data.get("file_size", 0),
                        "tokens_used": summary_data.get("tokens_used", 0),
                        "summarized_at": summary_data.get("summarized_at", ""),
                        "model_used": summary_data.get("model_used", ""),
                        "file_hash": summary_data.get("file_hash", ""),
                        "chunk_index": idx,
                        "total_chunks": total
                    }
                    if group_name:
                        meta["group_name"] = group_name
                    documents.append(Document(page_content=chunk, metadata=meta))
                    # Stable ID per file+chunk
                    base = f"{analysis_id}|{file_path}|{summary_data.get('file_hash','')}|{idx}"
                    ids.append(hashlib.sha1(base.encode('utf-8')).hexdigest())
            
            if not documents:
                logger.warning(f"No valid summary documents created for analysis {analysis_id}")
                return {"status": "no_valid_summaries", "count": 0}
            
            # 문서들을 Chroma에 저장
            doc_ids = self.vectorstore.add_documents(documents, ids=ids if ids else None)
            
            logger.info(f"Successfully embedded {len(documents)} source summary documents for analysis {analysis_id}")
            
            return {
                "status": "success",
                "count": len(documents),
                "document_ids": doc_ids,
                "analysis_id": analysis_id,
                "source_type": "source_summary"
            }
            
        except Exception as e:
            logger.error(f"Failed to embed source summaries for analysis {analysis_id}: {str(e)}")
            raise
    
    def _create_documents_from_analysis(self, analysis_result: AnalysisResult) -> List[Document]:
        """
        분석 결과로부터 Document 객체들을 생성
        
        Args:
            analysis_result: 분석 결과 객체
            
        Returns:
            Document 객체 리스트
        """
        documents = []
        
        for repo_analysis in analysis_result.repositories:
            # 1. 레포지토리 기본 정보 문서
            repo_summary = self._create_repository_summary(repo_analysis)
            if repo_summary:
                documents.append(Document(
                    page_content=repo_summary,
                    metadata={
                        "analysis_id": analysis_result.analysis_id,
                        "repository_url": str(repo_analysis.repository.url),
                        "repository_name": repo_analysis.repository.name or "unknown",
                        "document_type": "repository_summary",
                        "created_at": analysis_result.created_at.isoformat() if analysis_result.created_at else None,
                        "group_name": getattr(analysis_result, 'group_name', None)
                    }
                ))
            
            # 2. 기술스펙 문서들
            for tech_spec in repo_analysis.tech_specs:
                tech_content = self._create_tech_spec_content(tech_spec)
                if tech_content:
                    documents.append(Document(
                        page_content=tech_content,
                        metadata={
                            "analysis_id": analysis_result.analysis_id,
                            "repository_url": str(repo_analysis.repository.url),
                            "repository_name": repo_analysis.repository.name or "unknown",
                            "document_type": "tech_spec",
                            "language": tech_spec.language,
                            "package_manager": tech_spec.package_manager,
                            "group_name": getattr(analysis_result, 'group_name', None)
                        }
                ))
            
            # 3. AST 분석 결과 문서들
            for file_path, ast_nodes in repo_analysis.ast_analysis.items():
                ast_content = self._create_ast_content(file_path, ast_nodes)
                if ast_content:
                    # 큰 AST 내용은 청크로 분할
                    chunks = self.text_splitter.split_text(ast_content)
                    for i, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "analysis_id": analysis_result.analysis_id,
                                "repository_url": str(repo_analysis.repository.url),
                                "repository_name": repo_analysis.repository.name or "unknown",
                                "document_type": "ast_analysis",
                                "file_path": file_path,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            }
                        ))
            
            # 4. 코드 메트릭 문서
            metrics_content = self._create_metrics_content(repo_analysis)
            if metrics_content:
                documents.append(Document(
                    page_content=metrics_content,
                    metadata={
                        "analysis_id": analysis_result.analysis_id,
                        "repository_url": str(repo_analysis.repository.url),
                        "repository_name": repo_analysis.repository.name or "unknown",
                        "document_type": "code_metrics"
                    }
                ))
        
        # 5. 연관도 분석 문서
        if analysis_result.correlation_analysis:
            correlation_content = self._create_correlation_content(analysis_result.correlation_analysis)
            if correlation_content:
                documents.append(Document(
                    page_content=correlation_content,
                    metadata={
                        "analysis_id": analysis_result.analysis_id,
                        "document_type": "correlation_analysis",
                        "repository_count": len(analysis_result.repositories),
                        "group_name": getattr(analysis_result, 'group_name', None)
                    }
                ))
        
        return documents
    
    def _create_repository_summary(self, repo_analysis: RepositoryAnalysis) -> str:
        """레포지토리 요약 텍스트 생성"""
        summary_parts = []
        
        # 기본 정보
        summary_parts.append(f"Repository: {repo_analysis.repository.name or 'Unknown'}")
        summary_parts.append(f"URL: {repo_analysis.repository.url}")
        if repo_analysis.repository.branch:
            summary_parts.append(f"Branch: {repo_analysis.repository.branch}")
        
        # 파일 통계
        if repo_analysis.files:
            file_count = len(repo_analysis.files)
            languages = set(f.language for f in repo_analysis.files if f.language)
            summary_parts.append(f"Total files: {file_count}")
            if languages:
                summary_parts.append(f"Languages: {', '.join(sorted(languages))}")
        
        # 문서 파일들
        if repo_analysis.documentation_files:
            summary_parts.append(f"Documentation files: {', '.join(repo_analysis.documentation_files)}")
        
        # 설정 파일들
        if repo_analysis.config_files:
            summary_parts.append(f"Configuration files: {', '.join(repo_analysis.config_files)}")
        
        return "\n".join(summary_parts)
    
    def _create_tech_spec_content(self, tech_spec) -> str:
        """기술스펙 내용 생성"""
        content_parts = []
        
        content_parts.append(f"Language: {tech_spec.language}")
        if tech_spec.package_manager:
            content_parts.append(f"Package Manager: {tech_spec.package_manager}")
        
        if tech_spec.dependencies:
            content_parts.append("Dependencies:")
            for dep in tech_spec.dependencies:
                content_parts.append(f"  - {dep}")
        
        return "\n".join(content_parts)
    
    def _create_ast_content(self, file_path: str, ast_nodes: List[ASTNode]) -> str:
        """AST 분석 내용 생성"""
        content_parts = []
        
        content_parts.append(f"File: {file_path}")
        content_parts.append("AST Analysis:")
        
        for node in ast_nodes:
            node_info = f"  {node.type}"
            if node.name:
                node_info += f" '{node.name}'"
            if node.line_start:
                if node.line_end and node.line_end != node.line_start:
                    node_info += f" (lines {node.line_start}-{node.line_end})"
                else:
                    node_info += f" (line {node.line_start})"
            content_parts.append(node_info)
            
            if node.metadata:
                for key, value in node.metadata.items():
                    content_parts.append(f"    {key}: {value}")
        
        return "\n".join(content_parts)
    
    def _create_metrics_content(self, repo_analysis: RepositoryAnalysis) -> str:
        """코드 메트릭 내용 생성"""
        if not repo_analysis.code_metrics:
            return ""
        
        metrics = repo_analysis.code_metrics
        content_parts = []
        
        content_parts.append("Code Metrics:")
        content_parts.append(f"  Lines of code: {metrics.lines_of_code}")
        if hasattr(metrics, 'total_files'):
            content_parts.append(f"  Total files: {metrics.total_files}")
        if hasattr(metrics, 'average_file_size'):
            content_parts.append(f"  Average file size: {metrics.average_file_size:.2f} lines")
        if metrics.cyclomatic_complexity:
            content_parts.append(f"  Cyclomatic complexity: {metrics.cyclomatic_complexity:.2f}")
        if metrics.maintainability_index:
            content_parts.append(f"  Maintainability index: {metrics.maintainability_index:.2f}")
        if metrics.comment_ratio:
            content_parts.append(f"  Comment ratio: {metrics.comment_ratio:.2f}")
        
        if hasattr(metrics, 'language_distribution') and metrics.language_distribution:
            content_parts.append("  Language distribution:")
            for lang, count in metrics.language_distribution.items():
                content_parts.append(f"    {lang}: {count} files")
        
        return "\n".join(content_parts)
    
    def _create_correlation_content(self, correlation_analysis) -> str:
        """연관도 분석 내용 생성"""
        content_parts = []
        
        content_parts.append("Repository Correlation Analysis:")
        
        if correlation_analysis.common_dependencies:
            content_parts.append("Common Dependencies:")
            for dep in correlation_analysis.common_dependencies:
                content_parts.append(f"  - {dep}")
        
        if correlation_analysis.shared_technologies:
            content_parts.append("Shared Technologies:")
            for tech in correlation_analysis.shared_technologies:
                content_parts.append(f"  - {tech}")
        
        if correlation_analysis.architecture_similarity > 0:
            content_parts.append(f"Architecture Similarity Score: {correlation_analysis.architecture_similarity:.2f}")
        
        return "\n".join(content_parts)
    
    def search_similar_documents(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None, repository_url: Optional[str] = None) -> List[Dict]:
        """
        유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter_metadata: 메타데이터 필터
            repository_url: 특정 레포지토리 URL (최신 commit 분석 결과 우선 검색)
            
        Returns:
            유사한 문서들과 점수
        """
        try:
            # 특정 레포지토리의 최신 commit 분석 결과를 우선 검색
            if repository_url and not filter_metadata:
                latest_analysis_id = self._get_latest_analysis_for_repository(repository_url)
                if latest_analysis_id:
                    filter_metadata = {"analysis_id": latest_analysis_id}
                    logger.info(f"Searching with latest analysis for repository {repository_url}: {latest_analysis_id}")
            
            # 필터 적용하여 검색 (초기 후보 확장)
            from config.settings import settings as _settings
            rerank_multiplier = int(getattr(_settings, "RERANK_MULTIPLIER", 5))
            rerank_max_candidates = int(getattr(_settings, "RERANK_MAX_CANDIDATES", 30))
            initial_k = min(max(k * rerank_multiplier, k), rerank_max_candidates)
            if filter_metadata:
                initial_results = self.vectorstore.similarity_search_with_score(
                    query, k=initial_k, filter=filter_metadata
                )
            else:
                initial_results = self.vectorstore.similarity_search_with_score(query, k=initial_k)
            
            if not initial_results:
                return []

            # LLM 기반 리랭킹 (옵션)
            reranked_results = []
            documents_to_rerank = []
            # 트렁케이션 길이
            rerank_content_chars = int(getattr(_settings, "RERANK_CONTENT_CHARS", 1000))
            for i, (doc, original_score) in enumerate(initial_results):
                documents_to_rerank.append({
                    "index": i,
                    "content": (doc.page_content[:rerank_content_chars] if doc.page_content else ""),
                    "metadata": doc.metadata,
                    "original_score": original_score
                })
            
            # LLM 리랭킹 비활성화 시, 원본 점수로 정렬 반환
            if not getattr(_settings, "ENABLE_RERANKING", False) or not self.llm_client:
                reranked_results = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "original_score": score,
                        "rerank_score": score
                    }
                    for doc, score in initial_results
                ]
                reranked_results.sort(key=lambda x: x["original_score"], reverse=True)
                return reranked_results[:k]

            # LLM에 리랭킹 요청 프롬프트 구성
            prompt_messages = [
                {"role": "system", "content": "You rerank documents by relevance to a query. Respond with a JSON array of objects, each with 'index' and 'rerank_score' (0-1)."},
                {"role": "user", "content": f"Query: {query}\n\nDocuments to rerank (JSON array of objects with 'index' and 'content'):\n{json.dumps(documents_to_rerank, ensure_ascii=False)}\n\nReturn only JSON array with 'index' and 'rerank_score'."}
            ]

            try:
                llm_model = getattr(_settings, "RERANK_MODEL", "gpt-4o-mini")
                llm_response = self.llm_client.chat.completions.create(
                    model=llm_model, # 리랭킹에 사용할 LLM 모델
                    messages=prompt_messages,
                    temperature=0.0, # 리랭킹은 창의성보다 정확성이 중요
                    max_tokens=1024 # 충분한 응답 길이
                )
                
                # LLM 응답 파싱
                rerank_output = llm_response.choices[0].message.content
                rerank_scores_list = json.loads(rerank_output)

                # 원본 문서와 리랭크 점수를 결합
                for item in rerank_scores_list:
                    original_doc_info = documents_to_rerank[item["index"]]
                    reranked_results.append({
                        "content": original_doc_info["content"],
                        "metadata": original_doc_info["metadata"],
                        "original_score": original_doc_info["original_score"],
                        "rerank_score": item["rerank_score"]
                    })
                
                # 리랭크 점수를 기준으로 내림차순 정렬하고 상위 k개 선택
                reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

            except Exception as llm_e:
                logger.error(f"LLM reranking failed, falling back to original scores: {llm_e}")
                # LLM 리랭킹 실패 시 원래 유사도 점수를 사용
                reranked_results = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "original_score": score,
                        "rerank_score": score # 리랭크 실패 시 원래 점수를 리랭크 점수로 사용
                    }
                    for doc, score in initial_results
                ]
                reranked_results.sort(key=lambda x: x["original_score"], reverse=True)

            return reranked_results[:k]
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def search_source_summaries(
        self, 
        query: str, 
        analysis_id: Optional[str] = None,
        k: int = 5,
        language_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        소스코드 요약에서 검색
        
        Args:
            query: 검색 쿼리
            analysis_id: 특정 분석 ID로 필터링 (선택사항)
            k: 반환할 결과 수
            language_filter: 특정 언어로 필터링 (선택사항)
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 필터 조건 구성
            filter_dict = {"source_type": "source_summary"}
            
            if analysis_id:
                filter_dict["analysis_id"] = analysis_id
                
            if language_filter:
                filter_dict["language"] = language_filter
            
            # 검색 수행
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # 결과 포맷팅
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                    "file_path": doc.metadata.get("file_path", ""),
                    "language": doc.metadata.get("language", "Unknown"),
                    "file_name": doc.metadata.get("file_name", "")
                })
            
            logger.info(f"Found {len(formatted_results)} source summary results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search source summaries: {str(e)}")
            return []
    
    def _get_latest_analysis_for_repository(self, repository_url: str) -> Optional[str]:
        """
        특정 레포지토리의 최신 commit 분석 ID를 가져옵니다.
        
        Args:
            repository_url: 레포지토리 URL
            
        Returns:
            최신 분석 ID 또는 None
        """
        try:
            from core.database import SessionLocal, RepositoryAnalysis, RepositoryStatus
            
            with SessionLocal() as db:
                # 해당 레포지토리의 완료된 분석 중 최신 것을 가져오기 (commit_date 기준)
                # MariaDB/MySQL에서는 NULLS LAST 대신 CASE WHEN을 사용
                from sqlalchemy import case
                latest_analysis = db.query(RepositoryAnalysis).filter(
                    RepositoryAnalysis.repository_url == repository_url,
                    RepositoryAnalysis.status == RepositoryStatus.COMPLETED
                ).order_by(
                    case(
                        (RepositoryAnalysis.commit_date.is_(None), 1),
                        else_=0
                    ),  # NULL 값을 마지막으로
                    RepositoryAnalysis.commit_date.desc(),  # commit_date가 있는 것을 우선
                    RepositoryAnalysis.updated_at.desc()  # 그 다음은 업데이트 시간 기준
                ).first()
                
                if latest_analysis:
                    logger.info(f"Found latest analysis for {repository_url}: {latest_analysis.analysis_id} (commit: {latest_analysis.commit_hash[:8] if latest_analysis.commit_hash else 'unknown'})")
                    return latest_analysis.analysis_id
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get latest analysis for repository {repository_url}: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 정보 반환"""
        try:
            # Chroma 컬렉션 정보 가져오기
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def get_all_group_names(self) -> List[str]:
        """DB에 저장된 모든 유니크한 group_name을 반환합니다."""
        try:
            collection = self.vectorstore._collection
            # include=['metadatas'] 를 사용하여 메타데이터만 가져옵니다.
            results = collection.get(include=["metadatas"])
            
            group_names = set()
            for metadata in results.get('metadatas', []):
                if metadata and 'group_name' in metadata:
                    group_names.add(metadata['group_name'])
            
            logger.info(f"Found {len(group_names)} unique group names.")
            return sorted(list(group_names))
        except Exception as e:
            logger.error(f"Failed to get all group names: {e}")
            return []

    def create_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a single text string.
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to create embedding for text: {e}")
            raise

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings.
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to create embeddings for texts: {e}")
            raise
