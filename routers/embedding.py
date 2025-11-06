import asyncio
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Body, UploadFile, File, status

from models.schemas import (
    SearchRequest, 
    EmbeddingRequest, 
    EmbeddingResponse, 
    EmbeddingData, 
    EmbeddingUsage
)
from services.embedding_service import (
    StructuredDatasetSpec,
    StructuredEmbeddingService,
    ITSD_DATASET_SPEC,
    get_structured_embedding_service,
)
from services.job_status_service import JobStatusStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1", 
    tags=["🔍 Vector Search"],
    responses={
        200: {"description": "검색 성공"},
        400: {"description": "잘못된 요청"},
        500: {"description": "서버 오류"}
    }
)


DATASET_SPECS: Dict[str, StructuredDatasetSpec] = {
    "itsd": ITSD_DATASET_SPEC,
}


def _resolve_dataset(dataset_name: str) -> StructuredDatasetSpec:
    spec = DATASET_SPECS.get(dataset_name.lower())
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_name}")
    return spec


def _resolve_embedding_service(dataset_name: str) -> StructuredEmbeddingService:
    spec = _resolve_dataset(dataset_name)
    return get_structured_embedding_service(spec)


@router.post(
    "/datasets/{dataset_name}/embed-excel",
    summary="Upload structured dataset Excel for embedding",
    tags=["🔍 Vector Search"],
)
async def embed_dataset_from_excel(
    dataset_name: str,
    file: UploadFile = File(..., description="Structured dataset Excel (.xlsx)"),
):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Excel(.xlsx) 파일만 업로드할 수 있습니다.",
        )

    service = _resolve_embedding_service(dataset_name)
    content = await file.read()
    try:
        count = await asyncio.to_thread(service.embed_from_excel_bytes, content)
        return {
            "message": f"{dataset_name} 데이터셋 임베딩이 성공적으로 완료되었습니다.",
            "embedded_count": count,
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        failed_dir = os.path.join("output", "failed_uploads", dataset_name.lower())
        os.makedirs(failed_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_path = os.path.join(failed_dir, f"{timestamp}_{file.filename}")
        try:
            with open(failed_path, "wb") as handle:
                handle.write(content)
        except Exception as write_exc:
            logger.error("Failed to persist failed upload: %s", write_exc)
        logger.error(
            "Dataset embedding failed (dataset=%s, file=%s): %s",
            dataset_name,
            file.filename,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 임베딩 중 오류가 발생했습니다. (오류 파일: {failed_path}) 원인: {exc}",
        )


@router.post(
    "/datasets/{dataset_name}/embed-excel-async",
    summary="Queue structured dataset embedding (async)",
    tags=["🔍 Vector Search"],
)
async def embed_dataset_async(
    dataset_name: str,
    file: UploadFile = File(..., description="Structured dataset Excel (.xlsx)"),
):
    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Excel(.xlsx) 파일만 업로드할 수 있습니다.",
        )

    service = _resolve_embedding_service(dataset_name)
    job_store = JobStatusStore()
    job = job_store.create_job(task=f"{dataset_name}_embed", filename=file.filename)

    uploads_dir = os.path.join("output", "uploads", dataset_name.lower())
    os.makedirs(uploads_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_path = os.path.join(uploads_dir, f"{job['job_id']}_{timestamp}_{file.filename}")

    content = await file.read()
    try:
        with open(saved_path, "wb") as handle:
            handle.write(content)
    except Exception as exc:
        job_store.fail_job(job["job_id"], error=f"파일 저장 실패: {exc}")
        raise HTTPException(status_code=500, detail=f"업로드 파일 저장 실패: {exc}")

    async def _process(job_id: str, path: str) -> None:
        try:
            job_store.start_job(job_id)

            def _progress(progress: float | int, stage: Optional[str] = None) -> None:
                try:
                    JobStatusStore().set_progress(job_id, progress, stage)
                except Exception:
                    pass

            with open(path, "rb") as handle:
                payload = handle.read()
            _progress(5, "file_loaded")
            count = await asyncio.to_thread(
                service.embed_from_excel_bytes,
                payload,
                _progress,
            )
            job_store.complete_job(job_id, result={"embedded_count": int(count) if count is not None else 0})
        except Exception as exc:
            logger.error("Async dataset embedding failed (dataset=%s, job=%s): %s", dataset_name, job_id, exc)
            job_store.fail_job(job_id, error=str(exc))
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    asyncio.create_task(_process(job["job_id"], saved_path))
    return {"job_id": job["job_id"], "status": "queued"}


@router.get(
    "/datasets/{dataset_name}/embed-jobs/{job_id}",
    summary="Retrieve dataset embedding job status",
    tags=["🔍 Vector Search"],
)
async def get_embed_job_status(dataset_name: str, job_id: str):
    job_store = JobStatusStore()
    data = job_store.get_job(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="해당 job_id를 찾을 수 없습니다.")
    return data


@router.post(
    "/datasets/{dataset_name}/embed-local",
    summary="Embed dataset from local default Excel",
    tags=["🔍 Vector Search"],
)
async def embed_dataset_from_local_file(dataset_name: str):
    spec = _resolve_dataset(dataset_name)
    if not spec.default_local_path:
        raise HTTPException(status_code=400, detail="로컬 임베딩이 지원되지 않습니다.")

    path = spec.default_local_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"'{path}' 파일을 찾을 수 없습니다.")

    service = _resolve_embedding_service(dataset_name)
    try:
        with open(path, "rb") as handle:
            content = handle.read()
        count = await asyncio.to_thread(service.embed_from_excel_bytes, content)
        return {
            "message": f"'{path}' 파일의 임베딩이 성공적으로 완료되었습니다.",
            "embedded_count": count,
        }
    except Exception as exc:
        logger.error("Local dataset embedding failed (dataset=%s, path=%s): %s", dataset_name, path, exc)
        raise HTTPException(status_code=500, detail=f"로컬 파일 임베딩 중 오류가 발생했습니다: {exc}")


@router.get(
    "/datasets/{dataset_name}/index-stats",
    summary="Retrieve dataset index statistics",
    tags=["🔍 Vector Search"],
)
async def dataset_index_stats(dataset_name: str):
    service = _resolve_embedding_service(dataset_name)
    try:
        return service.get_index_stats()
    except Exception as exc:
        logger.error("Failed to get dataset index stats (dataset=%s): %s", dataset_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/datasets/{dataset_name}/sample",
    summary="Sample documents for a dataset variant",
    tags=["🔍 Vector Search"],
)
async def dataset_sample(
    dataset_name: str,
    variant: str = "title",
    limit: int = 3,
):
    spec = _resolve_dataset(dataset_name)
    variant = (variant or "").strip().lower()
    allowed_variants = {
        spec.title_variant_name,
        spec.content_variant_name,
        spec.combined_variant_name,
    }
    if variant not in allowed_variants:
        raise HTTPException(
            status_code=400,
            detail=f"variant must be one of: {', '.join(sorted(allowed_variants))}",
        )
    service = _resolve_embedding_service(dataset_name)
    safe_limit = max(1, min(50, int(limit)))
    try:
        return service.sample_documents(variant, limit=safe_limit)
    except Exception as exc:
        logger.error("Failed to sample dataset documents (dataset=%s, variant=%s): %s", dataset_name, variant, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/search", 
    response_model=List[dict],
    summary="벡터 유사도 검색",
    description="""
    **ChromaDB 벡터 데이터베이스에서 유사한 문서를 검색합니다.**
    
    ### 🔍 검색 기능
    - **의미적 검색**: 텍스트의 의미를 이해하여 관련 문서 검색
    - **메타데이터 필터링**: 파일 타입, 언어, 태그 등으로 결과 필터링
    - **분석 결과별 검색**: analysis_id로 특정 분석 결과만 검색
    - **최신 commit 우선 검색**: repository_url로 최신 commit 분석 결과 우선 검색 ⭐ **NEW**
    - **그룹명으로 검색**: group_name으로 특정 그룹에 속한 레포지토리 분석 결과 검색 ⭐ **NEW**
    - **유사도 점수**: 각 결과의 관련성 점수 제공
    
    ### 📝 사용 예시
    ```bash
    # 일반 검색
    curl -X POST "http://localhost:8001/api/v1/search" \
      -H "Content-Type: application/json" \
      -d '{ 
        "query": "Python 함수 정의",
        "k": 5,
        "filter_metadata": {
          "file_type": "python"
        }
      }'
    
    # 특정 분석 결과에서만 검색
    curl -X POST "http://localhost:8001/api/v1/search" \
      -H "Content-Type: application/json" \
      -d '{ 
        "query": "Python 함수 정의",
        "k": 5,
        "analysis_id": "3cbf3db0-fd9e-410c-bdaa-30cdeb9d7d6c"
      }'
    
    # 특정 레포지토리의 최신 commit 분석 결과에서 검색 (NEW!)
    curl -X POST "http://localhost:8001/api/v1/search" \
      -H "Content-Type: application/json" \
      -d '{ 
        "query": "Python 함수 정의",
        "k": 5,
        "repository_url": "https://github.com/octocat/Hello-World.git"
      }

    # 특정 그룹명으로 검색 (NEW!)
    curl -X POST "http://localhost:8001/api/v1/search" \
      -H "Content-Type: application/json" \
      -d '{ 
        "query": "결제 모듈",
        "k": 5,
        "group_name": "PaymentServiceTeam"
      }'
    
    ### 🎯 검색 팁
    - 구체적인 키워드 사용 (예: "FastAPI 라우터" vs "웹 개발")
    - analysis_id로 특정 분석 결과만 검색하여 정확도 향상
    - group_name으로 특정 그룹에 속한 레포지토리 분석 결과만 검색 ⭐ **NEW** # <-- 설명 추가
    - 메타데이터 필터로 결과 범위 제한
    - k 값 조정으로 결과 수 조절 (기본값: 5)
    """,
    response_description="유사한 문서 목록과 유사도 점수"
)
async def search_embeddings(request: SearchRequest = Body(...)):
    """벡터 유사도 검색 - 의미적 검색으로 관련 문서를 찾습니다."""
    try:
        from services.embedding_service import get_embedding_service
        from config.settings import settings
        
        query = request.query
        k = request.k
        filter_metadata = request.filter_metadata
        analysis_id = request.analysis_id
        repository_url = request.repository_url
        group_name = request.group_name

        # analysis_id가 제공된 경우 필터에 추가
        if analysis_id:
            if filter_metadata is None:
                filter_metadata = {}
            filter_metadata["analysis_id"] = analysis_id
            logger.info(f"Searching with analysis_id filter: {analysis_id}")
        
        # group_name이 제공된 경우 필터에 추가
        if group_name:
            if filter_metadata is None:
                filter_metadata = {}
            filter_metadata["group_name"] = group_name
            logger.info(f"Searching with group_name filter: {group_name}")

        embedding_service = get_embedding_service()
        results = embedding_service.search_similar_documents(
            query, 
            k=k, 
            filter_metadata=filter_metadata, 
            repository_url=repository_url  # 최신 commit 분석 결과 우선 검색
        )
        return results
    except Exception as e:
        logger.error(f"Failed to search embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")


@router.get(
    "/stats", 
    response_model=dict,
    summary="벡터 데이터베이스 통계",
    description="""
    **ChromaDB 벡터 데이터베이스의 통계 정보를 조회합니다.**
    
    ### 📊 제공 정보
    - **총 문서 수**: 저장된 문서의 개수
    - **벡터 차원**: 임베딩 벡터의 차원 수
    - **컬렉션 정보**: 데이터베이스 컬렉션 상태
    - **인덱스 상태**: 검색 인덱스 정보
    
    ### 📝 사용 예시
    ```bash
    curl -X GET "http://localhost:8001/api/v1/stats"
    ```
    
    ### 💡 활용 방법
    - 데이터베이스 상태 모니터링
    - 검색 성능 최적화 참고
    - 저장 용량 관리
    """,
    response_description="벡터 데이터베이스 통계 정보"
)
async def get_embedding_stats():
    """벡터 데이터베이스 통계 조회 - ChromaDB의 상태와 통계를 확인합니다."""
    try:
        from services.embedding_service import get_embedding_service
        from config.settings import settings
        
        embedding_service = get_embedding_service()
        stats = embedding_service.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get embedding stats: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}")

@router.post(
    "/embed_rdb_schema",
    response_model=Dict[str, Any],
    summary="RDB 스키마 임베딩",
    description="""
    **MariaDB 데이터베이스의 스키마(테이블, 컬럼) 정보를 추출하여 임베딩하고 ChromaDB에 저장합니다.**
    
    이 작업을 통해 RDB의 구조를 자연어 쿼리로 검색할 수 있게 됩니다.
    """,
    response_description="임베딩 작업 결과"
)
async def embed_rdb_schema():
    """RDB 스키마를 임베딩하여 벡터 저장소에 추가합니다."""
    try:
        from services.rdb_embedding_service import RDBEmbeddingService
        rdb_embedding_service = RDBEmbeddingService()
        result = rdb_embedding_service.extract_and_embed_schema()
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        return result
    except Exception as e:
        logger.error(f"Failed to embed RDB schema: {e}")
        raise HTTPException(status_code=500, detail=f"RDB 스키마 임베딩 중 오류가 발생했습니다: {str(e)}")

@router.post("/embeddings", response_model=EmbeddingResponse, summary="텍스트 임베딩 생성 (OpenAI 호환)")
async def create_text_embeddings(request: EmbeddingRequest):
    """
    OpenAI 호환 형식으로 텍스트 임베딩을 생성합니다.
    """
    try:
        from services.embedding_service import get_embedding_service
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        embedding_service = get_embedding_service()
        
        # Use the new create_embeddings method
        embeddings_vectors = embedding_service.create_embeddings(texts)
        
        embedding_data = []
        total_tokens = 0

        for i, embedding_vector in enumerate(embeddings_vectors):
            embedding_data.append(EmbeddingData(embedding=embedding_vector, index=i))
            total_tokens += len(texts[i].split()) # Simple token estimation

        return EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens)
        )
    except Exception as e:
        logger.error(f"Failed to create text embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"텍스트 임베딩 생성 중 오류가 발생했습니다: {str(e)}")

@router.get("/groups", response_model=List[str], summary="등록된 모든 그룹 이름 조회")
async def get_all_group_names():
    """
    ChromaDB에 저장된 모든 문서에서 유니크한 `group_name` 목록을 조회합니다.
    """
    try:
        from services.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        group_names = embedding_service.get_all_group_names()
        return group_names
    except Exception as e:
        logger.error(f"Failed to get all group names: {e}")
        raise HTTPException(status_code=500, detail=f"그룹 이름 조회 중 오류가 발생했습니다: {str(e)}")
