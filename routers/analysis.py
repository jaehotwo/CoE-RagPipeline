from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
import uuid
import logging

from models.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisStatus,
    ItsdRecommendationRequest,
)
from analyzers.git_analyzer import GitAnalyzer
from core.database import get_db
from services.rdb_embedding_service import RDBEmbeddingService # RDBEmbeddingService 임포트
from services.embedding_service import (
    StructuredDatasetSpec,
    ITSD_DATASET_SPEC,
    get_structured_embedding_service,
)
from services.rag_analysis_service import AssigneeRecommendationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1", 
    tags=["🔍 Git Analysis"],
    responses={
        200: {"description": "분석 성공"},
        400: {"description": "잘못된 요청"},
        404: {"description": "분석 결과를 찾을 수 없음"},
        500: {"description": "서버 오류"}
    }
)

# 메모리 캐시 (성능 향상을 위해 유지, 데이터베이스와 함께 사용)
analysis_results = {}


DATASET_SPECS: Dict[str, StructuredDatasetSpec] = {
    "itsd": ITSD_DATASET_SPEC,
}


def _resolve_dataset(dataset_name: str) -> StructuredDatasetSpec:
    spec = DATASET_SPECS.get(dataset_name.lower())
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_name}")
    return spec


def _resolve_recommendation_service(dataset_name: str) -> AssigneeRecommendationService:
    spec = _resolve_dataset(dataset_name)
    embedding_service = get_structured_embedding_service(spec)
    return AssigneeRecommendationService(embedding_service=embedding_service, dataset_spec=spec)

@router.post(
    "/analyze", 
    response_model=dict,
    summary="Git 레포지토리 분석 시작",
    description="""
    **Git 레포지토리들을 심층 분석하여 개발 가이드를 생성합니다.**
    
    ### 🔍 분석 항목
    - **AST 분석**: 코드 구조, 함수, 클래스 추출
    - **기술스펙 분석**: 의존성, 프레임워크, 라이브러리 감지
    - **레포지토리간 연관도**: 공통 패턴, 아키텍처 유사성
    - **문서 수집**: README, doc 폴더, 참조 URL 자동 수집
    
    ### 📝 사용 예시
    ```bash
    curl -X POST "http://localhost:8001/api/v1/analyze" \
      -H "Content-Type: application/json" \
      -d 
      '{ 
        "repositories": [
          {
            "url": "https://github.com/octocat/Hello-World.git",
            "branch": "master"
          }
        ],
        "include_ast": true,
        "include_tech_spec": true,
        "include_correlation": true,
        "group_name": "MyTeamA" # <-- 이 줄 추가
      }'
    ```
    
    ### 🔧 고급 옵션(Enhanced)
    - `include_tree_sitter`: Tree-sitter 기반 AST 분석 포함 여부 (기본 true)
    - `include_static_analysis`: 정적 분석 포함 여부 (기본 true)
    - `include_dependency_analysis`: 의존성 분석 포함 여부 (기본 true)
    - `generate_report`: 분석 리포트 생성 여부 (기본 true)

    기존 `/api/v1/enhanced/*` 엔드포인트는 곧 중단 예정이며, 본 API의 플래그로 대체됩니다.

    ### ⏱️ 처리 시간
    - 소규모 레포지토리: 1-3분
    - 대규모 레포지토리: 5-15분
    - 다중 레포지토리: 레포지토리 수에 비례
    """,
    response_description="분석 시작 확인 및 analysis_id 반환"
)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Git 레포지토리 분석 시작 - 백그라운드에서 비동기 처리됩니다."""
    try:
        from services.analysis_service import AnalysisService, RagRepositoryAnalysisService
        
        # 중복 레포지토리 체크 및 최신 commit 확인
        from analyzers.git_analyzer import GitAnalyzer
        
        existing_analysis_ids = []
        new_repositories = []
        git_analyzer = GitAnalyzer()
        
        for repo in request.repositories:
            repo_url = repo.url
            branch = getattr(repo, 'branch', 'main')
            
            try:
                # 최신 commit 정보 가져오기
                latest_commit_info = git_analyzer.get_latest_commit_info(repo_url, branch)
                latest_commit_hash = latest_commit_info.get('commit_hash')
                
                # 분석이 필요한지 확인 (commit hash 비교)
                analysis_needed, existing_analysis_id = RagRepositoryAnalysisService.check_if_analysis_needed(
                    db, repo_url, branch, latest_commit_hash
                )
                
                if analysis_needed:
                    new_repositories.append(repo)
                    logger.info(f"New analysis needed for {repo_url}:{branch} - commit: {latest_commit_hash[:8] if latest_commit_hash else 'unknown'}")
                else:
                    existing_analysis_ids.append(existing_analysis_id)
                    logger.info(f"Reusing existing analysis for {repo_url}:{branch}: {existing_analysis_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to get commit info for {repo_url}: {e}")
                # commit 정보를 가져올 수 없는 경우 기존 방식으로 fallback
                existing_analysis_id = RagRepositoryAnalysisService.get_analysis_by_repository_url(
                    db, repo_url, branch
                )
                
                if existing_analysis_id:
                    existing_analysis_ids.append(existing_analysis_id)
                    logger.info(f"Found existing analysis for repository {repo_url}: {existing_analysis_id}")
                else:
                    new_repositories.append(repo)
        
        # 모든 레포지토리가 이미 분석된 경우, 가장 최신 분석 결과 반환
        if not new_repositories and existing_analysis_ids:
            latest_analysis_id = existing_analysis_ids[0]  # 가장 최신 것 사용
            logger.info(f"All repositories already analyzed. Returning latest analysis: {latest_analysis_id}")
            return {
                "analysis_id": latest_analysis_id,
                "status": "existing",
                "message": f"모든 레포지토리가 이미 분석되었습니다. 기존 분석 결과를 사용합니다: {latest_analysis_id}"
            }
        
        # 새로운 분석이 필요한 경우
        analysis_id = str(uuid.uuid4())
        
        # 새로운 레포지토리만 포함하는 요청 생성
        if new_repositories:
            new_request = AnalysisRequest(
                repositories=new_repositories,
                include_ast=request.include_ast,
                include_tech_spec=request.include_tech_spec,
                include_correlation=request.include_correlation,
                group_name=request.group_name # <-- 이 줄 추가
            )
        else:
            new_request = request
        
        # 데이터베이스에 AnalysisRequest 레코드 생성 (foreign key constraint를 위해 필요)
        from services.analysis_service import RagAnalysisService
        try:
            # 레포지토리 정보를 딕셔너리 형태로 변환
            repositories_data = []
            for repo in new_request.repositories:
                repositories_data.append({
                    "url": str(repo.url),
                    "branch": repo.branch or "main",
                    "name": repo.name
                })
            
            # 데이터베이스에 AnalysisRequest 생성
            db_analysis_request = RagAnalysisService.create_analysis_request(
                db=db,
                repositories=repositories_data,
                include_ast=new_request.include_ast,
                include_tech_spec=new_request.include_tech_spec,
                include_correlation=new_request.include_correlation,
                analysis_id=analysis_id,
                group_name=new_request.group_name # <-- 이 줄 추가
            )
            logger.info(f"Created AnalysisRequest in database: {analysis_id}")
            
        except Exception as e:
            logger.error(f"Failed to create AnalysisRequest in database: {e}")
            raise HTTPException(status_code=500, detail=f"데이터베이스에 분석 요청을 생성하는 중 오류가 발생했습니다: {str(e)}")
        
        # 분석 결과 초기화 (메모리 캐시용)
        analysis_result = AnalysisResult(
            analysis_id=analysis_id,
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(),
            repositories=[],
            correlation_analysis=None,
            source_summaries_used=False,
            group_name=request.group_name # <-- 이 줄 추가
        )
        
        # 메모리 캐시에 저장
        analysis_results[analysis_id] = analysis_result
        
        # 백그라운드에서 분석 실행 (데이터베이스 세션도 전달)
        analysis_service = AnalysisService()
        background_tasks.add_task(analysis_service.perform_analysis, analysis_id, new_request, analysis_results, db)
        
        message = "분석이 시작되었습니다."
        if existing_analysis_ids:
            message += f" 일부 레포지토리는 기존 분석 결과를 재사용합니다."
        
        return {
            "analysis_id": analysis_id,
            "status": "started",
            "message": f"{message} /results/{analysis_id} 엔드포인트로 결과를 확인하세요.",
            "existing_analyses": existing_analysis_ids if existing_analysis_ids else None
        }
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=f"분석 시작 중 오류가 발생했습니다: {str(e)}")


@router.post(
    "/ingest_rdb_schema",
    summary="RDB 스키마 정보 임베딩",
    description="""
    **MariaDB의 스키마 정보를 추출하여 벡터 데이터베이스에 임베딩합니다.**
    이를 통해 RDB 구조에 대한 질문에 RAG 기반으로 답변할 수 있습니다.
    """,
    response_description="RDB 스키마 임베딩 결과"
)
async def ingest_rdb_schema():
    """RDB 스키마 정보 임베딩 - MariaDB의 테이블 및 컬럼 정보를 벡터화합니다."""
    try:
        rdb_embedding_service = RDBEmbeddingService()
        result = rdb_embedding_service.extract_and_embed_schema()
        return result
    except Exception as e:
        logger.error(f"Failed to ingest RDB schema: {e}")
        raise HTTPException(status_code=500, detail=f"RDB 스키마 임베딩 중 오류가 발생했습니다: {str(e)}")


@router.get("/results/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str, db: Session = Depends(get_db)):
    """분석 결과 조회"""
    try:
        from services.analysis_service import AnalysisService
        
        analysis_service = AnalysisService()
        
        # 먼저 메모리 캐시에서 확인
        if analysis_id in analysis_results:
            return analysis_results[analysis_id]
        
        # 메모리에 없으면 데이터베이스에서 로드 시도
        result = analysis_service.load_analysis_result_from_db(analysis_id, db)
        if result:
            analysis_results[analysis_id] = result  # 메모리에 캐시
            return result
        
        # 데이터베이스에도 없으면 디스크에서 로드 시도 (백워드 호환성)
        result = analysis_service.load_analysis_result(analysis_id)
        if result:
            analysis_results[analysis_id] = result  # 메모리에 캐시
            return result
        
        # 모든 곳에서 찾지 못하면 404 에러
        available_ids = list(analysis_results.keys())
        error_detail = {
            "message": "분석 결과를 찾을 수 없습니다.",
            "analysis_id": analysis_id,
            "available_analysis_ids": available_ids[:5],  # 최대 5개만 표시
            "total_available": len(available_ids),
            "suggestions": [
                "1. 올바른 analysis_id를 사용하고 있는지 확인하세요.",
                "2. /results 엔드포인트로 사용 가능한 분석 결과 목록을 확인하세요.",
                "3. 분석이 아직 진행 중이거나 실패했을 수 있습니다.",
                "4. 분석 ID 형식이 올바른지 확인하세요 (UUID 형식)."
            ]
        }
        raise HTTPException(status_code=404, detail=error_detail)
    except HTTPException:
        # HTTPException은 그대로 재발생
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis result for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=f"분석 결과 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/results", response_model=List[dict])
async def list_analysis_results(db: Session = Depends(get_db)):
    """모든 분석 결과 목록 조회"""
    try:
        from services.analysis_result_service import AnalysisResultService
        
        analysis_result_service = AnalysisResultService()
        
        # 데이터베이스에서 모든 분석 결과 조회
        db_results = analysis_result_service.get_all_analysis_results_from_db(db)
        
        # 메모리 캐시의 결과와 병합
        all_results = {}
        
        # 데이터베이스 결과 추가
        for result in db_results:
            all_results[result.analysis_id] = {
                "analysis_id": result.analysis_id,
                "status": result.status,
                "created_at": result.analysis_date,
                "completed_at": result.completed_at,
                "repository_count": result.repository_count,
                "source": "database"
            }
        
        # 메모리 캐시 결과 추가/업데이트
        for result in analysis_results.values():
            all_results[result.analysis_id] = {
                "analysis_id": result.analysis_id,
                "status": result.status,
                "created_at": result.created_at,
                "completed_at": result.completed_at,
                "repository_count": len(result.repositories),
                "source": "memory"
            }
        
        return list(all_results.values())
    except Exception as e:
        logger.error(f"Failed to list analysis results: {e}")
        raise HTTPException(status_code=500, detail=f"분석 결과 목록 조회 중 오류가 발생했습니다: {str(e)}")


@router.post(
    "/analysis/{dataset_name}/recommend-assignee",
    summary="구조화 데이터셋 담당자 추천",
    response_model=str,
    tags=["Assignee Recommendation"],
)
async def recommend_assignee_for_dataset(
    dataset_name: str,
    request: ItsdRecommendationRequest,
    page: int = 1,
    page_size: int = 5,
    use_rrf: bool | None = None,
    w_title: float | None = None,
    w_content: float | None = None,
    rrf_k0: int | None = None,
    top_k_each: int | None = None,
):
    service = _resolve_recommendation_service(dataset_name)
    try:
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(50, int(page_size)))
        return await service.recommend_assignee(
            title=request.title,
            description=request.description,
            page=safe_page,
            page_size=safe_page_size,
            use_rrf=use_rrf,
            w_title=w_title,
            w_content=w_content,
            rrf_k0=rrf_k0,
            top_k_each=top_k_each,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("담당자 추천 중 오류 발생 (dataset=%s): %s", dataset_name, exc)
        raise HTTPException(status_code=500, detail=f"담당자 추천 중 서버 오류가 발생했습니다: {exc}")


@router.get("/cache/stats", summary="캐시 통계 조회", description="Git 레포지토리 캐시 디렉토리의 통계 정보를 조회합니다.")
async def get_cache_stats() -> Dict[str, Any]:
    """Git 레포지토리 캐시 통계 정보 조회"""
    try:
        git_analyzer = GitAnalyzer()
        stats = git_analyzer.get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"캐시 통계 조회 중 오류가 발생했습니다: {str(e)}")


@router.post("/cache/cleanup", summary="캐시 정리", description="오래된 Git 레포지토리 캐시를 정리합니다.")
async def cleanup_cache(max_age_hours: int = 24) -> Dict[str, Any]:
    """오래된 Git 레포지토리 캐시 정리"""
    try:
        git_analyzer = GitAnalyzer()
        cleaned_count = git_analyzer.cleanup_old_repositories(max_age_hours)
        return {
            "message": f"캐시 정리 완료",
            "cleaned_repositories": cleaned_count,
            "max_age_hours": max_age_hours
        }
    except Exception as e:
        logger.error(f"Failed to cleanup cache: {e}")
        raise HTTPException(status_code=500, detail=f"캐시 정리 중 오류가 발생했습니다: {str(e)}")
