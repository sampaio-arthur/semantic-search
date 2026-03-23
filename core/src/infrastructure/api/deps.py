from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from application.auth_use_cases import (
    ConfirmPasswordResetUseCase,
    RefreshTokenUseCase,
    RequestPasswordResetUseCase,
    SignInUseCase,
    SignUpUseCase,
)
from application.chat_use_cases import AddMessageUseCase, CreateChatUseCase, DeleteChatUseCase, GetChatUseCase, ListChatsUseCase, RenameChatUseCase
from application.ir_use_cases import BuildAssistantRetrievalMessageUseCase, EvaluateUseCase, IndexDatasetUseCase, SearchUseCase, UpsertGroundTruthUseCase
from domain.exceptions import UnauthorizedError
from infrastructure.config import Settings, get_settings
from infrastructure.datasets.beir_local_provider import BeirLocalDatasetProvider
from infrastructure.db.session import db_session
from infrastructure.email.dev_notifier import DevLogNotifier
from infrastructure.encoders.base import SharedSbertBase
from infrastructure.encoders.classical import ClassicalPipelineEncoder
from infrastructure.encoders.quantum import QuantumPipelineEncoder
from infrastructure.encoders.statistical import StatisticalPipelineEncoder
from infrastructure.metrics.answer_similarity import AnswerSimilarityService
from infrastructure.metrics.ir_measures_adapter import IrMeasuresAdapter
from infrastructure.repositories.sqlalchemy_repositories import (
    SqlAlchemyChatRepository,
    SqlAlchemyDatasetSnapshotRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemyGroundTruthRepository,
    SqlAlchemyPasswordResetRepository,
    SqlAlchemyUserRepository,
)
from infrastructure.security.adapters import BcryptPasswordHasher, JoseJwtProvider, Sha256ResetTokenGenerator

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# ── Encoder singletons ────────────────────────────────────────────────────────
# Encoders are created once and reused across all requests. This is required because
# the PCA/SVD transformations are fitted during indexing and must persist in memory
# for subsequent search requests. A new instance per request would lose fitted state.
_base_encoder: SharedSbertBase | None = None
_classical_encoder: ClassicalPipelineEncoder | None = None
_quantum_encoder: QuantumPipelineEncoder | None = None
_statistical_encoder: StatisticalPipelineEncoder | None = None


def _get_encoders(settings: Settings):
    global _base_encoder, _classical_encoder, _quantum_encoder, _statistical_encoder
    if _classical_encoder is None:
        _base_encoder = SharedSbertBase(settings.classical_model_name)
        _classical_encoder = ClassicalPipelineEncoder(
            base=_base_encoder,
            dim=settings.vector_dim,
            seed=settings.seed,
        )
        _quantum_encoder = QuantumPipelineEncoder(
            base=_base_encoder,
            n_qubits=settings.quantum_n_qubits,
            dim=settings.vector_dim,
            seed=settings.seed,
        )
        _statistical_encoder = StatisticalPipelineEncoder(
            base=_base_encoder,
            dim=settings.vector_dim,
            pca_intermediate_dim=settings.pca_intermediate_dim,
            seed=settings.seed,
        )
        # Try to restore fitted state persisted from a previous indexing run.
        state_dir = settings.encoder_state_dir
        _classical_encoder.load_state(f"{state_dir}/classical.joblib")
        _quantum_encoder.load_state(f"{state_dir}/quantum.joblib")
        _statistical_encoder.load_state(f"{state_dir}/statistical.joblib")
    return _classical_encoder, _quantum_encoder, _statistical_encoder


@dataclass(slots=True)
class Services:
    settings: Settings
    users: SqlAlchemyUserRepository
    chats: SqlAlchemyChatRepository
    documents: SqlAlchemyDocumentRepository
    dataset_snapshots: SqlAlchemyDatasetSnapshotRepository
    ground_truths: SqlAlchemyGroundTruthRepository
    sign_up: SignUpUseCase
    sign_in: SignInUseCase
    request_reset: RequestPasswordResetUseCase
    confirm_reset: ConfirmPasswordResetUseCase
    refresh: RefreshTokenUseCase
    create_chat: CreateChatUseCase
    list_chats: ListChatsUseCase
    get_chat: GetChatUseCase
    add_message: AddMessageUseCase
    rename_chat: RenameChatUseCase
    delete_chat: DeleteChatUseCase
    index_dataset: IndexDatasetUseCase
    search: SearchUseCase
    upsert_ground_truth: UpsertGroundTruthUseCase
    evaluate: EvaluateUseCase
    build_assistant_message: BuildAssistantRetrievalMessageUseCase
    answer_similarity: AnswerSimilarityService
    jwt: JoseJwtProvider


def build_services(session: Session, settings: Settings) -> Services:
    users = SqlAlchemyUserRepository(session)
    resets = SqlAlchemyPasswordResetRepository(session)
    chats = SqlAlchemyChatRepository(session)
    docs = SqlAlchemyDocumentRepository(session)
    dataset_snaps = SqlAlchemyDatasetSnapshotRepository(session)
    gts = SqlAlchemyGroundTruthRepository(session)

    hasher = BcryptPasswordHasher()
    jwt = JoseJwtProvider(settings)
    notifier = DevLogNotifier()
    reset_tokens = Sha256ResetTokenGenerator()

    classical_encoder, quantum_encoder, statistical_encoder = _get_encoders(settings)

    datasets = BeirLocalDatasetProvider()
    metrics = IrMeasuresAdapter()
    answer_sim = AnswerSimilarityService(classical_encoder.base)

    search_uc = SearchUseCase(docs, classical_encoder, quantum_encoder, statistical_encoder)
    return Services(
        settings=settings,
        users=users,
        chats=chats,
        documents=docs,
        ground_truths=gts,
        sign_up=SignUpUseCase(users, hasher),
        sign_in=SignInUseCase(users, hasher, jwt),
        request_reset=RequestPasswordResetUseCase(users, resets, reset_tokens, notifier, settings.password_reset_expire_minutes),
        confirm_reset=ConfirmPasswordResetUseCase(users, resets, reset_tokens, hasher),
        refresh=RefreshTokenUseCase(jwt),
        create_chat=CreateChatUseCase(chats),
        list_chats=ListChatsUseCase(chats),
        get_chat=GetChatUseCase(chats),
        add_message=AddMessageUseCase(chats),
        rename_chat=RenameChatUseCase(chats),
        delete_chat=DeleteChatUseCase(chats),
        index_dataset=IndexDatasetUseCase(datasets, docs, classical_encoder, quantum_encoder, statistical_encoder, dataset_snaps, gts, encoder_state_dir=settings.encoder_state_dir),
        search=search_uc,
        upsert_ground_truth=UpsertGroundTruthUseCase(gts),
        evaluate=EvaluateUseCase(gts, search_uc, metrics, answer_sim),
        build_assistant_message=BuildAssistantRetrievalMessageUseCase(),
        answer_similarity=answer_sim,
        dataset_snapshots=dataset_snaps,
        jwt=jwt,
    )


def get_services(session: Session = Depends(db_session), settings: Settings = Depends(get_settings)) -> Services:
    return build_services(session, settings)


def get_current_user_id(
    token: str | None = Depends(oauth2_scheme),
    services: Services = Depends(get_services),
) -> int:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        claims = services.jwt.decode_token(token)
    except UnauthorizedError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    if claims.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token")
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")
    return int(sub)
