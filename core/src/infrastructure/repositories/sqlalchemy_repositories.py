from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import and_, delete, select
from sqlalchemy.orm import Session

from audit import audit_print, preview_results, preview_text, preview_vector
from domain.entities import Chat, ChatMessage, DatasetSnapshot, Document, GroundTruth, MessageRole, SearchResult, User
from domain.ports import (
    ChatRepositoryPort,
    DatasetSnapshotRepositoryPort,
    DocumentRepositoryPort,
    GroundTruthRepositoryPort,
    PasswordResetRepositoryPort,
    UserRepositoryPort,
)
from infrastructure.db.models import (
    ChatMessageModel,
    ChatModel,
    DatasetSnapshotModel,
    DocumentModel,
    PasswordResetModel,
    QrelModel,
    QueryModel,
    UserModel,
)
class SqlAlchemyUserRepository(UserRepositoryPort):
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user: User) -> User:
        model = UserModel(
            email=user.email.lower().strip(),
            password_hash=user.password_hash,
            name=user.name,
            is_active=user.is_active,
            is_admin=user.is_admin,
        )
        self.session.add(model)
        self.session.commit()
        self.session.refresh(model)
        return User(
            id=model.id,
            email=model.email,
            password_hash=model.password_hash,
            name=model.name,
            is_active=model.is_active,
            is_admin=model.is_admin,
            created_at=model.created_at,
        )

    def get_by_email(self, email: str) -> User | None:
        model = self.session.scalar(select(UserModel).where(UserModel.email == email.lower().strip()))
        if not model:
            return None
        return User(model.id, model.email, model.password_hash, model.name, model.is_active, model.is_admin, model.created_at)

    def get_by_id(self, user_id: int) -> User | None:
        model = self.session.get(UserModel, user_id)
        if not model:
            return None
        return User(model.id, model.email, model.password_hash, model.name, model.is_active, model.is_admin, model.created_at)

    def update_password_hash(self, user_id: int, password_hash: str) -> None:
        model = self.session.get(UserModel, user_id)
        if not model:
            return
        model.password_hash = password_hash
        self.session.commit()


class SqlAlchemyPasswordResetRepository(PasswordResetRepositoryPort):
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, user_id: int, token_hash: str, expires_at: datetime) -> None:
        self.session.add(PasswordResetModel(user_id=user_id, token_hash=token_hash, expires_at=expires_at))
        self.session.commit()

    def consume_valid(self, token_hash: str, now: datetime) -> int | None:
        stmt = (
            select(PasswordResetModel)
            .where(
                and_(
                    PasswordResetModel.token_hash == token_hash,
                    PasswordResetModel.used_at.is_(None),
                    PasswordResetModel.expires_at > now,
                )
            )
            .order_by(PasswordResetModel.created_at.desc())
        )
        model = self.session.scalar(stmt)
        if not model:
            return None
        model.used_at = now
        user_id = model.user_id
        self.session.commit()
        return user_id


class SqlAlchemyChatRepository(ChatRepositoryPort):
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_chat(self, chat: Chat) -> Chat:
        model = ChatModel(user_id=chat.user_id, title=chat.title)
        self.session.add(model)
        self.session.commit()
        self.session.refresh(model)
        return Chat(model.id, model.user_id, model.title, model.created_at, model.updated_at, model.deleted_at)

    def list_chats(self, user_id: int, offset: int, limit: int) -> list[Chat]:
        stmt = (
            select(ChatModel)
            .where(ChatModel.user_id == user_id, ChatModel.deleted_at.is_(None))
            .order_by(ChatModel.updated_at.desc(), ChatModel.id.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = self.session.scalars(stmt).all()
        return [Chat(r.id, r.user_id, r.title, r.created_at, r.updated_at, r.deleted_at) for r in rows]

    def get_chat(self, user_id: int, chat_id: int) -> Chat | None:
        row = self.session.scalar(select(ChatModel).where(ChatModel.id == chat_id, ChatModel.user_id == user_id, ChatModel.deleted_at.is_(None)))
        if not row:
            return None
        return Chat(row.id, row.user_id, row.title, row.created_at, row.updated_at, row.deleted_at)

    def rename_chat(self, user_id: int, chat_id: int, title: str) -> Chat | None:
        row = self.session.scalar(select(ChatModel).where(ChatModel.id == chat_id, ChatModel.user_id == user_id, ChatModel.deleted_at.is_(None)))
        if not row:
            return None
        row.title = title or row.title
        self.session.commit()
        self.session.refresh(row)
        return Chat(row.id, row.user_id, row.title, row.created_at, row.updated_at, row.deleted_at)

    def soft_delete_chat(self, user_id: int, chat_id: int) -> bool:
        row = self.session.scalar(select(ChatModel).where(ChatModel.id == chat_id, ChatModel.user_id == user_id, ChatModel.deleted_at.is_(None)))
        if not row:
            return False
        row.deleted_at = datetime.now(UTC)
        self.session.commit()
        return True

    def create_message(self, message: ChatMessage) -> ChatMessage:
        model = ChatMessageModel(chat_id=message.chat_id, role=message.role.value, content=message.content)
        self.session.add(model)
        self.session.commit()
        self.session.refresh(model)
        chat = self.session.get(ChatModel, message.chat_id)
        if chat:
            chat.updated_at = datetime.now(UTC)
            self.session.commit()
        return ChatMessage(model.id, model.chat_id, MessageRole(model.role), model.content, model.created_at)

    def list_messages(self, user_id: int, chat_id: int, offset: int, limit: int) -> list[ChatMessage]:
        if not self.session.scalar(select(ChatModel.id).where(ChatModel.id == chat_id, ChatModel.user_id == user_id, ChatModel.deleted_at.is_(None))):
            return []
        stmt = select(ChatMessageModel).where(ChatMessageModel.chat_id == chat_id).order_by(ChatMessageModel.id.asc()).offset(offset).limit(limit)
        rows = self.session.scalars(stmt).all()
        return [ChatMessage(r.id, r.chat_id, MessageRole(r.role), r.content, r.created_at) for r in rows]


class SqlAlchemyDocumentRepository(DocumentRepositoryPort):
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_documents(self, documents: list[Document]) -> int:
        audit_print("repository.documents.upsert.start", batch_size=len(documents))
        count = 0
        for item in documents:
            audit_print(
                "repository.documents.upsert.item",
                dataset_id=item.dataset,
                doc_id=item.doc_id,
                title=preview_text(item.title, limit=80),
                text=preview_text(item.text),
                embedding=preview_vector(item.embedding_vector),
                quantum=preview_vector(item.quantum_vector),
                statistical=preview_vector(item.statistical_vector),
            )
            row = self.session.scalar(select(DocumentModel).where(DocumentModel.dataset == item.dataset, DocumentModel.doc_id == item.doc_id))
            if row is None:
                row = DocumentModel(dataset=item.dataset, doc_id=item.doc_id, title=item.title, text=item.text)
                self.session.add(row)
            row.title = item.title
            row.text = item.text
            row.metadata_json = item.metadata
            row.embedding_vector = item.embedding_vector
            row.quantum_vector = item.quantum_vector
            row.statistical_vector = item.statistical_vector
            count += 1
        self.session.commit()
        audit_print("repository.documents.upsert.completed", batch_size=len(documents), persisted=count)
        return count

    def count_by_dataset(self, dataset: str) -> int:
        return len(self.list_document_ids(dataset))

    def list_document_ids(self, dataset: str) -> list[str]:
        return list(self.session.scalars(select(DocumentModel.doc_id).where(DocumentModel.dataset == dataset)))

    def search_by_embedding(self, dataset: str, query_vector: list[float], top_k: int) -> list[SearchResult]:
        return self._search(dataset, query_vector, top_k, field="embedding_vector")

    def search_by_quantum(self, dataset: str, query_vector: list[float], top_k: int) -> list[SearchResult]:
        return self._search(dataset, query_vector, top_k, field="quantum_vector")

    def search_by_statistical(self, dataset: str, query_vector: list[float], top_k: int) -> list[SearchResult]:
        return self._search(dataset, query_vector, top_k, field="statistical_vector")

    def _search(self, dataset: str, query_vector: list[float], top_k: int, field: str) -> list[SearchResult]:
        dialect = self.session.bind.dialect.name if self.session.bind else ""
        column = getattr(DocumentModel, field)
        if dialect != "postgresql" or not hasattr(column, "cosine_distance"):
            raise RuntimeError("Search requires PostgreSQL + pgvector cosine_distance support.")
        audit_print(
            "repository.documents.search.start",
            dataset_id=dataset,
            field=field,
            top_k=top_k,
            query_vector=preview_vector(query_vector),
        )
        stmt = (
            select(DocumentModel, (1 - column.cosine_distance(query_vector)).label("score"))
            .where(DocumentModel.dataset == dataset)
            .where(column.is_not(None))
            .order_by(column.cosine_distance(query_vector))
            .limit(top_k)
        )
        rows = self.session.execute(stmt).all()
        results = [
            SearchResult(doc_id=row.DocumentModel.doc_id, text=row.DocumentModel.text, score=float(row.score), metadata=row.DocumentModel.metadata_json or {})
            for row in rows
        ]
        audit_print(
            "repository.documents.search.completed",
            dataset_id=dataset,
            field=field,
            top_k=top_k,
            results=preview_results(results),
        )
        return results


class SqlAlchemyGroundTruthRepository(GroundTruthRepositoryPort):
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, item: GroundTruth) -> GroundTruth:
        qrels = {doc_id: 1 for doc_id in item.relevant_doc_ids}
        self.upsert_qrels(
            dataset=item.dataset,
            split="test",
            query_id=item.query_id,
            query_text=item.query_text,
            qrels=qrels,
        )
        row = self.session.scalar(
            select(QueryModel).where(QueryModel.dataset == item.dataset, QueryModel.split == "test", QueryModel.query_id == item.query_id)
        )
        row.user_id = item.user_id
        if item.ideal_answer is not None:
            row.ideal_answer = item.ideal_answer
        self.session.commit()
        self.session.refresh(row)
        return GroundTruth(row.query_id, row.query_text, list(item.relevant_doc_ids), row.dataset, row.user_id, row.created_at, row.ideal_answer)

    def upsert_qrels(self, dataset: str, split: str, query_id: str, query_text: str, qrels: dict[str, int]) -> None:
        row = self.session.scalar(
            select(QueryModel).where(
                QueryModel.dataset == dataset,
                QueryModel.split == split,
                QueryModel.query_id == query_id,
            )
        )
        if row is None:
            row = QueryModel(dataset=dataset, split=split, query_id=query_id)
            self.session.add(row)
        row.query_text = query_text
        self.session.execute(
            delete(QrelModel).where(
                QrelModel.dataset == dataset,
                QrelModel.split == split,
                QrelModel.query_id == query_id,
            )
        )
        for doc_id, relevance in qrels.items():
            self.session.add(
                QrelModel(
                    dataset=dataset,
                    split=split,
                    query_id=query_id,
                    doc_id=doc_id,
                    relevance=int(relevance),
                )
            )
        self.session.commit()

    def get(self, dataset: str, query_id: str) -> GroundTruth | None:
        row = self.session.scalar(
            select(QueryModel).where(
                QueryModel.dataset == dataset,
                QueryModel.split == "test",
                QueryModel.query_id == query_id,
            )
        )
        if not row:
            return None
        qrels = self.session.scalars(
            select(QrelModel)
            .where(QrelModel.dataset == dataset, QrelModel.split == "test", QrelModel.query_id == query_id)
            .order_by(QrelModel.relevance.desc(), QrelModel.doc_id.asc())
        ).all()
        relevant_doc_ids = [q.doc_id for q in qrels if int(q.relevance) > 0]
        return GroundTruth(row.query_id, row.query_text, relevant_doc_ids, row.dataset, row.user_id, row.created_at, row.ideal_answer)

    def list_by_dataset(self, dataset: str) -> list[GroundTruth]:
        rows = self.session.scalars(
            select(QueryModel)
            .where(QueryModel.dataset == dataset, QueryModel.split == "test")
            .order_by(QueryModel.id.asc())
        ).all()
        items: list[GroundTruth] = []
        for row in rows:
            qrels = self.session.scalars(
                select(QrelModel)
                .where(QrelModel.dataset == dataset, QrelModel.split == "test", QrelModel.query_id == row.query_id)
                .order_by(QrelModel.relevance.desc(), QrelModel.doc_id.asc())
            ).all()
            relevant_doc_ids = [q.doc_id for q in qrels if int(q.relevance) > 0]
            items.append(GroundTruth(row.query_id, row.query_text, relevant_doc_ids, row.dataset, row.user_id, row.created_at, row.ideal_answer))
        return items

    def delete(self, dataset: str, query_id: str) -> bool:
        row = self.session.scalar(
            select(QueryModel).where(
                QueryModel.dataset == dataset,
                QueryModel.split == "test",
                QueryModel.query_id == query_id,
            )
        )
        if row is None:
            return False
        self.session.execute(
            delete(QrelModel).where(
                QrelModel.dataset == dataset,
                QrelModel.split == "test",
                QrelModel.query_id == query_id,
            )
        )
        self.session.delete(row)
        self.session.commit()
        return True


class SqlAlchemyDatasetSnapshotRepository(DatasetSnapshotRepositoryPort):
    def __init__(self, session: Session) -> None:
        self.session = session

    def _to_entity(self, row: DatasetSnapshotModel) -> DatasetSnapshot:
        return DatasetSnapshot(
            dataset_id=row.dataset_id,
            name=row.name,
            provider=row.provider,
            description=row.description,
            source_url=row.source_url,
            reference_urls=list(row.reference_urls or []),
            max_docs=row.max_docs,
            max_queries=row.max_queries,
            document_count=row.document_count,
            query_count=row.query_count,
            document_ids=list(row.document_ids or []),
            queries=list(row.queries_json or []),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def upsert(self, item: DatasetSnapshot) -> DatasetSnapshot:
        row = self.session.scalar(select(DatasetSnapshotModel).where(DatasetSnapshotModel.dataset_id == item.dataset_id))
        if row is None:
            row = DatasetSnapshotModel(dataset_id=item.dataset_id)
            self.session.add(row)
        row.name = item.name
        row.provider = item.provider
        row.description = item.description
        row.source_url = item.source_url
        row.reference_urls = item.reference_urls
        row.max_docs = item.max_docs
        row.max_queries = item.max_queries
        row.document_count = item.document_count
        row.query_count = item.query_count
        row.document_ids = item.document_ids
        row.queries_json = item.queries
        self.session.commit()
        self.session.refresh(row)
        return self._to_entity(row)

    def get(self, dataset_id: str) -> DatasetSnapshot | None:
        row = self.session.scalar(select(DatasetSnapshotModel).where(DatasetSnapshotModel.dataset_id == dataset_id))
        if not row:
            return None
        return self._to_entity(row)

    def list_all(self) -> list[DatasetSnapshot]:
        rows = self.session.scalars(select(DatasetSnapshotModel).order_by(DatasetSnapshotModel.dataset_id.asc())).all()
        return [self._to_entity(row) for row in rows]
