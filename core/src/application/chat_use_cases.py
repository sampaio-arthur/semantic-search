from __future__ import annotations

import json

from domain.entities import Chat, ChatMessage, MessageRole
from domain.exceptions import NotFoundError, ValidationError
from domain.ports import ChatRepositoryPort


class CreateChatUseCase:
    def __init__(self, chats: ChatRepositoryPort) -> None:
        self.chats = chats

    def execute(self, user_id: int, title: str | None = None) -> Chat:
        resolved_title = (title or "Nova conversa").strip() or "Nova conversa"
        return self.chats.create_chat(Chat(id=None, user_id=user_id, title=resolved_title))


class ListChatsUseCase:
    def __init__(self, chats: ChatRepositoryPort) -> None:
        self.chats = chats

    def execute(self, user_id: int, page: int = 1, page_size: int = 20) -> list[Chat]:
        offset = max(page - 1, 0) * page_size
        return self.chats.list_chats(user_id, offset, page_size)


class GetChatUseCase:
    def __init__(self, chats: ChatRepositoryPort) -> None:
        self.chats = chats

    def execute(self, user_id: int, chat_id: int, page: int = 1, page_size: int = 200) -> tuple[Chat, list[ChatMessage]]:
        chat = self.chats.get_chat(user_id, chat_id)
        if not chat:
            raise NotFoundError("Chat not found.")
        offset = max(page - 1, 0) * page_size
        messages = self.chats.list_messages(user_id, chat_id, offset, page_size)
        return chat, messages


class AddMessageUseCase:
    def __init__(self, chats: ChatRepositoryPort) -> None:
        self.chats = chats

    def execute(self, user_id: int, chat_id: int, role: str, content: str) -> ChatMessage:
        chat = self.chats.get_chat(user_id, chat_id)
        if not chat:
            raise NotFoundError("Chat not found.")
        try:
            enum_role = MessageRole(role)
        except ValueError as exc:
            raise ValidationError("Invalid role.") from exc
        return self.chats.create_message(ChatMessage(id=None, chat_id=chat_id, role=enum_role, content=content))

    def save_assistant_retrieval_result(self, user_id: int, chat_id: int, payload: dict) -> ChatMessage:
        return self.execute(user_id, chat_id, MessageRole.ASSISTANT.value, json.dumps(payload, ensure_ascii=False))


class RenameChatUseCase:
    def __init__(self, chats: ChatRepositoryPort) -> None:
        self.chats = chats

    def execute(self, user_id: int, chat_id: int, title: str) -> Chat:
        updated = self.chats.rename_chat(user_id, chat_id, title.strip())
        if not updated:
            raise NotFoundError("Chat not found.")
        return updated


class DeleteChatUseCase:
    def __init__(self, chats: ChatRepositoryPort) -> None:
        self.chats = chats

    def execute(self, user_id: int, chat_id: int) -> None:
        if not self.chats.soft_delete_chat(user_id, chat_id):
            raise NotFoundError("Chat not found.")
