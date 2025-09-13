"""
Интеграция с YandexGPT API для LangGraph
Обновленная версия с YandexCloudML SDK и асинхронными операциями
"""
import json
import requests
from typing import Any, Dict, List, Optional, AsyncIterator
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult, Generation
from pydantic import Field
import os
import asyncio
import aiohttp
from datetime import datetime

# Импорт Streamlit для секретов
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Warning: Streamlit не доступен, используем переменные окружения")

# Импорт YandexCloudML SDK
try:
    from yandex_cloud_ml_sdk import YCloudML
    YANDEX_SDK_AVAILABLE = True
except ImportError:
    YANDEX_SDK_AVAILABLE = False
    print("Warning: yandex_cloud_ml_sdk не установлен. Используется fallback к HTTP API")

def get_yandex_config(key: str, default: str = "") -> str:
    """Получает конфигурацию из Streamlit secrets или переменных окружения"""
    if STREAMLIT_AVAILABLE and hasattr(st, 'secrets'):
        try:
            return st.secrets["yandex"][key]
        except (KeyError, AttributeError):
            # Fallback к переменным окружения
            env_key = f"YANDEX_{key.upper()}"
            return os.getenv(env_key, default)
    else:
        # Используем переменные окружения
        env_key = f"YANDEX_{key.upper()}"
        return os.getenv(env_key, default)


class YandexGPT(LLM):
    """Кастомная LLM для работы с YandexGPT API через YandexCloudML SDK с асинхронными операциями"""
    
    api_key: str = Field(default_factory=lambda: get_yandex_config("api_key", ""))
    folder_id: str = Field(default_factory=lambda: get_yandex_config("folder_id", ""))
    model_id: str = Field(default_factory=lambda: get_yandex_config("model_id", "yandexgpt"))
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    
    # Метаданные для LangGraph
    request_count: int = Field(default=0)
    total_tokens: int = Field(default=0)
    last_request_time: Optional[datetime] = Field(default=None)
    
    # YandexCloudML SDK объект (инициализируется лениво)
    sdk: Optional[Any] = Field(default=None, exclude=True)
    
    @property
    def _llm_type(self) -> str:
        return "yandex_gpt_sdk" if YANDEX_SDK_AVAILABLE else "yandex_gpt_http"
    
    def _get_sdk(self):
        """Получает или создает экземпляр YandexCloudML SDK"""
        if not YANDEX_SDK_AVAILABLE:
            raise RuntimeError("YandexCloudML SDK не установлен")
            
        if self.sdk is None:
            self.sdk = YCloudML(
                folder_id=self.folder_id,
                auth=self.api_key,
            )
        return self.sdk
    
    def _call_with_sdk(self, prompt: str, **kwargs) -> str:
        """Вызов через YandexCloudML SDK с асинхронным ожиданием"""
        try:
            # Получаем SDK
            sdk = self._get_sdk()
            
            # Обновляем параметры из kwargs
            temperature = kwargs.get('temperature', self.temperature)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            
            # Получаем модель
            model = sdk.models.completions(self.model_id)
            
            # Формируем сообщения в формате YandexCloudML
            messages = [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
            
            # Запускаем асинхронную операцию с конфигурацией
            operation = model.configure(
                temperature=temperature,
                max_tokens=max_tokens
            ).run_deferred(messages)
            
            # Ждем завершения операции используя wait() - как во втором варианте из async_variant.py
            result = operation.wait()
            
            # Извлекаем текст ответа
            # Структура ответа: GPTModelResult(alternatives=(Alternative(text='...'),), ...)
            if hasattr(result, 'alternatives') and result.alternatives:
                # В асинхронном режиме: result.alternatives[0].text (не .message.text)
                text_result = result.alternatives[0].text
            elif hasattr(result, 'text'):
                text_result = result.text
            else:
                text_result = str(result)
            
            return text_result
            
        except Exception as e:
            raise Exception(f"Ошибка YandexCloudML SDK: {str(e)}")
    
    def _call_with_http(self, prompt: str, **kwargs) -> str:
        """Fallback к HTTP API если SDK недоступен"""
        print("🔍 [YandexGPT HTTP] Подготовка HTTP запроса к API...")
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        # Обновляем параметры из kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        print(f"🔍 [YandexGPT HTTP] URL: {url}")
        print(f"🔍 [YandexGPT HTTP] ModelUri: gpt://{self.folder_id}/{self.model_id}")
        print(f"🔍 [YandexGPT HTTP] Temperature: {temperature}, MaxTokens: {max_tokens}")
        print("🔍 [YandexGPT HTTP] Отправка POST запроса...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"🔍 [YandexGPT HTTP] Получен статус ответа: {response.status_code}")
        
        response.raise_for_status()
        
        result = response.json()
        text_result = result["result"]["alternatives"][0]["message"]["text"]
        print(f"✅ [YandexGPT HTTP] Успешно получен текст длиной {len(text_result)} символов")
        
        return text_result
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Вызов YandexGPT API"""
        
        print(f"🔍 [YandexGPT] Начало API вызова (запрос #{self.request_count + 1})")
        print(f"🔍 [YandexGPT] Длина промпта: {len(prompt)} символов")
        print(f"🔍 [YandexGPT] Промпт: {prompt[:200]}...")
        
        try:
            # Проверяем credentials перед вызовом
            if not self.api_key:
                raise ValueError("API ключ Yandex не установлен")
            if not self.folder_id:
                raise ValueError("Folder ID Yandex не установлен")
            
            # Обновляем метаданные
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            # Пытаемся использовать SDK, если доступен
            if YANDEX_SDK_AVAILABLE:
                print("🔍 [YandexGPT] Использование YandexCloudML SDK...")
                text_result = self._call_with_sdk(prompt, **kwargs)
            else:
                print("🔍 [YandexGPT] Использование HTTP API fallback...")
                text_result = self._call_with_http(prompt, **kwargs)
            
            # Примерная оценка токенов
            estimated_tokens = len(text_result.split()) * 1.3
            self.total_tokens += int(estimated_tokens)
            
            print(f"✅ [YandexGPT] API вызов успешен. Получено {len(text_result)} символов")
            print(f"✅ [YandexGPT] Ответ: {text_result[:200]}...")
            
            return text_result
            
        except Exception as e:
            error_msg = f"Ошибка YandexGPT: {str(e)}"
            print(f"❌ [YandexGPT] {error_msg}")
            if run_manager:
                run_manager.on_llm_error(Exception(error_msg))
            return error_msg
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        return {
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'sdk_available': YANDEX_SDK_AVAILABLE,
            'sdk_type': 'yandex_cloud_ml_sdk' if YANDEX_SDK_AVAILABLE else 'http_api'
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None


class YandexGPTChat(BaseChatModel):
    """Чат-модель для YandexGPT с поддержкой сообщений"""
    
    api_key: str = Field(default_factory=lambda: get_yandex_config("api_key", ""))
    folder_id: str = Field(default_factory=lambda: get_yandex_config("folder_id", ""))
    model_id: str = Field(default_factory=lambda: get_yandex_config("model_id", "yandexgpt-lite"))
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    
    # Метаданные для LangGraph
    request_count: int = Field(default=0)
    total_tokens: int = Field(default=0)
    last_request_time: Optional[datetime] = Field(default=None)
    
    @property
    def _llm_type(self) -> str:
        return "yandex_gpt_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Генерация ответа на основе списка сообщений"""
        
        # Конвертируем сообщения в формат YandexGPT
        yandex_messages = self._convert_messages_to_yandex_format(messages)
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        # Обновляем параметры из kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": yandex_messages
        }
        
        try:
            # Обновляем метаданные
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            text_result = result["result"]["alternatives"][0]["message"]["text"]
            
            # Примерная оценка токенов
            estimated_tokens = len(text_result.split()) * 1.3
            self.total_tokens += int(estimated_tokens)
            
            # Создаем ChatGeneration
            message = AIMessage(content=text_result)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            error_msg = f"Ошибка в YandexGPT Chat: {str(e)}"
            if run_manager:
                run_manager.on_llm_error(Exception(error_msg))
            
            # Возвращаем ошибку как сообщение
            error_message = AIMessage(content=error_msg)
            error_generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[error_generation])
    
    def _convert_messages_to_yandex_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Конвертирует сообщения LangChain в формат YandexGPT"""
        yandex_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                # YandexGPT не поддерживает системные сообщения напрямую,
                # поэтому добавляем их как пользовательские с префиксом
                yandex_messages.append({
                    "role": "user",
                    "text": f"Системная инструкция: {message.content}"
                })
            elif isinstance(message, HumanMessage):
                yandex_messages.append({
                    "role": "user",
                    "text": message.content
                })
            elif isinstance(message, AIMessage):
                yandex_messages.append({
                    "role": "assistant",
                    "text": message.content
                })
            else:
                # Для других типов сообщений используем пользовательскую роль
                yandex_messages.append({
                    "role": "user",
                    "text": str(message.content)
                })
        
        return yandex_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        return {
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None


class AsyncYandexGPT:
    """Асинхронная версия YandexGPT для LangGraph"""
    
    def __init__(
        self,
        api_key: str = None,
        folder_id: str = None,
        model_id: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.api_key = api_key or get_yandex_config("api_key", "")
        self.folder_id = folder_id or get_yandex_config("folder_id", "")
        self.model_id = model_id or get_yandex_config("model_id", "yandexgpt-lite")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Метаданные
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Асинхронная генерация текста"""
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        # Обновляем параметры из kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        try:
            # Обновляем метаданные
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            text_result = result["result"]["alternatives"][0]["message"]["text"]
            
            # Примерная оценка токенов
            estimated_tokens = len(text_result.split()) * 1.3
            self.total_tokens += int(estimated_tokens)
            
            return text_result
            
        except Exception as e:
            return f"Ошибка в AsyncYandexGPT: {str(e)}"
    
    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Асинхронная генерация в чат-режиме"""
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        # Обновляем параметры из kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": messages
        }
        
        try:
            # Обновляем метаданные
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            text_result = result["result"]["alternatives"][0]["message"]["text"]
            
            # Примерная оценка токенов
            estimated_tokens = len(text_result.split()) * 1.3
            self.total_tokens += int(estimated_tokens)
            
            return text_result
            
        except Exception as e:
            return f"Ошибка в AsyncYandexGPT chat: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        return {
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None


# Функции-помощники для LangGraph

def create_yandex_llm(
    temperature: float = 0.7,
    max_tokens: int = 2000,
    model_id: str = None
) -> YandexGPT:
    """Создает экземпляр YandexGPT LLM"""
    print("🔍 [YandexGPT] Создание экземпляра YandexGPT LLM...")
    
    # Проверяем переменные окружения
    api_key = get_yandex_config("api_key", "")
    folder_id = get_yandex_config("folder_id", "")
    model = model_id or get_yandex_config("model_id", "yandexgpt")
    
    print(f"🔍 [YandexGPT] API ключ: {'✅ установлен' if api_key else '❌ отсутствует'}")
    print(f"🔍 [YandexGPT] Folder ID: {'✅ установлен' if folder_id else '❌ отсутствует'}")
    print(f"🔍 [YandexGPT] Модель: {model}")
    print(f"🔍 [YandexGPT] SDK доступен: {'✅ да' if YANDEX_SDK_AVAILABLE else '❌ нет'}")
    
    llm = YandexGPT(
        temperature=temperature,
        max_tokens=max_tokens,
        model_id=model
    )
    
    print("✅ [YandexGPT] Экземпляр LLM создан")
    return llm

def create_yandex_chat(
    temperature: float = 0.7,
    max_tokens: int = 2000,
    model_id: str = None
) -> YandexGPTChat:
    """Создает экземпляр YandexGPT Chat"""
    return YandexGPTChat(
        temperature=temperature,
        max_tokens=max_tokens,
        model_id=model_id or get_yandex_config("model_id", "yandexgpt-lite")
    )

def create_async_yandex_llm(
    temperature: float = 0.7,
    max_tokens: int = 2000,
    model_id: str = None
) -> AsyncYandexGPT:
    """Создает экземпляр асинхронного YandexGPT"""
    return AsyncYandexGPT(
        temperature=temperature,
        max_tokens=max_tokens,
        model_id=model_id or get_yandex_config("model_id", "yandexgpt-lite")
    )

def validate_yandex_config() -> Dict[str, Any]:
    """Валидирует конфигурацию YandexGPT"""
    config = {
        'api_key': bool(get_yandex_config("api_key")),
        'folder_id': bool(get_yandex_config("folder_id")),
        'model_id': get_yandex_config("model_id", "yandexgpt"),
        'sdk_available': YANDEX_SDK_AVAILABLE
    }
    
    config['is_valid'] = config['api_key'] and config['folder_id']
    
    if not config['is_valid']:
        missing = []
        if not config['api_key']:
            missing.append('YANDEX_API_KEY')
        if not config['folder_id']:
            missing.append('YANDEX_FOLDER_ID')
        config['missing_vars'] = missing
    
    return config