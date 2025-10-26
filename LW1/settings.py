import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Загрузка .env из корня проекта 
load_dotenv()


@dataclass(frozen=True)
class NLPCloudConfig:
    base_url: str = os.getenv("NLPCLOUD_BASE_URL", "https://api.nlpcloud.io/v1")
    api_key: str = os.getenv("NLPCLOUD_API_KEY", "")
    model: str = os.getenv("NLPCLOUD_MODEL", "en_core_web_lg")
    timeout: float = float(os.getenv("TIMEOUT_SECONDS", "15.0"))


@dataclass(frozen=True)
class HuggingFaceConfig:
    model: str = os.getenv("HF_MODEL", "dbmdz/bert-large-cased-finetuned-conll03-english")
    token: str = os.getenv("HF_API_TOKEN", "")
    use_local_fallback: bool = bool(int(os.getenv("HF_USE_LOCAL_FALLBACK", "0")))


@dataclass(frozen=True)
class MetricsConfig:
    file_path: str = os.getenv("METRICS_FILE", "./logs/metrics.jsonl")


@dataclass(frozen=True)
class AppSettings:
    """Глобальные настройки приложения."""
    env: str = os.getenv("APP_ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    nlpcloud: NLPCloudConfig = NLPCloudConfig()
    huggingface: HuggingFaceConfig = HuggingFaceConfig()
    metrics: MetricsConfig = MetricsConfig()


# Глобальный объект конфигурации 
settings = AppSettings()
