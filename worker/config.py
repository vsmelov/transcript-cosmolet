"""Конфиг воркера — всё из env (docker-compose .env)."""
import os
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", "/data"))
INBOX = DATA / "inbox"
IN_PROGRESS = DATA / "in_progress"
DONE = DATA / "done"
FAILED = DATA / "failed"
ARTIFACTS = DATA / "artifacts"
AUDIO = DATA / "audio"          # постоянное хранилище аудио (не удаляем — решение пользователя)

DATABASE_URL = os.environ["DATABASE_URL"]

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]

DRAFT_MODEL = os.environ.get("DRAFT_MODEL", "openai/whisper-large-v3-turbo")
# Язык черновика НЕ фиксируем: речь бывает двуязычной (ru+en в одной фразе), а жёсткий
# язык калечит вторую половину. Текст черновика мы всё равно не используем — из него
# берётся только карта речи (границы + no_speech_prob/logprob). Что whisper местами
# уезжает в перевод — на границы речи не влияет, а галлюцинации ловит speechmap.
# Задать язык принудительно можно через env, если запись заведомо моноязычная.
DRAFT_LANGUAGE = os.environ.get("DRAFT_LANGUAGE", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-2.5-flash-lite")
SCRIBE_MODEL = os.environ.get("SCRIBE_MODEL", "scribe_v2")

DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "2.0"))

EMBED_MODEL = os.environ.get("EMBED_MODEL", "/models/campplus_zh_en_advanced.onnx")

# Карта речи из чернового verbose_json: сегмент считается речью, если НЕ похож на
# тишину/галлюцинацию. Пороги — стартовые, крутить по факту на реальных записях.
# Детектор речи (локальный, бесплатный): Silero VAD + классификатор аудио-событий.
# Заменил платный whisper-черновик: на 11-часовой ночной записи whisper пометил бы
# речью 86% длительности, VAD+CED дают 2.6% — и это подтверждается прослушиванием.
VAD_MODEL = os.environ.get("VAD_MODEL", "/models/silero_vad.onnx")
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE = 0.8         # пауза, после которой кусок речи считается законченным
VAD_MIN_SPEECH = 0.25
TAG_MODEL = os.environ.get("TAG_MODEL", "/models/sherpa-onnx-ced-tiny-audio-tagging-2024-04-19")
TAG_SPEECH_MIN = 0.35         # ниже — классификатор считает кусок не речью (шум, ТВ, тишина)
REGION_MERGE_GAP_SEC = 8.0    # склеиваем речевые регионы через паузы короче этого
REGION_PAD_SEC = 4.0          # поля региона — чтобы не срезать начало/конец фразы

# Качественный проход: файл длиннее — режем на куски (лимиты API по размеру запроса)
QUALITY_CHUNK_SEC = 1200.0    # 20 мин на запрос к Scribe
SCRIBE_PRICE_PER_HOUR = 0.22

# Диаризация/опознание
UTTER_GAP_SEC = 0.8           # пауза внутри реплики одного speaker_id
MIN_EMBED_SEC = 1.5           # короче — эмбеддинг не считаем, спикер наследуется
EMBED_CLIP_MAX_SEC = 45.0
CLUSTER_JOIN = 0.70           # склейка меток в человека
CLUSTER_MIN_SEC = 15.0        # метка короче — не образует свой кластер, а липнет к ближайшему
CONF_OK = 0.70                # ниже — ambiguous
TOP2_MARGIN = 0.15            # отрыв top1-top2 меньше — ambiguous
# Переназначение отдельной реплики из её кластера к другому человеку: только когда
# голос говорит об этом уверенно (иначе доверяем кластеру, он усреднён и надёжнее).
REASSIGN_MIN_COS = 0.65       # насколько близко к новому кандидату
REASSIGN_MIN_GAP = 0.20       # и насколько он дальше отрывается от дефолта кластера

# Детектор смены говорящего ВНУТРИ реплики: Scribe иногда не замечает перехват и
# склеивает двух людей в один speaker_id. Ищем точку разреза по границам слов
# (эмбеддинг считаем по склеенной речи, без пауз — см. AudioCache.embed_spans).
SPLIT_MIN_SEC = 8.0           # реплики короче не трогаем: мало данных для двух половин
SPLIT_MIN_SIDE_SEC = 2.5      # каждая половина должна содержать столько речи
SPLIT_MAX_COS = 0.60          # косинус половин ниже — считаем, что говорят разные люди
SPLIT_MAX_DEPTH = 2           # рекурсия: максимум 4 части из одной реплики
