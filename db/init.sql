CREATE EXTENSION IF NOT EXISTS vector;

-- Записи (одна строка = один аудиофайл, прошедший через inbox)
CREATE TABLE recordings (
    id           serial PRIMARY KEY,
    filename     text NOT NULL,
    source       text NOT NULL DEFAULT 'inbox',      -- inbox | plaud | manual
    audio_path   text,                               -- где лежит аудио (data/done/...), NULL если удалено
    duration_sec real,
    size_bytes   bigint,
    started_at   timestamptz,                        -- время начала записи (из метаданных, если есть)
    status       text NOT NULL DEFAULT 'new',        -- new | processing | done | failed
    meta         jsonb NOT NULL DEFAULT '{}',
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Этапы обработки (draft / quality / resolve / store) с телеметрией — экран Jobs в UI
CREATE TABLE jobs (
    id            serial PRIMARY KEY,
    recording_id  int NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    stage         text NOT NULL,                     -- draft | quality | resolve | store
    status        text NOT NULL DEFAULT 'pending',   -- pending | running | done | failed | skipped
    started_at    timestamptz,
    finished_at   timestamptz,
    cost_usd      numeric(10,6) NOT NULL DEFAULT 0,
    error         text,
    artifact_path text,                              -- json-артефакт этапа в data/artifacts/
    meta          jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX jobs_recording_idx ON jobs(recording_id);

-- Люди и эталоны голосов (переезд voiceprints-базы из transcribe-mcp)
CREATE TABLE speakers (
    id         serial PRIMARY KEY,
    name       text NOT NULL UNIQUE,
    aliases    text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE speaker_samples (
    id           serial PRIMARY KEY,
    speaker_id   int NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    path         text,                               -- клип на диске (для прослушивания в UI)
    kind         text NOT NULL DEFAULT 'reference',  -- reference (2-10с, годен для API) | embed
    duration_sec real,
    embedding    vector(192),
    source       text,
    added_at     timestamptz NOT NULL DEFAULT now()
);

-- Сегменты финального транскрипта.
-- Компактная схема по умолчанию: speaker_name + confidence.
-- Разворот (detail) заполняется ТОЛЬКО у неоднозначных сегментов:
--   {"top": [{"name","cos"}...], "cluster_default": {"name","cos"}, "resolution": "..."}
CREATE TABLE segments (
    id           bigserial PRIMARY KEY,
    recording_id int NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    start_sec    real NOT NULL,
    end_sec      real NOT NULL,
    text         text NOT NULL,
    speaker_id   int REFERENCES speakers(id),
    speaker_name text NOT NULL,                      -- имя из базы или S1/S2/S?
    confidence   real,                               -- косинус к финальному спикеру; NULL = inherited
    inherited    boolean NOT NULL DEFAULT false,     -- коротыш, унаследовал спикера от контекста
    ambiguous    boolean NOT NULL DEFAULT false,
    detail       jsonb,                              -- разворот только при ambiguous
    embedding    vector(192),                        -- вектор сегмента (если считался)
    words        jsonb,                              -- пословные таймстемпы+logprob от Scribe
    asr_logprob  real,                               -- средний ASR-confidence сегмента
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX segments_recording_idx ON segments(recording_id, start_sec);

-- Очередь конфликтов диаризации на ручной разбор (low_confidence_queue)
CREATE TABLE conflicts (
    id            serial PRIMARY KEY,
    segment_id    bigint NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    reason        text NOT NULL,                     -- low_confidence | top2_close | cluster_mismatch
    status        text NOT NULL DEFAULT 'open',      -- open | resolved | dismissed
    judge_verdict jsonb,                             -- вердикт текст-судьи (если был)
    resolved_name text,
    resolved_by   text,                              -- user | judge
    created_at    timestamptz NOT NULL DEFAULT now(),
    resolved_at   timestamptz
);
CREATE INDEX conflicts_open_idx ON conflicts(status) WHERE status = 'open';

-- Журнал расходов — бюджетный предохранитель DAILY_BUDGET_USD
CREATE TABLE costs (
    id           serial PRIMARY KEY,
    ts           timestamptz NOT NULL DEFAULT now(),
    day          date NOT NULL DEFAULT current_date,
    usd          numeric(10,6) NOT NULL,
    kind         text NOT NULL,                      -- draft | quality | judge
    model        text,
    recording_id int,
    note         text
);
CREATE INDEX costs_day_idx ON costs(day);
ALTER TABLE recordings ADD COLUMN IF NOT EXISTS title text;  -- человеческое название (из Plaud или имени файла)
