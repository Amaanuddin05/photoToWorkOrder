-- Migration 001: ServiceTextEmbedding (per-model text embeddings of the service catalog)
--
-- One row per (service, text model). Lets the service run against WHATEVER model the
-- TEI embedder serves (e.g. BAAI/bge-base-en-v1.5 768-d, BAAI/bge-m3 1024-d) without
-- touching the original "WoService"."titleEmb"/"descEmb" vector(768) columns, which
-- stay the English bge-base vectors used by the English stack and WoSemanticSearch.
-- Same pattern as "ServiceImageProfile" (keyed by "imageModel").
--
-- Rows are written by the siglip service itself (auto re-embed on startup / when the
-- TEI model changes, and on catalog create/upsert); this file only creates the table.
-- Idempotent: safe to run repeatedly. Applied by scripts/migrate_text_embeddings.py.

CREATE TABLE IF NOT EXISTS "ServiceTextEmbedding" (
    "serviceId"   integer      NOT NULL REFERENCES "WoService"(id) ON DELETE CASCADE,
    "textModel"   text         NOT NULL,                 -- TEI /info model_id
    "dim"         integer      NOT NULL CHECK ("dim" > 0),
    "titleEmb"    vector       NOT NULL,                 -- unconstrained: any model's dimension
    "descEmb"     vector       NOT NULL,
    "contentHash" text         NOT NULL,                 -- sha256(title \n description) when embedded
    "updatedAt"   timestamptz  NOT NULL DEFAULT now(),
    PRIMARY KEY ("serviceId", "textModel")
);

COMMENT ON TABLE "ServiceTextEmbedding" IS
    'Per-model text embeddings of WoService title/description (siglip multilingual). Original WoService.titleEmb/descEmb are untouched.';
