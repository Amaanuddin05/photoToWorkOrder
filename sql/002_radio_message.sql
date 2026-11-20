-- Radio channel demo: messages workers send (text and/or photo). Messages with photo + text run through the
-- pipeline; decisionId / action / serviceTitle record the outcome. Created at startup (IF NOT EXISTS).
CREATE TABLE IF NOT EXISTS "RadioMessage" (
    id             serial PRIMARY KEY,
    "createdAt"    timestamptz NOT NULL DEFAULT now(),
    sender         text        NOT NULL DEFAULT 'Worker',
    text           text,
    "photoPath"    text,
    "decisionId"   integer     REFERENCES "DecisionLog"(id) ON DELETE SET NULL,
    action         text,
    "serviceTitle" text,
    reason         text,
    llm            jsonb
);
CREATE INDEX IF NOT EXISTS "RadioMessage_createdAt_idx" ON "RadioMessage" ("createdAt");
