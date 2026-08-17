CREATE TABLE IF NOT EXISTS model_versions (
    model_name    TEXT NOT NULL,
    version       TEXT NOT NULL,
    stage         TEXT NOT NULL DEFAULT 'staging',
    precision     TEXT NOT NULL DEFAULT 'fp32',
    artifact_path TEXT NOT NULL,
    metrics       JSONB NOT NULL DEFAULT '{}',
    traffic_pct   INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_stage ON model_versions (model_name, stage);

CREATE TABLE IF NOT EXISTS deployment_events (
    id            BIGSERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    version       TEXT NOT NULL,
    action        TEXT NOT NULL, -- deploy | rollback | archive
    reason        TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
