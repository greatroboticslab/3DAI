CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL,
    total_steps INTEGER NOT NULL,
    completed_steps INTEGER NOT NULL DEFAULT 0,
    sample_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS sample_id TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS steps (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id),
    step_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    claimed BOOLEAN NOT NULL DEFAULT FALSE
);

ALTER TABLE steps ALTER COLUMN session_id SET NOT NULL;
ALTER TABLE steps ALTER COLUMN step_index SET NOT NULL;
ALTER TABLE steps ALTER COLUMN claimed SET NOT NULL;

CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id),
    step_id UUID NOT NULL REFERENCES steps(id),
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE images ALTER COLUMN session_id SET NOT NULL;
ALTER TABLE images ALTER COLUMN step_id SET NOT NULL;

CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('kinect', 'fringe', 'fusion', 'calibration')),
    role TEXT NOT NULL,
    file_path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    array_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE artifacts ALTER COLUMN session_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_steps_session ON steps(session_id);
CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_sample_id ON sessions(sample_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_unique_file
    ON artifacts(session_id, category, role, file_path);
