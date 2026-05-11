CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL,
    total_steps INTEGER NOT NULL,
    completed_steps INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS steps (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    step_index INTEGER,
    status TEXT NOT NULL,
    claimed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    step_id UUID REFERENCES steps(id),
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_steps_session ON steps(session_id);
CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id);