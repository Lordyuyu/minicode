CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_codebase_path TEXT NOT NULL,
    test_command TEXT NOT NULL DEFAULT 'pytest',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS skills (
    skill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    embedding vector(768),
    success_rate DOUBLE PRECISION DEFAULT 0.0,
    invocation_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_skills_embedding ON skills USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS skill_invocations (
    invocation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    skill_id UUID NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
    input_context TEXT,
    output_result TEXT,
    success BOOLEAN NOT NULL DEFAULT false,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    invoked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_entries (
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_type TEXT NOT NULL,
    task_id UUID REFERENCES tasks(task_id) ON DELETE SET NULL,
    task_description TEXT NOT NULL DEFAULT '',
    actions JSONB DEFAULT '[]',
    outcome TEXT NOT NULL DEFAULT '',
    embedding vector(768),
    procedural_pattern TEXT DEFAULT '',
    timestamp DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now())
);

CREATE INDEX idx_memory_embedding ON memory_entries USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS patches (
    patch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    original_content TEXT NOT NULL,
    patched_content TEXT NOT NULL,
    diff TEXT NOT NULL,
    verified BOOLEAN NOT NULL DEFAULT false,
    verification_output TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bug_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
