-- Enable pgcrypto (for gen_random_uuid)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Clean slate (drop in dependency order)
DO $$
BEGIN
  IF to_regclass('"Element"') IS NOT NULL THEN DROP TABLE "Element" CASCADE; END IF;
  IF to_regclass('"Feedback"') IS NOT NULL THEN DROP TABLE "Feedback" CASCADE; END IF;
  IF to_regclass('"Step"') IS NOT NULL THEN DROP TABLE "Step" CASCADE; END IF;
  IF to_regclass('"Thread"') IS NOT NULL THEN DROP TABLE "Thread" CASCADE; END IF;
  IF to_regclass('"User"') IS NOT NULL THEN DROP TABLE "User" CASCADE; END IF;
  IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'StepType') THEN DROP TYPE "StepType"; END IF;
END $$;

-- Enum first
DO $$ BEGIN
  CREATE TYPE "StepType" AS ENUM (
    'assistant_message',
    'embedding',
    'llm',
    'retrieval',
    'rerank',
    'run',
    'system_message',
    'tool',
    'undefined',
    'user_message'
  );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- User
CREATE TABLE "User" (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  createdAt  timestamp NOT NULL DEFAULT now(),
  updatedAt  timestamp NOT NULL DEFAULT now(),
  metadata   jsonb     NOT NULL,
  identifier text      NOT NULL,
  CONSTRAINT uq_user_identifier UNIQUE (identifier)
);
CREATE INDEX idx_user_identifier ON "User"(identifier);

-- Thread (depends on User)
CREATE TABLE "Thread" (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  createdAt  timestamp NOT NULL DEFAULT now(),
  updatedAt  timestamp NOT NULL DEFAULT now(),
  deletedAt  timestamp,
  name       text,
  metadata   jsonb     NOT NULL,
  tags       text[]    DEFAULT '{}',
  userId     uuid,
  CONSTRAINT fk_thread_user FOREIGN KEY (userId) REFERENCES "User"(id)
);
CREATE INDEX idx_thread_createdAt ON "Thread"(createdAt);
CREATE INDEX idx_thread_name      ON "Thread"(name);

-- Step (depends on Thread, self-ref parent)
CREATE TABLE "Step" (
  id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  createdAt timestamp NOT NULL DEFAULT now(),
  updatedAt timestamp NOT NULL DEFAULT now(),
  parentId  uuid,
  threadId  uuid,
  input     text,
  metadata  jsonb NOT NULL,
  name      text,
  output    text,
  type      "StepType" NOT NULL,
  showInput text DEFAULT 'json',
  isError   boolean DEFAULT false,
  startTime timestamp NOT NULL,
  endTime   timestamp NOT NULL,
  CONSTRAINT fk_step_parent FOREIGN KEY (parentId) REFERENCES "Step"(id) ON DELETE CASCADE,
  CONSTRAINT fk_step_thread FOREIGN KEY (threadId) REFERENCES "Thread"(id) ON DELETE CASCADE
);
CREATE INDEX idx_step_createdAt        ON "Step"(createdAt);
CREATE INDEX idx_step_endTime          ON "Step"(endTime);
CREATE INDEX idx_step_parentId         ON "Step"(parentId);
CREATE INDEX idx_step_startTime        ON "Step"(startTime);
CREATE INDEX idx_step_threadId         ON "Step"(threadId);
CREATE INDEX idx_step_type             ON "Step"(type);
CREATE INDEX idx_step_name             ON "Step"(name);
CREATE INDEX idx_step_threadId_time    ON "Step"(threadId, startTime, endTime);

-- Element (depends on Step, Thread)
CREATE TABLE "Element" (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  createdAt  timestamp NOT NULL DEFAULT now(),
  updatedAt  timestamp NOT NULL DEFAULT now(),
  threadId   uuid,
  stepId     uuid NOT NULL,
  metadata   jsonb NOT NULL,
  mime       text,
  name       text NOT NULL,
  objectKey  text,
  url        text,
  chainlitKey text,
  display    text,
  size       text,
  language   text,
  page       integer,
  props      jsonb,
  CONSTRAINT fk_element_step   FOREIGN KEY (stepId)   REFERENCES "Step"(id)   ON DELETE CASCADE,
  CONSTRAINT fk_element_thread FOREIGN KEY (threadId) REFERENCES "Thread"(id) ON DELETE CASCADE
);
CREATE INDEX idx_element_stepId   ON "Element"(stepId);
CREATE INDEX idx_element_threadId ON "Element"(threadId);

-- Feedback (depends on Step)
CREATE TABLE "Feedback" (
  id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  createdAt timestamp NOT NULL DEFAULT now(),
  updatedAt timestamp NOT NULL DEFAULT now(),
  stepId    uuid,
  name      text NOT NULL,
  value     double precision NOT NULL,
  comment   text,
  CONSTRAINT fk_feedback_step FOREIGN KEY (stepId) REFERENCES "Step"(id)
);
CREATE INDEX idx_feedback_createdAt  ON "Feedback"(createdAt);
CREATE INDEX idx_feedback_name       ON "Feedback"(name);
CREATE INDEX idx_feedback_stepId     ON "Feedback"(stepId);
CREATE INDEX idx_feedback_value      ON "Feedback"(value);
CREATE INDEX idx_feedback_name_value ON "Feedback"(name, value);
