-- connect: psql postgresql://root:root@localhost:5432/chainlit

BEGIN;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='Thread' AND column_name='deletedat'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='Thread' AND column_name='deletedAt'
  ) THEN
    ALTER TABLE "Thread" RENAME COLUMN deletedat TO "deletedAt";
  END IF;
END $$;

COMMIT;
