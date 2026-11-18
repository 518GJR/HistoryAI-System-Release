-- Additive migration for the separately keyed signed textbook table.
-- Run only after a production backup and separate approval.
CREATE TABLE IF NOT EXISTS signed_textbook_knowledge (
  knowledge_id VARCHAR(64) PRIMARY KEY,
  lesson_number INTEGER NOT NULL,
  title VARCHAR(255) NOT NULL,
  content TEXT NOT NULL,
  content_status VARCHAR(64) NOT NULL,
  signoff_status VARCHAR(64) NOT NULL,
  source_status VARCHAR(64) NOT NULL,
  provenance_json TEXT NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY ix_signed_textbook_knowledge_lesson_number (lesson_number)
);

CREATE TABLE IF NOT EXISTS signed_textbook_source_fragment (
  fragment_id VARCHAR(128) PRIMARY KEY,
  source_id VARCHAR(128) NOT NULL,
  lesson_number INTEGER NOT NULL,
  book_id VARCHAR(32) NOT NULL,
  lesson_id VARCHAR(64) NOT NULL,
  excerpt TEXT NOT NULL,
  textbook_pages_json TEXT NOT NULL,
  pdf_pages_json TEXT NOT NULL,
  source_type VARCHAR(64) NOT NULL,
  verification_status VARCHAR(128) NOT NULL,
  support_score DOUBLE NULL,
  historical_source_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
  metadata_json TEXT NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY ix_signed_textbook_source_fragment_lesson_number (lesson_number)
);

CREATE TABLE IF NOT EXISTS signed_textbook_knowledge_source (
  knowledge_id VARCHAR(64) NOT NULL,
  fragment_id VARCHAR(128) NOT NULL,
  source_id VARCHAR(128) NOT NULL,
  relation_status VARCHAR(64) NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (knowledge_id, fragment_id, source_id),
  KEY ix_signed_textbook_knowledge_source_fragment_id (fragment_id),
  KEY ix_signed_textbook_knowledge_source_source_id (source_id),
  KEY ix_signed_textbook_knowledge_source_relation_status (relation_status),
  CONSTRAINT fk_signed_source_knowledge
    FOREIGN KEY (knowledge_id) REFERENCES signed_textbook_knowledge (knowledge_id),
  CONSTRAINT fk_signed_source_fragment
    FOREIGN KEY (fragment_id) REFERENCES signed_textbook_source_fragment (fragment_id)
);
