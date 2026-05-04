/**
 * Wire types mirroring docs/openapi.yaml.
 *
 * Single source of truth: when the spec changes, regenerate this file by hand
 * (the project deliberately avoids `openapi-typescript` for now to keep the
 * dependency surface minimal). Each interface here lines up 1:1 with a
 * component schema in `docs/openapi.yaml`; if you add a field, add it in the
 * spec first.
 */

export type AnswerSource =
  | "RAG_LLM"
  | "EXTRACTIVE"
  | "FALLBACK"
  | "NO_INFO"
  | "LLM_TIMEOUT"
  | null;

export interface AskRequest {
  question: string;
  conversation_id?: number | null;
  debug?: boolean;
  strict_rag_verify?: boolean;
}

export interface QueryFilters {
  faculty: string | null;
  department: string | null;
  course_code: string | null;
  semester: number | null;
  content_types: string[];
  matched_terms: string[];
}

export interface RetrievedChunk {
  chunk_id: number;
  score: number;
  title: string;
  url: string;
  /** Promoted metadata fields surface as first-class keys (optional). */
  content_type?: string;
  department?: string;
  faculty?: string;
  course_code?: string;
  semester?: number;
  /** Full chunk metadata blob; key set is open-ended. */
  metadata: Record<string, unknown>;
  /** Only populated when the request set `debug: true`. */
  text?: string;
}

export interface LatencyMs {
  retrieve?: number;
  llm?: number;
  total?: number;
}

export interface AskResponse {
  conversation_id: number;
  answer: string;
  answer_source: AnswerSource;
  retrieved_chunks: RetrievedChunk[];
  filters: QueryFilters;
  latency_ms: LatencyMs;
}

export interface ApiError {
  error: {
    code:
      | "invalid_json"
      | "empty_question"
      | "invalid_conversation_id"
      | "conversation_not_found"
      | "llm_error"
      | "internal_error"
      | "upstream_unreachable"
      | "rate_limited";
    message: string;
  };
  conversation_id?: number | null;
}

export interface ConversationSummary {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: Message[];
}

export interface ConversationsListResponse {
  results: ConversationSummary[];
}
