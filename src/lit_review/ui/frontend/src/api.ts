export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    const message = typeof detail.detail === 'object' ? detail.detail.message : detail.detail;
    if(detail.detail?.readiness && typeof window !== 'undefined')window.dispatchEvent(new CustomEvent('readiness-needed',{detail:detail.detail.readiness}));
    throw new Error(message || response.statusText);
  }
  return response.json();
}

export const json = (method: string, value: unknown): RequestInit => ({
  method,
  body: JSON.stringify(value),
});

export type Project = {
  id: string;
  name: string;
  description: string;
  updated_at: string;
  progress: {
    defined: boolean;
    taxonomy: boolean;
    searched: boolean;
    downloaded: number;
    extracted: boolean;
  };
  jobs?: Job[];
  setup?: {request:string;status:string;message?:string} | null;
};

export type Job = {
  id: string;
  kind: string;
  status: string;
  error: string;
  created_at: string;
};

export type WorkflowStep = {
  complete: boolean;
  running: boolean;
  stale?: boolean;
  missing: string[];
  materials: Record<string, any>;
  action_label: string;
};
export type Workflow = {
  steps: Record<"define" | "collect" | "candidates" | "extract" | "filter" | "annotate" | "merge" | "score" | "results", WorkflowStep>;
};

export type ExtractionPromptPreview = {
  system_prompt: string;
  user_prompt: string;
  attachment_filename: string;
  paper: Record<string, any>;
  paper_count: number;
  max_features_per_pdf: number;
};

export type Family = { id: string; label: string; description: string; aliases: string[] };
export type Taxonomy = { version: 1; title: string; families: Family[] };
export type RelevanceProfile = {
  version: 1;
  title: string;
  description: string;
  criteria: { id: string; label: string; description: string; weight: number; terms: string[] }[];
  field_weights: { title: number; abstract: number; matched_keywords: number };
  metadata_weights: { source_relevance: number; citation_impact: number; recency: number; open_access: number; doi: number };
  exclusions: { term: string; penalty: number; reason: string }[];
};


export type Readiness={ready:boolean;message:string;requirements:{id:string;kind:string;label:string;message:string;field?:string}[]};
