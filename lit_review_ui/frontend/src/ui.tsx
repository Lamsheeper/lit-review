import { useEffect, useMemo, useState } from "react";
import { Link, Outlet, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as Tabs from "@radix-ui/react-tabs";
import CodeMirror from "@uiw/react-codemirror";
import { markdown } from "@codemirror/lang-markdown";
import ReactMarkdown from "react-markdown";
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, CheckCircle2, ChevronDown, Download, FileSearch, LoaderCircle, Plus, Save, Sparkles, Square, X } from "lucide-react";
import clsx from "clsx";
import { api, Family, Job, json, Project, RelevanceProfile, Taxonomy, Workflow, WorkflowStep } from "./api";

export function AppShell() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-6 py-4">
          <Link to="/" className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-slate-950 text-sm font-bold text-white">LR</div>
            <div><div className="font-semibold text-slate-950">Lit Review Studio</div><div className="text-xs text-slate-500">Evidence-bound feature extraction</div></div>
          </Link>
          <Credentials />
        </div>
      </header>
      <Outlet />
    </div>
  );
}

function Credentials() {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState<Record<string, string>>({});
  const query = useQuery({ queryKey: ["credentials"], queryFn: () => api<Record<string, { available: boolean; source: string | null }>>("/api/settings/credential-status") });
  const save = useMutation({
    mutationFn: () => api("/api/settings/session-credentials", json("PUT", { credentials: values })),
    onSuccess: () => { query.refetch(); setValues({}); setOpen(false); },
  });
  const ready = Object.values(query.data || {}).filter((item) => item.available).length;
  const credentials = [
    { key: "GEMINI_API_KEY", label: "Gemini API key" },
    { key: "SEMANTIC_SCHOLAR_API_KEY", label: "Semantic Scholar API key" },
  ];
  return <div className="relative">
    <button className="button-secondary" onClick={() => setOpen(!open)}>Credentials <span className="badge">{ready} ready</span></button>
    {open && <div className="absolute right-0 z-50 mt-2 w-96 isolate rounded-2xl border border-slate-300 bg-white p-5 opacity-100 shadow-2xl">
      <h3 className="font-semibold">Session credentials</h3>
      <p className="mt-1 text-xs text-slate-500">Overrides stay in memory and are never written to project files.</p>
      {credentials.map(({ key, label }) =>
        <label className="mt-4 block text-xs font-semibold text-slate-600" key={key}>{label}
          <input type="password" className="input mt-1 bg-white opacity-100" value={values[key] || ""} placeholder={query.data?.[key]?.available ? `${query.data[key].source} credential available` : "Not configured"} onChange={(event) => setValues({ ...values, [key]: event.target.value })} />
        </label>)}
      <button className="button-primary mt-5 w-full" onClick={() => save.mutate()}>Use for this session</button>
    </div>}
  </div>;
}

export function ProjectsPage() {
  const navigate = useNavigate();
  const client = useQueryClient();
  const [name, setName] = useState("");
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => api<Project[]>("/api/projects") });
  const create = useMutation({
    mutationFn: () => api<Project>("/api/projects", json("POST", { name })),
    onSuccess: (project) => { client.invalidateQueries({ queryKey: ["projects"] }); navigate(`/projects/${project.id}`); },
  });
  return <main className="mx-auto max-w-7xl px-6 py-14">
    <div className="grid gap-12 lg:grid-cols-[1fr_420px]">
      <section>
        <p className="eyebrow">Your reviews</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-slate-950">From a broad question to evidence-backed features.</h1>
        <p className="mt-4 max-w-2xl text-slate-600">Define a taxonomy, choose the literature that matters, and extract features directly from full PDFs.</p>
        <div className="mt-9 grid gap-4 md:grid-cols-2">
          {(projects.data || []).map((project) => <Link className="card group" to={`/projects/${project.id}`} key={project.id}>
            <div className="flex items-start justify-between"><h2 className="text-lg font-semibold group-hover:text-blue-700">{project.name}</h2><span className="badge">{project.progress.downloaded} PDFs</span></div>
            <p className="mt-2 line-clamp-2 text-sm text-slate-500">{project.description || "No description yet."}</p>
            <Progress project={project} />
          </Link>)}
        </div>
      </section>
      <aside className="card h-fit bg-slate-950 text-white">
        <p className="text-xs font-semibold uppercase tracking-[.2em] text-blue-300">Start a project</p>
        <h2 className="mt-4 text-2xl font-semibold">What feature set are you investigating?</h2>
        <input className="input mt-7 border-slate-700 bg-slate-900 text-white" placeholder="e.g. Human factors in aviation incidents" value={name} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => event.key === "Enter" && name.trim() && create.mutate()} />
        <button className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-blue-500 px-4 py-3 font-semibold hover:bg-blue-400 disabled:opacity-50" disabled={!name.trim() || create.isPending} onClick={() => create.mutate()}><Plus size={17} /> Create review</button>
      </aside>
    </div>
  </main>;
}

const workflowSteps = ["define", "collect", "candidates", "extract", "results"] as const;
type WorkflowStepName = typeof workflowSteps[number];

export function Progress({ project, workflow }: { project: Project; workflow?: Workflow }) {
  const steps = workflow ? workflowSteps.map(step => workflow.steps[step].complete) : [project.progress.defined, project.progress.taxonomy, project.progress.searched, project.progress.downloaded > 0, project.progress.extracted];
  return <div className="mt-6 flex gap-1.5">{steps.map((done, index) => <div key={index} className={clsx("h-1.5 flex-1 rounded-full", done ? "bg-blue-500" : "bg-slate-200")} />)}</div>;
}

export function ProjectPage() {
  const { projectId = "" } = useParams();
  const client = useQueryClient();
  const [activeStep, setActiveStep] = useState<WorkflowStepName>("define");
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api<Project>(`/api/projects/${projectId}`), refetchInterval: 3000 });
  const workflow = useQuery({ queryKey: ["workflow", projectId], queryFn: () => api<Workflow>(`/api/projects/${projectId}/workflow`), refetchInterval: 1500 });
  const rename = useMutation({
    mutationFn: (name: string) => api<Project>(`/api/projects/${projectId}`, json("PATCH", { name })),
    onSuccess: () => { client.invalidateQueries({ queryKey: ["project", projectId] }); client.invalidateQueries({ queryKey: ["projects"] }); },
  });
  if (!project.data || !workflow.data) return <Loading />;
  const stepUnlocked = (index: number) => index === 0 || workflowSteps.slice(0, index).every(step => workflow.data!.steps[step].complete && !workflow.data!.steps[step].running);
  const goNext = () => {
    const index = workflowSteps.indexOf(activeStep);
    if (index < workflowSteps.length - 1) setActiveStep(workflowSteps[index + 1]);
  };
  return <main className="mx-auto max-w-[1500px] px-6 py-8">
    <Link to="/" className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-950"><ArrowLeft size={15} /> All reviews</Link>
    <div className="mt-5 flex items-end justify-between"><div><p className="eyebrow">Review project</p><div className="mt-1 flex items-center gap-3"><h1 className="text-3xl font-semibold tracking-tight">{project.data.name}</h1><button className="button-quiet" onClick={() => { const name = window.prompt("Rename project", project.data?.name); if (name?.trim()) rename.mutate(name.trim()); }}>Rename</button></div></div><Progress project={project.data} workflow={workflow.data} /></div>
    <Tabs.Root value={activeStep} onValueChange={value => setActiveStep(value as WorkflowStepName)} className="mt-8">
      <Tabs.List className="tabs">
        {workflowSteps.map((value, index) => <Tabs.Trigger className="tab" disabled={!stepUnlocked(index)} value={value} key={value}><span>{workflow.data.steps[value].complete ? <CheckCircle2 size={12} /> : index + 1}</span>{value}</Tabs.Trigger>)}
      </Tabs.List>
      <Tabs.Content value="define"><Define projectId={projectId} status={workflow.data.steps.define} onNext={goNext} /></Tabs.Content>
      <Tabs.Content value="collect"><Collect projectId={projectId} status={workflow.data.steps.collect} onNext={goNext} /></Tabs.Content>
      <Tabs.Content value="candidates"><Candidates projectId={projectId} status={workflow.data.steps.candidates} onNext={goNext} /></Tabs.Content>
      <Tabs.Content value="extract"><Extract projectId={projectId} status={workflow.data.steps.extract} onNext={goNext} /></Tabs.Content>
      <Tabs.Content value="results"><Results projectId={projectId} status={workflow.data.steps.results} /></Tabs.Content>
    </Tabs.Root>
  </main>;
}

export function StepHeader({ eyebrow, title, description, status, action, onNext, ready = status.complete && !status.running }: { eyebrow: string; title: string; description: string; status: WorkflowStep; action: any; onNext?: () => void; ready?: boolean }) {
  return <div className="stage-head"><div className="max-w-3xl"><p className="eyebrow">{eyebrow}</p><h2>{title}</h2><p className="mt-2 text-sm text-slate-500">{description}</p>{status.running && <p className="mt-3 text-xs font-semibold text-blue-700">This step is running. Next will appear when the required output is ready.</p>}{!status.running && !ready && status.missing.length > 0 && <p className="mt-3 text-xs font-semibold text-amber-700">Before proceeding: {status.missing.join(", ")}.</p>}</div><div className="flex flex-wrap items-center gap-2">{action}{ready && onNext && <button className="button-next" onClick={onNext}><CheckCircle2 size={16} /> Next step <ArrowRight size={16} /></button>}</div></div>;
}

function Define({ projectId, status, onNext }: { projectId: string; status: WorkflowStep; onNext: () => void }) {
  const client = useQueryClient();
  const [request, setRequest] = useState("");
  const [dirty, setDirty] = useState(false);
  const [generator, setGenerator] = useState({ provider: "gemini", model: "gemini-2.5-flash", reasoning_effort: "low", base_url: "" });
  const docs = useQuery({ queryKey: ["docs", projectId], queryFn: () => api<{ draft_markdown: string; goal_markdown: string }>(`/api/projects/${projectId}/documents`) });
  const taxonomy = useQuery({ queryKey: ["taxonomy", projectId], queryFn: () => api<Taxonomy | null>(`/api/projects/${projectId}/taxonomy`) });
  const relevance = useQuery({ queryKey: ["relevance", projectId], queryFn: () => api<RelevanceProfile | null>(`/api/projects/${projectId}/relevance-scoring`) });
  const [draft, setDraft] = useState(""); const [goal, setGoal] = useState(""); const [tax, setTax] = useState<Taxonomy>({ version: 1, title: "", families: [] }); const [relevanceJson, setRelevanceJson] = useState("");
  useEffect(() => { if (docs.data) { setDraft(docs.data.draft_markdown); setGoal(docs.data.goal_markdown); } }, [docs.data]);
  useEffect(() => { if (taxonomy.data) setTax(taxonomy.data); }, [taxonomy.data]);
  useEffect(() => { if (relevance.data) setRelevanceJson(JSON.stringify(relevance.data, null, 2)); }, [relevance.data]);
  const generate = useMutation({
    mutationFn: () => api<{ draft_markdown: string; goal_markdown: string; taxonomy: Taxonomy; relevance_scoring: RelevanceProfile }>(`/api/projects/${projectId}/documents/generate`, json("POST", { request, ...generator, base_url: generator.base_url || null })),
    onSuccess: (data) => { setDraft(data.draft_markdown); setGoal(data.goal_markdown); setTax(data.taxonomy); setRelevanceJson(JSON.stringify(data.relevance_scoring, null, 2)); setDirty(false); client.invalidateQueries({ queryKey: ["workflow", projectId] }); },
  });
  const save = useMutation({
    mutationFn: async () => {
      const parsedRelevance = relevanceJson.trim() ? JSON.parse(relevanceJson) : null;
      await api(`/api/projects/${projectId}/documents`, json("PUT", { draft_markdown: draft, goal_markdown: goal }));
      await api(`/api/projects/${projectId}/taxonomy`, json("PUT", tax));
      if (parsedRelevance) await api(`/api/projects/${projectId}/relevance-scoring`, json("PUT", parsedRelevance));
    },
    onSuccess: () => { setDirty(false); client.invalidateQueries({ queryKey: ["project", projectId] }); client.invalidateQueries({ queryKey: ["workflow", projectId] }); },
  });
  const updateFamily = (index: number, value: Partial<Family>) => { setDirty(true); setTax({ ...tax, families: tax.families.map((family, i) => i === index ? { ...family, ...value } : family) }); };
  const moveFamily = (index: number, direction: number) => {
    const target = index + direction;
    if (target < 0 || target >= tax.families.length) return;
    const families = [...tax.families];
    [families[index], families[target]] = [families[target], families[index]];
    setDirty(true);
    setTax({ ...tax, families });
  };
  return <section className="stage">
    <StepHeader eyebrow="Step 1 · Define the target" title="Generate and approve the project definition." description="Describe the inventory once, run the definition generator, then review or edit its saved materials." status={status} ready={status.complete && !dirty} onNext={onNext} action={<><button className="button-secondary" disabled={!dirty || save.isPending} onClick={() => save.mutate()}><Save size={16} /> Save changes</button><button className="button-primary" disabled={!request.trim() || generate.isPending} onClick={() => generate.mutate()}>{generate.isPending ? <LoaderCircle className="animate-spin" size={16} /> : <Sparkles size={16} />} Generate project definition</button></>} />
    <div className="card">
      <label className="label">Broad feature request</label>
      <textarea className="input min-h-28" placeholder="Describe the concepts, evidence, distinctions, and intended use in ordinary language." value={request} onChange={(event) => setRequest(event.target.value)} />
      <p className="mt-3 text-xs text-slate-500">The primary action above creates the collection draft, extraction goal, feature families, and project-specific relevance scoring profile.</p>
      <details className="details"><summary>Generation model <ChevronDown size={15} /></summary><div className="form-grid mt-4">
        <Field label="Provider"><select className="input" value={generator.provider} onChange={event => setGenerator({ ...generator, provider: event.target.value, model: event.target.value === "gemini" ? "gemini-2.5-flash" : "gpt-4o-mini" })}><option value="gemini">Gemini</option><option value="openai">OpenAI-compatible</option></select></Field>
        <Field label="Model"><input className="input" value={generator.model} onChange={event => setGenerator({ ...generator, model: event.target.value })} /></Field>
        <Field label="Reasoning effort"><select className="input" value={generator.reasoning_effort} onChange={event => setGenerator({ ...generator, reasoning_effort: event.target.value })}>{["none", "minimal", "low", "medium", "high"].map(value => <option key={value}>{value}</option>)}</select></Field>
        <Field label="Base URL"><input className="input" placeholder="Use provider default" value={generator.base_url} onChange={event => setGenerator({ ...generator, base_url: event.target.value })} /></Field>
      </div></details>
      <Error mutation={generate} />
    </div>
    <div className="grid gap-5 xl:grid-cols-2"><MarkdownEditor title="Collection draft" value={draft} onChange={value => { setDirty(true); setDraft(value); }} /><MarkdownEditor title="Extraction goal" value={goal} onChange={value => { setDirty(true); setGoal(value); }} /></div>
    <div className="card"><div className="flex items-center justify-between"><div><p className="label">Feature families</p><h3 className="text-xl font-semibold">{tax.title || "Untitled taxonomy"}</h3></div><button className="button-secondary" onClick={() => { setDirty(true); setTax({ ...tax, families: [...tax.families, { id: "new_family", label: "New family", description: "", aliases: [] }] }); }}><Plus size={15} /> Add family</button></div>
      <input className="input mt-5" value={tax.title} placeholder="Taxonomy title" onChange={(event) => { setDirty(true); setTax({ ...tax, title: event.target.value }); }} />
      <div className="mt-5 grid gap-3">{tax.families.map((family, index) => <div className="rounded-xl border border-slate-200 bg-slate-50 p-4" key={`${family.id}-${index}`}>
        <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto_auto_auto]"><input className="input" value={family.id} onChange={(event) => updateFamily(index, { id: event.target.value })} /><input className="input" value={family.label} onChange={(event) => updateFamily(index, { label: event.target.value })} /><button className="icon-button" onClick={() => moveFamily(index, -1)}><ArrowUp size={16} /></button><button className="icon-button" onClick={() => moveFamily(index, 1)}><ArrowDown size={16} /></button><button className="icon-button" onClick={() => { setDirty(true); setTax({ ...tax, families: tax.families.filter((_, i) => i !== index) }); }}><X size={16} /></button></div>
        <textarea className="input mt-3 min-h-20" value={family.description} placeholder="What belongs in this family?" onChange={(event) => updateFamily(index, { description: event.target.value })} />
        <input className="input mt-3" value={family.aliases.join(", ")} placeholder="Aliases, comma separated" onChange={(event) => updateFamily(index, { aliases: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) })} />
      </div>)}</div><Error mutation={save} /></div>
    <div className="card overflow-hidden p-0"><div className="border-b border-slate-200 px-5 py-4"><p className="label">Candidate relevance algorithm</p><h3 className="mt-1 text-lg font-semibold">Generated, deterministic scoring profile</h3><p className="mt-2 text-sm text-slate-500">The LLM chooses topical criteria, signal weights, and exclusions for this project. LitHarvest applies the saved profile reproducibly and records a breakdown for every candidate.</p></div><CodeMirror value={relevanceJson} height="520px" onChange={value => { setDirty(true); setRelevanceJson(value); }} /></div>
  </section>;
}

function MarkdownEditor({ title, value, onChange }: { title: string; value: string; onChange: (value: string) => void }) {
  const [preview, setPreview] = useState(false);
  return <div className="card overflow-hidden p-0"><div className="flex items-center justify-between border-b border-slate-200 px-5 py-4"><h3 className="font-semibold">{title}</h3><button className="button-quiet" onClick={() => setPreview(!preview)}>{preview ? "Edit" : "Preview"}</button></div>
    {preview ? <article className="prose max-w-none p-6"><ReactMarkdown>{value}</ReactMarkdown></article> : <CodeMirror value={value} height="520px" extensions={[markdown()]} onChange={onChange} />}
  </div>;
}

function Collect({ projectId, status, onNext }: { projectId: string; status: WorkflowStep; onNext: () => void }) {
  const client = useQueryClient();
  const config = useQuery({ queryKey: ["collectionConfig", projectId], queryFn: () => api<Record<string, any>>(`/api/projects/${projectId}/collection/config`) });
  const [value, setValue] = useState<Record<string, any>>({});
  const [dirty, setDirty] = useState(false);
  useEffect(() => { if (config.data) setValue(config.data); }, [config.data]);
  const save = useMutation({ mutationFn: () => api(`/api/projects/${projectId}/collection/config`, json("PUT", value)), onSuccess: () => { setDirty(false); client.invalidateQueries({ queryKey: ["workflow", projectId] }); } });
  const run = useMutation({ mutationFn: async () => { await save.mutateAsync(); return api<Job>(`/api/projects/${projectId}/collection/search`, { method: "POST" }); }, onSuccess: () => client.invalidateQueries({ queryKey: ["workflow", projectId] }) });
  const set = (key: string, next: any) => { setDirty(true); setValue({ ...value, [key]: next }); };
  const cached = Boolean(status.materials.uses_cached_candidates);
  return <section className="stage"><StepHeader eyebrow="Step 2 · Collect papers" title={cached ? "Apply the latest ranking to your saved candidates." : "Find and rank candidate papers."} description={cached ? "This run uses the existing candidate cache and recalculates relevance locally. It does not repeat the external metadata search unless Refresh metadata candidates is enabled." : "Run metadata search across the configured scholarly sources, then rank the candidates with this project's relevance profile."} status={status} ready={status.complete && !dirty && !status.running} onNext={onNext} action={<><button className="button-secondary" disabled={!dirty || save.isPending || status.running} onClick={() => save.mutate()}><Save size={15} /> Save controls</button><button className="button-primary" disabled={run.isPending || status.running} onClick={() => run.mutate()}>{run.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} {status.action_label}</button></>} />
    <div className="grid gap-5 lg:grid-cols-[1fr_420px]"><div className="card"><h3 className="section-title">Search scope</h3><div className="form-grid">
      <Field label="Sources"><input className="input" value={(value.sources || []).join(", ")} onChange={(e) => set("sources", e.target.value.split(",").map(x => x.trim()).filter(Boolean))} /></Field>
      <Field label="API contact email (recommended)" description="Optional, but useful. It is sent as contact metadata to OpenAlex and Crossref, included in the collector User-Agent, and used to enable Unpaywall DOI lookups for open-access PDFs. It is saved in this project's collection config and is not treated as a secret."><input className="input" type="email" placeholder="researcher@example.edu" value={value.email || ""} onChange={(e) => set("email", e.target.value)} /></Field>
      <Field label="Earliest year"><input className="input" type="number" value={value.year_from || ""} onChange={(e) => set("year_from", e.target.value ? Number(e.target.value) : null)} /></Field>
      <Field label="Latest year"><input className="input" type="number" value={value.year_to || ""} onChange={(e) => set("year_to", e.target.value ? Number(e.target.value) : null)} /></Field>
      <Field label="Generated queries"><input className="input" type="number" value={value.max_queries || 8} onChange={(e) => set("max_queries", Number(e.target.value))} /></Field>
      <Field label="Results per query"><input className="input" type="number" value={value.top_k_per_query || 10} onChange={(e) => set("top_k_per_query", Number(e.target.value))} /></Field>
      <Field label="Maximum downloads"><input className="input" type="number" value={value.max_downloads || 50} onChange={(e) => set("max_downloads", Number(e.target.value))} /></Field>
    </div><details className="details"><summary>Advanced controls <ChevronDown size={15} /></summary><div className="form-grid mt-4">
      <Field label="Web search sources"><input className="input" value={(value.web_search_sources || []).join(", ")} onChange={(e) => set("web_search_sources", e.target.value.split(",").map(x => x.trim()).filter(Boolean))} /></Field>
      <Field label="Download workers"><input className="input" type="number" value={value.download_workers || 4} onChange={(e) => set("download_workers", Number(e.target.value))} /></Field>
      <Field label="Timeout seconds"><input className="input" type="number" value={value.timeout || 30} onChange={(e) => set("timeout", Number(e.target.value))} /></Field>
      <Field label="Retries"><input className="input" type="number" value={value.retries || 2} onChange={(e) => set("retries", Number(e.target.value))} /></Field>
      <Field label="Rate-limit delay"><input className="input" type="number" step=".1" value={value.rate_limit_delay ?? .1} onChange={(e) => set("rate_limit_delay", Number(e.target.value))} /></Field>
      <Field label="Extra queries (one per line)"><textarea className="input min-h-24" value={(value.extra_queries || []).map((item: any) => item.text).join("\n")} onChange={(e) => set("extra_queries", e.target.value.split("\n").map(text => text.trim()).filter(Boolean).map(text => ({ bucket: "ui_extra", text, terms: [] })))} /></Field>
      <label className="flex items-center gap-2 text-sm font-semibold text-slate-700"><input type="checkbox" checked={Boolean(value.refresh_candidates)} onChange={(e) => set("refresh_candidates", e.target.checked)} /> Refresh metadata candidates</label>
      <label className="flex items-center gap-2 text-sm font-semibold text-slate-700"><input type="checkbox" checked={Boolean(value.retry_failed_downloads)} onChange={(e) => set("retry_failed_downloads", e.target.checked)} /> Retry failed downloads</label>
    </div></details><Error mutation={run} /></div><JobPanel projectId={projectId} kind="collection_search" /></div>
  </section>;
}

type Candidate = { candidate_id: string; title: string; authors: string[]; year: number; doi: string; source_apis: string[]; relevance_score: number; relevance_score_breakdown: Record<string, any>; citation_count: number; matched_keywords: string[]; download_status: string; selected: boolean };
function ScoreBreakdown({ candidate }: { candidate: Candidate }) {
  const breakdown = candidate.relevance_score_breakdown || {};
  if (breakdown.mode !== "generated_profile") return <div><div className="font-semibold">{Number(candidate.relevance_score || 0).toFixed(3)}</div><div className="mt-1 text-[10px] text-slate-500">legacy ranking</div></div>;
  const criteria = (breakdown.criteria || []).filter((item: any) => item.contribution > 0);
  return <details className="min-w-28 text-xs"><summary className="cursor-pointer font-semibold">{Number(candidate.relevance_score || 0).toFixed(3)} <span className="font-normal text-blue-700">why?</span></summary><div className="mt-2 w-64 space-y-2 rounded-lg border border-slate-200 bg-white p-3 shadow-lg">
    <div className="font-semibold text-slate-800">{breakdown.profile_title}</div>
    {criteria.length ? criteria.slice(0, 4).map((item: any) => <div key={item.id}><span className="font-semibold">{item.label}</span>: {Number(item.score || 0).toFixed(2)}</div>) : <div className="text-slate-500">No topical criteria matched.</div>}
    {breakdown.exclusion_penalty > 0 && <div className="text-red-700">Exclusion penalty: -{Number(breakdown.exclusion_penalty).toFixed(2)}</div>}
  </div></details>;
}
function Candidates({ projectId, status, onNext }: { projectId: string; status: WorkflowStep; onNext: () => void }) {
  const client = useQueryClient();
  const [query, setQuery] = useState(""); const [page, setPage] = useState(1); const [selected, setSelected] = useState<Set<string>>(new Set());
  const [selectionDirty, setSelectionDirty] = useState(false);
  const pageSize = 100;
  const savedSelection = useQuery({ queryKey: ["candidateSelection", projectId], queryFn: () => api<{ candidate_ids: string[] }>(`/api/projects/${projectId}/collection/selection`) });
  const candidates = useQuery({ queryKey: ["candidates", projectId, query, page], queryFn: () => api<{ items: Candidate[]; total: number }>(`/api/projects/${projectId}/collection/candidates?page=${page}&page_size=${pageSize}&query=${encodeURIComponent(query)}`) });
  useEffect(() => { if (savedSelection.data) setSelected(new Set(savedSelection.data.candidate_ids)); }, [savedSelection.data]);
  useEffect(() => { if (candidates.data) setSelected(current => new Set([...current, ...candidates.data.items.filter(row => row.selected).map(row => row.candidate_id)])); }, [candidates.data]);
  const save = useMutation({ mutationFn: () => api(`/api/projects/${projectId}/collection/selection`, json("PUT", { candidate_ids: [...selected] })), onSuccess: () => { setSelectionDirty(false); client.invalidateQueries({ queryKey: ["workflow", projectId] }); } });
  const download = useMutation({ mutationFn: async () => { await save.mutateAsync(); return api<Job>(`/api/projects/${projectId}/collection/download`, { method: "POST" }); }, onSuccess: () => client.invalidateQueries({ queryKey: ["workflow", projectId] }) });
  const rows = candidates.data?.items || [];
  const helper = createColumnHelper<Candidate>();
  const columns = useMemo(() => [
    helper.display({ id: "select", header: () => <input type="checkbox" checked={rows.length > 0 && rows.every(row => selected.has(row.candidate_id))} onChange={(event) => { setSelectionDirty(true); setSelected(event.target.checked ? new Set(rows.map(row => row.candidate_id)) : new Set()); }} />, cell: info => <input type="checkbox" checked={selected.has(info.row.original.candidate_id)} onChange={(event) => { const next = new Set(selected); event.target.checked ? next.add(info.row.original.candidate_id) : next.delete(info.row.original.candidate_id); setSelectionDirty(true); setSelected(next); }} /> }),
    helper.accessor("title", { header: "Paper", cell: info => <div><div className="font-semibold text-slate-900">{info.getValue()}</div><div className="mt-1 text-xs text-slate-500">{info.row.original.authors?.slice(0, 3).join(", ")}</div></div> }),
    helper.accessor("year", { header: "Year" }), helper.accessor("relevance_score", { header: "Relevance", cell: info => <ScoreBreakdown candidate={info.row.original} /> }),
    helper.accessor("citation_count", { header: "Citations" }), helper.accessor("source_apis", { header: "Sources", cell: info => <span className="text-xs">{info.getValue()?.join(", ")}</span> }),
    helper.accessor("download_status", { header: "Status", cell: info => <span className="badge">{info.getValue() || "not downloaded"}</span> }),
  ], [rows, selected]);
  const table = useReactTable({ data: rows, columns, getCoreRowModel: getCoreRowModel() });
  return <section className="stage"><StepHeader eyebrow="Step 3 · Choose the corpus" title="Review the ranking and download the papers you want." description="Select candidate papers, then run the step to download their full PDFs. Next appears after at least one selected full PDF is available." status={status} ready={status.complete && !selectionDirty && !status.running} onNext={onNext} action={<><button className="button-secondary" onClick={() => candidates.refetch()}>Refresh table</button><button className="button-secondary" disabled={!selectionDirty || save.isPending || status.running} onClick={() => save.mutate()}><Save size={15} /> Save {selected.size}</button><button className="button-primary" disabled={selected.size === 0 || download.isPending || status.running} onClick={() => download.mutate()}>{download.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <Download size={16} />} Download selected PDFs</button></>} />
    <div className="grid gap-3 sm:grid-cols-3">
      <MaterialCount label="Candidates found" value={Number(status.materials.candidate_count || 0)} />
      <MaterialCount label="Papers selected" value={Number(status.materials.selected_count || 0)} />
      <MaterialCount label="Full PDFs downloaded" value={Number(status.materials.selected_downloaded_count || 0)} detail={`${Number(status.materials.downloaded_count || 0)} downloaded across all candidates`} accent />
    </div>
    <div className="grid gap-5 xl:grid-cols-[1fr_380px]"><div className="card overflow-hidden p-0"><div className="border-b border-slate-200 p-4"><input className="input" placeholder="Filter by title, author, DOI, keyword..." value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} /></div><DataTable table={table} /><div className="flex items-center justify-between border-t border-slate-200 p-4 text-xs text-slate-500"><span>{candidates.data?.total || 0} candidates · page {page}</span><div className="flex gap-2"><button className="button-secondary" disabled={page === 1} onClick={() => setPage(page - 1)}>Previous</button><button className="button-secondary" disabled={page * pageSize >= (candidates.data?.total || 0)} onClick={() => setPage(page + 1)}>Next</button></div></div></div><JobPanel projectId={projectId} kind="collection_download" /></div><Error mutation={download} />
  </section>;
}

function Extract({ projectId, status, onNext }: { projectId: string; status: WorkflowStep; onNext: () => void }) {
  const client = useQueryClient();
  const config = useQuery({ queryKey: ["extractConfig", projectId], queryFn: () => api<Record<string, any>>(`/api/projects/${projectId}/extraction/config`) });
  const [value, setValue] = useState<Record<string, any>>({});
  const candidates = useQuery({ queryKey: ["extractCandidateCounts", projectId], queryFn: () => api<{ items: Candidate[]; total: number }>(`/api/projects/${projectId}/collection/candidates?page_size=500`) });
  useEffect(() => { if (config.data) setValue(config.data); }, [config.data]);
  const save = useMutation({ mutationFn: () => api(`/api/projects/${projectId}/extraction/config`, json("PUT", value)) });
  const run = useMutation({ mutationFn: async () => { await save.mutateAsync(); return api<Job>(`/api/projects/${projectId}/extraction/run`, { method: "POST" }); }, onSuccess: () => client.invalidateQueries({ queryKey: ["workflow", projectId] }) });
  const set = (key: string, next: any) => setValue({ ...value, [key]: next });
  return <section className="stage"><StepHeader eyebrow="Step 4 · Direct PDF extraction" title="Extract raw features from the selected full PDFs." description="This runs lightweight direct-PDF preparation followed by evidence-bound feature extraction. No Markdown conversion, chunk indexing, or curation is performed." status={status} onNext={onNext} action={<button className="button-primary" disabled={run.isPending || status.running || Number(status.materials.selected_downloaded_count || 0) === 0} onClick={() => run.mutate()}>{run.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <Sparkles size={16} />} Extract raw features</button>} />
    <div className="grid gap-5 lg:grid-cols-[1fr_420px]"><div className="card"><div className="rounded-xl border border-blue-200 bg-blue-50 p-4 text-sm text-blue-900"><strong>Input mode: Direct PDF.</strong> The lightweight preparation step creates only the paper index required by extraction.<div className="mt-2 text-xs font-semibold">{candidates.data?.items.filter(row => row.selected).length || 0} selected · {candidates.data?.items.filter(row => ["downloaded", "already_exists"].includes(row.download_status)).length || 0} downloaded</div></div><div className="form-grid mt-6">
      <Field label="Provider"><select className="input" value={value.provider || "gemini"} onChange={(e) => set("provider", e.target.value)}><option value="gemini">Gemini</option><option value="openai">OpenAI-compatible</option></select></Field>
      <Field label="Model"><input className="input" value={value.model || ""} onChange={(e) => set("model", e.target.value)} /></Field>
      <Field label="Reasoning effort"><select className="input" value={value.reasoning_effort || "low"} onChange={(e) => set("reasoning_effort", e.target.value)}>{["none", "minimal", "low", "medium", "high"].map(x => <option key={x}>{x}</option>)}</select></Field>
      <Field label="Maximum features per PDF"><input className="input" type="number" value={value.max_features_per_pdf || 40} onChange={(e) => set("max_features_per_pdf", Number(e.target.value))} /></Field>
      <Field label="Temperature"><input className="input" type="number" step=".1" value={value.temperature ?? .1} onChange={(e) => set("temperature", Number(e.target.value))} /></Field>
      <Field label="Timeout seconds"><input className="input" type="number" value={value.timeout || 600} onChange={(e) => set("timeout", Number(e.target.value))} /></Field>
    </div><details className="details"><summary>Advanced transport <ChevronDown size={15} /></summary><Field label="Base URL"><input className="input mt-3" placeholder="Use provider default" value={value.base_url || ""} onChange={(e) => set("base_url", e.target.value)} /></Field></details><Error mutation={run} /></div><JobPanel projectId={projectId} kind="direct_pdf_extraction" /></div>
  </section>;
}

type Feature = { category: string; feature_name: string; definitions: string[]; synonyms: string[]; examples: string[]; paper_count: number; mention_count: number; confidence: number; source_titles: string[]; evidence_types: string[] };
function Results({ projectId, status }: { projectId: string; status: WorkflowStep }) {
  const [query, setQuery] = useState(""); const [category, setCategory] = useState(""); const [detail, setDetail] = useState<Feature | null>(null);
  const features = useQuery({ queryKey: ["features", projectId, query, category], queryFn: () => api<{ items: Feature[]; total: number }>(`/api/projects/${projectId}/results/features?page_size=500&query=${encodeURIComponent(query)}&category=${encodeURIComponent(category)}`) });
  const errors = useQuery({ queryKey: ["errors", projectId], queryFn: () => api<{ items: Record<string, any>[]; total: number }>(`/api/projects/${projectId}/results/errors?page_size=50`) });
  const mentions = useQuery({ queryKey: ["detailMentions", projectId, detail?.feature_name, detail?.category], enabled: Boolean(detail), queryFn: () => api<{ items: Record<string, any>[] }>(`/api/projects/${projectId}/results/mentions?page_size=100&query=${encodeURIComponent(detail?.feature_name || "")}&category=${encodeURIComponent(detail?.category || "")}`) });
  const rows = features.data?.items || []; const categories = [...new Set(rows.map(row => row.category))];
  const helper = createColumnHelper<Feature>();
  const columns = useMemo(() => [
    helper.accessor("feature_name", { header: "Feature", cell: info => <button className="text-left font-semibold text-blue-700 hover:underline" onClick={() => setDetail(info.row.original)}>{info.getValue()}</button> }),
    helper.accessor("category", { header: "Family", cell: info => <span className="badge">{info.getValue()}</span> }),
    helper.accessor("paper_count", { header: "Papers" }), helper.accessor("mention_count", { header: "Mentions" }),
    helper.accessor("confidence", { header: "Confidence", cell: info => Number(info.getValue() || 0).toFixed(2) }),
    helper.accessor("evidence_types", { header: "Evidence", cell: info => <span className="text-xs">{info.getValue()?.join(", ")}</span> }),
  ], []);
  const table = useReactTable({ data: rows, columns, getCoreRowModel: getCoreRowModel() });
  return <section className="stage"><StepHeader eyebrow="Step 5 · Raw extraction results" title="Inspect and export the uncurated feature inventory." description={`${features.data?.total || 0} features · ${errors.data?.total || 0} failed PDFs. This is the final v1 step; results remain read-only.`} status={status} action={<a className="button-primary" href={`/api/projects/${projectId}/exports/features.jsonl`}><Download size={16} /> Download raw features</a>} />
    <div className="card flex flex-wrap gap-2">{["features.csv", "feature_mentions.jsonl", "feature_mentions.csv"].map(file => <a className="button-secondary" href={`/api/projects/${projectId}/exports/${file}`} key={file}><Download size={14} /> {file}</a>)}</div>
    <div className="card overflow-hidden p-0"><div className="grid gap-3 border-b border-slate-200 p-4 md:grid-cols-[1fr_260px]"><input className="input" placeholder="Filter extracted features..." value={query} onChange={e => setQuery(e.target.value)} /><select className="input" value={category} onChange={e => setCategory(e.target.value)}><option value="">All families</option>{categories.map(x => <option key={x}>{x}</option>)}</select></div><DataTable table={table} /></div>
    {(errors.data?.items.length || 0) > 0 && <details className="details bg-white"><summary>Extraction failures ({errors.data?.total}) <ChevronDown size={15} /></summary><div className="mt-4 space-y-2">{errors.data?.items.map((error, index) => <div className="rounded-xl bg-red-50 p-3 text-xs text-red-800" key={index}><strong>{error.paper_id || "Unknown paper"}</strong><div className="mt-1">{error.error}</div></div>)}</div></details>}
    {detail && <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30" onClick={() => setDetail(null)}><aside className="h-full w-full max-w-2xl overflow-auto bg-white p-8 shadow-2xl" onClick={e => e.stopPropagation()}><button className="icon-button float-right" onClick={() => setDetail(null)}><X size={18} /></button><span className="badge">{detail.category}</span><h2 className="mt-4 text-3xl font-semibold">{detail.feature_name}</h2><Detail label="Definitions" values={detail.definitions} /><Detail label="Synonyms" values={detail.synonyms} /><Detail label="Examples" values={detail.examples} /><Detail label="Source papers" values={detail.source_titles} /><Detail label="Evidence snippets" values={(mentions.data?.items || []).map(row => row.source_quote || row.snippet).filter(Boolean)} /></aside></div>}
  </section>;
}

function Detail({ label, values }: { label: string; values: string[] }) { return <div className="mt-8"><h3 className="label">{label}</h3><div className="mt-2 space-y-2">{(values || []).map((value, i) => <p className="rounded-xl bg-slate-50 p-3 text-sm text-slate-700" key={i}>{value}</p>)}</div></div>; }
function MaterialCount({ label, value, detail, accent = false }: { label: string; value: number; detail?: string; accent?: boolean }) { return <div className={clsx("rounded-2xl border p-5 shadow-sm", accent ? "border-emerald-200 bg-emerald-50" : "border-slate-200 bg-white")}><p className="label">{label}</p><p className={clsx("mt-2 text-3xl font-semibold", accent ? "text-emerald-800" : "text-slate-950")}>{value}</p>{detail && <p className="mt-1 text-xs text-slate-500">{detail}</p>}</div>; }
function DataTable({ table }: { table: any }) { return <div className="overflow-auto"><table className="data-table"><thead>{table.getHeaderGroups().map((group: any) => <tr key={group.id}>{group.headers.map((header: any) => <th key={header.id}>{flexRender(header.column.columnDef.header, header.getContext())}</th>)}</tr>)}</thead><tbody>{table.getRowModel().rows.map((row: any) => <tr key={row.id}>{row.getVisibleCells().map((cell: any) => <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>)}</tr>)}</tbody></table></div>; }

function JobPanel({ projectId, kind }: { projectId: string; kind: string }) {
  const project = useQuery({ queryKey: ["projectJobs", projectId], queryFn: () => api<Project>(`/api/projects/${projectId}`), refetchInterval: 1500 });
  const job = project.data?.jobs?.find(item => item.kind === kind);
  const [log, setLog] = useState("");
  useEffect(() => {
    if (!job || !["queued", "running", "cancelling"].includes(job.status)) return;
    setLog("");
    const events = new EventSource(`/api/jobs/${job.id}/events`);
    events.addEventListener("log", (event) => setLog(current => current + JSON.parse((event as MessageEvent).data).text));
    events.addEventListener("status", (event) => { const status = JSON.parse((event as MessageEvent).data).status; if (["completed", "failed", "cancelled", "interrupted"].includes(status)) { events.close(); project.refetch(); } });
    return () => events.close();
  }, [job?.id, job?.status]);
  return <aside className="card h-fit bg-slate-950 text-slate-100"><div className="flex items-center justify-between"><div><p className="text-xs font-semibold uppercase tracking-[.16em] text-blue-300">Latest job</p><h3 className="mt-1 font-semibold">{job?.kind?.replaceAll("_", " ") || "No run yet"}</h3></div>{job && <span className="badge border-slate-700 bg-slate-800 text-slate-200">{job.status}</span>}</div>
    {job && ["queued", "running", "cancelling"].includes(job.status) && <div className="mt-5"><div className="flex items-center gap-2 text-sm text-blue-200"><LoaderCircle size={15} className="animate-spin" /> Pipeline is working</div><pre className="mt-4 max-h-80 overflow-auto whitespace-pre-wrap rounded-xl bg-black/30 p-3 text-[11px] leading-5 text-slate-300">{log || "Waiting for output..."}</pre><button className="mt-4 flex items-center gap-2 text-xs font-semibold text-red-300" onClick={() => api(`/api/jobs/${job.id}/cancel`, { method: "POST" })}><Square size={12} /> Cancel job</button></div>}
    {job?.error && <p className="mt-4 rounded-xl bg-red-950/70 p-3 text-xs text-red-200">{job.error}</p>}
    {!job && <p className="mt-4 text-sm text-slate-400">Start this stage to see live logs and status here.</p>}
  </aside>;
}

function Field({ label, description, children }: { label: string; description?: string; children: any }) { return <label className="block"><span className="label">{label}</span><div className="mt-2">{children}</div>{description && <span className="mt-2 block text-xs leading-5 text-slate-500">{description}</span>}</label>; }
function Error({ mutation }: { mutation: any }) { return mutation.error ? <p className="mt-4 rounded-xl bg-red-50 p-3 text-sm text-red-700">{mutation.error.message}</p> : null; }
function Loading() { return <div className="grid min-h-[50vh] place-items-center"><LoaderCircle className="animate-spin text-blue-600" /></div>; }
