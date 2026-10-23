import { useEffect, useMemo, useState } from "react";
import { Link, Outlet, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as Tabs from "@radix-ui/react-tabs";
import CodeMirror from "@uiw/react-codemirror";
import { markdown } from "@codemirror/lang-markdown";
import ReactMarkdown from "react-markdown";
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, CheckCircle2, Download, Eye, FileSearch, LoaderCircle, Plus, Save, Square, Trash2, X } from "lucide-react";
import clsx from "clsx";
import { api, ExtractionPromptPreview, Family, Job, json, Project, RelevanceProfile, Taxonomy, Workflow, WorkflowStep } from "./api";

export function AppShell() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-6 py-4">
          <Link to="/" className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-sm border border-slate-900 text-sm font-semibold text-slate-900">LR</div>
            <div><div className="font-semibold text-slate-950">Lit Review Studio</div><div className="hidden text-xs text-slate-500 sm:block">Papers, features, and supporting evidence</div></div>
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
    { key: "OPENAI_API_KEY", label: "OpenAI compatible API key" },
    { key: "SEMANTIC_SCHOLAR_API_KEY", label: "Semantic Scholar API key" },
  ];
  return <div className="relative">
    <button className="button-secondary whitespace-nowrap" onClick={() => setOpen(!open)}>Credentials <span className="badge">{ready} ready</span></button>
    {open && <div className="absolute right-0 z-50 mt-2 w-96 isolate rounded-sm border border-slate-300 bg-white p-5 opacity-100 shadow-lg">
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
  const deleteProject = useMutation({
    mutationFn: (project: Project) => api<{ deleted: boolean }>(`/api/projects/${project.id}`, { method: "DELETE" }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["projects"] }),
  });
  return <main className="mx-auto max-w-7xl px-6 py-14">
    <div className="grid gap-12 lg:grid-cols-[1fr_420px]">
      <section>

        <h1 className="text-3xl font-semibold tracking-tight text-slate-950">Projects</h1>
        <p className="mt-4 max-w-2xl text-slate-600">Start a review or continue an existing project. Each project keeps its papers, decisions, and results together.</p>
        <div className="mt-9 divide-y divide-slate-200">
          {(projects.data || []).map((project) => <article className="panel" key={project.id}>
            <Link className="block group" to={`/projects/${project.id}`}>
              <div className="flex items-start justify-between gap-4"><h2 className="text-lg font-semibold group-hover:text-blue-700">{project.name}</h2><span className="badge">{project.progress.downloaded} PDFs</span></div>
              <p className="mt-2 line-clamp-2 text-sm text-slate-500">{project.description || "No description yet."}</p>
              <Progress project={project} />
            </Link>
            <div className="mt-5 flex justify-end border-t border-slate-100 pt-4">
              <button className="button-secondary border-red-100 text-red-700 hover:border-red-200 hover:bg-red-50" disabled={deleteProject.isPending} onClick={() => { if (window.confirm(`Delete "${project.name}" and its local files?`)) deleteProject.mutate(project); }}><Trash2 size={14} /> Delete</button>
            </div>
          </article>)}
        </div>
        <Error mutation={deleteProject} />
      </section>
      <aside className="panel h-fit">
        <p className="label">Start a project</p>
        <h2 className="mt-4 text-2xl font-semibold">New project</h2>
        <input className="input mt-5" aria-label="Project name" placeholder="e.g. Human factors in aviation incidents" value={name} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => event.key === "Enter" && name.trim() && create.mutate()} />
        <button className="button-primary mt-4" disabled={!name.trim() || create.isPending} onClick={() => create.mutate()}><Plus size={17} /> Create project</button>
      </aside>
    </div>
  </main>;
}

const workflowSteps = ["define", "collect", "candidates", "extract", "filter", "annotate", "merge", "results"] as const;
type WorkflowStepName = typeof workflowSteps[number];
const stepLabels: Record<WorkflowStepName, string> = {
  define: "Define", collect: "Collect", candidates: "Select PDFs", extract: "Extract",
  filter: "Filter", annotate: "Annotate", merge: "Merge", results: "Results",
};

export function nextWorkflowStep(workflow: Workflow): WorkflowStepName {
  return workflowSteps.find(step => !workflow.steps[step].complete || workflow.steps[step].running) || "results";
}

export function Progress({ project, workflow }: { project: Project; workflow?: Workflow }) {
  if (!workflow) {
    const label = project.progress.extracted ? "Features extracted" : project.progress.downloaded > 0 ? "PDFs ready for extraction" : project.progress.searched ? "Paper search ready" : project.progress.defined && project.progress.taxonomy ? "Ready to collect" : "Project definition needed";
    return <p className="mt-3 text-sm text-slate-500">{label}</p>;
  }
  const completed = workflowSteps.filter(step => workflow.steps[step].complete).length;
  return <p className="text-sm tabular-nums text-slate-500">{completed} of {workflowSteps.length} steps complete</p>;
}

export function ProjectPage() {
  const { projectId = "" } = useParams();
  const client = useQueryClient();
  const [activeStep, setActiveStep] = useState<WorkflowStepName | null>(null);
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api<Project>(`/api/projects/${projectId}`), refetchInterval: 3000 });
  const workflow = useQuery({ queryKey: ["workflow", projectId], queryFn: () => api<Workflow>(`/api/projects/${projectId}/workflow`), refetchInterval: 1500 });
  useEffect(() => { setActiveStep(null); }, [projectId]);
  useEffect(() => {
    if (workflow.data) setActiveStep(current => current || nextWorkflowStep(workflow.data!));
  }, [workflow.data, projectId]);
  useEffect(() => {
    document.querySelector('[aria-current="step"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [activeStep]);
  const rename = useMutation({
    mutationFn: (name: string) => api<Project>(`/api/projects/${projectId}`, json("PATCH", { name })),
    onSuccess: () => { client.invalidateQueries({ queryKey: ["project", projectId] }); client.invalidateQueries({ queryKey: ["projects"] }); },
  });
  if (project.error || workflow.error) return <main className="mx-auto max-w-5xl px-6 py-10"><h1 className="text-2xl font-semibold">Unable to open this project</h1><p role="alert" className="mt-3 text-sm text-red-800">{(project.error || workflow.error)?.message}</p><Link className="button-quiet mt-5 inline-block" to="/">Back to projects</Link></main>;
  if (!project.data || !workflow.data || !activeStep) return <Loading />;
  const stepUnlocked = (index: number) => index === 0 || workflowSteps.slice(0, index).every(step => workflow.data!.steps[step].complete && !workflow.data!.steps[step].running);
  const goNext = () => {
    const index = workflowSteps.indexOf(activeStep);
    if (index < workflowSteps.length - 1) setActiveStep(workflowSteps[index + 1]);
  };
  return <main className="mx-auto max-w-[1500px] px-6 py-8">
    <Link to="/" className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-950"><ArrowLeft size={15} /> All projects</Link>
    <div className="mt-5 flex flex-wrap items-end justify-between gap-4"><div><p className="eyebrow">Project</p><div className="mt-1 flex items-center gap-3"><h1 className="text-3xl font-semibold tracking-tight">{project.data.name}</h1><button className="button-quiet" onClick={() => { const name = window.prompt("Rename project", project.data?.name); if (name?.trim()) rename.mutate(name.trim()); }}>Rename</button></div></div><Progress project={project.data} workflow={workflow.data} /></div>
    <Tabs.Root value={activeStep} onValueChange={value => setActiveStep(value as WorkflowStepName)} className="mt-8">
      <Tabs.List className="tabs" aria-label="Review workflow">
        {workflowSteps.map((value, index) => <Tabs.Trigger className="tab" disabled={!stepUnlocked(index)} value={value} key={value} aria-current={activeStep === value ? "step" : undefined} title={!stepUnlocked(index) ? `Complete ${stepLabels[workflowSteps[index - 1]]} first` : undefined}><span aria-hidden="true">{workflow.data.steps[value].complete ? <CheckCircle2 size={12} /> : index + 1}</span>{stepLabels[value]}</Tabs.Trigger>)}
      </Tabs.List>
      <Tabs.Content value="define"><Define projectId={projectId} status={workflow.data.steps.define} onNext={goNext} /></Tabs.Content>
      <Tabs.Content value="collect"><Collect projectId={projectId} status={workflow.data.steps.collect} onNext={goNext} /></Tabs.Content>
      <Tabs.Content value="candidates"><Candidates projectId={projectId} status={workflow.data.steps.candidates} onNext={goNext} /></Tabs.Content>
      <Tabs.Content value="extract"><Extract projectId={projectId} status={workflow.data.steps.extract} onNext={goNext} /></Tabs.Content>
      {(["filter", "annotate", "merge"] as const).map(stage => <Tabs.Content value={stage} key={stage}><ProcessingStage projectId={projectId} stage={stage} status={workflow.data!.steps[stage]} onNext={goNext} /></Tabs.Content>)}
      <Tabs.Content value="results"><Results projectId={projectId} status={workflow.data.steps.results} scoreStatus={workflow.data.steps.score} /></Tabs.Content>
    </Tabs.Root>
  </main>;
}

export function StepHeader({ eyebrow, title, description, status, action, onNext, nextLabel, ready = status.complete && !status.running }: { eyebrow: string; title: string; description: string; status: WorkflowStep; action: any; onNext?: () => void; nextLabel?: string; ready?: boolean }) {
  ready = ready && !status.stale && !status.running;
  const state = status.running ? "Running" : status.stale ? "Update needed" : ready ? "Complete" : status.complete ? "Unsaved changes" : "Not run";
  return <header className="stage-head">
    <div className="flex flex-wrap items-center justify-between gap-3"><p className="eyebrow">{eyebrow}</p><span className={clsx("text-sm font-medium", status.running ? "text-blue-700" : ready ? "text-emerald-700" : "text-slate-500")}>{state}</span></div>
    <div className="max-w-3xl"><h2>{title}</h2><p className="mt-2 text-sm leading-6 text-slate-600">{description}</p></div>
    {status.running && <p role="status" className="text-sm text-blue-700">This step is running. Follow its progress below; you can continue when it finishes.</p>}
    {!status.running && status.stale && <p role="status" className="border-l-2 border-amber-600 pl-3 text-sm text-amber-900">Inputs or settings changed. These results need to be recomputed before continuing.</p>}
    {!status.running && !status.stale && !ready && status.missing.length > 0 && <p className="text-sm text-slate-500">Needed before the next step: {status.missing.map(item => item === "successful upstream stage and current artifacts" ? "a successful run with the current inputs" : item).join(", ")}.</p>}
    {status.complete && !ready && !status.running && !status.stale && <p className="text-sm text-amber-800">Save your changes and run this step again before continuing.</p>}
    <div className="stage-actions" data-ready={ready}>{action}{ready && onNext && <button className="button-next" onClick={onNext}>{nextLabel ? `Next: ${nextLabel}` : "Next step"} <ArrowRight size={16} /></button>}</div>
  </header>;
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
    <StepHeader eyebrow="Step 1 of 8 · Define" title="Define what you want to find." description="Describe your question. Generate a search brief and feature families, then review them before collecting papers." status={{ ...status, running: status.running || generate.isPending }} ready={status.complete && !dirty && !generate.isPending} onNext={onNext} nextLabel="Collect" action={<><button className="button-secondary" disabled={!dirty || save.isPending} onClick={() => save.mutate()}><Save size={16} /> Save changes</button><button className="button-primary" disabled={!request.trim() || generate.isPending} onClick={() => generate.mutate()}>{generate.isPending ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} Generate project definition</button></>} />
    <div className="panel">
      <label className="label" htmlFor="research-request">What are you researching?</label>
      <textarea id="research-request" className="input min-h-28" placeholder="Describe the concepts, evidence, distinctions, and intended use in ordinary language." value={request} onChange={(event) => setRequest(event.target.value)} />
      <p className="mt-3 text-xs text-slate-500">Generate the definition, then review the search brief, feature families, and paper ranking rules below.</p>
      <details className="details"><summary>Generation model </summary><div className="form-grid mt-4">
        <Field label="Provider"><select className="input" value={generator.provider} onChange={event => setGenerator({ ...generator, provider: event.target.value, model: event.target.value === "gemini" ? "gemini-2.5-flash" : "gpt-4o-mini" })}><option value="gemini">Gemini</option><option value="openai">OpenAI-compatible</option></select></Field>
        <Field label="Model"><input className="input" value={generator.model} onChange={event => setGenerator({ ...generator, model: event.target.value })} /></Field>
        <Field label="Reasoning effort"><select className="input" value={generator.reasoning_effort} onChange={event => setGenerator({ ...generator, reasoning_effort: event.target.value })}>{["none", "minimal", "low", "medium", "high"].map(value => <option key={value}>{value}</option>)}</select></Field>
        <Field label="Base URL"><input className="input" placeholder="Use provider default" value={generator.base_url} onChange={event => setGenerator({ ...generator, base_url: event.target.value })} /></Field>
      </div></details>
      <Error mutation={generate} />
    </div>
    <div className="grid gap-5 xl:grid-cols-2"><MarkdownEditor title="Search brief" value={draft} onChange={value => { setDirty(true); setDraft(value); }} /><MarkdownEditor title="Extraction goal" value={goal} onChange={value => { setDirty(true); setGoal(value); }} /></div>
    <div className="panel"><div className="flex items-center justify-between"><div><p className="label">Feature families</p><h3 className="text-xl font-semibold">{tax.title || "Untitled taxonomy"}</h3></div><button className="button-secondary" onClick={() => { setDirty(true); setTax({ ...tax, families: [...tax.families, { id: "new_family", label: "New family", description: "", aliases: [] }] }); }}><Plus size={15} /> Add family</button></div>
      <input className="input mt-5" value={tax.title} placeholder="Taxonomy title" onChange={(event) => { setDirty(true); setTax({ ...tax, title: event.target.value }); }} />
      <div className="mt-5 grid gap-3">{tax.families.map((family, index) => <div className="rounded-sm border border-slate-200 bg-slate-50 p-4" key={`${family.id}-${index}`}>
        <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto_auto_auto]"><input className="input" value={family.id} onChange={(event) => updateFamily(index, { id: event.target.value })} /><input className="input" value={family.label} onChange={(event) => updateFamily(index, { label: event.target.value })} /><button className="icon-button" onClick={() => moveFamily(index, -1)}><ArrowUp size={16} /></button><button className="icon-button" onClick={() => moveFamily(index, 1)}><ArrowDown size={16} /></button><button className="icon-button" onClick={() => { setDirty(true); setTax({ ...tax, families: tax.families.filter((_, i) => i !== index) }); }}><X size={16} /></button></div>
        <textarea className="input mt-3 min-h-20" value={family.description} placeholder="What belongs in this family?" onChange={(event) => updateFamily(index, { description: event.target.value })} />
        <input className="input mt-3" value={family.aliases.join(", ")} placeholder="Aliases, comma separated" onChange={(event) => updateFamily(index, { aliases: event.target.value.split(",").map((value) => value.trim()).filter(Boolean) })} />
      </div>)}</div><Error mutation={save} /></div>
    <div className="panel overflow-hidden p-0"><div className="border-b border-slate-200 px-5 py-4"><p className="label">Collection settings</p><h3 className="mt-1 text-lg font-semibold">Paper ranking rules</h3><p className="mt-2 text-sm text-slate-500">These rules rank papers against your question. Each paper shows which criteria contributed to its score.</p></div><CodeMirror value={relevanceJson} height="520px" onChange={value => { setDirty(true); setRelevanceJson(value); }} /></div>
  </section>;
}

function MarkdownEditor({ title, value, onChange }: { title: string; value: string; onChange: (value: string) => void }) {
  const [preview, setPreview] = useState(false);
  return <div className="panel overflow-hidden p-0"><div className="flex items-center justify-between border-b border-slate-200 px-5 py-4"><h3 className="font-semibold">{title}</h3><button className="button-quiet" onClick={() => setPreview(!preview)}>{preview ? "Edit" : "Preview"}</button></div>
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
  return <section className="stage"><StepHeader eyebrow="Step 2 of 8 · Collect" title={cached ? "Apply the latest ranking to your saved candidates." : "Find and rank candidate papers."} description={cached ? "Re-rank your saved papers using the current rules. Enable Refresh metadata candidates in advanced settings to search the services again." : "Search the selected services and rank papers against your question. You will choose which PDFs to download next."} status={status} ready={status.complete && !dirty && !status.running} onNext={onNext} nextLabel="Select PDFs" action={<><button className="button-secondary" disabled={!dirty || save.isPending || status.running} onClick={() => save.mutate()}><Save size={15} /> Save settings</button><button className="button-primary" disabled={run.isPending || status.running} onClick={() => run.mutate()}>{run.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} {status.action_label}</button></>} />
    <div className="grid gap-5 lg:grid-cols-[1fr_420px]"><div className="panel"><h3 className="section-title">Search scope</h3><div className="form-grid">
      <fieldset><legend className="label">Sources</legend><div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3">{[["openalex", "OpenAlex"], ["semantic_scholar", "Semantic Scholar"], ["crossref", "Crossref"], ["europe_pmc", "Europe PMC"]].map(([source, label]) => <label className="flex items-center gap-2 text-sm" key={source}><input type="checkbox" checked={(value.sources || []).includes(source)} onChange={event => set("sources", event.target.checked ? [...(value.sources || []), source] : value.sources.filter((item: string) => item !== source))} />{label}</label>)}</div></fieldset>
      <Field label="Contact email (optional)" description="Helps scholarly services identify your requests and find open-access PDFs. Saved with the project."><input className="input" type="email" placeholder="researcher@example.edu" value={value.email || ""} onChange={(e) => set("email", e.target.value)} /></Field>
      <Field label="Earliest year"><input className="input" type="number" value={value.year_from || ""} onChange={(e) => set("year_from", e.target.value ? Number(e.target.value) : null)} /></Field>
      <Field label="Latest year"><input className="input" type="number" value={value.year_to || ""} onChange={(e) => set("year_to", e.target.value ? Number(e.target.value) : null)} /></Field>
      <Field label="Generated queries"><input className="input" type="number" value={value.max_queries || 8} onChange={(e) => set("max_queries", Number(e.target.value))} /></Field>
      <Field label="Results per query"><input className="input" type="number" value={value.top_k_per_query || 10} onChange={(e) => set("top_k_per_query", Number(e.target.value))} /></Field>
      <Field label="Maximum downloads"><input className="input" type="number" value={value.max_downloads || 50} onChange={(e) => set("max_downloads", Number(e.target.value))} /></Field>
    </div><details className="details"><summary>Advanced controls </summary><div className="form-grid mt-4">
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
  return <details className="min-w-28 text-xs"><summary className="cursor-pointer font-semibold">{Number(candidate.relevance_score || 0).toFixed(3)} <span className="font-normal text-blue-700">why?</span></summary><div className="mt-2 w-64 space-y-2 rounded-sm border border-slate-200 bg-white p-3 shadow-lg">
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
    helper.display({ id: "select", header: () => <input aria-label="Select all papers on this page" type="checkbox" checked={rows.length > 0 && rows.every(row => selected.has(row.candidate_id))} onChange={(event) => { setSelectionDirty(true); setSelected(event.target.checked ? new Set(rows.map(row => row.candidate_id)) : new Set()); }} />, cell: info => <input aria-label={`Select ${info.row.original.title}`} type="checkbox" checked={selected.has(info.row.original.candidate_id)} onChange={(event) => { const next = new Set(selected); event.target.checked ? next.add(info.row.original.candidate_id) : next.delete(info.row.original.candidate_id); setSelectionDirty(true); setSelected(next); }} /> }),
    helper.accessor("title", { header: "Paper", cell: info => <div><div className="font-semibold text-slate-900">{info.getValue()}</div><div className="mt-1 text-xs text-slate-500">{info.row.original.authors?.slice(0, 3).join(", ")}</div></div> }),
    helper.accessor("year", { header: "Year" }), helper.accessor("relevance_score", { header: "Relevance", cell: info => <ScoreBreakdown candidate={info.row.original} /> }),
    helper.accessor("citation_count", { header: "Citations" }), helper.accessor("source_apis", { header: "Sources", cell: info => <span className="text-xs">{info.getValue()?.join(", ")}</span> }),
    helper.accessor("download_status", { header: "Status", cell: info => <span className="badge">{info.getValue() || "not downloaded"}</span> }),
  ], [rows, selected]);
  const table = useReactTable({ data: rows, columns, getCoreRowModel: getCoreRowModel() });
  return <section className="stage"><StepHeader eyebrow="Step 3 of 8 · Select PDFs" title="Choose which papers to use." description="Select papers from the ranked list and download their full PDFs. At least one selected PDF is needed for extraction." status={status} ready={status.complete && !selectionDirty && !status.running} onNext={onNext} nextLabel="Extract" action={<><button className="button-secondary" onClick={() => candidates.refetch()}>Refresh table</button><button className="button-secondary" disabled={!selectionDirty || save.isPending || status.running} onClick={() => save.mutate()}><Save size={15} /> Save {selected.size}</button><button className="button-primary" disabled={selected.size === 0 || download.isPending || status.running} onClick={() => download.mutate()}>{download.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <Download size={16} />} Download selected PDFs</button></>} />
    <div className="grid gap-3 sm:grid-cols-3">
      <MaterialCount label="Candidates found" value={Number(status.materials.candidate_count || 0)} />
      <MaterialCount label="Papers selected" value={Number(status.materials.selected_count || 0)} />
      <MaterialCount label="Full PDFs downloaded" value={Number(status.materials.selected_downloaded_count || 0)} detail={`${Number(status.materials.downloaded_count || 0)} downloaded across all candidates`} accent />
    </div>
    <div className="grid gap-5 xl:grid-cols-[1fr_380px]"><div className="panel overflow-hidden p-0"><div className="border-b border-slate-200 p-4"><input className="input" placeholder="Filter by title, author, DOI, keyword..." value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} /></div><DataTable table={table} /><div className="flex items-center justify-between border-t border-slate-200 p-4 text-xs text-slate-500"><span>{candidates.data?.total || 0} candidates · page {page}</span><div className="flex gap-2"><button className="button-secondary" disabled={page === 1} onClick={() => setPage(page - 1)}>Previous</button><button className="button-secondary" disabled={page * pageSize >= (candidates.data?.total || 0)} onClick={() => setPage(page + 1)}>Next</button></div></div></div><JobPanel projectId={projectId} kind="collection_download" /></div><Error mutation={download} />
  </section>;
}

function Extract({ projectId, status, onNext }: { projectId: string; status: WorkflowStep; onNext: () => void }) {
  const client = useQueryClient();
  const config = useQuery({ queryKey: ["extractConfig", projectId], queryFn: () => api<Record<string, any>>(`/api/projects/${projectId}/extraction/config`) });
  const [dirty, setDirty] = useState(false);
  const [value, setValue] = useState<Record<string, any>>({});
  const [promptPreview, setPromptPreview] = useState<ExtractionPromptPreview | null>(null);
  const candidates = useQuery({ queryKey: ["extractCandidateCounts", projectId], queryFn: () => api<{ items: Candidate[]; total: number }>(`/api/projects/${projectId}/collection/candidates?page_size=500`) });
  useEffect(() => { if (config.data) { setValue(config.data); setDirty(false); } }, [config.data]);
  const save = useMutation({ mutationFn: () => api(`/api/projects/${projectId}/extraction/config`, json("PUT", value)) });
  const run = useMutation({ mutationFn: async () => { await save.mutateAsync(); return api<Job>(`/api/projects/${projectId}/extraction/run`, { method: "POST" }); }, onSuccess: () => { setDirty(false); client.invalidateQueries({ queryKey: ["workflow", projectId] }); } });
  const preview = useMutation({
    mutationFn: () => api<ExtractionPromptPreview>(`/api/projects/${projectId}/extraction/prompt-preview`, json("POST", value)),
    onSuccess: (data) => setPromptPreview(data),
  });
  const set = (key: string, next: any) => { setDirty(true); setValue({ ...value, [key]: next }); };
  return <section className="stage"><StepHeader eyebrow="Step 4 of 8 · Extract" title="Extract raw features from the selected full PDFs." description="Read your selected PDFs and save each feature with its supporting evidence. You will review relevance in the next step." status={{ ...status, running: status.running || run.isPending }} ready={status.complete && !status.running && !dirty && !run.isPending} onNext={onNext} nextLabel="Filter" action={<><button className="button-secondary" disabled={preview.isPending || status.running || Number(status.materials.selected_downloaded_count || 0) === 0} onClick={() => preview.mutate()}>{preview.isPending ? <LoaderCircle className="animate-spin" size={16} /> : <Eye size={16} />} View prompt</button><button className="button-primary" disabled={run.isPending || status.running || Number(status.materials.selected_downloaded_count || 0) === 0} onClick={() => run.mutate()}>{run.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} Extract raw features</button></>} />
    <div className="grid gap-5 lg:grid-cols-[1fr_420px]"><div className="panel"><div className="border-l-2 border-slate-300 pl-4 text-sm text-slate-700"><strong>Input: selected full PDFs.</strong> Each extracted feature keeps a reference to its source paper.<div className="mt-2 text-xs font-semibold">{candidates.data?.items.filter(row => row.selected).length || 0} selected · {candidates.data?.items.filter(row => ["downloaded", "already_exists"].includes(row.download_status)).length || 0} downloaded</div></div><div className="form-grid mt-6">
      <Field label="Provider"><select className="input" value={value.provider || "gemini"} onChange={(e) => set("provider", e.target.value)}><option value="gemini">Gemini</option><option value="openai">OpenAI-compatible</option></select></Field>
      <Field label="Model"><input className="input" value={value.model || ""} onChange={(e) => set("model", e.target.value)} /></Field>
      <Field label="Maximum features per PDF"><input className="input" type="number" value={value.max_features_per_pdf || 40} onChange={(e) => set("max_features_per_pdf", Number(e.target.value))} /></Field>
    </div><details className="details"><summary>Advanced model settings</summary><div className="form-grid">      <Field label="Reasoning effort"><select className="input" value={value.reasoning_effort || "low"} onChange={(e) => set("reasoning_effort", e.target.value)}>{["none", "minimal", "low", "medium", "high"].map(x => <option key={x}>{x}</option>)}</select></Field>
      <Field label="Temperature"><input className="input" type="number" step=".1" value={value.temperature ?? .1} onChange={(e) => set("temperature", Number(e.target.value))} /></Field>
      <Field label="Timeout seconds"><input className="input" type="number" value={value.timeout || 600} onChange={(e) => set("timeout", Number(e.target.value))} /></Field>
</div><Field label="Base URL"><input className="input mt-3" placeholder="Use provider default" value={value.base_url || ""} onChange={(e) => set("base_url", e.target.value)} /></Field></details><Error mutation={preview} /><Error mutation={run} /></div><JobPanel projectId={projectId} kind="direct_pdf_extraction" /></div>
    {promptPreview && <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30" onClick={() => setPromptPreview(null)}><aside className="h-full w-full max-w-4xl overflow-auto bg-white p-8 shadow-lg" onClick={event => event.stopPropagation()}>
      <button className="icon-button float-right" onClick={() => setPromptPreview(null)}><X size={18} /></button>
      <p className="eyebrow">Direct PDF extraction</p>
      <h2 className="mt-3 text-2xl font-semibold">Feature extraction prompt</h2>
      <div className="mt-4 flex flex-wrap gap-2"><span className="badge">{promptPreview.attachment_filename}</span><span className="badge">{promptPreview.paper_count} eligible PDFs</span><span className="badge">{promptPreview.max_features_per_pdf} max features</span></div>
      <div className="mt-7 space-y-5">
        <section><h3 className="label">System message</h3><pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-sm bg-slate-950 p-4 text-xs leading-5 text-slate-100">{promptPreview.system_prompt}</pre></section>
        <section><h3 className="label">User message</h3><pre className="mt-2 max-h-[70vh] overflow-auto whitespace-pre-wrap break-words rounded-sm border border-slate-200 bg-slate-50 p-4 text-xs leading-5 text-slate-700">{promptPreview.user_prompt}</pre></section>
      </div>
    </aside></div>}
  </section>;
}

export function ProcessingStage({ projectId, stage, status, onNext }: { projectId: string; stage: "filter" | "annotate" | "merge" | "score"; status: WorkflowStep; onNext?: () => void }) {
  const client = useQueryClient();
  const [dirty, setDirty] = useState(false);
  const [value, setValue] = useState<Record<string, any>>({});
  const [preview, setPreview] = useState<any>(null);
  const artifacts = {
    filter: ["filtered_features.jsonl", "rejected_features.jsonl", "curation_decisions.jsonl"],
    annotate: ["annotated_features.jsonl", "excluded_annotations.jsonl", "annotation_decisions.jsonl"],
    merge: ["llm_merged_features.jsonl", "llm_merge_candidates.jsonl", "llm_merge_decisions.jsonl"],
    score: ["scored_features.jsonl", "scoring_decisions.jsonl"],
  };
  const artifactLabels: Record<string, string> = {
    "filtered_features.jsonl": "Retained features", "rejected_features.jsonl": "Rejected features", "curation_decisions.jsonl": "Filter decisions",
    "annotated_features.jsonl": "Annotated features", "excluded_annotations.jsonl": "Annotation exclusions", "annotation_decisions.jsonl": "Annotation decisions",
    "llm_merged_features.jsonl": "Merged inventory", "llm_merge_candidates.jsonl": "Proposed pairs", "llm_merge_decisions.jsonl": "Merge decisions",
    "scored_features.jsonl": "Scored inventory", "scoring_decisions.jsonl": "Scoring decisions",
  };
  const [artifact, setArtifact] = useState(artifacts[stage][0]);
  const config = useQuery({ queryKey: ["stageConfig", projectId, stage], queryFn: () => api<Record<string, any>>(`/api/projects/${projectId}/stages/${stage}/config`) });
  useEffect(() => { if (config.data) { setValue(config.data); setDirty(false); } }, [config.data]);
  const save = () => api(`/api/projects/${projectId}/stages/${stage}/config`, json("PUT", value));
  const inspect = useMutation({ mutationFn: async () => { await save(); return api(`/api/projects/${projectId}/stages/${stage}/preview`, { method: "POST" }); }, onSuccess: data => { setPreview(data); setDirty(false); client.invalidateQueries({ queryKey: ["workflow", projectId] }); } });
  const run = useMutation({ mutationFn: async () => { await save(); return api(`/api/projects/${projectId}/stages/${stage}/run`, { method: "POST" }); }, onSuccess: () => { setDirty(false); setPreview(null); client.invalidateQueries({ queryKey: ["workflow", projectId] }); } });
  const rows = useQuery({ queryKey: ["stageRows", projectId, stage, artifact], queryFn: () => api<{ items: Record<string, any>[]; total: number }>(`/api/projects/${projectId}/stages/${stage}/results/${artifact}`), refetchInterval: status.running ? 1500 : false });
  const descriptions = {
    filter: "Keep or reject candidates against your goal. Inspect the retained inventory and rejection reasons before continuing.",
    annotate: "Write definitions, aliases, and examples. Inspect exclusions and features that need review.",
    merge: "Review similar features and combine equivalent terms. Keep the original evidence for every merged feature.",
    score: "Optionally score the merged inventory for literature support, annotation reliability, and distinctiveness.",
  };
  return <section className="stage">
    <StepHeader eyebrow={stage === "score" ? "Optional scoring" : { filter: "Step 5 of 8 · Filter", annotate: "Step 6 of 8 · Annotate", merge: "Step 7 of 8 · Merge" }[stage]} title={{ filter: "Filter feature candidates", annotate: "Annotate retained features", merge: "Merge equivalent features", score: "Score the final inventory" }[stage]} description={descriptions[stage]} status={{ ...status, running: status.running || run.isPending }} ready={status.complete && !status.running && !dirty && !run.isPending} onNext={onNext} nextLabel={{ filter: "Annotate", annotate: "Merge", merge: "Results", score: "" }[stage]} action={<><button className="button-secondary" disabled={status.running || inspect.isPending || run.isPending} onClick={() => inspect.mutate()}>Preview settings</button><button className="button-primary" disabled={status.running || run.isPending || inspect.isPending} onClick={() => run.mutate()}>{status.running ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} Run {stage}</button></>} />
    <div className="grid gap-5 lg:grid-cols-[1fr_420px]"><div className="panel"><h3 className="section-title">Settings</h3><div className="form-grid">
      <Field label="Provider"><select className="input" value={value.provider || "gemini"} onChange={e => { setDirty(true); setValue({ ...value, provider: e.target.value, base_url: null }); }}><option value="gemini">Gemini</option><option value="openai">OpenAI compatible</option></select></Field>
      <Field label="Model"><input className="input" value={value.model || ""} onChange={e => { setDirty(true); setValue({ ...value, model: e.target.value }); }} /></Field>
      <Field label="Domain rules"><select className="input" value={value.profile || ""} onChange={e => { setDirty(true); setValue({ ...value, profile: e.target.value || null }); }}><option value="">Use project taxonomy</option><option value="media_attributes">Media attributes</option></select></Field>
      <Field label={stage === "merge" ? "Neighbor candidates per feature" : "Features per batch"}><input className="input" type="number" min="1" value={value[stage === "merge" ? "top_k" : "batch_features"] || (stage === "merge" ? 10 : 25)} onChange={e => { setDirty(true); setValue({ ...value, [stage === "merge" ? "top_k" : "batch_features"]: Number(e.target.value) }); }} /></Field>
      {stage === "merge" && <Field label="Embedding model"><input className="input" value={value.embedding_model || "gemini-embedding-001"} onChange={e => { setDirty(true); setValue({ ...value, embedding_model: e.target.value }); }} /></Field>}
    </div><Error mutation={inspect} /><Error mutation={run} />
    {preview && <details className="details" open><summary>Planned inputs and outputs</summary><pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap text-xs">{JSON.stringify(preview, null, 2)}</pre></details>}
    </div><JobPanel projectId={projectId} kind={stage} /></div>
    <div className="panel"><h3 className="section-title mb-4">Inspect results</h3><div className="flex flex-wrap items-center gap-3"><select aria-label="Stage artifact" className="input max-w-md" value={artifact} onChange={e => setArtifact(e.target.value)}>{artifacts[stage].map(name => <option key={name} value={name}>{artifactLabels[name]}</option>)}</select><button className="button-secondary" onClick={() => rows.refetch()}>Refresh</button><a className="button-secondary" href={`/api/projects/${projectId}/stages/${stage}/exports/${artifact}`}><Download size={14} /> Export</a></div><p className="mt-4 text-sm text-slate-500">{rows.isLoading ? "Loading records…" : rows.data ? `${rows.data.total} records. Open a record to inspect its evidence and decisions.` : "No results yet. Run this step to create them."}</p><div className="mt-4 max-h-[600px] overflow-auto">{rows.data?.items.map((row, index) => <details className="details mb-2" key={index}><summary>{String(row.feature_name || row.feature_key || row.pair_id || `Decision ${index + 1}`)}</summary>{row.source_paper_ids?.length > 0 && <p className="mt-3 text-sm text-slate-600">Source papers: {row.source_paper_ids.join(", ")}</p>}{(row.source_quote || row.definition) && <p className="mt-2 text-sm leading-6 text-slate-700">{row.source_quote || row.definition}</p>}<pre className="mt-3 overflow-auto whitespace-pre-wrap text-xs text-slate-600">{JSON.stringify(row, null, 2)}</pre></details>)}</div></div>
  </section>;
}


type Feature = { category: string; feature_name: string; definitions: string[]; synonyms: string[]; examples: string[]; paper_count: number; mention_count: number; confidence: number; source_titles: string[]; evidence_types: string[] };
function Results({ projectId, status, scoreStatus }: { projectId: string; status: WorkflowStep; scoreStatus: WorkflowStep }) {
  const [query, setQuery] = useState(""); const [category, setCategory] = useState(""); const [detail, setDetail] = useState<Feature | null>(null);
  const [inventory, setInventory] = useState("merge");
  const inventories: Record<string, string> = { extract: "features.jsonl", filter: "filtered_features.jsonl", annotate: "annotated_features.jsonl", merge: "llm_merged_features.jsonl", score: "scored_features.jsonl" };
  const features = useQuery({ queryKey: ["features", projectId, inventory, query, category], queryFn: () => api<{ items: Feature[]; total: number }>(`/api/projects/${projectId}/stages/${inventory}/results/${inventories[inventory]}?page_size=500&query=${encodeURIComponent(query)}&category=${encodeURIComponent(category)}`) });
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
  return <section className="stage"><StepHeader eyebrow="Step 8 of 8 · Results" title="Inspect and export the merged feature inventory." description={`${features.data?.total || 0} features · ${errors.data?.total || 0} failed PDFs. Scoring is optional.`} status={status} action={<a className="button-primary" href={`/api/projects/${projectId}/stages/${inventory}/exports/${inventories[inventory]}`}><Download size={16} /> Download inventory</a>} />
    <div className="panel"><Field label="Inventory"><select className="input" value={inventory} onChange={e => setInventory(e.target.value)}>{Object.keys(inventories).filter(stage => stage !== "score" || scoreStatus.complete).map(stage => <option key={stage}>{stage}</option>)}</select></Field><a className="button-secondary mt-3" href={`/api/projects/${projectId}/stages/${inventory}/exports/${inventories[inventory].replace(".jsonl", ".csv")}`}>Download CSV</a></div>
    <details className="details bg-white"><summary>Optional feature scoring</summary><ProcessingStage projectId={projectId} stage="score" status={scoreStatus} /></details>
    <div className="panel overflow-hidden p-0"><div className="grid gap-3 border-b border-slate-200 p-4 md:grid-cols-[1fr_260px]"><input className="input" placeholder="Search features..." value={query} onChange={e => setQuery(e.target.value)} /><select className="input" value={category} onChange={e => setCategory(e.target.value)}><option value="">All families</option>{categories.map(x => <option key={x}>{x}</option>)}</select></div><DataTable table={table} /></div>
    {(errors.data?.items.length || 0) > 0 && <details className="details bg-white"><summary>Extraction failures ({errors.data?.total}) </summary><div className="mt-4 space-y-2">{errors.data?.items.map((error, index) => <div className="rounded-sm bg-red-50 p-3 text-xs text-red-800" key={index}><strong>{error.paper_id || "Unknown paper"}</strong><div className="mt-1">{error.error}</div></div>)}</div></details>}
    {detail && <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30" onClick={() => setDetail(null)}><aside className="h-full w-full max-w-2xl overflow-auto bg-white p-8 shadow-lg" onClick={e => e.stopPropagation()}><button className="icon-button float-right" onClick={() => setDetail(null)}><X size={18} /></button><span className="badge">{detail.category}</span><h2 className="mt-4 text-3xl font-semibold">{detail.feature_name}</h2><Detail label="Definitions" values={detail.definitions} /><Detail label="Synonyms" values={detail.synonyms} /><Detail label="Examples" values={detail.examples} /><Detail label="Source papers" values={detail.source_titles} /><Detail label="Evidence snippets" values={(mentions.data?.items || []).map(row => row.source_quote || row.snippet).filter(Boolean)} /></aside></div>}
  </section>;
}

function Detail({ label, values }: { label: string; values: string[] }) { return <div className="mt-8"><h3 className="label">{label}</h3><div className="mt-2 space-y-2">{(values || []).map((value, i) => <p className="rounded-sm bg-slate-50 p-3 text-sm text-slate-700" key={i}>{value}</p>)}</div></div>; }
function MaterialCount({ label, value, detail }: { label: string; value: number; detail?: string; accent?: boolean }) { return <div className="border-l border-slate-200 pl-4"><p className="text-sm text-slate-500">{label}</p><p className="mt-1 text-2xl font-semibold tabular-nums text-slate-950">{value}</p>{detail && <p className="mt-1 text-xs text-slate-500">{detail}</p>}</div>; }

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
  const labels: Record<string, string> = { queued: "Waiting to start", running: "Running", cancelling: "Stopping", completed: "Completed", failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted" };
  const active = Boolean(job && ["queued", "running", "cancelling"].includes(job.status));
  return <aside className="run-status" aria-label="Run status">
    <h3 className="text-sm font-medium text-slate-600">Run status</h3>
    <p className="mt-2 flex items-center gap-2 font-medium text-slate-950" role="status">{active && <LoaderCircle size={15} className="animate-spin" />}{job ? labels[job.status] || job.status : "Not started"}</p>
    {!job && <p className="mt-3 text-sm leading-6 text-slate-500">Check the settings, then run this step. Progress and any errors will appear here.</p>}
    {job?.status === "completed" && <p className="mt-3 text-sm text-slate-600">Results are ready to inspect.</p>}
    {job && ["cancelled", "interrupted", "failed"].includes(job.status) && <p className="mt-3 text-sm leading-6 text-slate-600">Progress is saved. Run this step again to resume.</p>}
    {job?.error && <p role="alert" className="mt-3 border-l-2 border-red-600 pl-3 text-sm text-red-800">{job.error}</p>}
    {active && <>
      <p className="mt-3 text-sm text-slate-600">{job?.status === "cancelling" ? "Stopping the current job. Please wait before starting another run." : "This project is processing. You can inspect the live log or cancel the run."}</p>
      <details className="details"><summary>Live log</summary><pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-600">{log || "Waiting for output…"}</pre></details>
      <button className="button-secondary mt-4" disabled={job?.status === "cancelling"} onClick={() => api(`/api/jobs/${job!.id}/cancel`, { method: "POST" })}><Square size={12} /> {job?.status === "cancelling" ? "Stopping…" : "Cancel run"}</button>
    </>}
  </aside>;
}

function Field({ label, description, children }: { label: string; description?: string; children: any }) { return <label className="block"><span className="label">{label}</span><div className="mt-2">{children}</div>{description && <span className="mt-2 block text-xs leading-5 text-slate-500">{description}</span>}</label>; }
function Error({ mutation }: { mutation: any }) { return mutation.error ? <p className="mt-4 rounded-sm bg-red-50 p-3 text-sm text-red-700">{mutation.error.message}</p> : null; }
function Loading() { return <div className="grid min-h-[50vh] place-items-center"><LoaderCircle className="animate-spin text-blue-600" /></div>; }
