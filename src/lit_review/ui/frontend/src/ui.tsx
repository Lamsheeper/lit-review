import { useContext, useEffect, useMemo, useRef, useState } from "react";
import { Link, Outlet, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import CodeMirror from "@uiw/react-codemirror";
import { markdown } from "@codemirror/lang-markdown";
import ReactMarkdown from "react-markdown";
import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, CheckCircle2, Download, Eye, FileSearch, LoaderCircle, Plus, Save, Square, X } from "lucide-react";
import clsx from "clsx";
import { api, ExtractionPromptPreview, Family, Job, json, Project, RelevanceProfile, Taxonomy, Workflow, WorkflowStep } from "./api";

import { GoogleCloudSettings } from "./google-cloud";

import {ReadyContext,ReadinessNotice,useReadiness,ReadinessButton} from "./readiness";

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
  const [credentialTab, setCredentialTab] = useState("keys");
  useEffect(() => {
    const showCloud = () => { setOpen(true); setCredentialTab("cloud"); };
    window.addEventListener("open-google-cloud", showCloud);
    return () => window.removeEventListener("open-google-cloud", showCloud);
  }, []);
  const [values, setValues] = useState<Record<string, string>>({});
  const query = useQuery({ queryKey: ["credentials"], queryFn: () => api<Record<string, { available: boolean; source: string | null }>>("/api/settings/credential-status") });
  const save = useMutation({
    mutationFn: () => api("/api/settings/session-credentials", json("PUT", { credentials: values })),
    onSuccess: () => { query.refetch(); setValues({}); setOpen(false); window.dispatchEvent(new Event("credentials-updated")); },
  });
  const ready = Object.values(query.data || {}).filter((item) => item.available).length;
  const credentials = [
    { key: "GEMINI_API_KEY", label: "Gemini API key" },
    { key: "OPENAI_API_KEY", label: "OpenAI compatible API key" },
    { key: "SEMANTIC_SCHOLAR_API_KEY", label: "Semantic Scholar API key" },
  ];
  return <div className="relative">
    <button className="button-secondary whitespace-nowrap" onClick={() => setOpen(!open)}>Credentials <span className="badge">{ready} ready</span></button>
    {open && <div className="absolute right-0 z-50 mt-2 max-h-[85vh] w-96 overflow-y-auto isolate rounded-sm border border-slate-300 bg-white p-5 opacity-100 shadow-lg">
      <h3 className="font-semibold">Credentials</h3>
      <div className="mt-3 flex gap-2" role="tablist" aria-label="Credential type">
        <button className="button-secondary" role="tab" aria-selected={credentialTab === "keys"} onClick={() => setCredentialTab("keys")}>API keys</button>
        <button className="button-secondary" role="tab" aria-selected={credentialTab === "cloud"} onClick={() => setCredentialTab("cloud")}>Google Cloud</button>
      </div>
      {credentialTab === "cloud" ? <GoogleCloudSettings /> : <>

      <p className="mt-1 text-xs text-slate-500">Overrides stay in memory and are never written to project files.</p>
      {credentials.map(({ key, label }) =>
        <label className="mt-4 block text-xs font-semibold text-slate-600" key={key}>{label}
          <input type="password" className="input mt-1 bg-white opacity-100" value={values[key] || ""} placeholder={query.data?.[key]?.available ? `${query.data[key].source} credential available` : "Not configured"} onChange={(event) => setValues({ ...values, [key]: event.target.value })} />
        </label>)}

      <Error mutation={save} />
      <button className="button-primary mt-5 w-full" onClick={() => save.mutate()}>Use for this session</button></>}
    </div>}
  </div>;
}

export function ProjectsPage() {
  const navigate = useNavigate();
  const client = useQueryClient();
  const [request, setRequest] = useState("");
  const readiness=useReadiness();
  const projects = useQuery({queryKey:["projects"],queryFn:()=>api<Project[]>("/api/projects")});
  const create = useMutation({
    mutationFn:()=>api<Project>("/api/projects/start",json("POST",{request:request.trim()})),
    onSuccess:project=>{client.invalidateQueries({queryKey:["projects"]});navigate(`/projects/${project.id}`)},
  });
  return <main className="prompt-home"><section className="research-start"><div className="research-symbol"><FileSearch size={27}/></div><h1>What would you like to discover?</h1><p>Describe a concept or research question. We’ll create your workspace and prepare a search for relevant literature.</p><form className="research-composer" onSubmit={event=>{event.preventDefault();if(request.trim()&&!create.isPending&&readiness.data?.ready)create.mutate()}}><textarea aria-label="Research question" placeholder="e.g. How do people express emotion in written communication?" rows={2} value={request} onChange={event=>setRequest(event.target.value)} onKeyDown={event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();if(request.trim()&&!create.isPending&&readiness.data?.ready)event.currentTarget.form?.requestSubmit()}}}/><button aria-label="Start research" disabled={!request.trim()||create.isPending||!readiness.data?.ready}>{create.isPending?<LoaderCircle className="animate-spin" size={20}/>:<ArrowUp size={20}/>}</button></form><ReadinessNotice report={readiness.data} error={readiness.error} checking={readiness.isPending} check={()=>readiness.refetch()}/><Error mutation={create}/>{create.isPending&&<p role="status" className="mt-5 text-sm text-slate-500">Creating your workspace…</p>}</section>{!!projects.data?.length&&<details className="recent-research"><summary>Continue a previous project</summary><div>{projects.data.map(project=><Link key={project.id} to={`/projects/${project.id}`}>{project.name}<ArrowRight size={14}/></Link>)}</div></details>}</main>;
}

const workflowSteps = ["define", "papers", "extract", "refine", "merge", "results"] as const;
type WorkflowStepName = typeof workflowSteps[number];
const stepLabels: Record<WorkflowStepName, string> = { define: "Your search", papers: "Papers", extract: "Extract", refine: "Refine", merge: "Merge", results: "Results" };
function stageStatus(workflow: Workflow, step: WorkflowStepName): WorkflowStep {
  if (step === "papers") return { ...workflow.steps.candidates, complete: workflow.steps.collect.complete && workflow.steps.candidates.complete, running: workflow.steps.collect.running || workflow.steps.candidates.running };
  if (step === "refine") return { ...workflow.steps.filter, complete: workflow.steps.filter.complete && workflow.steps.annotate.complete, running: workflow.steps.filter.running || workflow.steps.annotate.running, stale: workflow.steps.filter.stale || workflow.steps.annotate.stale };
  return workflow.steps[step];
}
export function nextWorkflowStep(workflow: Workflow): WorkflowStepName {
  if(workflow.steps.results.materials.has_saved_inventory&&!Object.values(workflow.steps).some(step=>step.running))return "results";
  return workflowSteps.find(step => { const status = stageStatus(workflow, step); return !status.complete || status.running || status.stale; }) || "results";
}
export function Progress({ project, workflow }: { project: Project; workflow?: Workflow }) {
  if (!workflow) {
    const label = project.progress.extracted ? "Features extracted" : project.progress.downloaded > 0 ? "PDFs ready for extraction" : project.progress.searched ? "Paper search ready" : project.progress.defined && project.progress.taxonomy ? "Ready to collect" : "Project definition needed";
    return <p className="flow-muted">{label}</p>;
  }
  return <span className="flow-muted">{workflowSteps.filter(step => stageStatus(workflow, step).complete).length} of {workflowSteps.length} steps complete</span>;
}
export function ProjectPage() {
  const { projectId = "" } = useParams();
  const client = useQueryClient();
  const [activeStep, setActiveStep] = useState<WorkflowStepName | null>(null);
  const [history, setHistory] = useState(false);
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api<Project>(`/api/projects/${projectId}`), refetchInterval: 3000 });
  const workflow = useQuery({ queryKey: ["workflow", projectId], queryFn: () => api<Workflow>(`/api/projects/${projectId}/workflow`), refetchInterval: 1500 });
  const action = activeStep === "papers" ? (workflow.data?.steps.collect.complete && !workflow.data.steps.collect.running ? "download" : "collect") : ({ define: "setup", results: "results" } as Record<string,string>)[activeStep || "define"] || activeStep || "setup";
  const readiness = useReadiness(projectId, action);
  useEffect(() => { setActiveStep(null); setHistory(false); }, [projectId]);
  useEffect(()=>{document.querySelector<HTMLElement>(".focus-card h2")?.focus({preventScroll:true})},[activeStep]);
  useEffect(()=>{const close=(event:KeyboardEvent)=>{if(event.key==="Escape")setHistory(false)};window.addEventListener("keydown",close);return()=>window.removeEventListener("keydown",close)},[]);
  useEffect(() => { if (workflow.data) setActiveStep(current => current || nextWorkflowStep(workflow.data!)); }, [workflow.data, projectId]);
  useEffect(() => { void client.invalidateQueries({queryKey:["readiness",projectId]}); }, [workflow.data?.steps.candidates.materials.selected_count, workflow.data?.steps.candidates.materials.selected_downloaded_count]);
  const rename = useMutation({ mutationFn: (name:string) => api<Project>(`/api/projects/${projectId}`,json("PATCH",{name})), onSuccess: () => { client.invalidateQueries({queryKey:["project",projectId]}); client.invalidateQueries({queryKey:["projects"]}); } });
  if (project.error || workflow.error) return <main className="flow-workspace"><h1>Unable to open this project</h1><p role="alert">{(project.error || workflow.error)?.message}</p><Link to="/">Back to projects</Link></main>;
  if (!project.data || !workflow.data || !activeStep) return <Loading />;
  const state = workflow.data;
  const unlocked = (index:number) => (workflowSteps[index]==="results"&&Boolean(state.steps.results.materials.has_saved_inventory)) || index === 0 || workflowSteps.slice(0,index).every(step => { const s=stageStatus(state,step); return s.complete && !s.running && !s.stale; });
  const navigate = (step:WorkflowStepName) => { setActiveStep(step); setHistory(false); window.scrollTo?.({top:0,behavior:"smooth"}); };
  const goNext = () => { const index=workflowSteps.indexOf(activeStep); if(index<workflowSteps.length-1)navigate(workflowSteps[index+1]); };
  return <main className="flow-workspace">
    <div className="flow-project"><Link to="/" className="flow-back"><ArrowLeft size={15}/> Projects</Link><button className="flow-project-name" onClick={() => { const name=window.prompt("Rename project",project.data.name); if(name?.trim())rename.mutate(name.trim()); }}>{project.data.name}</button>
      <div className="flow-history"><button className="flow-history-toggle" aria-expanded={history} aria-controls="workflow-history" onClick={()=>setHistory(!history)}>{stepLabels[activeStep]} <span>{workflowSteps.indexOf(activeStep)+1} / {workflowSteps.length}</span></button>{history&&<nav id="workflow-history" aria-label="Research steps">{workflowSteps.map((step,index)=><button key={step} disabled={!unlocked(index)} aria-current={step===activeStep?"step":undefined} onClick={()=>navigate(step)}><span>{stageStatus(state,step).complete?<CheckCircle2 size={14}/>:index+1}</span>{stepLabels[step]}</button>)}</nav>}</div>
    </div>
    <div className="focus-card" aria-label={stepLabels[activeStep]}>
      <ReadinessNotice report={readiness.data} error={readiness.error} checking={readiness.isPending} check={()=>readiness.refetch()} onFiles={()=>navigate("papers")}/>
      <ReadyContext.Provider value={readiness.data?.ready===true}>
        {activeStep==="define"&&<Define projectId={projectId} status={state.steps.define} onNext={goNext} setup={project.data.setup}/>}
        {activeStep==="papers"&&(state.steps.collect.complete&&!state.steps.collect.running?<><Candidates projectId={projectId} status={state.steps.candidates} onNext={goNext}/><details className="details flow-options"><summary>Change the paper search</summary><Collect projectId={projectId} status={state.steps.collect}/></details></>:<Collect projectId={projectId} status={state.steps.collect}/>)}
        {activeStep==="extract"&&<Extract projectId={projectId} status={state.steps.extract} onNext={goNext}/>}
        {activeStep==="refine"&&<ProcessingStage projectId={projectId} stage="filter" automatic status={stageStatus(state,"refine")} onNext={goNext}/>}
        {activeStep==="merge"&&<ProcessingStage projectId={projectId} stage="merge" status={state.steps.merge} onNext={goNext}/>}
        {activeStep==="results"&&<Results projectId={projectId} status={{...state.steps.results,stale:state.steps.merge.stale}} scoreStatus={state.steps.score}/>}
      </ReadyContext.Provider>
    </div>
    <p className="flow-footnote">Progress is saved automatically. You can leave and return to this project.</p>
  </main>;
}

export function StepHeader({ eyebrow, title, description, status, action, onNext, nextLabel, inspection = false, rerun = true, ready = status.complete && !status.running }: { eyebrow: string; title: string; description: string; status: WorkflowStep; action: any; onNext?: () => void; nextLabel?: string; ready?: boolean; inspection?: boolean; rerun?: boolean }) {
  ready = ready && !status.stale && !status.running;
  const state = inspection ? "Saved inventory" : status.running ? "Running" : status.stale ? "Update needed" : ready ? "Complete" : status.complete ? "Unsaved changes" : "Not run";
  return <header className="stage-head">
    <div className="flex flex-wrap items-center justify-between gap-3"><p className="eyebrow">{eyebrow}</p><span className={clsx("text-sm font-medium", status.running ? "text-blue-700" : ready ? "text-emerald-700" : "text-slate-500")}>{state}</span></div>
    <div className="max-w-3xl"><h2 tabIndex={-1}>{title}</h2><p className="mt-2 text-sm leading-6 text-slate-600">{description}</p></div>
    {status.running && <p role="status" className="text-sm text-blue-700">This step is running. Follow its progress below; you can continue when it finishes.</p>}
    {!inspection && !status.running && status.stale && <p role="status" className="border-l-2 border-amber-600 pl-3 text-sm text-amber-900">Inputs or settings changed. These results need to be recomputed before continuing.</p>}
    {!inspection && !status.running && !status.stale && !ready && status.missing.length > 0 && <p className="text-sm text-slate-500">Needed before the next step: {status.missing.map(item => item === "successful upstream stage and current artifacts" ? "a successful run with the current inputs" : item).join(", ")}.</p>}
    {!inspection && status.complete && !ready && !status.running && !status.stale && <p className="text-sm text-amber-800">Save your changes and run this step again before continuing.</p>}
    {!status.running && <div className="stage-actions" data-ready={ready}>{ready && onNext ? <button className="button-next" onClick={onNext}>{nextLabel || "Continue"} <ArrowRight size={16}/></button> : action}</div>}{rerun && ready && onNext && <details className="flow-rerun"><summary>Run this step again</summary><div className="stage-actions">{action}</div></details>}
  </header>;
}

function Define({ projectId, status, onNext, setup }: { projectId: string; status: WorkflowStep; onNext: () => void; setup?:Project["setup"] }) {
  const client = useQueryClient();
  const [request, setRequest] = useState(setup?.request || "");
  const setupAttempted = useRef(false);
  const inputsReady=useContext(ReadyContext);
  const credentialStatus = useQuery({queryKey:["credentials"],queryFn:()=>api<Record<string,{available:boolean}>>("/api/settings/credential-status")});
  const automaticSetup = useMutation({mutationFn:()=>api<{status:string;message?:string}>(`/api/projects/${projectId}/setup`,json("POST",{})),onSuccess:()=>{for(const key of ["project","workflow","docs","taxonomy","relevance"])client.invalidateQueries({queryKey:[key,projectId]})}});
  const connected = !!credentialStatus.data?.GEMINI_API_KEY?.available || !!credentialStatus.data?.OPENAI_API_KEY?.available || !!credentialStatus.data?.GOOGLE_CLOUD_ADC?.available;
  useEffect(()=>{if(setup && !status.complete && ["pending","needs_credentials"].includes(setup.status) && connected && inputsReady && !setupAttempted.current){setupAttempted.current=true;automaticSetup.mutate()}},[setup?.status,connected,inputsReady]);
  const [dirty, setDirty] = useState(false);
  const [generator, setGenerator] = useState({ provider: "gemini", model: "gemini-2.5-flash", reasoning_effort: "low", base_url: "", auth_mode: "auto", vertex_project: "", vertex_location: "" });
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
  const preparing = automaticSetup.isPending || setup?.status === 'running';
  const setupState = automaticSetup.data?.status || setup?.status;
  if (setup && !status.complete) return <section className="stage"><div className="setup-card" role="status">{preparing?<LoaderCircle size={24} className="animate-spin"/>:<FileSearch size={24}/>}<h2>{preparing?'Preparing your research workspace':setupState==='needs_credentials'||!connected?'Your workspace is ready':'Let’s prepare your literature search'}</h2><p>{preparing?'We’re turning your question into a search brief and organizing the feature families automatically.':automaticSetup.error?'Your question is saved. Check your model connection and try again.':!connected?'Add a Gemini or OpenAI key under Credentials, or configure Google Cloud, to prepare your search automatically.':setup?.message||'We’ll organize the search around your question.'}</p><blockquote>{setup.request}</blockquote>{!preparing&&connected&&inputsReady&&<button className="button-primary" onClick={()=>automaticSetup.mutate()}>{automaticSetup.isError?'Try setup again':'Prepare search'}</button>}</div><Error mutation={automaticSetup}/></section>;
  return <section className="stage">
    <StepHeader rerun={false} eyebrow="YOUR SEARCH" title="Does this match your question?" description="Review what we’ll search for and what to take from each paper." status={status} ready={status.complete&&!dirty} onNext={onNext} nextLabel="Find papers" action={<button className="button-primary" disabled={!dirty||save.isPending} onClick={()=>save.mutate()}>Save changes</button>}/>
    {setup?.request&&<blockquote className="flow-question">{setup.request}</blockquote>}
    <div className="plan-summary"><div><h3>Search for</h3><p>{planSummary(draft)}</p></div><div><h3>Extract from each paper</h3><p>{planSummary(goal)}</p></div></div>
    <details className="details"><summary>Read the full search and extraction plans</summary><div className="plan-previews"><article className="prose"><h3>Search brief</h3><ReactMarkdown>{draft}</ReactMarkdown></article><article className="prose"><h3>Extraction goal</h3><ReactMarkdown>{goal}</ReactMarkdown></article></div></details>
    <details><summary className="label">Edit search, extraction and category settings</summary>
    <StepHeader eyebrow="Search settings" title="Define what you want to find." description="Describe your question. Generate a search brief and feature families, then review them before collecting papers." status={{ ...status, running: status.running || generate.isPending }} ready={status.complete && !dirty && !generate.isPending} onNext={onNext} nextLabel="Collect" action={<><button className="button-secondary" disabled={!dirty || save.isPending} onClick={() => save.mutate()}><Save size={16} /> Save changes</button><ReadinessButton className="button-primary" disabled={!request.trim() || generate.isPending} onClick={() => generate.mutate()}>{generate.isPending ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} Generate project definition</ReadinessButton></>} />
    <div className="panel">
      <label className="label" htmlFor="research-request">What are you researching?</label>
      <textarea id="research-request" className="input min-h-28" placeholder="Describe the concepts, evidence, distinctions, and intended use in ordinary language." value={request} onChange={(event) => setRequest(event.target.value)} />
      <p className="mt-3 text-xs text-slate-500">Generate the definition, then review the search brief, feature families, and paper ranking rules below.</p>
      <details className="details"><summary>Generation model </summary><div className="form-grid mt-4">
        <Field label="Provider"><select className="input" value={generator.provider} onChange={event => setGenerator({ ...generator, provider: event.target.value, auth_mode: "auto", vertex_project: "", vertex_location: "", model: event.target.value === "gemini" ? "gemini-2.5-flash" : "gpt-4o-mini" })}><option value="gemini">Gemini</option><option value="openai">OpenAI-compatible</option></select></Field>
        {generator.provider === "gemini" && <GoogleAuthFields value={generator} onChange={next => setGenerator({ ...generator, ...next })} />}
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
    </details>
  </section>;
}

function MarkdownEditor({ title, value, onChange }: { title: string; value: string; onChange: (value: string) => void }) {
  const [preview, setPreview] = useState(false);
  return <div className="panel overflow-hidden p-0"><div className="flex items-center justify-between border-b border-slate-200 px-5 py-4"><h3 className="font-semibold">{title}</h3><button className="button-quiet" onClick={() => setPreview(!preview)}>{preview ? "Edit" : "Preview"}</button></div>
    {preview ? <article className="prose max-w-none p-6"><ReactMarkdown>{value}</ReactMarkdown></article> : <CodeMirror value={value} height="520px" extensions={[markdown()]} onChange={onChange} />}
  </div>;
}

function Collect({ projectId, status, onNext }: { projectId: string; status: WorkflowStep; onNext?: () => void }) {
  const client = useQueryClient();
  const config = useQuery({ queryKey: ["collectionConfig", projectId], queryFn: () => api<Record<string, any>>(`/api/projects/${projectId}/collection/config`) });
  const [value, setValue] = useState<Record<string, any>>({});
  const [dirty, setDirty] = useState(false);
  useEffect(() => { if (config.data) setValue(config.data); }, [config.data]);
  const save = useMutation({ mutationFn: () => api(`/api/projects/${projectId}/collection/config`, json("PUT", value)), onSuccess: () => { setDirty(false); client.invalidateQueries({ queryKey: ["workflow", projectId] }); } });
  const run = useMutation({ mutationFn: async () => { await save.mutateAsync(); return api<Job>(`/api/projects/${projectId}/collection/search`, { method: "POST" }); }, onSuccess: () => client.invalidateQueries({ queryKey: ["workflow", projectId] }) });
  const set = (key: string, next: any) => { setDirty(true); setValue({ ...value, [key]: next }); };
  const cached = Boolean(status.materials.uses_cached_candidates);
  return <section className="stage"><StepHeader eyebrow="PAPERS" title={cached ? "Update your paper search" : "Let’s find relevant papers"} description={cached ? "Re-rank your saved papers using the current rules. Enable Refresh metadata candidates in advanced settings to search the services again." : "Search the selected services and rank papers against your question. You will choose which PDFs to download next."} status={status} ready={status.complete && !dirty && !status.running} onNext={onNext} nextLabel="Select PDFs" action={<><ReadinessButton className="button-primary" disabled={run.isPending || status.running} onClick={() => run.mutate()}>{run.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} {status.action_label}</ReadinessButton></>} />
    <JobPanel projectId={projectId} kind="collection_search" /><Error mutation={run}/>
    <LivePapers projectId={projectId} />
    <details className="details"><summary>Search settings</summary><div className="mt-4"><div className="panel"><h3 className="section-title">Search scope</h3><div className="form-grid">
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
    </div></details><Error mutation={run} /></div></div></details>
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
  const [metadata, setMetadata] = useState(false);
  const pageSize = 100;
  const candidates = useQuery({ queryKey: ["candidates", projectId, query, page], queryFn: () => api<{ items: Candidate[]; total: number }>(`/api/projects/${projectId}/collection/candidates?page=${page}&page_size=${pageSize}&query=${encodeURIComponent(query)}`), refetchInterval: 1500 });
  const savedSelection = useQuery({ enabled:Boolean(candidates.data), queryKey: ["candidateSelection", projectId], queryFn: () => api<{ candidate_ids: string[] }>(`/api/projects/${projectId}/collection/selection`) });
  useEffect(() => { if (savedSelection.data&&!selectionDirty) setSelected(new Set(savedSelection.data.candidate_ids)); }, [savedSelection.data]);
  const save = useMutation({ mutationFn: () => api(`/api/projects/${projectId}/collection/selection`, json("PUT", { candidate_ids: [...selected] })), onSuccess: () => { setSelectionDirty(false); client.invalidateQueries({queryKey:["candidateSelection",projectId]}); client.invalidateQueries({queryKey:["candidates",projectId]}); client.invalidateQueries({ queryKey: ["workflow", projectId] }); } });
  const download = useMutation({ mutationFn: async () => { await save.mutateAsync(); return api<Job>(`/api/projects/${projectId}/collection/download`, { method: "POST" }); }, onSuccess: () => client.invalidateQueries({ queryKey: ["workflow", projectId] }) });
  const rows = useMemo(() => candidates.data?.items || [], [candidates.data]);
  const helper = createColumnHelper<Candidate>();
  const columns = useMemo(() => [
    helper.display({ id: "select", header: () => <input aria-label="Select all papers on this page" type="checkbox" checked={rows.length > 0 && rows.every(row => selected.has(row.candidate_id))} onChange={(event) => { setSelectionDirty(true); setSelected(current=>{const next=new Set(current);for(const row of rows)event.target.checked?next.add(row.candidate_id):next.delete(row.candidate_id);return next}); }} />, cell: info => <input aria-label={`Select ${info.row.original.title}`} type="checkbox" checked={selected.has(info.row.original.candidate_id)} onChange={(event) => { const next = new Set(selected); event.target.checked ? next.add(info.row.original.candidate_id) : next.delete(info.row.original.candidate_id); setSelectionDirty(true); setSelected(next); }} /> }),
    helper.accessor("title", { header: "Paper", cell: info => <div><div className="font-semibold text-slate-900">{info.getValue()}</div><div className="mt-1 text-xs text-slate-500">{info.row.original.authors?.slice(0, 3).join(", ")}</div></div> }),
    helper.accessor("year", { header: "Year" }), helper.accessor("relevance_score", { header: "Relevance", cell: info => <ScoreBreakdown candidate={info.row.original} /> }),
    helper.accessor("citation_count", { header: "Citations" }), helper.accessor("source_apis", { header: "Sources", cell: info => <span className="text-xs">{info.getValue()?.join(", ")}</span> }),
    helper.accessor("download_status", { header: "Status", cell: info => <span className="badge">{info.getValue() || "not downloaded"}</span> }),
  ], [rows, selected]);
  const table = useReactTable({ data: rows, columns, state:{columnVisibility:metadata?{}:{relevance_score:false,citation_count:false,source_apis:false}}, getCoreRowModel: getCoreRowModel() });
  return <section className="stage"><StepHeader eyebrow="PAPERS" title="Choose which papers to use." description="Select papers from the ranked list and download their full PDFs. At least one selected PDF is needed for extraction." status={status} ready={status.complete && !selectionDirty && !status.running} onNext={onNext} nextLabel="Read these papers" action={<><ReadinessButton className="button-primary" disabled={selected.size === 0 || download.isPending || status.running} onClick={() => download.mutate()}>{download.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <Download size={16} />} Download selected PDFs</ReadinessButton></>} />
    <p className="flow-muted">{selected.size} selected · {Number(status.materials.selected_downloaded_count||0)} PDFs available</p><details className="details"><summary>Ranking details</summary><label className="flex items-center gap-2 mt-3"><input type="checkbox" checked={metadata} onChange={event=>setMetadata(event.target.checked)}/> Show relevance, citations and sources</label></details>
    <div className="space-y-4"><div className="panel overflow-hidden p-0"><div className="border-b border-slate-200 p-4"><input className="input" placeholder="Filter by title, author, DOI, keyword..." value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} /></div><DataTable table={table} /><div className="flex items-center justify-between border-t border-slate-200 p-4 text-xs text-slate-500"><span>{candidates.data?.total || 0} candidates · page {page}</span><div className="flex gap-2"><button className="button-secondary" disabled={page === 1} onClick={() => setPage(page - 1)}>Previous</button><button className="button-secondary" disabled={page * pageSize >= (candidates.data?.total || 0)} onClick={() => setPage(page + 1)}>Next</button></div></div></div><JobPanel projectId={projectId} kind="collection_download" /></div><Error mutation={download} />
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
  return <section className="stage"><StepHeader eyebrow="EXTRACT" title="Read the selected papers" description="Read your selected PDFs and save each feature with its supporting evidence. You will review relevance in the next step." status={{ ...status, running: status.running || run.isPending }} ready={status.complete && !status.running && !dirty && !run.isPending} onNext={onNext} nextLabel="Refine these concepts" action={<><ReadinessButton className="button-primary" disabled={run.isPending || status.running || Number(status.materials.selected_downloaded_count || 0) === 0} onClick={() => run.mutate()}>{run.isPending || status.running ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} Extract raw features</ReadinessButton></>} />
    <JobPanel projectId={projectId} kind="direct_pdf_extraction" /><Error mutation={run}/><Error mutation={preview}/>
    <LiveFeatures projectId={projectId} running={status.running} />
    <details className="details"><summary>Extraction settings and prompt</summary><div className="mt-4"><button className="button-secondary" disabled={preview.isPending || status.running || Number(status.materials.selected_downloaded_count || 0) === 0} onClick={() => preview.mutate()}>{preview.isPending ? <LoaderCircle className="animate-spin" size={16} /> : <Eye size={16} />} View prompt</button><div className="panel"><div className="border-l-2 border-slate-300 pl-4 text-sm text-slate-700"><strong>Input: selected full PDFs.</strong> Each extracted feature keeps a reference to its source paper.<div className="mt-2 text-xs font-semibold">{candidates.data?.items.filter(row => row.selected).length || 0} selected · {candidates.data?.items.filter(row => ["downloaded", "already_exists"].includes(row.download_status)).length || 0} downloaded</div></div><div className="form-grid mt-6">
      <Field label="Provider"><select className="input" value={value.provider || "gemini"} onChange={(e) => { setDirty(true); setValue({ ...value, provider: e.target.value, auth_mode: "auto", vertex_project: null, vertex_location: null }); }}><option value="gemini">Gemini</option><option value="openai">OpenAI-compatible</option></select></Field>
      {(value.provider || "gemini") === "gemini" && <GoogleAuthFields value={value} onChange={next => { setDirty(true); setValue({ ...value, ...next }); }} />}
      <Field label="Model"><input className="input" value={value.model || ""} onChange={(e) => set("model", e.target.value)} /></Field>
      <Field label="Maximum features per PDF"><input className="input" type="number" value={value.max_features_per_pdf || 40} onChange={(e) => set("max_features_per_pdf", Number(e.target.value))} /></Field>
    </div><details className="details"><summary>Advanced model settings</summary><div className="form-grid">      <Field label="Reasoning effort"><select className="input" value={value.reasoning_effort || "low"} onChange={(e) => set("reasoning_effort", e.target.value)}>{["none", "minimal", "low", "medium", "high"].map(x => <option key={x}>{x}</option>)}</select></Field>
      <Field label="Temperature"><input className="input" type="number" step=".1" value={value.temperature ?? .1} onChange={(e) => set("temperature", Number(e.target.value))} /></Field>
      <Field label="Timeout seconds"><input className="input" type="number" value={value.timeout || 600} onChange={(e) => set("timeout", Number(e.target.value))} /></Field>
</div><Field label="Base URL"><input className="input mt-3" placeholder="Use provider default" value={value.base_url || ""} onChange={(e) => set("base_url", e.target.value)} /></Field></details><Error mutation={preview} /><Error mutation={run} /></div></div></details>
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

export function ProcessingStage({ projectId, stage, status, onNext, automatic = false, settingsOnly = false, onDirtyChange }: { projectId: string; stage: "filter" | "annotate" | "merge" | "score"; status: WorkflowStep; onNext?: () => void; automatic?: boolean; settingsOnly?: boolean; onDirtyChange?: (dirty:boolean)=>void }) {
  const client = useQueryClient();
  const [dirty, setDirty] = useState(false);
  const [annotationDirty,setAnnotationDirty]=useState(false);
  useEffect(()=>{onDirtyChange?.(dirty)},[dirty,onDirtyChange]);
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
  const inspect = useMutation({ mutationFn: async () => { await save(); return api(`/api/projects/${projectId}/stages/${stage}/preview`, { method: "POST" }); }, onSuccess: data => { setPreview(data); setDirty(false); client.invalidateQueries({ queryKey: ["workflow", projectId] }); client.invalidateQueries({queryKey:["readiness",projectId]}); } });
  const saveSettings = useMutation({ mutationFn: save, onSuccess: () => { setDirty(false); client.invalidateQueries({queryKey:["workflow",projectId]}); client.invalidateQueries({queryKey:["readiness",projectId]}); } });
  const run = useMutation({ mutationFn: async () => { await save(); return api(automatic ? `/api/projects/${projectId}/refine/run` : `/api/projects/${projectId}/stages/${stage}/run`, { method: "POST" }); }, onSuccess: () => { setDirty(false); setPreview(null); client.invalidateQueries({ queryKey: ["workflow", projectId] }); } });
  const rows = useQuery({ queryKey: ["stageRows", projectId, stage, artifact], queryFn: () => api<{ items: Record<string, any>[]; total: number }>(`/api/projects/${projectId}/stages/${stage}/results/${artifact}`), refetchInterval: 1500 });
  const descriptionsResult=useQuery({queryKey:["automaticDescriptions",projectId],enabled:automatic&&status.complete,queryFn:()=>api<{items:Record<string,any>[];total:number}>(`/api/projects/${projectId}/stages/annotate/results/annotated_features.jsonl`),refetchInterval:1500});
  const descriptions = {
    filter: "Keep or reject candidates against your goal. Inspect the retained inventory and rejection reasons before continuing.",
    annotate: "Write definitions, aliases, and examples. Inspect exclusions and features that need review.",
    merge: "Review similar features and combine equivalent terms. Keep the original evidence for every merged feature.",
    score: "Optionally score the merged inventory for literature support, annotation reliability, and distinctiveness.",
  };
  return <section className="stage">
    {!settingsOnly&&<StepHeader eyebrow={stage === "score" ? "Optional scoring" : { filter: "REFINE", annotate: "AUTOMATIC DESCRIPTIONS", merge: "MERGE" }[stage]} title={automatic ? "Refine the concepts" : {filter:"Check relevance",annotate:"Add definitions and examples",merge:"Combine equivalent concepts",score:"Score the final inventory"}[stage]} description={automatic ? "Keep concepts that fit your question, then add definitions, aliases and examples automatically." : descriptions[stage]} status={{ ...status, running: status.running || run.isPending }} ready={status.complete && !status.running && !dirty && !annotationDirty && !run.isPending} onNext={onNext} nextLabel={{filter:"Review equivalent concepts",annotate:"Review equivalent concepts",merge:"View results",score:""}[stage]} action={<><ReadinessButton className="button-primary" disabled={status.running || run.isPending || inspect.isPending || annotationDirty} onClick={() => run.mutate()}>{status.running ? <LoaderCircle className="animate-spin" size={16} /> : <FileSearch size={16} />} {automatic?"Refine and add descriptions":stage==="merge"?"Compare concepts":"Run "+stage}</ReadinessButton></>} />}
    {!settingsOnly&&<><JobPanel projectId={projectId} kind={automatic?"refine":stage} message={automatic?(status.materials.refinement_phase==="annotate"?"Adding definitions and examples":"Checking concepts against your question"):undefined}/><Error mutation={run}/></>}
    {annotationDirty&&<p className="text-sm text-amber-800" role="status">Save the description settings below before continuing.</p>}
    {!settingsOnly&&<ConceptPreview items={automatic&&status.complete?(descriptionsResult.data?.items??rows.data?.items??[]):rows.data?.items??[]} count={automatic&&status.complete?(descriptionsResult.data?.total??rows.data?.total??0):rows.data?.total??0}/>}
    <details className="details" open={settingsOnly}><summary>{settingsOnly?"Description settings":"Settings and detailed records"}</summary>
    <div className="grid gap-5"><div className="panel"><h3 className="section-title">Settings</h3><div className="form-grid">
      <Field label="Provider"><select className="input" value={value.provider || "gemini"} onChange={e => { setDirty(true); setValue({ ...value, provider: e.target.value, base_url: null, auth_mode: "auto", vertex_project: null, vertex_location: null }); }}><option value="gemini">Gemini</option><option value="openai">OpenAI compatible</option></select></Field>
      {(value.provider || "gemini") === "gemini" && <GoogleAuthFields value={value} onChange={next => { setDirty(true); setValue({ ...value, ...next }); }} />}
      <Field label="Model"><input className="input" value={value.model || ""} onChange={e => { setDirty(true); setValue({ ...value, model: e.target.value }); }} /></Field>
      <Field label="Domain rules"><select className="input" value={value.profile || ""} onChange={e => { setDirty(true); setValue({ ...value, profile: e.target.value || null }); }}><option value="">Use project taxonomy</option><option value="media_attributes">Media attributes</option></select></Field>
      <Field label={stage === "merge" ? "Neighbor candidates per feature" : "Features per batch"}><input className="input" type="number" min="1" value={value[stage === "merge" ? "top_k" : "batch_features"] || (stage === "merge" ? 10 : 25)} onChange={e => { setDirty(true); setValue({ ...value, [stage === "merge" ? "top_k" : "batch_features"]: Number(e.target.value) }); }} /></Field>
      {stage === "merge" && <GoogleAuthFields value={value} prefix="embedding_" onChange={next => { setDirty(true); setValue({ ...value, ...next }); }} />}
      {stage === "merge" && <Field label="Embedding model"><input className="input" value={value.embedding_model || "gemini-embedding-001"} onChange={e => { setDirty(true); setValue({ ...value, embedding_model: e.target.value }); }} /></Field>}
    </div><Error mutation={inspect} /><Error mutation={run} />
    {preview && <details className="details" open><summary>Planned inputs and outputs</summary><pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap text-xs">{JSON.stringify(preview, null, 2)}</pre></details>}
    </div></div>
    <div className="stage-settings-actions"><button className="button-secondary" disabled={!dirty||saveSettings.isPending||status.running} onClick={()=>saveSettings.mutate()}>Save settings</button><button className="button-secondary" disabled={status.running||inspect.isPending} onClick={()=>inspect.mutate()}>Preview settings</button></div><Error mutation={saveSettings}/>
    {automatic&&<ProcessingStage projectId={projectId} stage="annotate" status={status} settingsOnly onDirtyChange={setAnnotationDirty}/>}
    {!settingsOnly&&<div className="panel"><h3 className="section-title mb-4">Inspect results</h3><div className="flex flex-wrap items-center gap-3"><select aria-label="Stage artifact" className="input max-w-md" value={artifact} onChange={e => setArtifact(e.target.value)}>{artifacts[stage].map(name => <option key={name} value={name}>{artifactLabels[name]}</option>)}</select><button className="button-secondary" onClick={() => rows.refetch()}>Refresh</button><a className="button-secondary" href={`/api/projects/${projectId}/stages/${stage}/exports/${artifact}`}><Download size={14} /> Export</a></div><p className="mt-4 text-sm text-slate-500">{rows.isLoading ? "Loading records…" : rows.data ? `${rows.data.total} records. Open a record to inspect its evidence and decisions.` : "No results yet. Run this step to create them."}</p><div className="mt-4 max-h-[600px] overflow-auto">{rows.data?.items.map((row, index) => <details className="details mb-2" key={index}><summary>{String(row.feature_name || row.feature_key || row.pair_id || `Decision ${index + 1}`)}</summary>{row.source_paper_ids?.length > 0 && <p className="mt-3 text-sm text-slate-600">Source papers: {row.source_paper_ids.join(", ")}</p>}{(row.source_quote || row.definition) && <p className="mt-2 text-sm leading-6 text-slate-700">{row.source_quote || row.definition}</p>}<pre className="mt-3 overflow-auto whitespace-pre-wrap text-xs text-slate-600">{JSON.stringify(row, null, 2)}</pre></details>)}</div></div>}
    </details>
  </section>;
}


type Feature = { category: string; feature_name: string; definitions: string[]; synonyms: string[]; examples: string[]; paper_count: number; mention_count: number; confidence: number; source_titles: string[]; evidence_types: string[]; annotation?: {formal_definition?:string;short_definition?:string} };
export function Results({ projectId, status, scoreStatus }: { projectId: string; status: WorkflowStep; scoreStatus: WorkflowStep }) {
  const [query, setQuery] = useState(""); const [category, setCategory] = useState(""); const [detail, setDetail] = useState<Feature | null>(null);
  const [inventory, setInventory] = useState("merge");
  const [moreColumns,setMoreColumns]=useState(false);
  const inventories: Record<string, string> = { extract: "features.jsonl", filter: "filtered_features.jsonl", annotate: "annotated_features.jsonl", merge: "llm_merged_features.jsonl", score: "scored_features.jsonl" };
  const features = useQuery({ queryKey: ["features", projectId, inventory, query, category], queryFn: () => api<{ items: Feature[]; total: number }>(`/api/projects/${projectId}/stages/${inventory}/results/${inventories[inventory]}?page_size=500&query=${encodeURIComponent(query)}&category=${encodeURIComponent(category)}`), refetchInterval: 1500 });
  const errors = useQuery({ queryKey: ["errors", projectId], queryFn: () => api<{ items: Record<string, any>[]; total: number }>(`/api/projects/${projectId}/results/errors?page_size=50`) });
  const mentions = useQuery({ queryKey: ["detailMentions", projectId, detail?.feature_name, detail?.category], enabled: Boolean(detail), queryFn: () => api<{ items: Record<string, any>[] }>(`/api/projects/${projectId}/results/mentions?page_size=100&query=${encodeURIComponent(detail?.feature_name || "")}&category=${encodeURIComponent(detail?.category || "")}`) });
  const rows = useMemo(() => features.data?.items || [], [features.data]); const categories = [...new Set(rows.map(row => row.category))];
  const helper = createColumnHelper<Feature>();
  const columns = useMemo(() => [
    helper.accessor("feature_name", { header: "Feature", cell: info => <button className="text-left font-semibold text-blue-700 hover:underline" onClick={() => setDetail(info.row.original)}>{info.getValue()}</button> }),
    helper.accessor("category", { header: "Family", cell: info => <span className="badge">{humanCategory(info.getValue())}</span> }),
    helper.accessor("paper_count", { header: "Papers" }), helper.accessor("mention_count", { header: "Mentions" }),
    helper.accessor("confidence", { header: "Confidence", cell: info => Number(info.getValue() || 0).toFixed(2) }),
    helper.accessor("evidence_types", { header: "Evidence", cell: info => <span className="text-xs">{info.getValue()?.join(", ")}</span> }),
  ], []);
  const table = useReactTable({ data: rows, columns, state:{columnVisibility:moreColumns?{}:{mention_count:false,confidence:false,evidence_types:false}}, getCoreRowModel: getCoreRowModel() });
  return <section className="stage"><StepHeader inspection eyebrow="RESULTS" title="Your concept inventory" description={`${features.data?.total||0} concepts. Open one to read its definition and source evidence.`} status={status} action={<a className="button-primary" href={`/api/projects/${projectId}/stages/${inventory}/exports/${inventories[inventory].replace(".jsonl", ".csv")}`}><Download size={16} /> Download CSV</a>} />
    {status.stale&&<p className="saved-inventory-note">This saved inventory may need updating after changes to your search. You can still inspect and export it.</p>}
    <details className="details"><summary>Other exports and optional scoring</summary><label className="flex items-center gap-2 mt-4"><input type="checkbox" checked={moreColumns} onChange={event=>setMoreColumns(event.target.checked)}/> Show detailed evidence columns</label><div className="panel"><Field label="Inventory"><select className="input" value={inventory} onChange={e => setInventory(e.target.value)}>{Object.keys(inventories).filter(stage => stage !== "score" || scoreStatus.complete).map(stage => <option key={stage}>{stage}</option>)}</select></Field><a className="button-secondary mt-3" href={`/api/projects/${projectId}/stages/${inventory}/exports/${inventories[inventory].replace(".jsonl", ".csv")}`}>Download CSV</a></div>
    <details className="details bg-white"><summary>Optional feature scoring</summary><ProcessingStage projectId={projectId} stage="score" status={scoreStatus} /></details>
    </details><div className="panel overflow-hidden p-0"><div className="grid gap-3 border-b border-slate-200 p-4 md:grid-cols-[1fr_260px]"><input className="input" placeholder="Search features..." value={query} onChange={e => setQuery(e.target.value)} /><select className="input" value={category} onChange={e => setCategory(e.target.value)}><option value="">All families</option>{categories.map(x => <option key={x} value={x}>{humanCategory(x)}</option>)}</select></div><DataTable table={table} /></div>
    {(errors.data?.items.length || 0) > 0 && <details className="details bg-white"><summary>Extraction failures ({errors.data?.total}) </summary><div className="mt-4 space-y-2">{errors.data?.items.map((error, index) => <div className="rounded-sm bg-red-50 p-3 text-xs text-red-800" key={index}><strong>{error.paper_id || "Unknown paper"}</strong><div className="mt-1">{error.error}</div></div>)}</div></details>}
    {detail && <div className="fixed inset-0 z-50 flex justify-end bg-slate-950/30" onClick={() => setDetail(null)}><aside className="h-full w-full max-w-2xl overflow-auto bg-white p-8 shadow-lg" onClick={e => e.stopPropagation()}><button className="icon-button float-right" onClick={() => setDetail(null)}><X size={18} /></button><span className="badge">{humanCategory(detail.category)}</span><h2 className="mt-4 text-3xl font-semibold">{detail.feature_name}</h2><Detail label="Definition" values={detail.annotation?.formal_definition?[detail.annotation.formal_definition]:detail.definitions} /><Detail label="Synonyms" values={detail.synonyms} /><Detail label="Examples" values={detail.examples} /><Detail label="Source papers" values={detail.source_titles} /><Detail label="Evidence snippets" values={(mentions.data?.items || []).map(row => row.source_quote || row.snippet).filter(Boolean)} /></aside></div>}
  </section>;
}

function humanCategory(value:string) { const text=value.replaceAll("_"," "); return text.charAt(0).toUpperCase()+text.slice(1); }
function planSummary(markdownText: string) {
  const lines=markdownText.split("\n").map(line=>line.trim()).filter(line=>line&&!line.startsWith("#"));
  return (lines[0]||"Review the full plan below.").replace(/\*\*/g,"");
}
function ConceptPreview({items,count}:{items:Record<string,any>[];count:number}) {
  if(!count)return null;
  return <section className="concept-preview" aria-label="Concepts"><p className="flow-muted">{count} concepts</p><div className="concept-preview-list">{items.map((row,index)=><div key={row.feature_key||row.feature_name||index}><strong>{row.feature_name||"Concept"}</strong><span>{humanCategory(String(row.category||""))}</span>{(row.annotation?.short_definition||row.definitions?.[0])&&<p>{row.annotation?.short_definition||row.definitions[0]}</p>}</div>)}</div></section>;
}
function Detail({ label, values }: { label: string; values: string[] }) { return <div className="mt-8"><h3 className="label">{label}</h3><div className="mt-2 space-y-2">{(values || []).map((value, i) => <p className="rounded-sm bg-slate-50 p-3 text-sm text-slate-700" key={i}>{value}</p>)}</div></div>; }

function DataTable({ table }: { table: any }) { return <div className="live-table" role="region" aria-label="Items" tabIndex={0}><table className="data-table"><thead>{table.getHeaderGroups().map((group: any) => <tr key={group.id}>{group.headers.map((header: any) => <th key={header.id}>{flexRender(header.column.columnDef.header, header.getContext())}</th>)}</tr>)}</thead><tbody>{table.getRowModel().rows.map((row: any) => <tr key={row.id}>{row.getVisibleCells().map((cell: any) => <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>)}</tr>)}</tbody></table></div>; }

function JobPanel({ projectId, kind, message: phaseMessage }: { projectId: string; kind: string; message?: string }) {
  const project = useQuery({ queryKey: ["projectJobs", projectId], queryFn: () => api<Project>(`/api/projects/${projectId}`), refetchInterval: 1500 });
  const kinds=kind==="refine"?["refine","filter","annotate"]:[kind];
  const matching=project.data?.jobs?.filter(item=>kinds.includes(item.kind))||[];
  const job=matching.find(item=>["queued","running","cancelling"].includes(item.status))||matching[0];
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
  const message = job ? (job.status === "running" ? ({collection_search:"Finding and ranking papers",collection_download:"Downloading full PDFs",direct_pdf_extraction:"Reading PDFs and finding features",filter:"Checking feature relevance",refine:"Checking relevance and adding definitions",annotate:"Adding feature descriptions",merge:"Combining equivalent features",score:"Measuring feature relevance"} as Record<string,string>)[kind] || "Processing saved results" : labels[job.status] || job.status) : "Ready to begin";
  if(!job||job.status==="completed")return null;
  return <aside className="run-status compact-status" aria-label="Run status">
    <p className="flex items-center gap-2 text-sm font-medium text-slate-950" role="status">{active && <LoaderCircle size={15} className="animate-spin" />}{active&&job?.kind==="annotate"?"Adding definitions and examples":active&&phaseMessage?phaseMessage:message}</p>
    {job && <details className="job-details"><summary>Details</summary><pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-600">{log || "No additional output."}</pre></details>}
    {active && <button className="button-secondary" disabled={job?.status === "cancelling"} onClick={() => api(`/api/jobs/${job!.id}/cancel`, { method: "POST" })}><Square size={12} />{job?.status === "cancelling" ? "Stopping…" : "Stop"}</button>}
    {job && ["cancelled","interrupted","failed"].includes(job.status) && <p className="w-full text-xs text-slate-600">Saved progress is available below. Run this step again to finish.</p>}
    {job?.error && <p role="alert" className="w-full border-l-2 border-red-600 pl-3 text-sm text-red-800">{job.error}</p>}
  </aside>;
}

function GoogleAuthFields({ value, onChange, prefix = "" }: { value: Record<string, any>; onChange: (next: Record<string, any>) => void; prefix?: string }) {
  const mode = value[prefix + "auth_mode"] || (prefix ? "" : "auto");
  const cloud = mode === "adc" || mode === "auto";
  return <>
    <Field label={prefix ? "Embedding authentication" : "Gemini authentication"} description="Automatic uses API keys when available, or Google Cloud when configured. ADC uses your Google Cloud login.">
      <select className="input" value={mode} onChange={e => onChange({ [prefix + "auth_mode"]: e.target.value || null })}>
        {prefix && <option value="">Use generation settings</option>}
        <option value="auto">Automatic</option><option value="api_key">API key</option><option value="adc">Google Cloud (ADC)</option>
      </select>
    </Field>
    {cloud && <>
      <Field label={prefix ? "Embedding Google Cloud project" : "Google Cloud project"} description="Defaults to GOOGLE_CLOUD_PROJECT. Entering a project selects Google Cloud in automatic mode.">
        <input className="input" placeholder="my-project-id" value={value[prefix + "vertex_project"] || ""} onChange={e => onChange({ [prefix + "vertex_project"]: e.target.value })} />
      </Field>
      <Field label={prefix ? "Embedding region" : "Google Cloud region"} description="Defaults to GOOGLE_CLOUD_LOCATION or us-central1. Choose a region that supports your model.">
        <input className="input" placeholder="us-central1" value={value[prefix + "vertex_location"] || ""} onChange={e => onChange({ [prefix + "vertex_location"]: e.target.value })} />
      </Field>
    </>}
  </>;
}

function Field({ label, description, children }: { label: string; description?: string; children: any }) { return <label className="block"><span className="label">{label}</span><div className="mt-2">{children}</div>{description && <span className="mt-2 block text-xs leading-5 text-slate-500">{description}</span>}</label>; }
function Error({ mutation }: { mutation: any }) { return mutation.error ? <p className="mt-4 rounded-sm bg-red-50 p-3 text-sm text-red-700">{mutation.error.message}</p> : null; }
function Loading() { return <div className="grid min-h-[50vh] place-items-center"><LoaderCircle className="animate-spin text-blue-600" /></div>; }


type PaperPreview = {items:Candidate[];total:number;partial:boolean;running:boolean;downloaded:number;progress?:{completed:number;total:number}};
export function LivePapers({projectId}:{projectId:string}) {
  const [query,setQuery]=useState(""); const [page,setPage]=useState(1);
  const result=useQuery({queryKey:["livePapers",projectId,query,page],placeholderData:previous=>previous,queryFn:()=>api<PaperPreview>(`/api/projects/${projectId}/collection/live?page=${page}&page_size=50&query=${encodeURIComponent(query)}`),refetchInterval:1500});
  const data=result.data; const rows=useMemo(()=>data?.items||[],[data]);
  const columns=useMemo(()=>[
    {accessorKey:"title",header:"Paper"}, {accessorKey:"year",header:"Year"},
    {accessorKey:"citation_count",header:"Citations"},
    {accessorKey:"relevance_score",header:"Relevance",cell:({getValue}:any)=>Number(getValue()||0).toFixed(3)},
    {accessorKey:"download_status",header:"PDF",cell:({getValue}:any)=>getValue()==="pending"?"Waiting":String(getValue()).replaceAll("_"," ")},
  ],[]);
  const table=useReactTable({data:rows,columns,getCoreRowModel:getCoreRowModel(),getRowId:row=>row.candidate_id});
  if(!query&&!data?.running&&!data?.total&&!result.error)return null;
  return <div className="panel overflow-hidden p-0" aria-label="Live paper results"><div className="p-4 border-b border-slate-200"><h3 className="font-semibold">Papers found so far <span className="text-slate-500">{data?.total||0}</span></h3><p className="mt-1 text-xs text-slate-500" role="status">{data?.running?`Updates automatically as searches finish. ${data.progress?.completed||0} of ${data.progress?.total||0} searches complete.`:data?.partial?"Saved partial search. Run collection again to finish.":"Search complete. Continue to choose PDFs."}</p><input className="input mt-3" aria-label="Search papers found so far" placeholder="Search papers" value={query} onChange={e=>{setQuery(e.target.value);setPage(1)}}/></div>{result.error&&<p className="p-4 text-red-800" role="alert">{result.error.message}</p>}<DataTable table={table}/><PreviewPages page={page} total={data?.total||0} setPage={setPage}/></div>;
}

export function LiveFeatures({projectId,running}:{projectId:string;running:boolean}) {
  const [query,setQuery]=useState(""); const [page,setPage]=useState(1);
  const result=useQuery({queryKey:["liveFeatures",projectId,query,page],placeholderData:previous=>previous,queryFn:()=>api<{items:Feature[];total:number}>(`/api/projects/${projectId}/stages/extract/results/features.jsonl?page=${page}&page_size=50&query=${encodeURIComponent(query)}`),refetchInterval:1500});
  const columns=useMemo(()=>[{accessorKey:"feature_name",header:"Feature"},{accessorKey:"category",header:"Family"},{accessorKey:"paper_count",header:"Papers"},{accessorKey:"mention_count",header:"Mentions"}],[]);
  const stableRows=useMemo(()=>result.data?.items||[],[result.data]);
  const table=useReactTable({data:stableRows,columns,getCoreRowModel:getCoreRowModel()});
  if(!query&&!running&&!result.data?.total&&!result.error)return null;
  return <div className="panel overflow-hidden p-0" aria-label="Live extracted features"><div className="p-4 border-b border-slate-200"><h3 className="font-semibold">Features found so far <span className="text-slate-500">{result.data?.total||0}</span></h3><p className="mt-1 text-xs text-slate-500" role="status">{running?"Updates automatically as each PDF or batch finishes. These features are provisional.":"Saved features from the latest extraction."}</p><input className="input mt-3" aria-label="Search features found so far" placeholder="Search features" value={query} onChange={e=>{setQuery(e.target.value);setPage(1)}}/></div>{result.error&&<p className="p-4 text-red-800" role="alert">{result.error.message}</p>}<DataTable table={table}/><PreviewPages page={page} total={result.data?.total||0} setPage={setPage}/></div>;
}
function PreviewPages({page,total,setPage}:{page:number;total:number;setPage:(page:number)=>void}) {return <div className="flex items-center justify-between border-t border-slate-200 p-4 text-xs text-slate-500"><span>{total} items · page {page}</span><div className="flex gap-2"><button className="button-secondary" disabled={page===1} onClick={()=>setPage(page-1)}>Previous</button><button className="button-secondary" disabled={page*50>=total} onClick={()=>setPage(page+1)}>Next</button></div></div>}
