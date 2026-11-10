import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Progress, StepHeader, nextWorkflowStep, ProjectsPage, ProjectPage, Results, FeatureDiscovery } from "./ui";
import type { Project, Workflow } from "./api";

vi.mock("@uiw/react-codemirror", () => ({default: ({value}: {value:string}) => <div data-testid="definition-editor">{value}</div>}));

afterEach(() => {cleanup();vi.unstubAllGlobals();vi.restoreAllMocks()});
const project: Project = { id: "p1", name: "Review", description: "", updated_at: "", progress: { defined: true, taxonomy: true, searched: true, downloaded: 2, extracted: false } };
const status = { complete: false, running: false, missing: ["ranked paper candidates"], materials: {}, action_label: "Find and rank papers" };
function workflow(completed: string[] = []): Workflow {
  return { steps: Object.fromEntries(["define", "collect", "candidates", "extract", "filter", "annotate", "merge", "results", "score"].map(step => [step, { ...status, complete: completed.includes(step) }])) } as Workflow;
}

describe("Workflow orientation", () => {
  it("describes progress using available records", () => {
    const { rerender } = render(<Progress project={project} />);
    expect(screen.getByText("PDFs ready for extraction")).toBeInTheDocument();
    rerender(<Progress project={project} workflow={workflow(["define", "collect", "candidates"])} />);
    expect(screen.getByText("2 of 3 steps complete")).toBeInTheDocument();
  });

  it("resumes at the first unfinished step, including stale upstream work", () => {
    const saved = workflow(["define", "collect", "candidates", "extract"]);
    expect(nextWorkflowStep(saved)).toBe("features");
    saved.steps.extract = { ...status, stale: true };
    expect(nextWorkflowStep(saved)).toBe("features");
    saved.steps.collect.running = true;
    expect(nextWorkflowStep(saved)).toBe("papers");
  });

  it("returns completed projects to results without requiring scoring", () => {
    expect(nextWorkflowStep(workflow(["define", "collect", "candidates", "extract", "filter", "annotate", "merge", "results"]))).toBe("features");
  });
});

describe("StepHeader", () => {
  it("names the next step only when the current results are ready", () => {
    const onNext = vi.fn();
    const base = { eyebrow: "PAPERS", title: "Choose papers", description: "Collect papers", action: <button>Download selected PDFs</button>, onNext, nextLabel: "Extract features" };
    const { rerender } = render(<StepHeader {...base} status={status} />);
    expect(screen.queryByRole("button", { name: "Extract features" })).not.toBeInTheDocument();
    expect(screen.getByText("Not run")).toBeInTheDocument();
    expect(screen.getByText(/Needed:/)).toHaveTextContent("ranked paper candidates");

    rerender(<StepHeader {...base} status={{ ...status, complete: true, running: true, missing: [] }} />);
    expect(screen.queryByRole("button", { name: "Extract features" })).not.toBeInTheDocument();
    expect(screen.getByText("Running")).toBeVisible();

    rerender(<StepHeader {...base} status={{ ...status, complete: true, missing: [] }} />);
    fireEvent.click(screen.getByRole("button", { name: "Extract features" }));
    expect(onNext).toHaveBeenCalledOnce();
  });

  it("blocks progression for unsaved settings and stale results", () => {
    const base = { eyebrow: "Filter", title: "Filter", description: "Review candidates", action: <button>Run filter</button>, onNext: vi.fn() };
    const { rerender } = render(<StepHeader {...base} status={{ ...status, complete: true }} ready={false} />);
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue" })).not.toBeInTheDocument();
    rerender(<StepHeader {...base} status={{ ...status, complete: true, stale: true }} ready />);
    expect(screen.getByText("Update needed")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Continue" })).not.toBeInTheDocument();
  });
});


import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import {MemoryRouter,Routes,Route} from 'react-router-dom';
function renderWorkspace(element:React.ReactNode,path='/') {
 const client=new QueryClient({defaultOptions:{queries:{retry:false,gcTime:0},mutations:{retry:false}}});
 const result=render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[path]}><Routes><Route path="/" element={element}/><Route path="/projects/:projectId" element={path==='/'?<div>Research workspace</div>:element}/></Routes></MemoryRouter></QueryClientProvider>);
 return {...result,client};
}

it('starts research from one question without a project-name form',async()=>{
 const fetchMock=vi.fn().mockImplementation(async(path,init)=>({ok:true,json:async()=>String(path).includes('/readiness')?{ready:true,message:'Ready',requirements:[]}:init?.method==='POST'?project:[]}));
 vi.stubGlobal('fetch',fetchMock);
 renderWorkspace(<ProjectsPage/>);
 expect(screen.getAllByRole('textbox')).toHaveLength(1);
 expect(screen.queryByLabelText('Project name')).not.toBeInTheDocument();
 fireEvent.change(screen.getByLabelText('Research question'),{target:{value:'  How do people express emotion?  '}});
 await waitFor(()=>expect(screen.getByRole('button',{name:'Start research'})).toBeEnabled());
 fireEvent.click(screen.getByRole('button',{name:'Start research'}));
 await screen.findByText('Research workspace');
 expect(fetchMock).toHaveBeenCalledWith('/api/projects/start',expect.objectContaining({method:'POST',body:JSON.stringify({request:'How do people express emotion?'})}));
});

it('prepares the research setup automatically with an available model connection',async()=>{
 let prepared=false;
 const fetchMock=vi.fn().mockImplementation(async(input,init)=>{const path=String(input);if(path.endsWith('/setup')&&init?.method==='POST')prepared=true;
 let data:unknown={};
 if(path.endsWith('/setup'))data={status:'ready'};
 else if(path.includes('/readiness'))data={ready:true,message:'Ready',requirements:[]};
 else if(path.endsWith('/credential-status'))data={GEMINI_API_KEY:{available:true}};
 else if(path.endsWith('/projects/p1'))data={...project,setup:{request:'Find emotion concepts',status:prepared?'ready':'pending'}};
 else if(path.endsWith('/workflow'))data=workflow(prepared?['define']:[]);
 else if(path.endsWith('/documents'))data={draft_markdown:prepared?'Search brief':'',goal_markdown:prepared?'Find concepts':''};
 else if(path.endsWith('/taxonomy'))data={version:1,title:'Emotion',families:[]};
 else if(path.endsWith('/relevance-scoring'))data=null;
 return {ok:true,json:async()=>data};});
 vi.stubGlobal('fetch',fetchMock);
 renderWorkspace(<ProjectPage/>,'/projects/p1');
 await waitFor(()=>expect(fetchMock).toHaveBeenCalledWith('/api/projects/p1/setup',expect.objectContaining({method:'POST'})));
 expect(await screen.findByRole('heading',{name:'Does this match your question?'})).toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Find papers'})).toBeInTheDocument();
 expect(fetchMock.mock.calls.filter(([path])=>String(path).endsWith('/setup'))).toHaveLength(1);
});


it('asks for required credentials before creating a research workspace',async()=>{
 const blocked={ready:false,message:'Complete required inputs',requirements:[{id:'GEMINI_API_KEY',field:'GEMINI_API_KEY',kind:'credential',label:'Gemini API key',message:'Add a Gemini API key for extraction and merging.'}]};
 let saved=false;
 const fetchMock=vi.fn().mockImplementation(async(path,init)=>{if(String(path).endsWith('/session-credentials'))saved=true;return {ok:true,json:async()=>String(path).includes('/readiness')?(saved?{ready:true,message:'Ready',requirements:[]}:blocked):[]}});
 vi.stubGlobal('fetch',fetchMock);renderWorkspace(<ProjectsPage/>);
 fireEvent.change(screen.getByLabelText('Research question'),{target:{value:'Emotion concepts'}});
 expect(await screen.findByLabelText('Gemini API key')).toBeInTheDocument();
 expect(screen.getByRole('button',{name:'Start research'})).toBeDisabled();
 expect(fetchMock.mock.calls.some(([path])=>String(path).endsWith('/projects/start'))).toBe(false);
 fireEvent.change(screen.getByLabelText('Gemini API key'),{target:{value:'session-only-key'}});
 fireEvent.click(screen.getByRole('button',{name:'Save and check'}));
 await waitFor(()=>expect(screen.getByRole('button',{name:'Start research'})).toBeEnabled());
 expect(screen.queryByDisplayValue('session-only-key')).not.toBeInTheDocument();
});

import {LivePapers,LiveFeatures} from './ui';
it('updates paper counts in a scrollable list while retaining the user search',async()=>{
 let count=1;
 vi.stubGlobal('fetch',vi.fn().mockImplementation(async()=>({ok:true,json:async()=>({total:count,running:true,partial:true,progress:{completed:count,total:4},items:[{candidate_id:'p1',title:'Live paper',year:2025,citation_count:count,relevance_score:1,download_status:'pending'}]})})));
 renderWorkspace(<LivePapers projectId="p1"/>);
 await screen.findByText('Live paper');
 fireEvent.change(screen.getByLabelText('Search papers found so far'),{target:{value:'Live'}});
 const panel=screen.getByRole('region',{name:'Items'});panel.scrollTop=180;
 count=2;
 await screen.findByText(/2 of 4 searches complete/,{},{timeout:4000});
 expect(screen.getByLabelText('Search papers found so far')).toHaveValue('Live');
 expect(panel.scrollTop).toBe(180);
 expect(panel).toHaveClass('live-table');
});
it('shows extracted features before the extraction job finishes',async()=>{
 let papers=1;
 vi.stubGlobal('fetch',vi.fn().mockImplementation(async()=>({ok:true,json:async()=>({total:1,items:[{feature_name:'Observed emotion',category:'Emotion',paper_count:papers,mention_count:papers+1}]})})));
 renderWorkspace(<LiveFeatures projectId="p1" running/>);
 await screen.findByText('Observed emotion');
 expect(screen.getByRole('status')).toHaveTextContent('Updates automatically');
 papers=3;
 await screen.findByText('3',{},{timeout:4000});
 expect(screen.getByRole('region',{name:'Items'})).toHaveClass('live-table');
});


const sampleFeature = {
  feature_name: "Loaded language", category: "persuasion", definitions: ["Emotionally charged words"],
  synonyms: [], examples: [], paper_count: 1, mention_count: 1, confidence: 1,
  source_titles: ["Paper Bullets"], evidence_types: ["definition"],
};
const ready = { ready: true, message: "Ready", requirements: [] };
function workspaceFetch(getWorkflow: () => Workflow, override?: (path: string, init?: RequestInit) => unknown) {
  return vi.fn().mockImplementation(async (input, init) => {
    const path = String(input);
    let data = override?.(path, init);
    if (data === undefined) {
      if (path.includes("/readiness")) data = ready;
      else if (path.endsWith("/workflow")) data = getWorkflow();
      else if (path.endsWith("/projects/p1")) data = { ...project, jobs: [] };
      else if (path.endsWith("/credential-status")) data = { GEMINI_API_KEY: { available: true } };
      else if (path.endsWith("/documents")) data = { draft_markdown: "Search brief", goal_markdown: "Extraction goal" };
      else if (path.endsWith("/taxonomy")) data = { version: 1, title: "Persuasion", families: [] };
      else if (path.endsWith("/relevance-scoring")) data = null;
      else if (path.includes("/discovery/live")) data = { items: [sampleFeature], total: 1, phase: "extract", running: false, partial: true, completed: 0, stages_total: 5, source_stage: "extract" };
      else if (path.endsWith("/config")) data = { provider: "gemini", model: "test-model" };
      else if (path.includes("/results/")) data = { items: [sampleFeature], total: 1 };
      else data = { items: [], total: 0 };
    }
    return { ok: true, json: async () => data };
  });
}

it("presents three research steps without separate processing tabs", async () => {
  const saved = workflow(["define", "collect", "candidates", "extract", "filter", "annotate", "merge", "results"]);
  vi.stubGlobal("fetch", workspaceFetch(() => saved));
  renderWorkspace(<ProjectPage />, "/projects/p1");
  await screen.findByRole("heading", { name: "Your concept inventory" });
  expect(screen.queryByRole("navigation", { name: "Research steps" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /Features 3 \/ 3/ }));
  const navigation = screen.getByRole("navigation", { name: "Research steps" });
  expect(within(navigation).getAllByRole("button").map(button => button.textContent?.trim())).toEqual([
    "Your search", "Papers", "Features",
  ]);
  expect(within(navigation).queryByRole("button", { name: /annotat/i })).not.toBeInTheDocument();
  expect(screen.queryByRole("tab", { name: /annotat/i })).not.toBeInTheDocument();
});


it("opens and exports a saved inventory even when upstream work is stale", async () => {
  const saved = workflow(["define", "collect", "candidates", "extract", "filter", "annotate", "merge"]);
  saved.steps.define.stale = true;
  saved.steps.merge.stale = true;
  saved.steps.results.materials = { has_saved_inventory: true };
  vi.stubGlobal("fetch", workspaceFetch(() => saved));
  renderWorkspace(<ProjectPage />, "/projects/p1");
  expect(nextWorkflowStep(saved)).toBe("features");
  await screen.findByRole("heading", { name: "Your concept inventory" });
  expect(await screen.findByRole("button", { name: "Loaded language" })).toBeInTheDocument();
  expect(screen.getByText(/This saved inventory may need updating/)).toBeVisible();
  expect(within(screen.getByRole("heading", { name: "Your concept inventory" }).closest("header")!).getByRole("button", { name: "Download CSV" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: /Features 3 \/ 3/ }));
  expect(within(screen.getByRole("navigation", { name: "Research steps" })).getByRole("button", { name: /Features$/ })).toBeEnabled();
});

it("keeps the results search responsive while a filtered request is pending", async () => {
  let resolveSearch: (() => void) | undefined;
  const fetchMock = workspaceFetch(() => workflow(), path => {
    if (path.includes("/stages/merge/results/")) {
      if (path.includes("query=loaded")) return new Promise(resolve => { resolveSearch = () => resolve({ items: [sampleFeature], total: 1 }); });
      return { items: [sampleFeature], total: 1 };
    }
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWorkspace(<Results projectId="p1" status={{ ...status, complete: true }} scoreStatus={status} />);
  await screen.findByRole("button", { name: "Loaded language" });
  fireEvent.change(screen.getByPlaceholderText("Search features..."), { target: { value: "loaded" } });
  await waitFor(() => expect(resolveSearch).toBeTypeOf("function"));
  expect(screen.getByPlaceholderText("Search features...")).toHaveValue("loaded");
  await act(async () => { resolveSearch!(); });
  expect(await screen.findByRole("button", { name: "Loaded language" })).toBeInTheDocument();
});

it("preserves a manually unchecked paper through polling and pagination", async () => {
  const saved = workflow(["define", "collect"]);
  let serverSelection = ["a", "b"];
  function paper(id: string, title: string) {
    return { candidate_id: id, title, authors: [], year: 2025, doi: "", source_apis: ["crossref"], relevance_score: 1, citation_count: 0, selected: true, download_status: "pending", matched_keywords: [], relevance_score_breakdown: {} };
  }
  const fetchMock = workspaceFetch(() => saved, (path, init) => {
    if (path.endsWith("/collection/selection")) {
      if (init?.method === "PUT") serverSelection = JSON.parse(init.body as string).candidate_ids;
      return { candidate_ids: serverSelection };
    }
    if (path.includes("/collection/candidates?")) {
      const page = new URL(path, "http://localhost").searchParams.get("page");
      return { items: [page === "2" ? paper("b", "Paper B") : paper("a", "Paper A")], total: 101 };
    }
    if (path.endsWith("/collection/download") && init?.method === "POST") return { id: "download-1" };
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  const { client } = renderWorkspace(<ProjectPage />, "/projects/p1");
  await screen.findByRole("checkbox", { name: "Select Paper A" });
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Select Paper A" })).toBeChecked());
  expect(fetchMock.mock.calls.findIndex(([path]) => String(path).endsWith("/collection/selection"))).toBeGreaterThan(fetchMock.mock.calls.findIndex(([path]) => String(path).includes("/collection/candidates?")));
  fireEvent.click(screen.getByRole("checkbox", { name: "Select Paper A" }));
  expect(screen.getByRole("checkbox", { name: "Select Paper A" })).not.toBeChecked();
  await act(async () => { await client.refetchQueries({ queryKey: ["candidates", "p1"] }); });
  expect(screen.getByRole("checkbox", { name: "Select Paper A" })).not.toBeChecked();
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByRole("checkbox", { name: "Select Paper B" })).toBeChecked();
  fireEvent.click(screen.getByRole("button", { name: "Previous" }));
  expect(await screen.findByRole("checkbox", { name: "Select Paper A" })).not.toBeChecked();
  fireEvent.click(screen.getByRole("button", { name: "Download selected PDFs" }));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/projects/p1/collection/selection", expect.objectContaining({ method: "PUT", body: JSON.stringify({ candidate_ids: ["b"] }) })));
  await waitFor(() => expect(serverSelection).toEqual(["b"]));
  expect(screen.getByRole("checkbox", { name: "Select Paper A" })).not.toBeChecked();
});


function candidate(id = "a", title = "Available paper", downloaded = false) {
  return { candidate_id: id, title, authors: ["Researcher"], year: 2025, doi: "", source_apis: ["crossref"], relevance_score: 1, citation_count: 0, selected: true, download_status: downloaded ? "downloaded" : "pending", matched_keywords: [], relevance_score_breakdown: {} };
}
function pdfsReady() {
  const saved = workflow(["define", "collect", "candidates"]);
  saved.steps.candidates.materials = { selected_count: 1, selected_downloaded_count: 1 };
  saved.steps.extract.materials = { selected_downloaded_count: 1 };
  return saved;
}
function runningDiscovery(phase = "extract") {
  const current = pdfsReady();
  for (const stage of ["extract", "filter", "annotate", "merge"] as const) current.steps[stage].running = true;
  current.steps.merge.materials = { discovery_phase: phase, discovery_status: "running", discovery_completed: phase === "merge" ? 4 : 2, discovery_total: 5 };
  return current;
}
function jobEvents() {
  const events = { addEventListener: vi.fn(), close: vi.fn() };
  const eventSource = vi.fn(function (_url: string) { return events; });
  vi.stubGlobal("EventSource", eventSource);
  return { eventSource, events };
}
function discoveryRows(phase = "extract", running = true) {
  return { items: [{ ...sampleFeature, feature_name: "Live technique", annotation: undefined as { short_definition: string } | undefined }], total: 1, phase, running, partial: true, completed: phase === "merge" ? 4 : 2, stages_total: 5, source_stage: phase === "merge" ? "annotate" : "extract" };
}

it("switches from a running paper search to PDF selection in the same workspace", async () => {
  let current = workflow(["define"]);
  current.steps.collect.running = true;
  const fetchMock = workspaceFetch(() => current, path => {
    if (path.includes("/collection/live?")) return { items: [candidate()], total: 1, running: true, partial: true, progress: { completed: 1, total: 2 } };
    if (path.includes("/collection/candidates?")) return { items: [candidate()], total: 1 };
    if (path.endsWith("/collection/selection")) return { candidate_ids: ["a"] };
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  const { client } = renderWorkspace(<ProjectPage />, "/projects/p1");
  await screen.findByText("Available paper");
  const workspace = screen.getByLabelText("Papers");
  expect(screen.queryByRole("checkbox", { name: "Select Available paper" })).not.toBeInTheDocument();
  current = workflow(["define", "collect"]);
  await act(async () => { await client.refetchQueries({ queryKey: ["workflow", "p1"] }); });
  expect(await screen.findByRole("checkbox", { name: "Select Available paper" })).toBeVisible();
  expect(screen.getByLabelText("Papers")).toBe(workspace);
  expect(screen.getByLabelText("Search found papers")).toBeVisible();
  expect(screen.getByRole("button", { name: "Download selected PDFs" })).toBeVisible();
  expect(screen.queryByRole("button", { name: /Select PDFs|Read these papers|Read PDFs/ })).not.toBeInTheDocument();
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === "POST")).toBe(false);
});

it("downloads selected PDFs, then starts the complete feature workflow only on Extract features", async () => {
  let current = workflow(["define", "collect"]);
  let downloaded = false;
  const fetchMock = workspaceFetch(() => current, (path, init) => {
    if (path.includes("/collection/candidates?")) return { items: [candidate("a", "Available paper", downloaded)], total: 1 };
    if (path.endsWith("/collection/selection")) return { candidate_ids: ["a"] };
    if (path.endsWith("/collection/download") && init?.method === "POST") { downloaded = true; current = pdfsReady(); return { id: "download-1", kind: "collection_download", status: "completed" }; }
    if (path.endsWith("/discovery/run") && init?.method === "POST") { current = runningDiscovery(); return { id: "discovery-1", kind: "discovery", status: "queued" }; }
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWorkspace(<ProjectPage />, "/projects/p1");
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Select Available paper" })).toBeChecked());
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === "POST")).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Download selected PDFs" }));
  const extract = await screen.findByRole("button", { name: "Extract features" });
  await waitFor(() => expect(extract).toBeEnabled());
  expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith("/discovery/run"))).toBe(false);
  expect(fetchMock.mock.calls.some(([path]) => String(path).includes("/readiness?action=discovery"))).toBe(true);
  fireEvent.click(extract);
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/projects/p1/discovery/run", expect.objectContaining({ method: "POST" })));
  expect(await screen.findByRole("heading", { name: "Features from your papers" })).toBeVisible();
  expect(fetchMock.mock.calls.some(([path]) => /\/(extraction|stages\/(filter|annotate|merge))\/run$/.test(String(path)))).toBe(false);
  expect(screen.queryByRole("button", { name: /Read these papers|Refine these concepts|Review equivalent concepts/ })).not.toBeInTheDocument();
});

it("checks whole-workflow inputs before enabling Extract features after PDF download", async () => {
  let current = workflow(["define", "collect"]);
  const blocked = { ready: false, message: "Complete required inputs", requirements: [{ id: "embedding_connection", field: "GEMINI_API_KEY", kind: "credential", label: "Gemini API key", message: "Configure embeddings before extracting features." }] };
  const fetchMock = workspaceFetch(() => current, path => {
    if (path.includes("/readiness?action=discovery")) return blocked;
    if (path.includes("/collection/candidates?")) return { items: [candidate("a", "Available paper", true)], total: 1 };
    if (path.endsWith("/collection/selection")) return { candidate_ids: ["a"] };
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  const { client } = renderWorkspace(<ProjectPage />, "/projects/p1");
  await screen.findByRole("checkbox", { name: "Select Available paper" });
  current = pdfsReady();
  await act(async () => { await client.refetchQueries({ queryKey: ["workflow", "p1"] }); });
  expect(await screen.findByText("Configure embeddings before extracting features.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Extract features" })).toBeDisabled();
  expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith("/discovery/run"))).toBe(false);
});

it("keeps live processing visible over an old inventory and opens final features when the chain finishes", async () => {
  jobEvents();
  let current = runningDiscovery("annotate");
  current.steps.results.materials = { has_saved_inventory: true };
  let live = discoveryRows("annotate");
  let state = "running";
  const fetchMock = workspaceFetch(() => current, path => {
    if (path.includes("/discovery/live?")) return live;
    if (path.endsWith("/projects/p1")) return { ...project, jobs: [{ id: "chain-1", kind: "discovery", status: state }] };
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  const { client } = renderWorkspace(<ProjectPage />, "/projects/p1");
  expect(nextWorkflowStep(current)).toBe("features");
  expect(await screen.findByRole("heading", { name: "Features from your papers" })).toBeVisible();
  expect(await screen.findByText("Live technique")).toBeVisible();
  expect(screen.queryByRole("heading", { name: "Your concept inventory" })).not.toBeInTheDocument();
  current = workflow(["define", "collect", "candidates", "extract", "filter", "annotate", "merge", "results"]);
  current.steps.results.materials = { has_saved_inventory: true };
  current.steps.merge.materials = { discovery_status: "completed", discovery_phase: "complete" };
  live = { ...discoveryRows("complete", false), partial: false, completed: 5 };
  state = "completed";
  await act(async () => { await client.refetchQueries({ queryKey: ["workflow", "p1"] }); });
  expect(await screen.findByRole("heading", { name: "Your concept inventory" })).toBeVisible();
  expect(screen.queryByRole("heading", { name: "Features from your papers" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Loaded language" })).toBeVisible();
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === "POST")).toBe(false);
});

it.each(["annotate", "merge"])("shows %s progress and cancels the discovery job without launching work on mount", async phase => {
  const { eventSource } = jobEvents();
  const current = runningDiscovery(phase);
  const fetchMock = workspaceFetch(() => current, path => {
    if (path.includes("/discovery/live?")) return discoveryRows(phase);
    if (path.endsWith("/projects/p1")) return { ...project, jobs: [{ id: "chain-1", kind: "discovery", status: "running" }] };
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWorkspace(<FeatureDiscovery projectId="p1" workflow={current} starting={false} resume={vi.fn()} />);
  expect(await screen.findByText("Live technique")).toBeVisible();
  const job = await screen.findByRole("complementary", { name: "Run status" });
  expect(within(job).getByRole("status")).toHaveTextContent(phase === "annotate" ? "Adding definitions and examples" : "Combining equivalent features");
  expect(eventSource).toHaveBeenCalledWith("/api/jobs/chain-1/events");
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === "POST")).toBe(false);
  fireEvent.click(within(job).getByRole("button", { name: "Stop" }));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/chain-1/cancel", expect.objectContaining({ method: "POST" })));
  expect(fetchMock.mock.calls.filter(([, options]) => options?.method === "POST").map(([path]) => path)).toEqual(["/api/jobs/chain-1/cancel"]);
});

it.each(["queued", "cancelling"])("keeps discovery controls safe while %s", async state => {
  jobEvents();
  const current = runningDiscovery();
  const fetchMock = workspaceFetch(() => current, path => {
    if (path.includes("/discovery/live?")) return discoveryRows();
    if (path.endsWith("/projects/p1")) return { ...project, jobs: [{ id: "chain-1", kind: "discovery", status: state }] };
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWorkspace(<FeatureDiscovery projectId="p1" workflow={current} starting={false} resume={vi.fn()} />);
  const stop = await screen.findByRole("button", { name: state === "queued" ? "Stop" : "Stopping…" });
  if (state === "queued") expect(stop).toBeEnabled(); else expect(stop).toBeDisabled();
  expect(screen.queryByRole("button", { name: "Extract features" })).not.toBeInTheDocument();
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === "POST")).toBe(false);
});

it.each(["cancelled", "interrupted", "failed"])("retains partial features after discovery is %s and waits for an explicit resume", async state => {
  const { eventSource } = jobEvents();
  const current = pdfsReady();
  current.steps.merge.materials = { discovery_status: state, discovery_phase: "annotate" };
  const resume = vi.fn();
  const fetchMock = workspaceFetch(() => current, path => {
    if (path.includes("/discovery/live?")) return discoveryRows("annotate", false);
    if (path.endsWith("/projects/p1")) return { ...project, jobs: [{ id: "chain-1", kind: "discovery", status: state, error: state === "failed" ? "Provider unavailable" : "" }] };
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWorkspace(<FeatureDiscovery projectId="p1" workflow={current} starting={false} resume={resume} />);
  expect(await screen.findByText("Live technique")).toBeVisible();
  expect(await screen.findByText("Saved progress is available below. Run this step again to finish.")).toBeVisible();
  expect(screen.queryByRole("button", { name: "Stop" })).not.toBeInTheDocument();
  const action = screen.getByRole("button", { name: "Resume feature extraction" });
  expect(action).toBeEnabled();
  expect(eventSource).not.toHaveBeenCalled();
  expect(resume).not.toHaveBeenCalled();
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === "POST")).toBe(false);
  fireEvent.click(action);
  expect(resume).toHaveBeenCalledOnce();
});


it("updates live feature definitions and progress without resetting the list search", async () => {
  let live = discoveryRows("extract");
  const current = runningDiscovery();
  const fetchMock = workspaceFetch(() => current, path => path.includes("/discovery/live?") ? live : undefined);
  vi.stubGlobal("fetch", fetchMock);
  const { client } = renderWorkspace(<FeatureDiscovery projectId="p1" workflow={current} starting={false} resume={vi.fn()} />);
  await screen.findByText("Live technique");
  fireEvent.change(screen.getByLabelText("Search extracted features"), { target: { value: "Live" } });
  await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => String(path).includes("/discovery/live?") && String(path).includes("query=Live"))).toBe(true));
  const list = screen.getByRole("region", { name: "Items" });
  list.scrollTop = 180;
  live = { ...discoveryRows("annotate"), completed: 3, source_stage: "annotate", items: [{ ...sampleFeature, feature_name: "Live technique", annotation: { short_definition: "A clearer source-based definition" } }] };
  await act(async () => { await client.refetchQueries({ queryKey: ["discoveryLive", "p1"] }); });
  expect(await screen.findByText("A clearer source-based definition")).toBeVisible();
  expect(screen.getByLabelText("Search extracted features")).toHaveValue("Live");
  expect(list.scrollTop).toBe(180);
  expect(screen.getByText("Adding definitions and examples")).toBeVisible();
  expect(fetchMock.mock.calls.some(([, options]) => options?.method === "POST")).toBe(false);
});

it("keeps an old inventory hidden after starting discovery while its first status refresh is pending", async () => {
  vi.spyOn(window, "scrollTo").mockImplementation(() => {});
  jobEvents();
  const saved = pdfsReady();
  saved.steps.results.materials = { has_saved_inventory: true };
  saved.steps.merge.materials = { discovery_status: "completed" };
  let started = false;
  let releaseStatus: (() => void) | undefined;
  const fetchMock = workspaceFetch(() => saved, (path, init) => {
    if (path.endsWith("/discovery/run") && init?.method === "POST") { started = true; return { id: "new-chain", kind: "discovery", status: "queued" }; }
    if (path.endsWith("/workflow") && started) return new Promise(resolve => { releaseStatus = () => resolve(runningDiscovery()); });
    if (path.endsWith("/projects/p1") && started) return { ...project, jobs: [{ id: "new-chain", kind: "discovery", status: "queued" }] };
    if (path.includes("/collection/candidates?")) return { items: [candidate("a", "Available paper", true)], total: 1 };
    if (path.endsWith("/collection/selection")) return { candidate_ids: ["a"] };
    return undefined;
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWorkspace(<ProjectPage />, "/projects/p1");
  await screen.findByRole("heading", { name: "Your concept inventory" });
  fireEvent.click(screen.getByRole("button", { name: /Features 3 \/ 3/ }));
  fireEvent.click(within(screen.getByRole("navigation", { name: "Research steps" })).getByRole("button", { name: "Papers" }));
  const extract = await screen.findByRole("button", { name: "Extract features" });
  await waitFor(() => expect(extract).toBeEnabled());
  fireEvent.click(extract);
  await waitFor(() => expect(releaseStatus).toBeTypeOf("function"));
  try {
    expect(await screen.findByRole("heading", { name: "Features from your papers" })).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Your concept inventory" })).not.toBeInTheDocument();
  } finally {
    await act(async () => { releaseStatus?.(); });
  }
});

it("shows selected PDF progress through completion in the Papers card without starting extraction", async () => {
  let current=workflow(["define","collect"]);
  let pdfs={selected:3,ready:0,failed:0,unavailable:0,pending:3,processed:0,running:false,status:null as string|null,job_id:null as string|null,complete:false,ready_to_extract:false};
  const papers=[candidate("a","First paper"),candidate("b","Second paper"),candidate("c","Third paper")];
  const fetchMock=workspaceFetch(()=>current,(path,init)=>{
    if(path.includes("/collection/candidates?"))return {items:path.includes("query=First")?[papers[0]]:papers,total:path.includes("query=First")?1:3};
    if(path.endsWith("/collection/selection"))return {candidate_ids:["a","b","c"]};
    if(path.endsWith("/collection/download-progress"))return pdfs;
    if(path.endsWith("/collection/download")&&init?.method==="POST"){
      current.steps.candidates.running=true;
      pdfs={...pdfs,running:true,status:"running",job_id:"pdf-job"};return {id:"pdf-job",kind:"collection_download",status:"running"};
    }
    if(path.endsWith("/discovery/run")&&init?.method==="POST"){current=runningDiscovery();return {id:"features",kind:"discovery",status:"queued"};}
    return undefined;
  });
  vi.stubGlobal("fetch",fetchMock);
  const {client}=renderWorkspace(<ProjectPage/>,"/projects/p1");
  await waitFor(()=>expect(screen.getByRole("checkbox",{name:"Select First paper"})).toBeChecked());
  expect(fetchMock.mock.calls.some(([,init])=>init?.method==="POST")).toBe(false);
  fireEvent.click(screen.getByRole("button",{name:"Download selected PDFs"}));
  expect(await screen.findByRole("progressbar",{name:"PDF download progress"})).toHaveAttribute("value","0");
  pdfs={...pdfs,ready:1,pending:2,processed:1};
  await act(async()=>{await client.refetchQueries({queryKey:["downloadProgress","p1"]})});
  await waitFor(()=>expect(screen.getByText("Downloading PDFs · 1 of 3 ready")).toBeVisible(),{timeout:3000});
  expect(screen.getByRole("progressbar",{name:"PDF download progress"})).toHaveAttribute("max","3");
  fireEvent.change(screen.getByLabelText("Search found papers"),{target:{value:"First"}});
  await waitFor(()=>expect(screen.queryByText("Second paper")).not.toBeInTheDocument());
  expect(screen.getByText("Downloading PDFs · 1 of 3 ready")).toBeVisible();
  expect(screen.getByRole("checkbox",{name:"Select First paper"})).toBeVisible();
  pdfs={...pdfs,ready:3,pending:0,processed:3,running:false,status:"completed",complete:true,ready_to_extract:true};
  current=pdfsReady();current.steps.candidates.materials={selected_count:3,selected_downloaded_count:3};
  await act(async()=>{await Promise.all([client.refetchQueries({queryKey:["downloadProgress","p1"]}),client.refetchQueries({queryKey:["workflow","p1"]})])});
  expect(screen.getByText("3 PDFs ready · ready to extract")).toBeVisible();
  expect(screen.getByRole("progressbar",{name:"PDF download progress"})).toHaveAttribute("value","3");
  // The separately polled workflow may finish refreshing after PDF progress.
  await waitFor(()=>expect(screen.getByRole("button",{name:"Extract features"})).toBeEnabled(),{timeout:5000});
  expect(fetchMock.mock.calls.some(([path])=>String(path).endsWith("/discovery/run"))).toBe(false);
  fireEvent.click(screen.getByRole("button",{name:"Extract features"}));
  await waitFor(()=>expect(fetchMock).toHaveBeenCalledWith("/api/projects/p1/discovery/run",expect.objectContaining({method:"POST"})));
});
