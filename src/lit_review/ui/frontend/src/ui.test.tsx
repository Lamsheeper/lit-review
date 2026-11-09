import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Progress, StepHeader, nextWorkflowStep, ProjectsPage, ProjectPage } from "./ui";
import type { Project, Workflow } from "./api";

vi.mock("@uiw/react-codemirror", () => ({default: ({value}: {value:string}) => <div data-testid="definition-editor">{value}</div>}));

afterEach(() => {cleanup();vi.unstubAllGlobals()});
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
    expect(screen.getByText("3 of 8 steps complete")).toBeInTheDocument();
  });

  it("resumes at the first unfinished step, including stale upstream work", () => {
    const saved = workflow(["define", "collect", "candidates", "extract"]);
    expect(nextWorkflowStep(saved)).toBe("filter");
    saved.steps.extract = { ...status, stale: true };
    expect(nextWorkflowStep(saved)).toBe("extract");
    saved.steps.collect.running = true;
    expect(nextWorkflowStep(saved)).toBe("collect");
  });

  it("returns completed projects to results without requiring scoring", () => {
    expect(nextWorkflowStep(workflow(["define", "collect", "candidates", "extract", "filter", "annotate", "merge", "results"]))).toBe("results");
  });
});

describe("StepHeader", () => {
  it("names the next step only when the current results are ready", () => {
    const onNext = vi.fn();
    const base = { eyebrow: "Step 2 of 8 · Collect", title: "Collect", description: "Collect papers", action: <button>Find and rank papers</button>, onNext, nextLabel: "Select PDFs" };
    const { rerender } = render(<StepHeader {...base} status={status} />);
    expect(screen.queryByText("Next: Select PDFs")).not.toBeInTheDocument();
    expect(screen.getByText("Not run")).toBeInTheDocument();
    expect(screen.getByText(/Needed before the next step/)).toHaveTextContent("ranked paper candidates");

    rerender(<StepHeader {...base} status={{ ...status, complete: true, running: true, missing: [] }} />);
    expect(screen.queryByText("Next: Select PDFs")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("This step is running");

    rerender(<StepHeader {...base} status={{ ...status, complete: true, missing: [] }} />);
    fireEvent.click(screen.getByText("Next: Select PDFs"));
    expect(onNext).toHaveBeenCalledOnce();
  });

  it("blocks progression for unsaved settings and stale results", () => {
    const base = { eyebrow: "Filter", title: "Filter", description: "Review candidates", action: <button>Run filter</button>, onNext: vi.fn() };
    const { rerender } = render(<StepHeader {...base} status={{ ...status, complete: true }} ready={false} />);
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    expect(screen.queryByText("Next step")).not.toBeInTheDocument();
    rerender(<StepHeader {...base} status={{ ...status, complete: true, stale: true }} ready />);
    expect(screen.getByText("Update needed")).toBeInTheDocument();
    expect(screen.queryByText("Next step")).not.toBeInTheDocument();
  });
});


import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import {MemoryRouter,Routes,Route} from 'react-router-dom';
function renderWorkspace(element:React.ReactNode,path='/') {
 const client=new QueryClient({defaultOptions:{queries:{retry:false},mutations:{retry:false}}});
 return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[path]}><Routes><Route path="/" element={element}/><Route path="/projects/:projectId" element={path==='/'?<div>Research workspace</div>:element}/></Routes></MemoryRouter></QueryClientProvider>);
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
 expect(await screen.findByText('Your research workspace is ready.')).toBeInTheDocument();
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
