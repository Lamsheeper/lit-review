import {createContext,useContext,useEffect,useState,type ButtonHTMLAttributes} from 'react';
import {useQuery,useQueryClient} from '@tanstack/react-query';
import {api,json,type Readiness} from './api';
export const ReadyContext=createContext(true);
export function ReadinessButton(props:ButtonHTMLAttributes<HTMLButtonElement>){const ready=useContext(ReadyContext);return <button {...props} disabled={props.disabled||!ready}/>;}
export function useReadiness(projectId?:string,action='flow'){
 const client=useQueryClient();
 const query=useQuery({queryKey:['readiness',projectId||'start',action],queryFn:()=>api<Readiness>(projectId?`/api/projects/${projectId}/readiness?action=${action}`:'/api/readiness'),retry:false});
 useEffect(()=>{const update=(event:Event)=>{client.setQueryData(['readiness',projectId||'start',action],(event as CustomEvent).detail)};const credentials=()=>{void client.invalidateQueries({queryKey:['readiness']})};window.addEventListener('readiness-needed',update);window.addEventListener('credentials-updated',credentials);return()=>{window.removeEventListener('readiness-needed',update);window.removeEventListener('credentials-updated',credentials)}},[projectId,action]);
 return query;
}
export function ReadinessNotice({report,check,onFiles,error:checkError,checking}:{report?:Readiness;error?:Error|null;checking?:boolean;check:()=>Promise<unknown>;onFiles?:()=>void}){
 const client=useQueryClient();
 const [values,setValues]=useState<Record<string,string>>({}),[busy,setBusy]=useState(false),[error,setError]=useState('');
 if(!report)return checkError?<section className="readiness-card" role="alert"><p>We couldn’t check your required inputs. Check the app connection and try again.</p><button className="button-secondary" onClick={()=>void check()}>Retry checks</button></section>:checking?<p className="text-xs text-slate-500 mt-3" role="status">Checking required inputs…</p>:null;
 if(report.ready)return null;
 const credentialFields=report.requirements.filter(r=>r.kind==='credential'&&['GEMINI_API_KEY','OPENAI_API_KEY','BRAVE_SEARCH_API_KEY','SERPAPI_API_KEY'].includes(r.field||''));
 const save=async()=>{setBusy(true);setError('');try{const credentials=Object.fromEntries(Object.entries(values).filter(([,v])=>v.trim()).map(([k,v])=>[k,v.trim()]));if(Object.keys(credentials).length){await api('/api/settings/session-credentials',json('PUT',{credentials}));setValues({});await client.invalidateQueries({queryKey:['credentials']});await client.invalidateQueries({queryKey:['readiness']})}await check()}catch(e){setError(String(e))}finally{setBusy(false)}};
 return <section className="readiness-card" aria-label="Required inputs"><h3>Before we start</h3>{report.requirements.map(r=><p key={r.id}>{r.message}</p>)}{credentialFields.map(r=><label key={r.id}>{r.label}<input className="input" type="password" autoComplete="off" value={values[r.field!]||''} onChange={e=>setValues({...values,[r.field!]:e.target.value})} placeholder="Used for this session only"/></label>)}<div>{report.requirements.some(r=>['GEMINI_API_KEY','OPENAI_API_KEY'].includes(r.field||'')||r.id.startsWith('google_'))&&<button className="button-secondary" onClick={()=>window.dispatchEvent(new Event('open-google-cloud'))}>Set up Google Cloud</button>}<button className="button-secondary" disabled={busy} onClick={()=>void save()}>{busy?'Checking…':credentialFields.length?'Save and check':'Check again'}</button>{report.requirements.some(r=>r.id==='pdfs')&&onFiles&&<button className="button-quiet" onClick={onFiles}>Go to selected PDFs</button>}</div>{error&&<p role="alert">{error}</p>}</section>
}
