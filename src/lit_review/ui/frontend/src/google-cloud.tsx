import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, json } from "./api";

type Login = { id: string; status: string; authorization_url: string | null; message: string };
type CloudStatus = {
  configured: boolean; configuration_error?: string; enabled: boolean; project: string; location: string;
  saved_project: string; saved_location: string; gcloud_installed: boolean;
  adc_file_detected: boolean; external_credentials: boolean; login: Login | null;
};
type Settings = { enabled: boolean; project: string; location: string };
const active = (login?: Login | null) => !!login && ["starting", "awaiting_code", "verifying"].includes(login.status);

export function GoogleCloudSettings() {
  const client = useQueryClient();
  const [settings, setSettings] = useState<Settings | null>(null);
  const [code, setCode] = useState("");
  const [notice, setNotice] = useState("");
  const query = useQuery({
    queryKey: ["google-cloud"], queryFn: () => api<CloudStatus>("/api/settings/google-cloud"), retry: false,
    refetchInterval: query => active(query.state.data?.login) ? 1000 : false,
  });
  const updated = () => {
    void client.invalidateQueries({ queryKey: ["credentials"] });
    void client.invalidateQueries({ queryKey: ["readiness"] });
    window.dispatchEvent(new Event("credentials-updated"));
  };
  useEffect(() => {
    if (query.data && !settings) setSettings({
      enabled: query.data.configured ? query.data.enabled : true,
      project: query.data.saved_project || query.data.project,
      location: query.data.saved_location || query.data.location || "us-central1",
    });
  }, [query.data, settings]);
  useEffect(() => {
    if (query.data?.login?.status === "succeeded") {
      setCode("");
      updated();
    }
  }, [query.data?.login?.status]);
  const setLogin = (login: Login) => {
    client.setQueryData<CloudStatus>(["google-cloud"], current => current ? { ...current, login } : current);
    void query.refetch();
  };
  const save = useMutation({
    mutationFn: (next: Settings) => api<CloudStatus>("/api/settings/google-cloud", json("PUT", next)),
    onSuccess: data => { client.setQueryData(["google-cloud"], data); setNotice("Google Cloud defaults saved for this app."); updated(); },
  });
  const login = useMutation({
    mutationFn: async () => {
      setNotice("");
      const next = { ...settings!, enabled: true };
      await save.mutateAsync(next);
      setSettings(next);
      return api<Login>("/api/settings/google-cloud/login", { method: "POST" });
    }, onSuccess: setLogin,
  });
  const submit = useMutation({
    mutationFn: () => api<Login>(`/api/settings/google-cloud/login/${query.data!.login!.id}/code`, json("POST", { code: code.trim() })),
    onSuccess: setLogin, onSettled: () => setCode(""),
  });
  const cancel = useMutation({
    mutationFn: () => api<Login>(`/api/settings/google-cloud/login/${query.data!.login!.id}`, { method: "DELETE" }),
    onSuccess: data => { setCode(""); setLogin(data); },
  });
  const check = useMutation({
    mutationFn: async () => {
      setNotice("");
      await save.mutateAsync(settings!);
      return api<{ message: string }>("/api/settings/google-cloud/check", { method: "POST" });
    }, onSuccess: data => { setNotice(data.message); updated(); void query.refetch(); },
  });
  const busy = active(query.data?.login) || login.isPending || submit.isPending || cancel.isPending;
  const error = query.error || save.error || login.error || submit.error || cancel.error || check.error;
  if (!settings) return <div className="mt-4">{error ? <p role="alert" className="text-sm text-red-700">{error.message}</p> : <p role="status">Loading Google Cloud settings…</p>}</div>;
  const valid = !!settings.project.trim() && !!settings.location.trim();
  const change = (next: Partial<Settings>) => { setNotice(""); setSettings({ ...settings, ...next }); };
  return <section className="mt-4" aria-label="Google Cloud setup">
    <p className="text-xs leading-5 text-slate-600">Sign in with Google, then use your project for Gemini and embeddings. gcloud stores your credentials on the machine running this app. Project and region defaults are saved across restarts.</p>
    {query.data?.configuration_error && <p role="alert" className="mt-3 text-sm text-red-700">{query.data.configuration_error}</p>}
    <label className="mt-4 block text-xs font-semibold">Google Cloud project
      <input className="input mt-1" placeholder="my-project-id" value={settings.project} disabled={busy} onChange={e => change({ project: e.target.value })} />
    </label>
    <label className="mt-3 block text-xs font-semibold">Google Cloud region
      <input className="input mt-1" placeholder="us-central1" value={settings.location} disabled={busy} onChange={e => change({ location: e.target.value })} />
    </label>
    <p className="mt-1 text-xs text-slate-500">Choose a region supported by your models. us-central1 supports the default embedding model.</p>
    <label className="mt-4 flex items-center gap-2 text-xs"><input type="checkbox" checked={settings.enabled} disabled={busy} onChange={e => change({ enabled: e.target.checked })} />Use Google Cloud by default</label>
    <p className="mt-2 text-xs text-slate-500">Individual stages can still select API key authentication.</p>
    <div className="mt-4 flex flex-wrap gap-2">
      <button className="button-secondary" disabled={!valid || busy || save.isPending || check.isPending} onClick={() => { setNotice(""); save.mutate(settings); }}>Save defaults</button>
      <button className="button-primary" disabled={!valid || busy || check.isPending || !query.data?.gcloud_installed || query.data?.external_credentials} onClick={() => login.mutate()}>Sign in with Google</button>
      <button className="button-secondary" disabled={!valid || busy || check.isPending || save.isPending} onClick={() => check.mutate()}>{check.isPending ? "Checking…" : "Check credentials"}</button>
    </div>
    {!query.data?.gcloud_installed && <p className="mt-3 text-xs text-slate-600">Install the <a className="underline" href="https://cloud.google.com/sdk/docs/install" target="_blank" rel="noreferrer">Google Cloud CLI</a> on the app’s machine to sign in here. Existing credentials can still be checked.</p>}
    {query.data?.external_credentials && <p className="mt-3 text-xs text-slate-600">This app uses the file selected by GOOGLE_APPLICATION_CREDENTIALS. Check those credentials, or unset that environment variable before starting a user sign-in.</p>}
    {query.data?.adc_file_detected && <p className="mt-3 text-xs text-slate-600">Google credential file detected. Use Check credentials to verify it.</p>}
    {query.data?.login && <div className="mt-4 border-l-2 border-slate-300 pl-3">
      <p role="status" className="text-sm">{query.data.login.message}</p>
      {query.data.login.status === "awaiting_code" && query.data.login.authorization_url && <>
        <a className="button-secondary mt-3" href={query.data.login.authorization_url} target="_blank" rel="noreferrer">Open Google sign-in</a>
        <form className="mt-3" onSubmit={e => { e.preventDefault(); if (code.trim() && !submit.isPending) submit.mutate(); }}>
          <label className="block text-xs font-semibold">Google authorization code<input className="input mt-1" type="password" autoComplete="off" value={code} onChange={e => setCode(e.target.value)} /></label>
          <button className="button-primary mt-2" disabled={!code.trim() || submit.isPending}>Complete sign-in</button>
        </form>
      </>}
      {active(query.data.login) && <button className="button-quiet mt-3" disabled={cancel.isPending} onClick={() => cancel.mutate()}>Cancel sign-in</button>}
    </div>}
    {notice && <p role="status" className="mt-3 text-sm text-slate-700">{notice}</p>}
    {error && <p role="alert" className="mt-3 text-sm text-red-700">{error.message}</p>}
  </section>;
}
