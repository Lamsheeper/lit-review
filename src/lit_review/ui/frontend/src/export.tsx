import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { CheckCircle2, Copy, Download, LoaderCircle, X } from "lucide-react";
import { api, json } from "./api";

type PreparedCSV = { filename: string; path: string; download_url: string };

export function InventoryExport({ projectId, inventory }: { projectId: string; inventory: string }) {
  const [prepared, setPrepared] = useState<PreparedCSV | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copyMessage, setCopyMessage] = useState("");
  const button = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLElement>(null);
  const titleId = useId();
  const descriptionId = useId();

  useEffect(() => {
    if (!prepared) return;
    dialog.current?.focus();
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        setPrepared(null);
      } else if (event.key === "Tab") {
        const controls = dialog.current?.querySelectorAll<HTMLElement>("button:not([disabled]), a[href]");
        if (!controls?.length) return;
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === dialog.current)) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = originalOverflow;
      button.current?.focus();
    };
  }, [prepared]);

  async function prepare() {
    setBusy(true);
    setError("");
    setCopyMessage("");
    try {
      const result = await api<PreparedCSV>(`/api/projects/${projectId}/exports/csv`, json("POST", { inventory }));
      const url = new URL(result.download_url, window.location.href);
      if (!result.filename || !result.path || url.origin !== window.location.origin || !url.pathname.startsWith("/api/projects/")) {
        throw new Error("The CSV download location could not be confirmed. Please try again.");
      }
      const link = document.createElement("a");
      link.href = result.download_url;
      link.download = result.filename;
      document.body.append(link);
      try {
        link.click();
      } finally {
        link.remove();
      }
      setPrepared(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function copyPath() {
    if (!prepared) return;
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(prepared.path);
      setCopyMessage("Path copied.");
    } catch {
      setCopyMessage("Select the path above to copy it manually.");
    }
  }

  return <>
    <button ref={button} className="button-primary" disabled={busy} onClick={() => void prepare()}>
      {busy ? <LoaderCircle className="animate-spin" size={16} /> : <Download size={16} />}
      {busy ? "Preparing CSV…" : "Download CSV"}
    </button>
    {error && <p className="export-error" role="alert">{error}</p>}
    {prepared && createPortal(<div className="export-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setPrepared(null); }}>
      <section ref={dialog} className="export-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={descriptionId} tabIndex={-1}>
        <button className="icon-button export-close" aria-label="Close export confirmation" onClick={() => setPrepared(null)}><X size={18} /></button>
        <CheckCircle2 className="export-success" size={26} />
        <h2 id={titleId}>CSV export complete</h2>
        <p id={descriptionId}>The CSV has been prepared and is available at this location on the machine running Lit Review.</p>
        <p className="export-filename">{prepared.filename}</p>
        <code className="export-path">{prepared.path}</code>
        <p>A browser download has also been requested. Your browser’s settings determine its Downloads folder.</p>
        <div className="export-actions">
          <button className="button-secondary" onClick={() => void copyPath()}><Copy size={14} /> Copy path</button>
          <a className="button-secondary" href={prepared.download_url} download={prepared.filename}><Download size={14} /> Download again</a>
          <button className="button-primary" onClick={() => setPrepared(null)}>Done</button>
        </div>
        <p className="export-copy-status" role="status">{copyMessage}</p>
      </section>
    </div>, document.body)}
  </>;
}
