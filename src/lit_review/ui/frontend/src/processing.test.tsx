import { fireEvent, render, screen, waitFor, cleanup } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { ProcessingStage } from "./ui";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return { ...original, api: vi.fn() };
});
const mockedApi = vi.mocked(api);

beforeEach(() => {
  mockedApi.mockImplementation(async (path) => {
    if (path.endsWith("/config")) return { provider: "gemini", model: "test-model", batch_features: 25 } as any;
    if (path.includes("/results/")) return { total: 1, items: [{ feature_name: "Neutral posture", source_paper_ids: ["paper-1"] }] } as any;
    if (path.endsWith("/preview")) return { dry_run: true, stages: [] } as any;
    if (path.endsWith("/run")) return { id: "job-1" } as any;
    return { jobs: [] } as any;
  });
});
afterEach(() => { cleanup(); vi.clearAllMocks(); });

function show(stage: "filter" | "score", status = { complete: false, running: false, stale: false, missing: [], materials: {}, action_label: "Run" }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(<QueryClientProvider client={client}><ProcessingStage projectId="p1" stage={stage} status={status} /></QueryClientProvider>);
}

describe("ProcessingStage", () => {
  it("inspects rejection artifacts and preserves evidence details", async () => {
    show("filter");
    expect(await screen.findByText("Neutral posture")).toBeInTheDocument();
    expect(screen.getByText("Source papers: paper-1")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Stage artifact"), { target: { value: "rejected_features.jsonl" } });
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/stages/filter/results/rejected_features.jsonl"));
    expect(screen.getByRole("link", { name: "Export" })).toHaveAttribute("href", "/api/projects/p1/stages/filter/exports/rejected_features.jsonl");
  });

  it("saves settings before previewing and running", async () => {
    show("filter");
    await screen.findByDisplayValue("test-model");
    fireEvent.click(screen.getByText("Preview settings"));
    expect(await screen.findByText(/Planned inputs and outputs/)).toBeInTheDocument();
    const calls = mockedApi.mock.calls.map(call => call[0]);
    expect(calls.indexOf("/api/projects/p1/stages/filter/preview")).toBeGreaterThan(calls.lastIndexOf("/api/projects/p1/stages/filter/config"));
    fireEvent.click(screen.getByText("Run filter"));
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/stages/filter/run", { method: "POST" }));
  });

  it("requires rerunning changed settings before offering the next step", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    render(<QueryClientProvider client={client}><ProcessingStage projectId="p1" stage="filter" status={{ complete: true, running: false, missing: [], materials: {}, action_label: "Run" }} onNext={() => {}} /></QueryClientProvider>);
    await screen.findByDisplayValue("test-model");
    expect(screen.getByText("Next: Annotate")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "changed-model" } });
    expect(screen.queryByText("Next: Annotate")).not.toBeInTheDocument();
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
  });

  it("shows stale results and keeps scoring separate from progression", async () => {
    show("score", { complete: false, running: false, stale: true, missing: [], materials: {}, action_label: "Run" });
    expect(screen.getByText("Optional scoring")).toBeInTheDocument();
    expect(screen.getByText(/These results need to be recomputed/)).toBeInTheDocument();
    expect(screen.queryByText("Next step")).not.toBeInTheDocument();
    await screen.findByText("Neutral posture");
  });
});
