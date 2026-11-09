import { fireEvent, render, screen, waitFor, cleanup, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, type WorkflowStep } from "./api";
import { ProcessingStage } from "./ui";

vi.mock("./api", async (importOriginal) => {
  const original = await importOriginal<typeof import("./api")>();
  return { ...original, api: vi.fn() };
});
const mockedApi = vi.mocked(api);
const idle: WorkflowStep = { complete: false, running: false, stale: false, missing: [], materials: {}, action_label: "Run" };

beforeEach(() => {
  mockedApi.mockImplementation(async (path) => {
    if (path.endsWith("/config")) return { provider: "gemini", model: "test-model", batch_features: 25 } as any;
    if (path.includes("/results/")) return { total: 1, items: [{ feature_name: "Neutral posture", source_paper_ids: ["paper-1"] }] } as any;
    if (path.endsWith("/preview")) return { dry_run: true, stages: [] } as any;
    if (path.endsWith("/run")) return { id: "job-1" } as any;
    return { jobs: [] } as any;
  });
});
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.unstubAllGlobals(); });

function show(stage: "filter" | "score", status: WorkflowStep = idle, options: { automatic?: boolean; onNext?: () => void } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
  const result = render(<QueryClientProvider client={client}><ProcessingStage projectId="p1" stage={stage} status={status} {...options} /></QueryClientProvider>);
  return { ...result, client };
}
function openSettings() {
  const summary = screen.getByText("Settings and detailed records");
  fireEvent.click(summary);
  return summary.closest("details")!;
}

describe("ProcessingStage", () => {
  it("keeps settings collapsed while allowing rejection and evidence inspection", async () => {
    show("filter");
    expect(await screen.findByRole("region", { name: "Concepts" })).toHaveTextContent("Neutral posture");
    expect(screen.getByText("Settings and detailed records").closest("details")).not.toHaveAttribute("open");
    expect(screen.getByLabelText("Model")).not.toBeVisible();
    openSettings();
    const records = screen.getByRole("heading", { name: "Inspect results" }).closest<HTMLElement>(".panel")!;
    fireEvent.click(within(records).getByText("Neutral posture"));
    expect(within(records).getByText("Source papers: paper-1")).toBeVisible();
    fireEvent.change(screen.getByRole("combobox", { name: "Stage artifact" }), { target: { value: "rejected_features.jsonl" } });
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/stages/filter/results/rejected_features.jsonl"));
    expect(screen.getByRole("link", { name: "Export" })).toHaveAttribute("href", "/api/projects/p1/stages/filter/exports/rejected_features.jsonl");
  });

  it("saves settings before previewing and running", async () => {
    show("filter");
    openSettings();
    await screen.findByDisplayValue("test-model");
    fireEvent.click(screen.getByRole("button", { name: "Preview settings" }));
    expect(await screen.findByText(/Planned inputs and outputs/)).toBeVisible();
    const saved = mockedApi.mock.calls.findIndex(([path, options]) => path.endsWith("/config") && options?.method === "PUT");
    const preview = mockedApi.mock.calls.findIndex(([path]) => path.endsWith("/preview"));
    expect(saved).toBeGreaterThanOrEqual(0);
    expect(preview).toBeGreaterThan(saved);
    fireEvent.click(screen.getByRole("button", { name: "Run filter" }));
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/stages/filter/run", { method: "POST" }));
  });

  it("saves Google Cloud authentication settings with the processing stage", async () => {
    show("filter");
    openSettings();
    await screen.findByDisplayValue("test-model");
    fireEvent.change(screen.getByLabelText(/Gemini authentication/), { target: { value: "adc" } });
    fireEvent.change(screen.getByLabelText(/Google Cloud project/), { target: { value: "my-project" } });
    fireEvent.change(screen.getByLabelText(/Google Cloud region/), { target: { value: "europe-west4" } });
    fireEvent.click(screen.getByRole("button", { name: "Run filter" }));
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/stages/filter/run", { method: "POST" }));
    const saved = mockedApi.mock.calls.find(([path, options]) => path.endsWith("/config") && options?.method === "PUT");
    expect(JSON.parse(saved![1]!.body as string)).toMatchObject({ auth_mode: "adc", vertex_project: "my-project", vertex_location: "europe-west4" });
  });

  it("requires rerunning changed settings before offering the next step", async () => {
    show("filter", { ...idle, complete: true }, { onNext: vi.fn() });
    await screen.findByDisplayValue("test-model");
    expect(screen.getByRole("button", { name: "Review equivalent concepts" })).toBeVisible();
    openSettings();
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "changed-model" } });
    expect(screen.queryByRole("button", { name: "Review equivalent concepts" })).not.toBeInTheDocument();
    expect(screen.getByText("Unsaved changes")).toBeVisible();
  });

  it("shows stale results and keeps scoring separate from progression", async () => {
    show("score", { ...idle, stale: true });
    expect(screen.getByText("Optional scoring")).toBeVisible();
    expect(screen.getByText(/These results need to be recomputed/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Continue" })).not.toBeInTheDocument();
    expect(await screen.findByRole("region", { name: "Concepts" })).toHaveTextContent("Neutral posture");
  });

  it("starts combined refinement only after the explicit action", async () => {
    show("filter", idle, { automatic: true });
    const action = await screen.findByRole("button", { name: "Refine and add descriptions" });
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/stages/annotate/config"));
    expect(mockedApi.mock.calls.some(([path, options]) => path.endsWith("/run") && options?.method === "POST")).toBe(false);
    fireEvent.click(action);
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/refine/run", { method: "POST" }));
    expect(mockedApi.mock.calls.some(([path]) => path === "/api/projects/p1/stages/filter/run" || path === "/api/projects/p1/stages/annotate/run")).toBe(false);
    const saveIndex = mockedApi.mock.calls.findIndex(([path, options]) => path === "/api/projects/p1/stages/filter/config" && options?.method === "PUT");
    const runIndex = mockedApi.mock.calls.findIndex(([path]) => path === "/api/projects/p1/refine/run");
    expect(saveIndex).toBeGreaterThanOrEqual(0);
    expect(runIndex).toBeGreaterThan(saveIndex);
  });

  it("requires explicitly saving edited description settings before continuing", async () => {
    show("filter", { ...idle, complete: true }, { automatic: true, onNext: vi.fn() });
    expect(await screen.findByRole("button", { name: "Review equivalent concepts" })).toBeEnabled();
    openSettings();
    const descriptionSettings = screen.getByText("Description settings").closest("details")!;
    await waitFor(() => expect(within(descriptionSettings).getByLabelText("Model")).toHaveValue("test-model"));
    fireEvent.change(within(descriptionSettings).getByLabelText("Model"), { target: { value: "new-description-model" } });
    expect(await screen.findByText("Save the description settings below before continuing.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Review equivalent concepts" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refine and add descriptions" })).toBeDisabled();
    fireEvent.click(within(descriptionSettings).getByRole("button", { name: "Save settings" }));
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/stages/annotate/config", expect.objectContaining({ method: "PUT", body: expect.stringContaining("new-description-model") })));
    await waitFor(() => expect(screen.queryByText("Save the description settings below before continuing.")).not.toBeInTheDocument());
    expect(mockedApi.mock.calls.some(([path, options]) => path.endsWith("/run") && options?.method === "POST")).toBe(false);
  });

  it("does not substitute retained counts for a measured empty annotation inventory", async () => {
    const defaultApi = mockedApi.getMockImplementation()!;
    mockedApi.mockImplementation(async (path, options) => {
      if (path.endsWith("/filter/results/filtered_features.jsonl")) return {
        total: 14, items: Array.from({ length: 14 }, (_, index) => ({ feature_name: `Retained ${index + 1}` })),
      } as any;
      if (path.endsWith("/annotate/results/annotated_features.jsonl")) return { total: 0, items: [] } as any;
      return defaultApi(path, options);
    });
    const { client } = show("filter", { ...idle, complete: true }, { automatic: true });
    await screen.findByText("14 records. Open a record to inspect its evidence and decisions.");
    await waitFor(() => expect(client.getQueryData(["automaticDescriptions", "p1"])).toEqual({ total: 0, items: [] }));
    expect(screen.queryByRole("region", { name: "Concepts" })).not.toBeInTheDocument();
    expect(screen.queryByText("14 concepts")).not.toBeInTheDocument();
  });

  it("shows and cancels a legacy annotation job without starting new refinement", async () => {
    const events = { addEventListener: vi.fn(), close: vi.fn() };
    const eventSource = vi.fn(function (_url: string) { return events; });
    vi.stubGlobal("EventSource", eventSource);
    const defaultApi = mockedApi.getMockImplementation()!;
    mockedApi.mockImplementation(async (path, options) => {
      if (path === "/api/projects/p1") return { jobs: [
        { id: "finished-filter", kind: "filter", status: "completed" },
        { id: "legacy-description", kind: "annotate", status: "running" },
      ] } as any;
      if (path.endsWith("/cancel")) return { status: "cancelling" } as any;
      return defaultApi(path, options);
    });
    show("filter", { ...idle, running: true, materials: { refinement_phase: "annotate" } }, { automatic: true });
    const stop = await screen.findByRole("button", { name: "Stop" });
    expect(stop).toBeEnabled();
    expect(screen.getByText("Adding definitions and examples")).toBeVisible();
    expect(eventSource).toHaveBeenCalledWith("/api/jobs/legacy-description/events");
    expect(mockedApi.mock.calls.some(([path, options]) => path.endsWith("/run") && options?.method === "POST")).toBe(false);
    fireEvent.click(stop);
    await waitFor(() => expect(mockedApi).toHaveBeenCalledWith("/api/jobs/legacy-description/cancel", { method: "POST" }));
    expect(mockedApi.mock.calls.filter(([, options]) => options?.method === "POST").map(([path]) => path)).toEqual(["/api/jobs/legacy-description/cancel"]);
  });

});
