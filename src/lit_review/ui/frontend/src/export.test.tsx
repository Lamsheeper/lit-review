import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { InventoryExport } from "./export";

vi.mock("./api", async importOriginal => {
  const original = await importOriginal<typeof import("./api")>();
  return { ...original, api: vi.fn() };
});
const mockedApi = vi.mocked(api);
const prepared = {
  filename: "llm_merged_features.csv",
  path: "/Users/researcher/projects/persuasion/exports/llm_merged_features.csv",
  download_url: "/api/projects/p1/exports/prepared/llm_merged_features.csv",
};
let nativeDownload: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  mockedApi.mockResolvedValue(prepared);
  nativeDownload = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
});
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("does no export work on mount and confirms the actual server path after preparing CSV", async () => {
  render(<InventoryExport projectId="p1" inventory="merge" />);
  expect(mockedApi).not.toHaveBeenCalled();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));
  const dialog = await screen.findByRole("dialog", { name: "CSV export complete" });
  expect(mockedApi).toHaveBeenCalledWith("/api/projects/p1/exports/csv", {
    method: "POST", body: JSON.stringify({ inventory: "merge" }),
  });
  expect(dialog).toHaveTextContent(prepared.path);
  expect(dialog).toHaveTextContent("A browser download has also been requested.");
  expect(dialog).toHaveTextContent("Your browser’s settings determine its Downloads folder.");
  expect(nativeDownload).toHaveBeenCalledOnce();
  const anchor = nativeDownload.mock.instances[0] as HTMLAnchorElement;
  expect(anchor.getAttribute("href")).toBe(prepared.download_url);
  expect(anchor.download).toBe(prepared.filename);
  expect(within(dialog).getByRole("link", { name: "Download again" })).toHaveAttribute("href", prepared.download_url);
  expect(dialog).toHaveFocus();
});

it("keeps a pending preparation busy and shows no completion before the server responds", async () => {
  let finish: ((value: typeof prepared) => void) | undefined;
  mockedApi.mockReturnValue(new Promise(resolve => { finish = resolve; }));
  render(<InventoryExport projectId="p1" inventory="filter" />);
  fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));
  expect(screen.getByRole("button", { name: "Preparing CSV…" })).toBeDisabled();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(nativeDownload).not.toHaveBeenCalled();
  finish!(prepared);
  expect(await screen.findByRole("dialog", { name: "CSV export complete" })).toBeVisible();
});

it("traps keyboard focus, closes with Escape, and returns focus to the export action", async () => {
  render(<InventoryExport projectId="p1" inventory="merge" />);
  const action = screen.getByRole("button", { name: "Download CSV" });
  fireEvent.click(action);
  const dialog = await screen.findByRole("dialog");
  expect(dialog).toHaveFocus();
  fireEvent.keyDown(document, { key: "Tab" });
  const close = within(dialog).getByRole("button", { name: "Close export confirmation" });
  expect(close).toHaveFocus();
  fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
  expect(within(dialog).getByRole("button", { name: "Done" })).toHaveFocus();
  fireEvent.keyDown(document, { key: "Tab" });
  expect(close).toHaveFocus();
  fireEvent.keyDown(document, { key: "Escape" });
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(action).toHaveFocus();
  expect(document.body.style.overflow).toBe("");
});

it("copies the actual prepared path and also closes with Done", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { clipboard: { writeText } });
  render(<InventoryExport projectId="p1" inventory="merge" />);
  fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));
  const dialog = await screen.findByRole("dialog");
  fireEvent.click(within(dialog).getByRole("button", { name: "Copy path" }));
  await waitFor(() => expect(writeText).toHaveBeenCalledWith(prepared.path));
  expect(await within(dialog).findByRole("status")).toHaveTextContent("Path copied.");
  fireEvent.click(within(dialog).getByRole("button", { name: "Done" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("reports failed preparation without claiming completion or triggering a native download", async () => {
  mockedApi.mockRejectedValue(new Error("This inventory has no CSV yet"));
  render(<InventoryExport projectId="p1" inventory="merge" />);
  fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("This inventory has no CSV yet");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(nativeDownload).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Download CSV" })).toBeEnabled();
});

it("rejects an unexpected remote download location instead of handing the CSV to another host", async () => {
  mockedApi.mockResolvedValue({ ...prepared, download_url: "https://example.org/download.csv" });
  render(<InventoryExport projectId="p1" inventory="merge" />);
  fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("download location could not be confirmed");
  expect(nativeDownload).not.toHaveBeenCalled();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});
