import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { DownloadProgress, type PDFDownloadProgress } from "./ui";

afterEach(cleanup);
const running:PDFDownloadProgress={selected:3,ready:1,failed:0,unavailable:0,pending:2,processed:1,running:true,status:"running",job_id:"job",complete:false,ready_to_extract:false};
it("keeps progress determinate and counts failed PDFs separately from ready PDFs",()=>{
  render(<DownloadProgress progress={{...running,failed:1,pending:1,processed:2}}/>);
  const bar=screen.getByRole("progressbar",{name:"PDF download progress"});
  expect(bar).toHaveAttribute("value","1");expect(bar).toHaveAttribute("max","3");
  expect(screen.getByText("Downloading PDFs · 1 of 3 ready")).toBeVisible();
  expect(screen.getByText("1 failed")).toBeVisible();
});
it("retains completed progress and identifies readiness for extraction",()=>{
  const {rerender}=render(<DownloadProgress progress={running}/>);
  rerender(<DownloadProgress progress={{...running,ready:3,pending:0,processed:3,running:false,status:"completed",complete:true,ready_to_extract:true}}/>);
  expect(screen.getByText("3 PDFs ready · ready to extract")).toBeVisible();
  expect(screen.getByRole("progressbar")).toHaveAttribute("value","3");
});
it("shows cancelled progress without claiming completion and clears old failures while a retry is queued",()=>{
  const {rerender}=render(<DownloadProgress progress={{...running,running:false,status:"cancelled",failed:1,pending:1}}/>);
  expect(screen.getByText("Downloads paused · 1 of 3 ready")).toBeVisible();
  expect(screen.getByText(/1 failed/)).toBeVisible();
  rerender(<DownloadProgress progress={{...running,running:false,status:"cancelled",failed:1,pending:1}} starting selectedCount={3}/>);
  expect(screen.getByText("Downloading PDFs · 1 of 3 ready")).toBeVisible();
  expect(screen.queryByText(/1 failed/)).not.toBeInTheDocument();
});
it("does not display a false completion for an empty selection",()=>{
  render(<DownloadProgress progress={{...running,selected:0,ready:0,pending:0,processed:0,running:false,status:"completed"}}/>);
  expect(screen.getByText("No papers selected")).toBeVisible();
  expect(screen.getByRole("progressbar")).toHaveAttribute("value","0");
  expect(screen.queryByText(/ready to extract/)).not.toBeInTheDocument();
});
