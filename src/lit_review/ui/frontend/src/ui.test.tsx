import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Progress, StepHeader, nextWorkflowStep } from "./ui";
import type { Project, Workflow } from "./api";

afterEach(cleanup);
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
