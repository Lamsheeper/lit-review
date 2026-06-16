import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Progress, StepHeader } from "./ui";

describe("Progress", () => {
  it("shows completed workflow checkpoints", () => {
    const { container } = render(
      <Progress project={{
        id: "p1",
        name: "Review",
        description: "",
        updated_at: "",
        progress: { defined: true, taxonomy: true, searched: true, downloaded: 2, extracted: false },
      }} />,
    );
    expect(container.querySelectorAll(".bg-blue-500")).toHaveLength(4);
    expect(container.querySelectorAll(".bg-slate-200")).toHaveLength(1);
  });
});

describe("StepHeader", () => {
  it("reveals Next only when the step materials are ready", () => {
    const onNext = vi.fn();
    const status = { complete: false, running: false, missing: ["ranked paper candidates"], materials: {}, action_label: "Find and rank papers" };
    const { rerender } = render(<StepHeader eyebrow="Step 2" title="Collect" description="Collect papers" status={status} action={<button>Find and rank papers</button>} onNext={onNext} />);
    expect(screen.queryByText("Next step")).not.toBeInTheDocument();
    expect(screen.getByText(/Before proceeding/)).toHaveTextContent("ranked paper candidates");

    rerender(<StepHeader eyebrow="Step 2" title="Collect" description="Collect papers" status={{ ...status, complete: true, running: true, missing: [] }} action={<button>Re-scoring</button>} onNext={onNext} />);
    expect(screen.queryByText("Next step")).not.toBeInTheDocument();
    expect(screen.getByText(/This step is running/)).toBeInTheDocument();

    rerender(<StepHeader eyebrow="Step 2" title="Collect" description="Collect papers" status={{ ...status, complete: true, missing: [] }} action={<button>Re-score cached candidates</button>} onNext={onNext} />);
    fireEvent.click(screen.getByText("Next step"));
    expect(onNext).toHaveBeenCalledOnce();
  });
});
