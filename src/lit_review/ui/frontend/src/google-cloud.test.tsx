import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GoogleCloudSettings } from "./google-cloud";
import { api } from "./api";

vi.mock("./api", async (original) => ({ ...await original<typeof import("./api")>(), api: vi.fn() }));
const mockedApi = vi.mocked(api);
const authUrl = "https://accounts.google.com/o/oauth2/auth?client_id=example";
let status: any;

beforeEach(() => {
  status = { configured: false, enabled: false, project: "", location: "us-central1", saved_project: "", saved_location: "", gcloud_installed: true, adc_file_detected: false, external_credentials: false, login: null };
  mockedApi.mockImplementation(async (path, options) => {
    if (path.endsWith("/code")) {
      status.login = { ...status.login, status: "succeeded", authorization_url: null, message: "Google sign-in complete." };
      return status.login;
    }
    if (path.endsWith("/login")) {
      status.login = { id: "login-1", status: "awaiting_code", authorization_url: authUrl, message: "Open Google sign-in, then paste the authorization code here." };
      return status.login;
    }
    if (path.endsWith("/login/login-1") && options?.method === "DELETE") {
      status.login = { ...status.login, status: "cancelled", authorization_url: null, message: "Google sign-in cancelled." };
      return status.login;
    }
    if (path.endsWith("/check")) return { verified: true, message: "Google credentials verified." };
    if (options?.method === "PUT") {
      const values = JSON.parse(options.body as string);
      status = { ...status, ...values, configured: true, saved_project: values.project, saved_location: values.location };
    }
    return status;
  });
});
afterEach(() => { cleanup(); vi.clearAllMocks(); });

function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
  const view = render(<QueryClientProvider client={client}><GoogleCloudSettings /></QueryClientProvider>);
  return { ...view, client };
}

async function project() {
  const field = await screen.findByLabelText("Google Cloud project");
  fireEvent.change(field, { target: { value: "example-project" } });
}

describe("Google Cloud setup", () => {
  it("saves defaults, completes Google browser sign-in, and clears the code", async () => {
    show();
    await project();
    fireEvent.click(screen.getByRole("button", { name: "Sign in with Google" }));
    const link = await screen.findByRole("link", { name: "Open Google sign-in" });
    expect(link).toHaveAttribute("href", authUrl);
    expect(mockedApi).toHaveBeenCalledWith("/api/settings/google-cloud", expect.objectContaining({ method: "PUT", body: JSON.stringify({ enabled: true, project: "example-project", location: "us-central1" }) }));
    fireEvent.change(screen.getByLabelText("Google authorization code"), { target: { value: "private-auth-code" } });
    fireEvent.click(screen.getByRole("button", { name: "Complete sign-in" }));
    await screen.findByText("Google sign-in complete.");
    expect(mockedApi).toHaveBeenCalledWith("/api/settings/google-cloud/login/login-1/code", expect.objectContaining({ method: "POST", body: JSON.stringify({ code: "private-auth-code" }) }));
    expect(screen.queryByLabelText("Google authorization code")).not.toBeInTheDocument();
    expect(screen.queryByDisplayValue("private-auth-code")).not.toBeInTheDocument();
  });

  it("allows checking existing credentials when gcloud is absent", async () => {
    status.gcloud_installed = false;
    status.project = "environment-project";
    status.location = "europe-west4";
    show();
    await screen.findByDisplayValue("environment-project");
    expect(screen.getByLabelText("Google Cloud region")).toHaveValue("europe-west4");
    expect(screen.getByRole("button", { name: "Sign in with Google" })).toBeDisabled();
    expect(screen.getByRole("link", { name: "Google Cloud CLI" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Check credentials" }));
    await screen.findByText("Google credentials verified.");
  });

  it("cancels a sign-in and supports restarting", async () => {
    show();
    await project();
    fireEvent.click(screen.getByRole("button", { name: "Sign in with Google" }));
    await screen.findByRole("link", { name: "Open Google sign-in" });
    fireEvent.click(screen.getByRole("button", { name: "Cancel sign-in" }));
    await screen.findByText("Google sign-in cancelled.");
    expect(screen.queryByRole("link", { name: "Open Google sign-in" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign in with Google" })).toBeEnabled();
  });

  it("shows login errors without claiming success", async () => {
    const normal = mockedApi.getMockImplementation()!;
    mockedApi.mockImplementation(async (path, options) => {
      if (path.endsWith("/login")) throw new Error("Google Cloud CLI could not start.");
      return normal(path, options);
    });
    show();
    await project();
    fireEvent.click(screen.getByRole("button", { name: "Sign in with Google" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Google Cloud CLI could not start."));
    expect(screen.queryByRole("link", { name: "Open Google sign-in" })).not.toBeInTheDocument();
  });
});
