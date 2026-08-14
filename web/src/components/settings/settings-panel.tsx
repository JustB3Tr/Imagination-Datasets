"use client";

import * as React from "react";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { Sheet } from "@/components/ui/sheet";
import { Input, Label } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useAppStore } from "@/lib/store";
import type { OllamaSettings } from "@/lib/types";

interface SettingsPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SettingsPanel({ open, onOpenChange }: SettingsPanelProps) {
  const settings = useAppStore((s) => s.settings);
  const setSettings = useAppStore((s) => s.setSettings);

  return (
    <Sheet
      open={open}
      onOpenChange={onOpenChange}
      title="Settings"
      description="Connection details for your Ollama instance."
    >
      {/* Remount on every open so local form state always starts from the latest saved settings. */}
      <SettingsForm
        key={open ? "open" : "closed"}
        settings={settings}
        onSave={(next) => {
          setSettings(next);
          onOpenChange(false);
        }}
        onCancel={() => onOpenChange(false)}
      />
    </Sheet>
  );
}

type CheckState = "idle" | "checking" | "ok" | "fail";

function SettingsForm({
  settings,
  onSave,
  onCancel,
}: {
  settings: OllamaSettings;
  onSave: (settings: OllamaSettings) => void;
  onCancel: () => void;
}) {
  const [baseUrl, setBaseUrl] = React.useState(settings.baseUrl);
  const [model, setModel] = React.useState(settings.model);
  const [authToken, setAuthToken] = React.useState(settings.authToken);
  const [check, setCheck] = React.useState<CheckState>("idle");
  const [checkDetail, setCheckDetail] = React.useState("");

  const testConnection = async () => {
    setCheck("checking");
    try {
      const params = new URLSearchParams({ baseUrl: baseUrl.trim(), authToken: authToken.trim() });
      const res = await fetch(`/api/health?${params.toString()}`);
      const data = await res.json();
      if (data.ok) {
        setCheck("ok");
        const found = Array.isArray(data.models) && data.models.includes(model.trim());
        setCheckDetail(
          found
            ? `Connected. "${model.trim()}" is available.`
            : `Connected, but "${model.trim()}" wasn't listed. It may still work if the tag is correct.`,
        );
      } else {
        setCheck("fail");
        setCheckDetail(data.error || "Unknown error.");
      }
    } catch (err) {
      setCheck("fail");
      setCheckDetail(err instanceof Error ? err.message : "Unknown error.");
    }
  };

  return (
    <div className="flex flex-col gap-5">
      <div>
        <Label htmlFor="baseUrl">Ollama base URL</Label>
        <Input
          id="baseUrl"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          placeholder="http://localhost:11434"
          spellCheck={false}
        />
        <p className="mt-1.5 text-[11px] text-muted-foreground">
          Update this whenever your cloudflared tunnel URL rotates — no redeploy needed.
        </p>
      </div>

      <div>
        <Label htmlFor="model">Model tag</Label>
        <Input
          id="model"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="imagination-2.1-pro-lmh"
          spellCheck={false}
        />
      </div>

      <div>
        <Label htmlFor="authToken">Auth token (optional)</Label>
        <Input
          id="authToken"
          type="password"
          value={authToken}
          onChange={(e) => setAuthToken(e.target.value)}
          placeholder="Token, with or without a leading 'Bearer '"
          spellCheck={false}
        />
      </div>

      <div className="flex flex-col gap-2">
        <Button type="button" variant="outline" onClick={testConnection} disabled={check === "checking"}>
          {check === "checking" && <Loader2 className="size-3.5 animate-spin" />}
          Test connection
        </Button>
        {check === "ok" && (
          <p className="flex items-start gap-1.5 text-xs text-foreground">
            <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-accent" />
            {checkDetail}
          </p>
        )}
        {check === "fail" && (
          <p className="flex items-start gap-1.5 text-xs text-destructive">
            <XCircle className="mt-0.5 size-3.5 shrink-0" />
            {checkDetail}
          </p>
        )}
      </div>

      <div className="mt-2 flex justify-end gap-2 border-t border-border pt-4">
        <Button type="button" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button
          type="button"
          onClick={() => onSave({ baseUrl: baseUrl.trim(), model: model.trim(), authToken: authToken.trim() })}
        >
          Save
        </Button>
      </div>
    </div>
  );
}
