"use client";

import * as React from "react";
import { useAppStore } from "@/lib/store";
import { useChat } from "@/hooks/use-chat";
import { useMounted } from "@/hooks/use-mounted";
import { ChatHeader } from "@/components/chat/chat-header";
import { MessageList } from "@/components/chat/message-list";
import { EmptyState } from "@/components/chat/empty-state";
import { Composer } from "@/components/chat/composer";
import { ConnectionError } from "@/components/chat/connection-error";
import { SettingsPanel } from "@/components/settings/settings-panel";

export default function Home() {
  const mounted = useMounted();
  const [settingsOpen, setSettingsOpen] = React.useState(false);

  const effort = useAppStore((s) => s.effort);
  const setEffort = useAppStore((s) => s.setEffort);
  const clearConversation = useAppStore((s) => s.clearConversation);

  const { messages, status, error, send, stop, retryLast } = useChat();

  if (!mounted) return null;

  return (
    <div className="flex min-h-dvh flex-col">
      <ChatHeader
        effort={effort}
        onEffortChange={setEffort}
        onNewConversation={clearConversation}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <main className="flex flex-1 flex-col">
        {messages.length === 0 ? (
          <EmptyState onPick={send} />
        ) : (
          <div className="flex-1">
            <MessageList messages={messages} />
          </div>
        )}
      </main>

      <div className="flex flex-col gap-3">
        {status === "error" && error && (
          <ConnectionError message={error} onRetry={retryLast} />
        )}
        <Composer onSend={send} onStop={stop} streaming={status === "streaming"} />
      </div>

      <SettingsPanel open={settingsOpen} onOpenChange={setSettingsOpen} />
    </div>
  );
}
