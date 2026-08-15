import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Mode, EffortLevel } from "@/lib/prompt";
import { DEFAULT_SETTINGS, type ChatMessage, type OllamaSettings } from "@/lib/types";

interface AppStore {
  settings: OllamaSettings;
  effort: EffortLevel;
  mode: Mode;
  messages: ChatMessage[];

  setSettings: (settings: Partial<OllamaSettings>) => void;
  setEffort: (effort: EffortLevel) => void;
  setMode: (mode: Mode) => void;

  addMessage: (message: ChatMessage) => void;
  updateMessage: (id: string, patch: Partial<ChatMessage>) => void;
  appendToMessage: (id: string, contentDelta: string) => void;
  removeMessage: (id: string) => void;
  clearConversation: () => void;
}

export const useAppStore = create<AppStore>()(
  persist(
    (set) => ({
      settings: DEFAULT_SETTINGS,
      effort: "medium",
      mode: "direct",
      messages: [],

      setSettings: (settings) =>
        set((state) => ({ settings: { ...state.settings, ...settings } })),

      setEffort: (effort) => set({ effort }),
      setMode: (mode) => set({ mode }),

      addMessage: (message) =>
        set((state) => ({ messages: [...state.messages, message] })),

      updateMessage: (id, patch) =>
        set((state) => ({
          messages: state.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)),
        })),

      appendToMessage: (id, contentDelta) =>
        set((state) => ({
          messages: state.messages.map((m) =>
            m.id === id ? { ...m, content: m.content + contentDelta } : m,
          ),
        })),

      removeMessage: (id) =>
        set((state) => ({ messages: state.messages.filter((m) => m.id !== id) })),

      clearConversation: () => set({ messages: [] }),
    }),
    {
      name: "imagination-ui-store",
      partialize: (state) => ({
        settings: state.settings,
        effort: state.effort,
        mode: state.mode,
        // Strip transient in-flight state — a message frozen at streaming:
        // true from a reload/crash mid-stream would otherwise show a
        // permanent loading cursor with no request actually running.
        messages: state.messages.map((m) => ({ ...m, streaming: undefined })),
      }),
    },
  ),
);
