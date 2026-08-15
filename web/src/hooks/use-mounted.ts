"use client";

import { useSyncExternalStore } from "react";

const subscribe = () => () => {};

/** True only after the client has hydrated — avoids SSR/CSR mismatches without setState-in-effect. */
export function useMounted(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
