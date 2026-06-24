// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { wrapDynamoStreamFn } from "./headers.js";

export default {
  id: "dynamo",
  name: "Dynamo Provider",
  description: "Send OpenClaw session identity to NVIDIA Dynamo.",
  register(api) {
    const parentSessionIds = new Map();
    const rememberParentSession = (sessionId, sessionKey) => {
      const session = api.runtime.agent.session;
      const entry = session.getSessionEntry({ sessionKey });
      const parentKey = entry?.parentSessionKey?.trim() || entry?.spawnedBy?.trim();
      const parentSessionId = parentKey
        ? session.getSessionEntry({ sessionKey: parentKey })?.sessionId?.trim()
        : undefined;
      parentSessionIds.set(
        sessionId,
        parentSessionId && parentSessionId !== sessionId ? parentSessionId : undefined,
      );
      return parentSessionIds.get(sessionId);
    };

    api.on("before_model_resolve", (_event, context) => {
      const sessionId = context.sessionId?.trim();
      const sessionKey = context.sessionKey?.trim();
      if (sessionId && sessionKey) rememberParentSession(sessionId, sessionKey);
    });
    api.on("session_end", (event) => {
      const sessionId = event.sessionId.trim();
      const nextSessionId = event.nextSessionId?.trim();
      if (nextSessionId && parentSessionIds.has(sessionId)) {
        parentSessionIds.set(nextSessionId, parentSessionIds.get(sessionId));
      }
      parentSessionIds.delete(sessionId);
    });

    const wrapStreamFn = ({ streamFn, agentId }) =>
      wrapDynamoStreamFn(streamFn, (sessionId) => {
        if (parentSessionIds.has(sessionId)) return parentSessionIds.get(sessionId);
        const match = api.runtime.agent.session
          .listSessionEntries({ agentId })
          .find(({ entry }) => entry.sessionId === sessionId);
        return match ? rememberParentSession(sessionId, match.sessionKey) : undefined;
      });
    api.registerProvider({
      id: "dynamo",
      label: "Dynamo",
      wrapStreamFn,
      wrapSimpleCompletionStreamFn: wrapStreamFn,
    });
  },
};
