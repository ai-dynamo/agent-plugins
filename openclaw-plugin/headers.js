// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const DYNAMO_SESSION_HEADER = "x-dynamo-session-id";
export const DYNAMO_PARENT_SESSION_HEADER = "x-dynamo-parent-session-id";

function hasHeader(headers, name) {
  const lowerName = name.toLowerCase();
  return Object.keys(headers).some((key) => key.toLowerCase() === lowerName);
}

export function withDynamoSessionHeaders(options, parentSessionId) {
  const sessionId = options?.sessionId?.trim();
  if (!sessionId) return options;

  const headers = { ...options.headers };
  if (!hasHeader(headers, DYNAMO_SESSION_HEADER)) {
    headers[DYNAMO_SESSION_HEADER] = sessionId;
  }
  if (
    parentSessionId &&
    parentSessionId !== sessionId &&
    !hasHeader(headers, DYNAMO_PARENT_SESSION_HEADER)
  ) {
    headers[DYNAMO_PARENT_SESSION_HEADER] = parentSessionId;
  }
  return { ...options, headers };
}

export function wrapDynamoStreamFn(streamFn, resolveParentSessionId) {
  if (!streamFn) return undefined;
  return (model, context, options) => {
    const sessionId = options?.sessionId?.trim();
    return streamFn(
      model,
      context,
      withDynamoSessionHeaders(options, sessionId && resolveParentSessionId?.(sessionId)),
    );
  };
}
