// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { wrapDynamoStreamFn } from "./headers.js";

export default {
  id: "dynamo",
  name: "Dynamo Provider",
  description: "Send OpenClaw session identity to NVIDIA Dynamo.",
  register(api) {
    const wrapStreamFn = ({ streamFn }) => wrapDynamoStreamFn(streamFn);
    api.registerProvider({
      id: "dynamo",
      label: "Dynamo",
      wrapStreamFn,
      wrapSimpleCompletionStreamFn: wrapStreamFn,
    });
  },
};
