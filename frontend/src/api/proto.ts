// Binary-protobuf API client. The Rust trainer serves these routes as
// application/x-protobuf; we decode them with the buf-generated schemas so the
// wire contract is shared and typed end to end.
import { fromBinary } from "@bufbuild/protobuf";
import type { DescMessage, MessageShape } from "@bufbuild/protobuf";
import {
  GameFileSchema,
  RunDetailSchema,
  RunListReplySchema,
} from "../gen/viewer_pb";

async function getProto<Desc extends DescMessage>(
  url: string,
  schema: Desc,
): Promise<MessageShape<Desc>> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  return fromBinary(schema, bytes);
}

export const getRuns = () => getProto("/api/runs", RunListReplySchema);

export const getRunDetail = (runId: string) =>
  getProto(`/api/runs/${encodeURIComponent(runId)}`, RunDetailSchema);

export const getGameFile = (runId: string, gen: number) =>
  getProto(`/api/runs/${encodeURIComponent(runId)}/games/${gen}`, GameFileSchema);
