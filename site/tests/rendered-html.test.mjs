import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(pathname) {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}-${pathname}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request(`http://localhost${pathname}`, { headers: { accept: "text/html" } }), { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } }, { waitUntil() {}, passThroughOnException() {} });
}

test("server-renders the Liquid Warp product page", async () => {
  const response = await render("/");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Liquid Warp/);
  assert.match(html, /GPU liquid\. Live in Blender/);
  assert.match(html, /ACTUAL ADD-ON OUTPUT/);
  assert.match(html, /Optional NPZ frame cache/);
  assert.match(html, /Blender 5\.1/);
  assert.match(html, /NVIDIA GPU with CUDA/);
  assert.match(html, /crown-splash-preview\.png/);
  assert.doesNotMatch(html, /Checkout coming soon/);
  assert.match(html, /View real demos/);
  assert.match(html, /Actual add-on output/);
  assert.match(html, /FluidSim Warp for Blender 5\.1/);
  assert.doesNotMatch(html, /perpetual Liquid Warp license/i);
});

test("server-renders buyer documentation", async () => {
  const response = await render("/docs");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Get Liquid Warp running/);
  assert.match(html, /First live simulation/);
  assert.match(html, /Export Current Frame/);
  assert.match(html, /warp-lang/);
  assert.match(html, /On this page/);
  assert.match(html, /id="troubleshooting"/);
  assert.match(html, /NPZ frame files/);
  assert.match(html, /Initialize Solver/);
  assert.match(html, /Live: Off/);
});

test("keeps checkout details centralized and explicitly unconfigured", async () => {
  const config = await readFile(new URL("../app/site-config.ts", import.meta.url), "utf8");
  assert.match(config, /priceLabel: "Set launch price"/);
  assert.match(config, /checkoutUrl: ""/);
});
