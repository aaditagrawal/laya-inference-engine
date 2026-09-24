import { chromium } from "playwright";
import { createServer } from "vite";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const output = fileURLToPath(new URL("../assets", import.meta.url));
const server = await createServer({root, server: {host: "127.0.0.1", port: 0}});
await server.listen();
const browser = await chromium.launch();
const errors: string[] = [];
try {
  await mkdir(output, {recursive: true});
  const page = await browser.newPage({viewport: {width: 1000, height: 2200}, deviceScaleFactor: 2, reducedMotion: "reduce"});
  page.on("pageerror", error => errors.push(error.message));
  const address = server.httpServer!.address();
  if (!address || typeof address === "string") throw new Error("No local chart server address");
  for (const theme of ["light", "dark"]) {
    await page.goto(`http://127.0.0.1:${address.port}/?theme=${theme}`);
    await page.waitForFunction(() => document.querySelectorAll("canvas[data-ready='true']").length === 22);
    await page.evaluate(() => document.fonts.ready);
    if (errors.length) throw new Error(errors.join("\n"));
    for (const chart of ["performance-overview", "public-modes", "warm-latency", "latest-paired", "experimental-gains", "startup"]) {
      const path = `${output}/${chart}-${theme}.png`;
      await page.locator(`#${chart}`).screenshot({path});
      console.log(path);
    }
    if (theme === "light") {
      const values = await page.evaluate(() => Reflect.get(window, "chartData"));
      const chartSources: Record<string, string[]> = await page.evaluate(() => Reflect.get(window, "chartSources"));
      const sources = Object.fromEntries(await Promise.all([...new Set(Object.values(chartSources).flat())].map(async path => [
        path, createHash("sha256").update(await readFile(new URL(`../../${path}`, import.meta.url))).digest("hex"),
      ])));
      await writeFile(`${output}/chart-data.json`, `${JSON.stringify({...values, provenance: {chartSources, sha256: sources}}, null, 2)}\n`);
    }
  }
} finally {
  await browser.close();
  await server.close();
}
