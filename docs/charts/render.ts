import { chromium } from "playwright";
import { createServer } from "vite";
import { mkdir, writeFile } from "node:fs/promises";
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
    await page.waitForFunction(() => document.querySelectorAll("canvas[data-ready='true']").length === 13);
    await page.evaluate(() => document.fonts.ready);
    if (errors.length) throw new Error(errors.join("\n"));
    for (const chart of ["warm-latency", "experimental-gains", "startup"]) {
      const path = `${output}/${chart}-${theme}.png`;
      await page.locator(`#${chart}`).screenshot({path});
      console.log(path);
    }
    if (theme === "light") {
      const values = await page.evaluate(() => Reflect.get(window, "chartData"));
      await writeFile(`${output}/chart-data.json`, `${JSON.stringify(values, null, 2)}\n`);
    }
  }
} finally {
  await browser.close();
  await server.close();
}
