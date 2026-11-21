import { build } from "esbuild";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const directory = fileURLToPath(new URL(".", import.meta.url));
await build({
  absWorkingDir: directory,
  entryPoints: ["client.jsx"],
  outfile: "client.bundle.js",
  bundle: true,
  format: "iife",
  minify: true,
  target: "es2020",
  define: { "process.env.NODE_ENV": '"production"' },
  legalComments: "inline",
});
const notices = [];
for (const [name, files] of [
  ["react", ["LICENSE"]],
  ["react-dom", ["LICENSE"]],
  ["scheduler", ["LICENSE"]],
  ["lightweight-charts", ["LICENSE"]],
]) {
  for (const file of files) {
    notices.push(
      `${name} — ${file}\n\n${await readFile(new URL(`node_modules/${name}/${file}`, import.meta.url), "utf8")}`,
    );
  }
}
notices.push(
  await readFile(new URL("TRADINGVIEW_NOTICE.txt", import.meta.url), "utf8"),
);
notices.push(
  "fancy-canvas 2.1.0 — MIT (notice absent from npm tarball; upstream tradingview/fancy-canvas LICENSE)\n\n" +
    (await readFile(
      new URL("FANCY_CANVAS_LICENSE.txt", import.meta.url),
      "utf8",
    )),
);
await writeFile(
  new URL("THIRD_PARTY_LICENSES.txt", import.meta.url),
  notices.join("\n\n-----\n\n").trimEnd() + "\n",
);
