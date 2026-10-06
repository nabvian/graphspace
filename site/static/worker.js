// Runs the unmodified graphspace package in Pyodide (self-hosted), off the main thread.
const base = new URL("./", self.location.href);
importScripts(new URL("pyodide/pyodide.js", base).href);
const status = (text) => postMessage({ type: "status", text });

const ready = (async () => {
  status("Loading Python (Pyodide)…");
  const pyodide = await loadPyodide({ indexURL: new URL("pyodide/", base).href });
  status("Loading graphspace…");
  const archive = await fetch(new URL("graphspace.zip", base));
  if (!archive.ok) throw new Error(`graphspace.zip: HTTP ${archive.status}`);
  pyodide.unpackArchive(await archive.arrayBuffer(), "zip", { extractDir: "/home/pyodide/site" });
  pyodide.runPython("import sys, json\nsys.path.insert(0, '/home/pyodide/site')\nimport gs_web, graphspace");
  return pyodide;
})();
ready.then((py) => postMessage({ type: "ready", version: py.runPython("graphspace.__version__"), python: py.runPython("sys.version.split()[0]") }),
  (error) => postMessage({ type: "failed", error: String(error?.message ?? error) }));

self.onmessage = async ({ data }) => {
  const { id, code } = data;
  try {
    const pyodide = await ready;
    pyodide.globals.set("_playground_code", code);
    const report = pyodide.runPython("json.dumps(gs_web.run(_playground_code), default=str)");
    postMessage({ type: "result", id, ok: true, report: JSON.parse(report) });
  } catch (error) {
    postMessage({ type: "result", id, ok: false, error: String(error?.message ?? error).trim().split("\n").at(-1) });
  }
};
