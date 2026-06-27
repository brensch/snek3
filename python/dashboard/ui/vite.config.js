import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built into ../static, which the FastAPI dashboard serves. `base: "/"` makes
// asset URLs absolute (/assets/...) so they resolve from any page depth,
// including deep links like /run/<name> (assets are mounted at /assets).
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: { outDir: "../static", emptyOutDir: true },
  // `npm run dev` (HMR) proxies the JSON API to the running dashboard server.
  server: { proxy: { "/api": "http://127.0.0.1:8050" } },
});
