import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built into ../static, which the FastAPI dashboard serves. `base: "./"` makes
// asset URLs relative so they resolve under the served root.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../static", emptyOutDir: true },
  // `npm run dev` (HMR) proxies the JSON API to the running dashboard server.
  server: { proxy: { "/api": "http://127.0.0.1:8050" } },
});
