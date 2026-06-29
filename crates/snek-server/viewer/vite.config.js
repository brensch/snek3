import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The viewer is embedded into the snek-server binary and served under /app, so
// `base: "/app/"` makes asset URLs resolve from there. `outDir: dist` is the
// folder rust-embed bundles. `npm run dev` proxies the /viewer JSON API to a
// locally running snek-server (set SNEK_DEV_API to point elsewhere).
const apiTarget = process.env.SNEK_DEV_API || "http://127.0.0.1:8141";

export default defineConfig({
  plugins: [react()],
  base: "/app/",
  build: { outDir: "dist", emptyOutDir: true },
  server: { proxy: { "/viewer": apiTarget } },
});
