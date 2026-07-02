import type { Config } from "tailwindcss";

// Color names map to the design tokens in src/lib/palette.ts — keep in sync.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        page: "#0d0d0d",
        surface: "#1a1a19",
        inset: "#141413",
        ink: {
          DEFAULT: "#ffffff",
          2: "#c3c2b7",
          3: "#898781",
        },
        grid: "#2c2c2a",
        axis: "#383835",
        accent: "#3987e5",
        good: "#0ca30c",
        warn: "#fab219",
        bad: "#e66767",
      },
    },
  },
  plugins: [],
} satisfies Config;
