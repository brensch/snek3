import { useEffect, useRef } from "react";
import type { GameSnapshot, Point } from "../types";

const colors = ["#38bdf8", "#22c55e", "#f59e0b", "#ec4899", "#a78bfa", "#f87171"];

type Props = { game: GameSnapshot };

export function BoardCanvas({ game }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    draw(canvas, game);
  }, [game]);
  return <canvas ref={ref} width={240} height={240} className="aspect-square w-full rounded border border-slate-800 bg-slate-950" />;
}

function draw(canvas: HTMLCanvasElement, game: GameSnapshot) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const size = canvas.width;
  const cell = Math.floor(size / Math.max(game.board_w, game.board_h, 1));
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, size, size);
  drawGrid(ctx, game, cell);
  game.food.forEach((p) => drawDot(ctx, game, cell, p, "#ef4444"));
  game.snakes.forEach((snake, si) => {
    ctx.globalAlpha = snake.alive ? 1 : 0.25;
    snake.body.forEach((p, bi) => drawCell(ctx, game, cell, p, colors[si % colors.length], bi === 0 ? String(si) : ""));
    ctx.globalAlpha = 1;
  });
}

function drawGrid(ctx: CanvasRenderingContext2D, game: GameSnapshot, cell: number) {
  ctx.strokeStyle = "#1e293b";
  for (let i = 0; i <= game.board_w; i++) line(ctx, i * cell, 0, i * cell, game.board_h * cell);
  for (let i = 0; i <= game.board_h; i++) line(ctx, 0, i * cell, game.board_w * cell, i * cell);
}

function drawCell(ctx: CanvasRenderingContext2D, game: GameSnapshot, cell: number, p: Point, color: string, label: string) {
  const x = p.x * cell, y = (game.board_h - 1 - p.y) * cell;
  ctx.fillStyle = color;
  ctx.fillRect(x + 2, y + 2, Math.max(1, cell - 4), Math.max(1, cell - 4));
  if (label) {
    ctx.fillStyle = "#fff";
    ctx.font = `bold ${Math.max(10, cell * 0.45)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x + cell / 2, y + cell / 2);
  }
}

function drawDot(ctx: CanvasRenderingContext2D, game: GameSnapshot, cell: number, p: Point, color: string) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x * cell + cell / 2, (game.board_h - 1 - p.y) * cell + cell / 2, cell * 0.22, 0, Math.PI * 2);
  ctx.fill();
}

function line(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}
