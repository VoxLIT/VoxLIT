/** Rounded rectangle path, falling back to a plain rect on browsers without
 *  CanvasRenderingContext2D.roundRect (Safari < 16, Firefox < 112). */
export const pillPath = (ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) => {
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
};
