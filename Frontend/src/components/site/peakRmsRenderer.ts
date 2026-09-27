/**
 * Custom wavesurfer.js renderer: rounded bars, each carrying two layers in
 * the style of pro audio editors (Audacity's "Show RMS", iZotope RX) — a
 * translucent full-height bar for the peak and a solid inner bar for RMS
 * loudness.
 *
 * wavesurfer sets ctx.fillStyle to waveColor before calling this, then tints a
 * copy with progressColor using `source-in`, which keeps our alpha — so both
 * layers survive in the played region as well.
 */

import { pillPath } from "./canvasShapes";

type Peaks = Array<Float32Array | number[]>;

const BAR_CSS_PX = 3;
const GAP_CSS_PX = 2;
/** < 1 lifts quiet passages so they stay readable next to loud ones. */
const GAMMA = 0.75;

/**
 * @param getGlobalPeak returns the clip's overall peak so that every canvas
 *   chunk (wavesurfer splits long audio) is normalised to the same scale.
 */
export const createPeakRmsRenderer =
  (getGlobalPeak: () => number) =>
  (peaks: Peaks, ctx: CanvasRenderingContext2D) => {
    const data = peaks[0];
    const { width: w, height: h } = ctx.canvas;
    if (!data?.length || !w) return;

    const dpr = window.devicePixelRatio || 1;
    const barW = BAR_CSS_PX * dpr;
    const step = (BAR_CSS_PX + GAP_CSS_PX) * dpr;
    const bars = Math.max(1, Math.floor(w / step));
    const perBar = data.length / bars;

    const peak = new Float32Array(bars);
    const rms = new Float32Array(bars);
    let localMax = 0;
    for (let b = 0; b < bars; b++) {
      const start = Math.floor(b * perBar);
      const end = Math.max(start + 1, Math.floor((b + 1) * perBar));
      let p = 0;
      let sq = 0;
      for (let i = start; i < end; i++) {
        const v = Number(data[i]) || 0;
        const a = Math.abs(v);
        if (a > p) p = a;
        sq += v * v;
      }
      peak[b] = p;
      rms[b] = Math.sqrt(sq / (end - start));
      if (p > localMax) localMax = p;
    }

    const mid = h / 2;
    const norm = getGlobalPeak() || localMax || 1;
    const maxHalf = mid * 0.92;
    const minHalf = 1 * dpr;
    const halfOf = (v: number) => Math.max(minHalf, Math.pow(Math.min(1, v / norm), GAMMA) * maxHalf);
    const radius = barW / 2;

    for (let b = 0; b < bars; b++) {
      const x = b * step;
      const ph = halfOf(peak[b]);
      const rh = Math.min(ph, halfOf(rms[b]));

      ctx.globalAlpha = 0.35;
      pillPath(ctx, x, mid - ph, barW, ph * 2, radius);
      ctx.fill();

      ctx.globalAlpha = 1;
      pillPath(ctx, x, mid - rh, barW, rh * 2, radius);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };
