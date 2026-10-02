/**
 * What the ASVspoof 2019 LA attack ids mean.
 *
 * The protocol only names each spoofing system A07–A19, which tells a reader
 * nothing. These are the systems as described in the database paper (Wang et
 * al., "ASVspoof 2019: A large-scale public database of synthesized, converted
 * and replayed speech", Computer Speech & Language, 2020; arXiv:1911.01601).
 * A16 and A19 reuse the training set's A04 and A06 algorithms; the other
 * eleven are unseen during training.
 *
 * Only used for dataset-level summaries: per-clip views never show an attack
 * id, because that would leak the ground truth.
 */

export interface AttackInfo {
  /** Text-to-speech, voice conversion, or a VC system fed TTS output. */
  kind: "TTS" | "VC" | "TTS + VC";
  /** Acoustic or conversion model. */
  model: string;
  /** Waveform generator (vocoder). */
  vocoder: string;
}

export const ASVSPOOF2019_LA_ATTACKS: Record<string, AttackInfo> = {
  A07: { kind: "TTS", model: "LSTM + WaveCycleGAN2 post-filter", vocoder: "WORLD" },
  A08: { kind: "TTS", model: "Autoregressive mixture density + VAE", vocoder: "Neural source-filter" },
  A09: { kind: "TTS", model: "LSTM", vocoder: "Vocaine" },
  A10: { kind: "TTS", model: "Tacotron 2 + speaker encoder", vocoder: "WaveRNN" },
  A11: { kind: "TTS", model: "Tacotron 2 + speaker encoder", vocoder: "Griffin-Lim" },
  A12: { kind: "TTS", model: "Autoregressive WaveNet", vocoder: "WaveNet" },
  A13: { kind: "TTS + VC", model: "Highway + feed-forward network", vocoder: "Waveform filtering" },
  A14: { kind: "TTS + VC", model: "LSTM on bottleneck features", vocoder: "STRAIGHT" },
  A15: { kind: "TTS + VC", model: "LSTM on bottleneck features", vocoder: "Speaker-dependent WaveNet" },
  A16: { kind: "TTS", model: "Unit selection (same as A04)", vocoder: "Waveform concatenation" },
  A17: { kind: "VC", model: "VAE", vocoder: "Waveform filtering" },
  A18: { kind: "VC", model: "i-vector / PLDA", vocoder: "MFCC vocoder" },
  A19: { kind: "VC", model: "GMM-UBM (same as A06)", vocoder: "Spectral filtering" },
};

/** Normalise "A7" / "a07" to the protocol's "A07". */
function canonical(attack: string): string {
  const match = /^a0*(\d+)$/i.exec(attack.trim());
  return match ? `A${match[1].padStart(2, "0")}` : attack;
}

/** The attack's description, or null for ids outside ASVspoof 2019 LA (custom datasets). */
export function describeAttack(attack: string): AttackInfo | null {
  return ASVSPOOF2019_LA_ATTACKS[canonical(attack)] ?? null;
}

/** One-line label for a bar or a sentence, e.g. "TTS · WaveRNN". */
export function attackShortLabel(attack: string): string | null {
  const info = describeAttack(attack);
  return info ? `${info.kind} · ${info.vocoder}` : null;
}

/** Full sentence for a tooltip. */
export function attackLongLabel(attack: string): string | null {
  const info = describeAttack(attack);
  return info ? `${attack}: ${info.kind}, ${info.model}, ${info.vocoder} waveform generator` : null;
}
