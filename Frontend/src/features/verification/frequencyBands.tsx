import { InfoTooltip } from "./InfoTooltip";

/** Neutral band label with its exact range, e.g. "Band 1 · 50–319 Hz". */
export const formatBandLabel = (bandIndex: number, lowHz: number, highHz: number) =>
  `Band ${bandIndex} · ${Math.round(lowHz)}–${Math.round(highHz)} Hz`;

/** Approximate literature guide to speech content by frequency. Bands
 *  themselves are never given a phonetic meaning. */
export const FrequencyBandGuideTooltip = () => (
  <InfoTooltip title="Typical speech content by frequency (approximate)">
    <p>
      Bands are mel-spaced and carry no fixed meaning. As rough, overlapping guides from the speech literature: adult
      F0 commonly lies around 85–255 Hz (Titze, 1994); the first formant typically around 250–1000 Hz and the second
      around 800–2500 Hz (Peterson &amp; Barney, 1952); sibilant fricatives such as /s/ and /ʃ/ concentrate energy
      above ~4 kHz (Jongman et al., 2000). Speaker-specific information has been reported in both low and ~4–5 kHz
      regions (Lu &amp; Dang, 2008). Ranges vary by speaker, sex and vowel.
    </p>
  </InfoTooltip>
);
