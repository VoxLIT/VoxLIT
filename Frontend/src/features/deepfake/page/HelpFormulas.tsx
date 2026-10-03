import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { BookOpen, ExternalLink, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

/**
 * Help & Formulas for this page: every number the deepfake page shows, with
 * the exact definition the backend computes (app/tasks/deepfake/metrics.py,
 * evaluation.py, silence_probe.py, saliency.py, projection.py). Kept here
 * rather than in the shared help portal so it always matches this task's code.
 */
export const HelpFormulas = () => (
  <Sheet>
    <Tooltip>
      <TooltipTrigger asChild>
        <SheetTrigger asChild>
          <Button variant="outline" size="sm" className="h-7 border-border bg-white text-xs text-foreground hover:bg-muted">
            <HelpCircle className="mr-1 h-3.5 w-3.5 text-primary" />
            Help & Formulas
          </Button>
        </SheetTrigger>
      </TooltipTrigger>
      <TooltipContent>
        <p>Definitions and formulas behind every number on this page</p>
      </TooltipContent>
    </Tooltip>
    <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-2xl">
      <SheetHeader>
        <SheetTitle className="flex items-center gap-2">
          <BookOpen className="h-5 w-5 text-primary" /> Audio Deepfake Detection: help & formulas
        </SheetTitle>
        <SheetDescription>
          The exact definitions this page computes. Spoof is the positive class throughout; s is a detector&apos;s spoof
          score for one clip.
        </SheetDescription>
      </SheetHeader>

      <div className="mt-4 space-y-5 text-sm text-muted-foreground">
        <Section title="1. Score and decision">
          <Formula>s = softmax(z)_spoof ∈ [0, 1]      decision = spoof ⇔ s ≥ τ</Formula>
          <p>
            z are the classifier head&apos;s logits. The shipped threshold τ<sub>op</sub> = 0.5 is <em>uncalibrated</em>:
            s ranks clips but is not a calibrated probability, so a threshold only means something on the dataset it
            was set on.
          </p>
        </Section>

        <Section title="2. Error rates (ASVspoof convention)">
          <Formula>
            FAR(τ) = |{'{'}spoof : s &lt; τ{'}'}| / N_spoof{"\n"}FRR(τ) = |{'{'}bona fide : s ≥ τ{'}'}| / N_bona
          </Formula>
          <p>
            &ldquo;Acceptance&rdquo; means accepted as genuine: a false acceptance is a spoof that got through, a false
            rejection a genuine voice that was flagged.
          </p>
        </Section>

        <Section title="3. Equal error rate (EER)">
          <Formula>τ* = argmin_τ |FAR(τ) − FRR(τ)|      EER = ½ (FAR(τ*) + FRR(τ*))</Formula>
          <p>
            τ is swept over every observed score plus one value below and above the range, so the result is exact, not
            a grid approximation. With n clips per class one error moves a rate by 1/n, so the EER is resolved to
            1/(2n): 0.5 % for the 100 + 100 clip subset.
          </p>
        </Section>

        <Section title="4. Uncertainty">
          <Formula>
            EER 95% CI: percentile bootstrap, B = 1000 stratified resamples (fixed seed){"\n"}Zero errors: EER ≤ ½
            [(1 − α^(1/N_bona)) + (1 − α^(1/N_spoof))],  α = 0.05  (≈ 3/n){"\n"}FAR, FRR at τ_op: Clopper–Pearson exact
            interval{"\n"}FAR, FRR at your τ: Wilson score interval
          </Formula>
          <p>
            Each class is resampled with replacement at its own size, so every replicate keeps the class balance. When
            the observed EER is 0 every replicate is 0 too, so the exact one-sided binomial bound is reported instead.
          </p>
        </Section>

        <Section title="5. ROC-AUC (threshold-free)">
          <Formula>AUC = P(s_spoof &gt; s_bona) + ½ P(s_spoof = s_bona) = U / (N_spoof · N_bona)</Formula>
          <p>The Mann–Whitney form. AUC judges the ranking; EER and FAR/FRR also depend on where the cut is.</p>
        </Section>

        <Section title="6. Confusion-matrix metrics at a threshold">
          <Formula>
            precision = TP/(TP+FP)   recall = TPR = TP/(TP+FN)   specificity = TN/(TN+FP){"\n"}F1 = 2·P·R/(P+R)
            balanced acc. = ½(TPR + TNR)   FAR = FN/N_spoof   FRR = FP/N_bona
          </Formula>
        </Section>

        <Section title="7. DET curve">
          <Formula>x = Φ⁻¹(FRR(τ)),  y = Φ⁻¹(FAR(τ))  for every τ</Formula>
          <p>
            Normal-deviate (probit) axes, as in NIST and ASVspoof evaluations: two Gaussian score distributions give a
            straight line. Φ⁻¹(0) = −∞, so rates of exactly zero sit on a separate floor labelled 0, set off by an
            axis break. The dashed diagonal is FAR = FRR; it crosses the curve at the EER.
          </p>
        </Section>

        <Section title="8. Evaluation conditions and the silence ablation">
          <Formula>
            as distributed: score(x){"\n"}silence trimmed: score(x[t_first speech : t_last speech])
          </Formula>
          <p>
            In ASVspoof 2019 LA the genuine clips carry much more leading and trailing silence than the spoofed ones
            (on this subset 51 % vs 23 % of duration), and detectors can score well by reading that alone (Müller et
            al., 2021). Re-scoring every clip with the outer silence removed measures how much of the EER is the
            voice and how much is the silence. The trimmed figure is the one to quote; the gap between the two is the
            size of the shortcut.
          </p>
        </Section>

        <Section title="9. Silence probe (per clip)">
          <Formula>
            speech = frames within 30 dB of the clip&apos;s own peak (librosa.effects.split, top_db = 30){"\n"}whole
            clip · silence cut (outer non-speech removed) · only silence (all non-speech, pauses included)
          </Formula>
          <p>
            The threshold is relative to the clip&apos;s peak, not an absolute noise floor. A variant shorter than 0.50 s
            is still scored but marked † (indicative only); under 0.10 s there is no usable signal (wav2vec2&apos;s
            encoder alone needs 25 ms) and the probe reports the measured length instead of a score.
          </p>
        </Section>

        <Section title="10. Listening heatmap (saliency)">
          <Formula>
            a(t) = |E_ε[∂(z_spoof − z_bonafide) / ∂x(t)] · x(t)|,  ε ~ N(0, (0.1·σ_x)²), 8 samples by default  →  smoothed, averaged into 60 equal segments, divided by the max segment
          </Formula>
          <p>
            SmoothGrad × input on the decision margin: how much each instant of the waveform contributes to the verdict, weighted by what is actually there (a bare gradient measures sensitivity, which piles onto near-silent stretches). It leans towards loud regions, so whether the score depends on silence is the ablation&apos;s question, not this map&apos;s. Values are
            relative within one clip (the peak is 1). &ldquo;Share in speech&rdquo; is the fraction of attribution mass
            inside the probe&apos;s speech intervals; compare it with the share of the clip that is speech.
          </p>
        </Section>

        <Section title="11. Voice map (embeddings)">
          <p>
            Each point is the vector the detector&apos;s classification head reads for one clip, reduced to 2 or 3 axes
            by PCA, t-SNE (perplexity = min(30, n − 1)) or UMAP (n_neighbors = min(15, n − 1)), all with random_state =
            42. Points are coloured by the detector&apos;s own score, never by the ground-truth label. t-SNE and UMAP
            preserve neighbourhoods, not distances: compare clusters, not gaps.
          </p>
        </Section>

        <Section title="12. Detectors">
          <table className="w-full text-left text-xs">
            <thead className="text-foreground">
              <tr>
                <th className="py-1 pr-2">Model</th>
                <th className="py-1 pr-2">Checkpoint</th>
                <th className="py-1">Input</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t">
                <td className="py-1 pr-2">A · wav2vec2 XLS-R</td>
                <td className="pr-2 font-mono">Gustking/wav2vec2-large-xlsr-deepfake-audio-classification</td>
                <td>raw 16 kHz waveform, ≤ 30 s</td>
              </tr>
              <tr className="border-t">
                <td className="py-1 pr-2">B · AST</td>
                <td className="pr-2 font-mono">MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection</td>
                <td>log-Mel spectrogram patches</td>
              </tr>
              <tr className="border-t">
                <td className="py-1 pr-2">C · XLSR-Mamba</td>
                <td className="pr-2 font-mono">AustinXiao/XLSR-Mamba-LA</td>
                <td>wav2vec2 features, 4.17 s window</td>
              </tr>
              <tr className="border-t">
                <td className="py-1 pr-2">D · Wav2Vec2-AASIST</td>
                <td className="pr-2 font-mono">SpeechAntiSpoofingBenchmarks/W2V2-AASIST</td>
                <td>wav2vec2 features → graph attention, 4.04 s window</td>
              </tr>
              <tr className="border-t">
                <td className="py-1 pr-2">E · XLSR-SLS</td>
                <td className="pr-2 font-mono">SpeechAntiSpoofingBenchmarks/XLSR-SLS</td>
                <td>all 24 wav2vec2 layers, gated, 4.04 s window</td>
              </tr>
              <tr className="border-t">
                <td className="py-1 pr-2">F · Nes2Net-X</td>
                <td className="pr-2 font-mono">SpeechAntiSpoofingBenchmarks/Nes2Net</td>
                <td>wav2vec2 features → nested Res2Net, 4.04 s window</td>
              </tr>
            </tbody>
          </table>
        </Section>

        <Section title="13. Built-in datasets">
          <p>
            Three labelled subsets of 200 clips each (100 bona fide, 100 spoof, fixed seed), chosen from the Dataset menu.
            Comparing them shows how far a detector&apos;s accuracy travels beyond the data it was trained on.
          </p>
          <ul className="list-disc space-y-1 pl-4">
            <li>
              <strong>ASVspoof 2019 LA</strong>: studio speech against 13 text-to-speech and voice-conversion attacks
              (A07–A19). Models B–F were trained on its training partition, so this is their in-domain test (Model A&apos;s
              training data is not documented).
            </li>
            <li>
              <strong>ASVspoof 5</strong>: crowdsourced audiobook speech against 16 newer attacks (A17–A32, a different
              catalogue from 2019&apos;s ids of the same name), some with codec or adversarial processing. Not in the
              training data of Models B–F.
            </li>
            <li>
              <strong>In-the-Wild</strong>: real and fake speech of public figures collected from the internet, spread
              across speakers. The release names no generator, so its fakes are grouped as &ldquo;unattributed&rdquo;.
              Not in the training data of Models B–F.
            </li>
          </ul>
          <p>
            The 51 % vs 23 % silence figure in section 8 is measured on the ASVspoof 2019 LA subset; run the silence
            ablation on each dataset rather than assuming the same shortcut.
          </p>
        </Section>

        <Section title="14. Custom datasets">
          <p>
            Manage Datasets stores your files as 16 kHz mono WAV. A label file (CSV <code>filename,label[,attack]</code>
            or ASVspoof protocol lines) is matched on file names without extension and used only for aggregate metrics;
            clips stay blind on the page. EER needs at least one labelled clip of each class, and the uncertainty
            figures above tell you how far a small dataset can be trusted.
          </p>
        </Section>

        <Section title="References">
          <ul className="list-disc space-y-1 pl-4 text-xs">
            <li>Wang, X. et al. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech. Computer Speech &amp; Language 64.</li>
            <li>Wang, X. et al. (2025). ASVspoof 5: Design, collection and validation of resources for spoofing, deepfake, and adversarial attack detection using crowdsourced speech. Computer Speech &amp; Language.</li>
            <li>Müller, N. M. et al. (2022). Does Audio Deepfake Detection Generalize? Interspeech 2022.</li>
            <li>Müller, N. M. et al. (2021). Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn? ASVspoof 2021 Workshop.</li>
            <li>Martin, A. et al. (1997). The DET curve in assessment of detection task performance. Eurospeech.</li>
            <li>Efron, B. &amp; Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman &amp; Hall.</li>
            <li>Tak, H. et al. (2022). Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation. Odyssey 2022.</li>
            <li>Zhang, Q., Wen, S. &amp; Hu, T. (2024). Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier. ACM Multimedia 2024.</li>
            <li>Liu, T. et al. (2025). Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-Spoofing. IEEE Transactions on Information Forensics and Security, 20.</li>
            <li>Dowerah, S. et al. (2025). Speech DF Arena: A Leaderboard for Speech DeepFake Detection Models. arXiv:2509.02859.</li>
          </ul>
        </Section>

        <Link to="/help" className="inline-flex items-center gap-1.5 text-xs font-medium text-primary hover:underline">
          VoxLIT help portal (all tasks) <ExternalLink className="h-3 w-3" />
        </Link>
      </div>
    </SheetContent>
  </Sheet>
);

const Section = ({ title, children }: { title: string; children: ReactNode }) => (
  <section className="space-y-2">
    <h3 className="text-sm font-semibold text-foreground">{title}</h3>
    {children}
  </section>
);

const Formula = ({ children }: { children: ReactNode }) => (
  <pre className="whitespace-pre-wrap rounded-md bg-slate-900 px-3 py-2.5 font-mono text-xs leading-relaxed text-slate-100">
    {children}
  </pre>
);
