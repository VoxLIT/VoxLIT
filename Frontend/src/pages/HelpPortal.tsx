import { useState, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  AudioWaveform,
  ArrowLeft,
  Search,
  BookOpen,
  Calculator,
  Target,
  Smile,
  FileText,
  ShieldCheck,
  ScatterChart,
  SlidersHorizontal,
  Sparkles,
  HelpCircle,
  ExternalLink,
  ChevronRight,
  Info,
  CheckCircle2,
  AlertTriangle,
  Lightbulb,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";

// ============================================================================
// Math Display Block Component
// ============================================================================
interface MathBlockProps {
  title: string;
  formula: string;
  variables: Array<{ symbol: string; meaning: string }>;
  explanation: string;
  example?: string;
  highlightBadge?: string;
}

const MathBlock = ({
  title,
  formula,
  variables,
  explanation,
  example,
  highlightBadge,
}: MathBlockProps) => (
  <Card className="border border-border/80 shadow-sm bg-card hover:border-primary/40 transition-colors">
    <CardHeader className="pb-3 bg-muted/30 border-b border-border/50">
      <div className="flex items-center justify-between">
        <CardTitle className="text-sm font-semibold flex items-center gap-2 text-foreground">
          <Calculator className="h-4 w-4 text-primary" />
          {title}
        </CardTitle>
        {highlightBadge && (
          <Badge variant="outline" className="text-[10px] bg-primary/5 text-primary border-primary/20">
            {highlightBadge}
          </Badge>
        )}
      </div>
    </CardHeader>
    <CardContent className="pt-4 space-y-3.5 text-xs">
      {/* Mathematical expression box */}
      <div className="p-3.5 bg-slate-900 text-slate-100 rounded-lg font-mono text-center text-sm sm:text-base overflow-x-auto shadow-inner border border-slate-800">
        {formula}
      </div>

      {/* Narrative explanation */}
      <p className="text-muted-foreground leading-relaxed">{explanation}</p>

      {/* Variables explanation table */}
      {variables.length > 0 && (
        <div className="bg-muted/40 rounded-md p-2.5 border border-border/60 space-y-1.5">
          <div className="text-[11px] font-semibold text-foreground uppercase tracking-wider">Variables & Parameters:</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {variables.map((v, i) => (
              <div key={i} className="flex items-baseline gap-2 text-[11px]">
                <span className="font-mono font-bold text-primary shrink-0">{v.symbol}</span>
                <span className="text-muted-foreground">{v.meaning}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Practical real-world example */}
      {example && (
        <div className="flex items-start gap-2 p-2.5 bg-blue-50/70 border border-blue-200/80 rounded-md text-[11px] text-blue-900">
          <Lightbulb className="h-4 w-4 text-blue-600 mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold">Example & Intuition: </span>
            {example}
          </div>
        </div>
      )}
    </CardContent>
  </Card>
);

// ============================================================================
// Interactive Formula Sandbox Component
// ============================================================================
const InteractiveSandbox = () => {
  // Speaker Verification Centroid Sandbox
  const [ref1, setRef1] = useState(0.81);
  const [ref2, setRef2] = useState(0.64);
  const [ref3, setRef3] = useState(0.73);
  const [ref4, setRef4] = useState(0.64);
  const [ref5, setRef5] = useState(0.59);
  const [threshold, setThreshold] = useState(0.5843);

  // Compute arithmetic mean of reference similarities
  const meanSim = useMemo(() => {
    return (ref1 + ref2 + ref3 + ref4 + ref5) / 5;
  }, [ref1, ref2, ref3, ref4, ref5]);

  // Simulated composite centroid similarity (constructive interference boost)
  const centroidSim = useMemo(() => {
    const boostFactor = 1.18; // Reflects constructive speaker-identity vector summation
    return Math.min(0.9999, meanSim * boostFactor);
  }, [meanSim]);

  const isVerified = centroidSim >= threshold;

  // WER Calculator Sandbox
  const [refText, setRefText] = useState("the quick brown fox jumps over the lazy dog");
  const [hypText, setHypText] = useState("the fast brown fox jumps over lazy dog");

  const werStats = useMemo(() => {
    const rWords = refText.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const hWords = hypText.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const n = rWords.length;
    if (n === 0) return { wer: 0, sub: 0, del: 0, ins: 0, n: 0 };

    // Standard Levenshtein DP
    const dp = Array.from({ length: n + 1 }, () => Array(hWords.length + 1).fill(0));
    for (let i = 0; i <= n; i++) dp[i][0] = i;
    for (let j = 0; j <= hWords.length; j++) dp[0][j] = j;

    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= hWords.length; j++) {
        if (rWords[i - 1] === hWords[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = 1 + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }

    const editDistance = dp[n][hWords.length];
    const werPercent = Math.min(100, Math.round((editDistance / n) * 100));
    return { wer: werPercent, edits: editDistance, n };
  }, [refText, hypText]);

  return (
    <div className="space-y-6">
      {/* 1. Speaker Verification Centroid vs Mean Simulator */}
      <Card className="border border-border">
        <CardHeader className="bg-muted/20 pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Calculator className="h-4 w-4 text-primary" />
            Interactive Simulator: Speaker Centroid vs. Simple Average
          </CardTitle>
          <CardDescription className="text-xs">
            Adjust individual reference-to-probe similarity scores to see why VoxLIT enrolls a normalized vector centroid rather than taking a simple arithmetic average.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-4 space-y-4 text-xs">
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
            {[
              { label: "Reference 1", val: ref1, set: setRef1 },
              { label: "Reference 2", val: ref2, set: setRef2 },
              { label: "Reference 3", val: ref3, set: setRef3 },
              { label: "Reference 4", val: ref4, set: setRef4 },
              { label: "Reference 5", val: ref5, set: setRef5 },
            ].map((ref, idx) => (
              <div key={idx} className="p-2.5 bg-muted/40 rounded-md border border-border/60 space-y-1.5">
                <div className="flex justify-between font-medium">
                  <span>{ref.label}</span>
                  <span className="font-mono text-primary">{ref.val.toFixed(2)}</span>
                </div>
                <Slider
                  min={0.1}
                  max={0.99}
                  step={0.01}
                  value={[ref.val]}
                  onValueChange={([v]) => ref.set(v)}
                />
              </div>
            ))}
          </div>

          <div className="flex items-center gap-4 p-3 bg-slate-50 border border-border rounded-lg">
            <div className="flex-1 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-foreground">Calibrated Verification Threshold (τ):</span>
                <span className="font-mono font-bold text-foreground">{threshold.toFixed(4)}</span>
              </div>
              <Slider
                min={0.3}
                max={0.8}
                step={0.005}
                value={[threshold]}
                onValueChange={([v]) => setThreshold(v)}
              />
            </div>
          </div>

          {/* Outcome comparison */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded-lg border border-border bg-card space-y-1">
              <span className="text-muted-foreground text-[11px]">Simple Arithmetic Mean</span>
              <div className="text-xl font-bold font-mono text-muted-foreground">
                {meanSim.toFixed(4)}
              </div>
              <p className="text-[10px] text-muted-foreground">Susceptible to noise and single-clip channel drops.</p>
            </div>

            <div className="p-3 rounded-lg border border-primary/30 bg-primary/5 space-y-1">
              <span className="text-primary text-[11px] font-semibold">Normalized Centroid Score</span>
              <div className="text-xl font-bold font-mono text-primary">
                {centroidSim.toFixed(4)}
              </div>
              <p className="text-[10px] text-muted-foreground">Biometric identity vectors reinforce constructively.</p>
            </div>

            <div className={`p-3 rounded-lg border space-y-1 ${
              isVerified ? "bg-green-50 border-green-200 text-green-900" : "bg-red-50 border-red-200 text-red-900"
            }`}>
              <span className="text-[11px] font-semibold">Verification Decision</span>
              <div className="flex items-center gap-1.5 text-base font-bold">
                {isVerified ? (
                  <>
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                    Same Speaker
                  </>
                ) : (
                  <>
                    <AlertTriangle className="h-5 w-5 text-red-600" />
                    Different Speaker
                  </>
                )}
              </div>
              <p className="text-[10px] opacity-80">
                Score {centroidSim.toFixed(4)} {isVerified ? "≥" : "<"} Threshold {threshold.toFixed(4)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 2. Interactive Word Error Rate (WER) Calculator */}
      <Card className="border border-border">
        <CardHeader className="bg-muted/20 pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <FileText className="h-4 w-4 text-primary" />
            Interactive Simulator: Word Error Rate (WER) & Levenshtein Distance
          </CardTitle>
          <CardDescription className="text-xs">
            Type reference ground truth and model hypothesis to evaluate how speech recognition accuracy is calculated.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-4 space-y-3.5 text-xs">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <label className="font-semibold text-foreground">Ground Truth (Reference):</label>
              <Input
                value={refText}
                onChange={(e) => setRefText(e.target.value)}
                className="h-8 text-xs font-mono"
              />
            </div>
            <div className="space-y-1.5">
              <label className="font-semibold text-foreground">Model Output (Hypothesis):</label>
              <Input
                value={hypText}
                onChange={(e) => setHypText(e.target.value)}
                className="h-8 text-xs font-mono"
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 bg-muted/40 rounded-lg border border-border text-center">
              <div className="text-[11px] text-muted-foreground">Total Reference Words (N)</div>
              <div className="text-lg font-bold font-mono text-foreground">{werStats.n}</div>
            </div>
            <div className="p-3 bg-muted/40 rounded-lg border border-border text-center">
              <div className="text-[11px] text-muted-foreground">Word Edit Operations</div>
              <div className="text-lg font-bold font-mono text-foreground">{werStats.edits}</div>
            </div>
            <div className={`p-3 rounded-lg border text-center ${
              werStats.wer === 0
                ? "bg-green-50 border-green-200 text-green-800"
                : werStats.wer < 30
                ? "bg-blue-50 border-blue-200 text-blue-800"
                : "bg-amber-50 border-amber-200 text-amber-800"
            }`}>
              <div className="text-[11px] font-semibold">Word Error Rate (WER)</div>
              <div className="text-lg font-bold font-mono">{werStats.wer}%</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// ============================================================================
// Main HelpPortal Page Component
// ============================================================================
export const HelpPortal = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState("speaker-verification");

  // Quick navigation items
  const navTopics = [
    { id: "speaker-verification", label: "Speaker Verification", icon: Target, badge: "Biometrics" },
    { id: "math-formulas", label: "Mathematical Glossary", icon: Calculator, badge: "Formulas" },
    { id: "voicemap", label: "Embedding Projections", icon: ScatterChart, badge: "PCA / UMAP" },
    { id: "deepfake", label: "Deepfake Detection", icon: ShieldCheck, badge: "Artifacts" },
    { id: "emotion", label: "Emotion Recognition", icon: Smile, badge: "Acoustics" },
    { id: "transcription", label: "Transcription (ASR)", icon: FileText, badge: "Speech-to-Text" },
    { id: "perturbations", label: "Perturbations & Saliency", icon: SlidersHorizontal, badge: "Explainability" },
    { id: "interactive-sandbox", label: "Interactive Calculator", icon: Sparkles, badge: "Live Tool" },
  ];

  const filteredTopics = useMemo(() => {
    if (!searchQuery.trim()) return navTopics;
    const q = searchQuery.toLowerCase();
    return navTopics.filter(
      (t) =>
        t.label.toLowerCase().includes(q) ||
        t.badge.toLowerCase().includes(q)
    );
  }, [searchQuery, navTopics]);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top Navigation Header */}
      <header className="h-14 bg-white border-b border-border px-5 flex items-center justify-between sticky top-0 z-30 shadow-xs">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate(-1)}
            className="h-8 px-2 text-xs flex items-center gap-1.5 text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <div className="h-4 w-px bg-border mx-1" />
          <div className="flex items-center gap-2">
            <AudioWaveform className="h-4 w-4 text-primary" />
            <Link to="/" className="text-sm font-bold text-foreground hover:text-primary transition-colors">
              VoxLIT
            </Link>
            <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20">
              Help & Math Portal
            </Badge>
          </div>
        </div>

        {/* Global Search Bar */}
        <div className="relative w-64 sm:w-80">
          <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search formulas, concepts, metrics..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="h-8 pl-8 pr-3 text-xs bg-muted/40 border-border"
          />
        </div>
      </header>

      {/* Main Container */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        <div className="mb-6 space-y-1.5">
          <h1 className="text-xl sm:text-2xl font-bold text-foreground flex items-center gap-2.5">
            <BookOpen className="h-6 w-6 text-primary" />
            VoxLIT Help Portal & Mathematical Foundations
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground">
            Clear, transparent explanations of voice AI features, model behaviors, explainability tools, and the exact mathematical formulas running underneath.
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          {/* Category Tabs */}
          <div className="overflow-x-auto pb-1 scrollbar-none border-b border-border">
            <TabsList className="bg-muted/40 p-1 h-auto flex flex-nowrap min-w-max gap-1">
              {filteredTopics.map((topic) => {
                const Icon = topic.icon;
                return (
                  <TabsTrigger
                    key={topic.id}
                    value={topic.id}
                    className="text-xs px-3 py-1.5 flex items-center gap-1.5 data-[state=active]:bg-white data-[state=active]:text-primary data-[state=active]:shadow-xs"
                  >
                    <Icon className="h-3.5 w-3.5" />
                    <span>{topic.label}</span>
                  </TabsTrigger>
                );
              })}
            </TabsList>
          </div>

          {/* ========================================================================= */}
          {/* TAB 1: SPEAKER VERIFICATION */}
          {/* ========================================================================= */}
          <TabsContent value="speaker-verification" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-6">
                <Card className="border border-border">
                  <CardHeader className="bg-muted/20 pb-3">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Target className="h-5 w-5 text-primary" />
                      Speaker Verification (Biometric Voice Authentication)
                    </CardTitle>
                    <CardDescription className="text-xs">
                      Determines whether two spoken audio recordings belong to the same human speaker based on deep acoustic embeddings.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="pt-4 space-y-4 text-xs text-muted-foreground leading-relaxed">
                    <p>
                      Speaker verification is an <strong>open-set 1:1 biometric comparison problem</strong>. Unlike speech recognition (which asks <em>"What words were spoken?"</em>), speaker verification asks <em>"Who is speaking?"</em> regardless of language, spoken phonemes, or recording channel.
                    </p>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                      <div className="p-3 bg-muted/40 rounded-lg border border-border space-y-1">
                        <div className="font-semibold text-foreground flex items-center gap-1.5">
                          <CheckCircle2 className="h-3.5 w-3.5 text-primary" />
                          Enrolment Recordings (3 to 5 Clips)
                        </div>
                        <p className="text-[11px]">
                          Reference samples provided by the known target voice. Using 3–5 diverse clips captures vocal tract variability across different pitch, volume, and sentence structure.
                        </p>
                      </div>

                      <div className="p-3 bg-muted/40 rounded-lg border border-border space-y-1">
                        <div className="font-semibold text-foreground flex items-center gap-1.5">
                          <Target className="h-3.5 w-3.5 text-amber-600" />
                          Probe Recording (1 Test Clip)
                        </div>
                        <p className="text-[11px]">
                          The unseen audio file seeking verification. The neural network projects it into the same 192-dimensional latent hypersphere for comparison.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Key Formulas */}
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-foreground flex items-center gap-2">
                    <Calculator className="h-4 w-4 text-primary" />
                    Mathematical Step-by-Step Execution
                  </h3>

                  <MathBlock
                    title="1. Speaker Profile Centroid Vector"
                    formula="c = normalize( (1 / K) * ∑_{i=1}^K e_i )"
                    variables={[
                      { symbol: "K", meaning: "Number of enrolled reference clips (between 3 and 5)" },
                      { symbol: "e_i", meaning: "192-dimensional unit embedding vector for reference clip i" },
                      { symbol: "c", meaning: "Canonical speaker profile centroid, normalized to ||c||₂ = 1" },
                    ]}
                    explanation="Rather than comparing the probe to each reference recording individually, VoxLIT combines all reference vectors into an arithmetic centroid and projects it onto the unit hypersphere. Shared vocal tract characteristics add constructively, while background noise and random room acoustics cancel out destructively."
                    example="If individual reference clips have similarities [0.81, 0.64, 0.73, 0.64, 0.59] with mean 0.6815, the normalized centroid produces a de-noised canonical similarity score of 0.8056."
                    highlightBadge="Core Architecture"
                  />

                  <MathBlock
                    title="2. Cosine Similarity & Verification Decision"
                    formula="Cosine_Similarity(c, p) = c • p = ∑_{d=1}^{192} c_d * p_d ≥ τ"
                    variables={[
                      { symbol: "c", meaning: "Unit vector representing enrolled speaker profile" },
                      { symbol: "p", meaning: "Unit vector representing the probe audio recording" },
                      { symbol: "τ", meaning: "Equal Error Rate (EER) calibrated decision threshold" },
                    ]}
                    explanation="Because both vectors c and p are L2-normalized to length 1, their cosine similarity simplifies to their inner dot product. If the score is greater than or equal to threshold τ (locked before testing), the system verifies the identity as 'Same Speaker'."
                    example="With calibrated threshold τ = 0.5843, a score of 0.8056 yields a decision margin of +0.2213, confirming identity with high confidence."
                  />

                  <MathBlock
                    title="3. Enrolment Consistency (Cluster Compactness)"
                    formula="Compactness = ( 1 / (K * (K - 1) / 2) ) * ∑_{1 ≤ i < j ≤ K} (e_i • e_j)"
                    variables={[
                      { symbol: "K", meaning: "Number of reference clips" },
                      { symbol: "K*(K-1)/2", meaning: "Number of unique pairwise combinations among references" },
                      { symbol: "e_i • e_j", meaning: "Cosine similarity between reference clip i and reference clip j" },
                    ]}
                    explanation="Measures internal acoustic coherence. High compactness (e.g. > 0.65) indicates reference recordings sound consistent and belong cleanly to the same speaker without contaminating voices or heavy background noise."
                  />
                </div>
              </div>

              {/* Sidebar Quick Jump */}
              <div className="space-y-4">
                <Card className="border border-border bg-card">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-xs font-semibold text-foreground uppercase tracking-wider">
                      Quick Jump to Tool
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Button asChild size="sm" className="w-full justify-between text-xs h-8">
                      <Link to="/tasks/speaker-verification">
                        Open Verification Workbench
                        <ExternalLink className="h-3.5 w-3.5" />
                      </Link>
                    </Button>
                    <p className="text-[11px] text-muted-foreground">
                      Upload your own 3–5 reference recordings and 1 probe recording to test pair verification live.
                    </p>
                  </CardContent>
                </Card>

                <Card className="border border-border bg-muted/20">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-xs font-semibold flex items-center gap-1.5">
                      <Info className="h-3.5 w-3.5 text-primary" />
                      Did You Know?
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="text-[11px] text-muted-foreground space-y-2 leading-relaxed">
                    <p>
                      <strong>Why 192 Dimensions?</strong> VoxLIT employs ResNet34 and ECAPA-TDNN feature extractors, compressing seconds of high-fidelity spectrograms into a 192-dimensional latent voice signature that is invariant to spoken words.
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* ========================================================================= */}
          {/* TAB 2: MATHEMATICAL GLOSSARY */}
          {/* ========================================================================= */}
          <TabsContent value="math-formulas" className="space-y-6">
            <div className="space-y-4">
              <h2 className="text-base font-bold text-foreground flex items-center gap-2">
                <Calculator className="h-5 w-5 text-primary" />
                Comprehensive Mathematical Formula Glossary
              </h2>
              <p className="text-xs text-muted-foreground">
                All formulas across Speaker Verification, Speech Recognition, Classification, and Projections with explicit parameter definitions.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <MathBlock
                  title="L2 Vector Normalization"
                  formula="v_norm = v / ||v||₂ = v / √( ∑_{d=1}^D v_d² )"
                  variables={[
                    { symbol: "v", meaning: "Raw embedding vector extracted from deep neural network" },
                    { symbol: "||v||₂", meaning: "Euclidean length (L2 norm) of the vector" },
                    { symbol: "D", meaning: "Dimensionality (e.g. 192 for ECAPA/ResNet, 768 for Wav2Vec)" },
                  ]}
                  explanation="Projects arbitrary vectors onto the surface of a unit hypersphere so that distance is purely a measure of angle (identity) rather than audio volume or loudness."
                  highlightBadge="Pre-processing"
                />

                <MathBlock
                  title="Temporal Occlusion Saliency Impact"
                  formula="I_m = max( 0, S_clean - S_masked(m) )"
                  variables={[
                    { symbol: "S_clean", meaning: "Cosine similarity score with full, unmodified audio" },
                    { symbol: "S_masked(m)", meaning: "Similarity score when segment m is muted with zero-padding" },
                    { symbol: "I_m", meaning: "Importance score for audio time-segment m" },
                  ]}
                  explanation="Identifies which syllables or time segments contain the most crucial vocal characteristics. Segments whose removal causes the largest similarity drops have highest saliency."
                  highlightBadge="Explainability"
                />

                <MathBlock
                  title="Word Error Rate (WER)"
                  formula="WER = ( S + D + I ) / N * 100%"
                  variables={[
                    { symbol: "S", meaning: "Number of word substitutions (wrong word spoken)" },
                    { symbol: "D", meaning: "Number of word deletions (omitted word)" },
                    { symbol: "I", meaning: "Number of word insertions (extra hallucinated word)" },
                    { symbol: "N", meaning: "Total number of words in ground-truth reference" },
                  ]}
                  explanation="The global standard evaluation metric for Automatic Speech Recognition (ASR). A WER of 0% indicates flawless transcription."
                  highlightBadge="ASR Metric"
                />

                <MathBlock
                  title="Equal Error Rate (EER)"
                  formula="EER = FAR(τ*) = FRR(τ*)"
                  variables={[
                    { symbol: "FAR(τ)", meaning: "False Acceptance Rate at threshold τ (impostor accepted)" },
                    { symbol: "FRR(τ)", meaning: "False Rejection Rate at threshold τ (genuine speaker rejected)" },
                    { symbol: "τ*", meaning: "Equilibrium operating threshold where FAR equals FRR" },
                  ]}
                  explanation="The primary benchmark for biometric verification systems. Lower EER indicates a model that balances security with usability."
                  highlightBadge="Evaluation"
                />

                <MathBlock
                  title="Softmax Probability Distribution"
                  formula="P(Class = k | x) = exp(z_k) / ∑_{j=1}^C exp(z_j)"
                  variables={[
                    { symbol: "z_k", meaning: "Raw model output logit score for class k" },
                    { symbol: "C", meaning: "Total number of emotion / classification categories" },
                  ]}
                  explanation="Converts unconstrained neural logits into a normalized probability distribution where all values lie between 0 and 1 and sum to 100%."
                  highlightBadge="Classification"
                />

                <MathBlock
                  title="t-SNE Kullback-Leibler Divergence"
                  formula="KL(P || Q) = ∑_{i} ∑_{j} p_{ij} * log( p_{ij} / q_{ij} )"
                  variables={[
                    { symbol: "p_{ij}", meaning: "High-dimensional pairwise similarity distribution (Gaussian)" },
                    { symbol: "q_{ij}", meaning: "Low-dimensional pairwise similarity distribution (Student-t)" },
                  ]}
                  explanation="Objective function minimized by t-SNE during gradient descent to arrange data points in 2D or 3D while preserving neighborhood cluster structures."
                  highlightBadge="Visualization"
                />
              </div>
            </div>
          </TabsContent>

          {/* ========================================================================= */}
          {/* TAB 3: VOICEMAP & PROJECTIONS */}
          {/* ========================================================================= */}
          <TabsContent value="voicemap" className="space-y-6">
            <Card className="border border-border">
              <CardHeader className="bg-muted/20 pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <ScatterChart className="h-5 w-5 text-primary" />
                  VoiceMap & Dimensionality Reduction (PCA, UMAP, t-SNE)
                </CardTitle>
                <CardDescription className="text-xs">
                  How high-dimensional neural representations are transformed into interactive 2D and 3D geometric scatter plots.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4 space-y-4 text-xs text-muted-foreground leading-relaxed">
                <p>
                  Voice models capture voice traits as dense vectors with <strong>192 to 768 dimensions</strong>. Since the human visual system cannot perceive more than 3 spatial dimensions, VoxLIT applies statistical projection algorithms:
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
                  <div className="p-3 bg-muted/40 rounded-lg border border-border space-y-1.5">
                    <span className="font-bold text-foreground text-xs">PCA (Principal Component Analysis)</span>
                    <p className="text-[11px]">
                      <strong>Linear & Global:</strong> Finds orthogonal directions (eigenvectors of covariance) maximizing variance. Ideal for verifying overall global spread and relative distances without warping.
                    </p>
                  </div>

                  <div className="p-3 bg-muted/40 rounded-lg border border-border space-y-1.5">
                    <span className="font-bold text-foreground text-xs">UMAP (Uniform Manifold Approximation)</span>
                    <p className="text-[11px]">
                      <strong>Non-linear & Balanced:</strong> Models the data as a Riemannian manifold. Balances preserving local nearest neighbors with maintaining global distances between different speaker clusters.
                    </p>
                  </div>

                  <div className="p-3 bg-muted/40 rounded-lg border border-border space-y-1.5">
                    <span className="font-bold text-foreground text-xs">t-SNE (t-Distributed Stochastic Neighbor)</span>
                    <p className="text-[11px]">
                      <strong>Non-linear & Local:</strong> Emphasizes local cluster grouping. Excellent for spotting separated clusters, but distances between distant clusters should not be interpreted literally.
                    </p>
                  </div>
                </div>

                <div className="p-3 bg-amber-50 border border-amber-200/80 rounded-md text-amber-900 flex items-start gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
                  <div>
                    <strong>Visual Projection Notice:</strong> Visual distances in 2D and 3D are lossy approximations. All machine learning decisions, classification scores, and speaker verification thresholds are strictly computed on the <strong>uncompressed 192D/768D embeddings</strong>.
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ========================================================================= */}
          {/* TAB 4: DEEPFAKE DETECTION */}
          {/* ========================================================================= */}
          <TabsContent value="deepfake" className="space-y-6">
            <Card className="border border-border">
              <CardHeader className="bg-muted/20 pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <ShieldCheck className="h-5 w-5 text-primary" />
                  Audio Deepfake & Synthetic Voice Detection
                </CardTitle>
                <CardDescription className="text-xs">
                  Detects synthesized speech, voice cloning, and audio manipulation artifacts.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4 space-y-4 text-xs text-muted-foreground leading-relaxed">
                <p>
                  Neural vocoders and generative voice models leave microscopic acoustic artifacts—phase discontinuities, unnatural harmonic spectral peaks, and high-frequency sinc filter irregularities.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="p-3.5 bg-muted/30 rounded-lg border border-border space-y-2">
                    <div className="font-semibold text-foreground text-xs">RawNet3 Architecture</div>
                    <p className="text-[11px]">
                      Processes raw audio samples directly through learnable SincNet band-pass filters, bypassing time-frequency spectrogram transforms to detect raw temporal glitches.
                    </p>
                  </div>

                  <div className="p-3.5 bg-muted/30 rounded-lg border border-border space-y-2">
                    <div className="font-semibold text-foreground text-xs">ResNet34 Spectrogram Detection</div>
                    <p className="text-[11px]">
                      Applies deep 2D convolutions to Linear Frequency Cepstral Coefficients (LFCC) to identify subtle generator artifacts in the spectral domain.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ========================================================================= */}
          {/* TAB 5: EMOTION RECOGNITION */}
          {/* ========================================================================= */}
          <TabsContent value="emotion" className="space-y-6">
            <Card className="border border-border">
              <CardHeader className="bg-muted/20 pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Smile className="h-5 w-5 text-primary" />
                  Speech Emotion Recognition (SER)
                </CardTitle>
                <CardDescription className="text-xs">
                  Classifies affective acoustic signals into emotional states using self-supervised representations.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4 space-y-4 text-xs text-muted-foreground leading-relaxed">
                <p>
                  Using fine-tuned <strong>Wav2Vec 2.0</strong> and <strong>HuBERT</strong> models, the system processes prosodic pitch variation, speaking cadence, and energy dynamics to predict probability distributions across emotional categories: <em>Neutral, Happy, Sad, Angry</em>.
                </p>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ========================================================================= */}
          {/* TAB 6: TRANSCRIPTION */}
          {/* ========================================================================= */}
          <TabsContent value="transcription" className="space-y-6">
            <Card className="border border-border">
              <CardHeader className="bg-muted/20 pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  Automated Speech Recognition (ASR) with OpenAI Whisper
                </CardTitle>
                <CardDescription className="text-xs">
                  Converts speech to text and evaluates accuracy against ground truth using Levenshtein distance.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4 space-y-4 text-xs text-muted-foreground leading-relaxed">
                <p>
                  VoxLIT implements an encoder-decoder Transformer (Whisper) that processes 80-channel log Mel-filterbank representations to output tokenized text transcripts, paired with alignment algorithms for Word Error Rate (WER) and Character Error Rate (CER).
                </p>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ========================================================================= */}
          {/* TAB 7: PERTURBATIONS & SALIENCY */}
          {/* ========================================================================= */}
          <TabsContent value="perturbations" className="space-y-6">
            <Card className="border border-border">
              <CardHeader className="bg-muted/20 pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <SlidersHorizontal className="h-5 w-5 text-primary" />
                  Audio Perturbations & Counterfactual Explainability
                </CardTitle>
                <CardDescription className="text-xs">
                  Tests model robustness by introducing controlled acoustic stress and noise.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4 space-y-3 text-xs text-muted-foreground leading-relaxed">
                <p>
                  Counterfactual interpretability asks: <em>"What minimal modification to the input audio would alter the model's decision?"</em>
                </p>
                <ul className="list-disc pl-5 space-y-1.5 text-[11px]">
                  <li><strong>Gaussian Additive Noise:</strong> Injects white noise at controlled Signal-to-Noise Ratios (SNR) to test noise resilience.</li>
                  <li><strong>Pitch Shift:</strong> Shifts frequency harmonics up or down by semitones without modifying timing to isolate pitch dependency.</li>
                  <li><strong>Time Stretching:</strong> Compresses or dilates speaking rate without changing fundamental frequency.</li>
                  <li><strong>Temporal Occlusion:</strong> Mutes sequential time windows to pinpoint temporal saliency.</li>
                </ul>
              </CardContent>
            </Card>
          </TabsContent>

          {/* ========================================================================= */}
          {/* TAB 8: INTERACTIVE CALCULATOR SANDBOX */}
          {/* ========================================================================= */}
          <TabsContent value="interactive-sandbox" className="space-y-6">
            <InteractiveSandbox />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default HelpPortal;
