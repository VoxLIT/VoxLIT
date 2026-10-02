import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle, Database, File as FileIcon, FolderPlus, Plus, Tags, Trash2, Upload, X } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { CustomDataset } from "../types";
import {
  type DatasetLimits,
  createDataset,
  deleteDataset,
  errorMessage,
  listDatasets,
  uploadDatasetFiles,
  uploadDatasetLabels,
} from "./api";

type FileStatus = { file: string; status: "pending" | "success" | "error"; error?: string };

/**
 * Manage Datasets for the deepfake task. Same dialog, tabs and actions as the
 * other tasks' CustomDatasetManager, backed by the deepfake task's own
 * /tasks/deepfake/datasets routes (the shared /upload/dataset store is not
 * readable by the detectors). One addition for researchers: an optional label
 * file, which lets the Detector report measure EER and the DET curve on the
 * dataset. Labels are stored server-side; only per-class counts come back.
 */
export const DatasetManager = ({
  activeDataset,
  onSelect,
  onChanged,
}: {
  /** Name of the custom dataset in use, or null for the built-in subset. */
  activeDataset: string | null;
  onSelect: (name: string) => void;
  /** After any create/upload/label/delete, so the page can refresh. */
  onChanged: (event: { type: "created" | "uploaded" | "labelled" | "deleted"; datasetName: string }) => void;
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("list");
  const [datasets, setDatasets] = useState<CustomDataset[]>([]);
  const [limits, setLimits] = useState<DatasetLimits | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [newDatasetName, setNewDatasetName] = useState("");
  const [createLoading, setCreateLoading] = useState(false);

  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [labelFile, setLabelFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<FileStatus[]>([]);
  const [labelSummary, setLabelSummary] = useState<string | null>(null);
  const filesInput = useRef<HTMLInputElement>(null);
  const labelsInput = useRef<HTMLInputElement>(null);

  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await listDatasets();
      setDatasets(payload.datasets);
      setLimits(payload.limits);
    } catch (caught) {
      setError(errorMessage(caught, "Failed to fetch datasets"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) fetchDatasets();
  }, [isOpen, fetchDatasets]);

  const create = async () => {
    const name = newDatasetName.trim();
    if (!name) {
      setError("Dataset name is required");
      return;
    }
    setCreateLoading(true);
    setError(null);
    try {
      await createDataset(name);
      setNewDatasetName("");
      await fetchDatasets();
      // Straight on to adding files, which is always the next step.
      setSelectedDataset(name);
      setActiveTab("upload");
      onChanged({ type: "created", datasetName: name });
    } catch (caught) {
      setError(errorMessage(caught, "Failed to create dataset"));
    } finally {
      setCreateLoading(false);
    }
  };

  const upload = async () => {
    if (!selectedDataset || (selectedFiles.length === 0 && !labelFile)) {
      setError("Please select a dataset and audio files or a label file");
      return;
    }
    setUploadLoading(true);
    setError(null);
    setUploadProgress(0);
    setLabelSummary(null);
    setUploadStatus(selectedFiles.map((file) => ({ file: file.name, status: "pending" })));
    try {
      if (selectedFiles.length > 0) {
        // Small batches, so progress moves and one huge request never times out.
        const statuses: FileStatus[] = [];
        const batch = 8;
        for (let start = 0; start < selectedFiles.length; start += batch) {
          const chunk = selectedFiles.slice(start, start + batch);
          const result = await uploadDatasetFiles(selectedDataset, chunk);
          for (const file of chunk) {
            const failure = result.errors.find((item) => item.filename === file.name);
            statuses.push(failure ? { file: file.name, status: "error", error: failure.error } : { file: file.name, status: "success" });
          }
          setUploadStatus([...statuses, ...selectedFiles.slice(statuses.length).map((file) => ({ file: file.name, status: "pending" as const }))]);
          setUploadProgress(Math.round((Math.min(start + batch, selectedFiles.length) / selectedFiles.length) * 100));
        }
        onChanged({ type: "uploaded", datasetName: selectedDataset });
      }
      if (labelFile) {
        const summary = await uploadDatasetLabels(selectedDataset, labelFile);
        setLabelSummary(
          `Labels matched ${summary.labels.matched_files} of ${summary.total_files} files ` +
            `(${summary.labels.bonafide} bona fide, ${summary.labels.spoof} spoof).`,
        );
        onChanged({ type: "labelled", datasetName: selectedDataset });
      }
      setUploadProgress(100);
      setSelectedFiles([]);
      setLabelFile(null);
      if (filesInput.current) filesInput.current.value = "";
      if (labelsInput.current) labelsInput.current.value = "";
      await fetchDatasets();
    } catch (caught) {
      const message = errorMessage(caught, "Upload failed");
      setError(message);
      setUploadStatus((current) => current.map((item) => (item.status === "pending" ? { ...item, status: "error", error: message } : item)));
    } finally {
      setUploadLoading(false);
    }
  };

  const remove = async (name: string) => {
    if (!confirm(`Are you sure you want to delete the dataset "${name}"? This action cannot be undone.`)) return;
    try {
      await deleteDataset(name);
      await fetchDatasets();
      onChanged({ type: "deleted", datasetName: name });
    } catch (caught) {
      setError(errorMessage(caught, "Failed to delete dataset"));
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 text-xs">
          <Database className="mr-2 h-4 w-4" />
          Manage Datasets
        </Button>
      </DialogTrigger>
      <DialogContent className="max-h-[80vh] max-w-4xl overflow-hidden">
        <DialogHeader>
          <DialogTitle>Custom Dataset Manager</DialogTitle>
          <DialogDescription>
            Create and manage custom audio datasets for deepfake analysis. Add a label file to measure EER on them.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="list" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              My Datasets
            </TabsTrigger>
            <TabsTrigger value="create" className="flex items-center gap-2">
              <FolderPlus className="h-4 w-4" />
              Create Dataset
            </TabsTrigger>
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload Files
            </TabsTrigger>
          </TabsList>

          <div className="mt-4 max-h-[50vh] overflow-y-auto">
            <TabsContent value="list" className="space-y-4">
              {loading && (
                <div className="py-8 text-center">
                  <div className="mx-auto mb-2 h-6 w-6 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
                  Loading datasets...
                </div>
              )}

              {!loading && datasets.length === 0 && (
                <div className="py-8 text-center text-muted-foreground">
                  <Database className="mx-auto mb-4 h-12 w-12 opacity-50" />
                  <p>No custom datasets found</p>
                  <p className="text-sm">Create your first dataset to get started</p>
                </div>
              )}

              {!loading &&
                datasets.map((dataset) => (
                  <Card key={dataset.dataset_name} className="transition-shadow hover:shadow-md">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2 text-lg">
                          {dataset.dataset_name}
                          {activeDataset === dataset.dataset_name && <Badge>in use</Badge>}
                        </CardTitle>
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary">{dataset.total_files} files</Badge>
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={dataset.total_files === 0}
                            onClick={() => {
                              onSelect(dataset.dataset_name);
                              setIsOpen(false);
                            }}
                          >
                            Select
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => remove(dataset.dataset_name)}
                            className="text-red-600 hover:text-red-700"
                            aria-label={`Delete ${dataset.dataset_name}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      <CardDescription>
                        Created {dataset.created_at ? new Date(dataset.created_at * 1000).toLocaleDateString() : "recently"}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Total Duration:</span> {dataset.total_duration_seconds.toFixed(1)}s
                        </div>
                        <div>
                          <span className="font-medium">Labels:</span>{" "}
                          {dataset.labels.provided
                            ? `${dataset.labels.matched_files} of ${dataset.total_files} matched (${dataset.labels.bonafide} bona fide / ${dataset.labels.spoof} spoof)`
                            : "none — EER needs a label file"}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
            </TabsContent>

            <TabsContent value="create" className="space-y-4">
              <div>
                <Label htmlFor="df-dataset-name">Dataset Name</Label>
                <Input
                  id="df-dataset-name"
                  value={newDatasetName}
                  onChange={(event) => setNewDatasetName(event.target.value)}
                  onKeyDown={(event) => event.key === "Enter" && create()}
                  placeholder="Enter dataset name..."
                  maxLength={48}
                  className="mt-1"
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  Letters, digits, spaces, - and _.{" "}
                  {limits && `Up to ${limits.max_datasets} datasets, kept for ${limits.ttl_days} days after the last change.`}
                </p>
              </div>
              <Button onClick={create} disabled={createLoading || !newDatasetName.trim()} className="w-full">
                {createLoading ? (
                  <>
                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus className="mr-2 h-4 w-4" />
                    Create Dataset
                  </>
                )}
              </Button>
            </TabsContent>

            <TabsContent value="upload" className="space-y-4">
              <div>
                <Label htmlFor="df-dataset-select">Select Dataset</Label>
                <select
                  id="df-dataset-select"
                  value={selectedDataset}
                  onChange={(event) => setSelectedDataset(event.target.value)}
                  className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Choose a dataset...</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.dataset_name} value={dataset.dataset_name}>
                      {dataset.dataset_name} ({dataset.total_files} files)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <Label htmlFor="df-files-input">Audio Files</Label>
                <Input
                  ref={filesInput}
                  id="df-files-input"
                  type="file"
                  multiple
                  accept="audio/*,.wav,.mp3,.flac,.ogg"
                  onChange={(event) => setSelectedFiles(Array.from(event.target.files ?? []))}
                  className="mt-1"
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  WAV, FLAC, OGG or MP3, 0.5–60 s each
                  {limits && `, up to ${limits.max_file_mb} MB and ${limits.max_files_per_dataset} files per dataset`}.
                  {selectedFiles.length > 0 && ` ${selectedFiles.length} file(s) selected.`}
                </p>
              </div>

              <div>
                <Label htmlFor="df-labels-input" className="flex items-center gap-1.5">
                  <Tags className="h-3.5 w-3.5" /> Labels (optional)
                </Label>
                <Input
                  ref={labelsInput}
                  id="df-labels-input"
                  type="file"
                  accept=".csv,.txt,text/csv,text/plain"
                  onChange={(event) => setLabelFile(event.target.files?.[0] ?? null)}
                  className="mt-1"
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  CSV lines <code>filename,label[,attack]</code> with label bonafide/spoof (or real/fake), or ASVspoof
                  protocol lines <code>SPEAKER FILE - ATTACK label</code>. Matched on the file name without extension.
                  Labels are only used for aggregate metrics (EER, DET, AUC); clips stay blind on the page.
                </p>
              </div>

              {uploadLoading && (
                <div className="space-y-2">
                  <Progress value={uploadProgress} className="w-full" />
                  <p className="text-center text-sm">Uploading and converting to 16 kHz mono...</p>
                </div>
              )}

              {uploadStatus.length > 0 && (
                <div className="max-h-32 space-y-2 overflow-y-auto">
                  <p className="text-sm font-medium">Upload Status:</p>
                  {uploadStatus.map((item, index) => (
                    <div key={`${item.file}-${index}`} className="flex items-center gap-2 text-xs">
                      {item.status === "success" && <CheckCircle className="h-4 w-4 text-green-500" />}
                      {item.status === "error" && <AlertCircle className="h-4 w-4 text-red-500" />}
                      {item.status === "pending" && <div className="h-4 w-4 rounded-full border border-gray-300" />}
                      <FileIcon className="h-3 w-3 text-muted-foreground" />
                      <span className="truncate">{item.file}</span>
                      {item.error && <span className="text-red-500">- {item.error}</span>}
                    </div>
                  ))}
                </div>
              )}
              {labelSummary && (
                <p className="flex items-center gap-2 text-xs">
                  <CheckCircle className="h-4 w-4 text-green-500" /> {labelSummary}
                </p>
              )}

              <Button
                onClick={upload}
                disabled={uploadLoading || !selectedDataset || (selectedFiles.length === 0 && !labelFile)}
                className="w-full"
              >
                {uploadLoading ? (
                  <>
                    <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" />
                    Upload Files
                  </>
                )}
              </Button>
            </TabsContent>
          </div>
        </Tabs>

        {error && (
          <div className="mt-4 flex items-center gap-2 rounded-md border border-red-200 bg-red-50 p-3">
            <AlertCircle className="h-4 w-4 text-red-500" />
            <span className="text-sm text-red-700">{error}</span>
            <Button variant="ghost" size="sm" onClick={() => setError(null)} className="ml-auto" aria-label="Dismiss">
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
