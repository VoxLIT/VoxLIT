import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { describe, it, expect, vi } from 'vitest';

vi.mock('@/tasks/registry', () => ({
  isDeepfakeDemoDataset: (dataset: string) => dataset === 'deepfake-demo',
}));

if (typeof window !== 'undefined' && !window.URL.createObjectURL) {
  window.URL.createObjectURL = vi.fn(() => 'mock-url');
}

import { AudioDataTable } from '../components/audio/AudioDataTable';

describe('AudioDataTable Search and Filter', () => {
  const mockMetadata = [
    { id: 'rec_001', path: 'audios/00010.wav', emotion: 'happy', duration_seconds: 3.2 },
    { id: 'rec_002', path: 'audios/00020.wav', emotion: 'angry', duration_seconds: 4.5 },
    { id: 'rec_003', path: 'audios/00030.wav', emotion: 'neutral', duration_seconds: 2.1 },
    { id: 'rec_004', path: 'audios/00040.wav', emotion: 'sad', duration_seconds: 5.0 },
  ];

  it('renders all rows when searchQuery is empty', () => {
    render(
      <AudioDataTable
        selectedRow={null}
        onRowSelect={vi.fn()}
        searchQuery=""
        model="wav2vec2"
        dataset="ravdess"
        datasetMetadata={mockMetadata}
      />
    );

    expect(screen.getByText('00010.wav')).toBeInTheDocument();
    expect(screen.getByText('00020.wav')).toBeInTheDocument();
    expect(screen.getByText('00030.wav')).toBeInTheDocument();
    expect(screen.getByText('00040.wav')).toBeInTheDocument();
  });

  it('filters rows by filename in dataset mode', () => {
    const { rerender } = render(
      <AudioDataTable
        selectedRow={null}
        onRowSelect={vi.fn()}
        searchQuery="00020"
        model="wav2vec2"
        dataset="ravdess"
        datasetMetadata={mockMetadata}
      />
    );

    expect(screen.getByText('00020.wav')).toBeInTheDocument();
    expect(screen.queryByText('00010.wav')).not.toBeInTheDocument();
    expect(screen.queryByText('00030.wav')).not.toBeInTheDocument();
    expect(screen.queryByText('00040.wav')).not.toBeInTheDocument();

    // Change search to another file
    rerender(
      <AudioDataTable
        selectedRow={null}
        onRowSelect={vi.fn()}
        searchQuery="00040"
        model="wav2vec2"
        dataset="ravdess"
        datasetMetadata={mockMetadata}
      />
    );

    expect(screen.getByText('00040.wav')).toBeInTheDocument();
    expect(screen.queryByText('00020.wav')).not.toBeInTheDocument();
  });

  it('filters rows by metadata (emotion) in dataset mode', () => {
    render(
      <AudioDataTable
        selectedRow={null}
        onRowSelect={vi.fn()}
        searchQuery="happy"
        model="wav2vec2"
        dataset="ravdess"
        datasetMetadata={mockMetadata}
      />
    );

    expect(screen.getByText('00010.wav')).toBeInTheDocument();
    expect(screen.queryByText('00020.wav')).not.toBeInTheDocument();
    expect(screen.queryByText('00030.wav')).not.toBeInTheDocument();
  });

  it('filters rows in verification mode', () => {
    render(
      <AudioDataTable
        selectedRow={null}
        onRowSelect={vi.fn()}
        searchQuery="00030"
        model=""
        dataset="custom"
        datasetMetadata={mockMetadata}
        selectionVariant="verification"
        checkedIds={[]}
        onCheckedIdsChange={vi.fn()}
      />
    );

    expect(screen.getByText('00030.wav')).toBeInTheDocument();
    expect(screen.queryByText('00010.wav')).not.toBeInTheDocument();
    expect(screen.queryByText('00020.wav')).not.toBeInTheDocument();
    expect(screen.queryByText('00040.wav')).not.toBeInTheDocument();
  });

  it('filters uploaded files correctly', () => {
    const uploadedFiles = [
      { file_id: 'up_1', filename: 'voice_sample_alpha.wav', file_path: '/path1', message: 'ok', duration: 2.0 },
      { file_id: 'up_2', filename: 'voice_sample_beta.wav', file_path: '/path2', message: 'ok', duration: 3.5 },
    ];

    render(
      <AudioDataTable
        selectedRow={null}
        onRowSelect={vi.fn()}
        searchQuery="beta"
        model=""
        dataset="custom"
        uploadedFiles={uploadedFiles}
      />
    );

    expect(screen.getByText('voice_sample_beta.wav')).toBeInTheDocument();
    expect(screen.queryByText('voice_sample_alpha.wav')).not.toBeInTheDocument();
  });

  it('shows "No results." when search query matches nothing', () => {
    render(
      <AudioDataTable
        selectedRow={null}
        onRowSelect={vi.fn()}
        searchQuery="nonexistent_audio_query_xyz"
        model="wav2vec2"
        dataset="ravdess"
        datasetMetadata={mockMetadata}
      />
    );

    expect(screen.getByText('No results.')).toBeInTheDocument();
    expect(screen.queryByText('00010.wav')).not.toBeInTheDocument();
  });
});
