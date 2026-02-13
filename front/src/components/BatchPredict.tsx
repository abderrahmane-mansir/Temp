import { useState, useCallback } from 'react';
import { FileUp, Download, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { predictFromFile, type PredictionResult } from '@/lib/api';

export function BatchPredict() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile);
        setError(null);
        setResults([]);
      } else {
        setError('Please upload a CSV file');
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResults([]);
    }
  };

  const handlePredict = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await predictFromFile(file);
      if (response.success) {
        setResults(response.predictions);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get predictions';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadResults = () => {
    if (results.length === 0) return;

    const csv = [
      ['Post_ID', 'Viral', 'Probability_Viral', 'Probability_Not_Viral'].join(','),
      ...results.map((r) =>
        [r.post_id, r.viral, r.probability_viral, r.probability_not_viral].join(',')
      ),
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const viralCount = results.filter((r) => r.viral === 1).length;
  const nonViralCount = results.filter((r) => r.viral === 0).length;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileUp className="h-5 w-5 text-primary" />
            Batch Predictions
          </CardTitle>
          <CardDescription>
            Upload a CSV file to predict virality for multiple posts at once
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Drop Zone */}
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
              dragActive
                ? 'border-primary bg-primary/5'
                : 'border-border hover:border-muted-foreground'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
              id="batch-file-upload"
            />
            <label htmlFor="batch-file-upload" className="cursor-pointer">
              <div className="flex flex-col items-center gap-3">
                <FileUp className="h-12 w-12 text-muted-foreground" />
                <div>
                  <p className="font-medium text-foreground">
                    {file ? file.name : 'Drop your CSV file here'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {file
                      ? `${(file.size / 1024).toFixed(1)} KB`
                      : 'or click to browse'}
                  </p>
                </div>
              </div>
            </label>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-destructive" />
              <p className="text-destructive">{error}</p>
            </div>
          )}

          {/* Buttons */}
          <div className="flex gap-3">
            <Button
              onClick={handlePredict}
              disabled={!file || isLoading}
              className="flex-1"
            >
              {isLoading ? 'Processing...' : 'Run Predictions'}
            </Button>
            {results.length > 0 && (
              <Button variant="outline" onClick={downloadResults}>
                <Download className="mr-2 h-4 w-4" />
                Download CSV
              </Button>
            )}
          </div>

          {isLoading && (
            <div className="space-y-2">
              <Progress value={50} className="animate-pulse" />
              <p className="text-sm text-muted-foreground text-center">
                Processing predictions...
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results */}
      {results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>
              {results.length} posts analyzed
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Summary */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold">{results.length}</p>
                <p className="text-sm text-muted-foreground">Total Posts</p>
              </div>
              <div className="bg-success/10 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-success">{viralCount}</p>
                <p className="text-sm text-muted-foreground">Predicted Viral</p>
              </div>
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold">{nonViralCount}</p>
                <p className="text-sm text-muted-foreground">Non-Viral</p>
              </div>
            </div>

            {/* Viral Rate */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Viral Rate</span>
                <span className="font-medium">
                  {((viralCount / results.length) * 100).toFixed(1)}%
                </span>
              </div>
              <Progress
                value={(viralCount / results.length) * 100}
                variant="success"
              />
            </div>

            {/* Results Table */}
            <div className="border rounded-lg overflow-hidden">
              <div className="max-h-80 overflow-y-auto">
                <table className="w-full">
                  <thead className="bg-secondary/50 sticky top-0">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-medium">
                        Post ID
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-medium">
                        Prediction
                      </th>
                      <th className="px-4 py-3 text-right text-sm font-medium">
                        Confidence
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {results.slice(0, 50).map((result, index) => (
                      <tr key={index} className="hover:bg-secondary/30 transition-colors">
                        <td className="px-4 py-3 text-sm">{result.post_id}</td>
                        <td className="px-4 py-3">
                          {result.viral === 1 ? (
                            <Badge variant="success" className="gap-1">
                              <CheckCircle className="h-3 w-3" />
                              Viral
                            </Badge>
                          ) : (
                            <Badge variant="secondary" className="gap-1">
                              <XCircle className="h-3 w-3" />
                              Non-Viral
                            </Badge>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm text-right">
                          {(
                            Math.max(
                              result.probability_viral,
                              result.probability_not_viral
                            ) * 100
                          ).toFixed(1)}
                          %
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {results.length > 50 && (
                <div className="px-4 py-2 bg-secondary/30 text-sm text-muted-foreground text-center">
                  Showing first 50 of {results.length} results
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
