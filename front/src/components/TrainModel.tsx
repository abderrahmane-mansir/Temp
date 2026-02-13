import { useState, useCallback } from 'react';
import { Upload, FileText, AlertCircle, CheckCircle2, Calendar } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { trainModel, type FeatureImportance, type TrainingMetrics, type BestDayAnalysis } from '@/lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface TrainModelProps {
  onTrainingComplete: () => void;
}

export function TrainModel({ onTrainingComplete }: TrainModelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [bestDays, setBestDays] = useState<BestDayAnalysis | null>(null);
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
      } else {
        setError('Please upload a CSV file');
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleTrain = async () => {
    if (!file) return;

    setIsTraining(true);
    setError(null);
    setMetrics(null);

    try {
      const result = await trainModel(file);
      if (result.success) {
        setMetrics(result.metrics);
        setFeatureImportance(result.feature_importance || []);
        setBestDays(result.best_days || null);
        onTrainingComplete();
      } else {
        setError(result.error || 'Training failed');
      }
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to train model';
      setError(errorMessage);
    } finally {
      setIsTraining(false);
    }
  };

  const chartData = featureImportance.slice(0, 10).map((f) => ({
    name: f.feature.replace('_encoded', '').replace('_', ' '),
    importance: Math.round(f.importance * 1000) / 10,
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5 text-primary" />
            Train New Model
          </CardTitle>
          <CardDescription>
            Upload a CSV dataset to train a new viral prediction model
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
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <div className="flex flex-col items-center gap-3">
                {file ? (
                  <>
                    <FileText className="h-12 w-12 text-success" />
                    <div>
                      <p className="font-medium text-foreground">{file.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </>
                ) : (
                  <>
                    <Upload className="h-12 w-12 text-muted-foreground" />
                    <div>
                      <p className="font-medium text-foreground">
                        Drop your CSV file here
                      </p>
                      <p className="text-sm text-muted-foreground">
                        or click to browse
                      </p>
                    </div>
                  </>
                )}
              </div>
            </label>
          </div>

          {/* Required Columns Info */}
          <div className="bg-secondary/50 rounded-lg p-4 text-sm">
            <p className="font-medium mb-2">Required columns:</p>
            <p className="text-muted-foreground">
              Viral, Views, Likes, Shares, Comments, Platform, Content_Type, Region, Post_Date
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-destructive" />
              <p className="text-destructive">{error}</p>
            </div>
          )}

          {/* Train Button */}
          <Button
            onClick={handleTrain}
            disabled={!file || isTraining}
            className="w-full"
            size="lg"
          >
            {isTraining ? 'Training Model...' : 'Start Training'}
          </Button>

          {isTraining && (
            <div className="space-y-2">
              <Progress value={66} className="animate-pulse" />
              <p className="text-sm text-muted-foreground text-center">
                Training in progress... This may take a few moments.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Results */}
      {metrics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-success" />
              Training Complete
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-primary">
                  {(metrics.test_f1_score * 100).toFixed(1)}%
                </p>
                <p className="text-sm text-muted-foreground">F1 Score</p>
              </div>
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-foreground">
                  {metrics.training_samples.toLocaleString()}
                </p>
                <p className="text-sm text-muted-foreground">Training Samples</p>
              </div>
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-foreground">
                  {metrics.test_samples.toLocaleString()}
                </p>
                <p className="text-sm text-muted-foreground">Test Samples</p>
              </div>
              <div className="bg-secondary/50 rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-foreground">
                  {metrics.top_features.length}
                </p>
                <p className="text-sm text-muted-foreground">Features Used</p>
              </div>
            </div>

            {/* Feature Importance Chart */}
            {chartData.length > 0 && (
              <div>
                <h4 className="font-medium mb-4">Feature Importance</h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartData}
                      layout="vertical"
                      margin={{ left: 100, right: 20, top: 10, bottom: 10 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis type="number" stroke="#9CA3AF" fontSize={12} />
                      <YAxis
                        dataKey="name"
                        type="category"
                        stroke="#9CA3AF"
                        fontSize={12}
                        width={90}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1f2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                        formatter={(value: number) => [`${value}%`, 'Importance']}
                      />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {chartData.map((_, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={index === 0 ? '#6366f1' : '#4f46e5'}
                            fillOpacity={1 - index * 0.08}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Best Day to Post Analysis */}
            {bestDays && (
              <div className="space-y-4">
                <h4 className="font-medium flex items-center gap-2">
                  <Calendar className="h-5 w-5 text-primary" />
                  Best Day to Post Analysis
                </h4>
                
                {/* Overall Best Day */}
                <div className="bg-primary/10 border border-primary/20 rounded-lg p-4">
                  <p className="text-sm text-muted-foreground mb-1">Overall Best Day</p>
                  <p className="text-2xl font-bold text-primary">{bestDays.overall_best_day}</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Based on {bestDays.total_posts_analyzed?.toLocaleString()} posts analyzed
                  </p>
                </div>

                {/* Best Days Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-2 px-3 font-medium">Platform</th>
                        <th className="text-left py-2 px-3 font-medium">Content</th>
                        <th className="text-left py-2 px-3 font-medium">Best Day</th>
                        <th className="text-left py-2 px-3 font-medium">Viral Rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bestDays.best_days_by_combination?.slice(0, 8).map((item, index) => (
                        <tr key={index} className="border-b border-border/50">
                          <td className="py-2 px-3">
                            <Badge variant="outline" className="capitalize">
                              {item.platform}
                            </Badge>
                          </td>
                          <td className="py-2 px-3 capitalize">{item.content_type}</td>
                          <td className="py-2 px-3 font-medium">{item.best_day}</td>
                          <td className="py-2 px-3">
                            <span className={`font-medium ${item.viral_rate >= 0.4 ? 'text-green-500' : 'text-yellow-500'}`}>
                              {(item.viral_rate * 100).toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
