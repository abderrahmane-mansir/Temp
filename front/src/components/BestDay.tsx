import { useState, useCallback, useEffect } from 'react';
import { Upload, Calendar, TrendingUp, BarChart3, Info, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { analyzeBestDay, getBestDays, type BestDayAnalysis } from '@/lib/api';
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

interface HeatmapCell {
  platform: string;
  content_type: string;
  day: string;
  day_number: number;
  viral_rate: number;
  post_count: number;
}

interface BestDayResult {
  platform: string;
  content_type: string;
  best_day: string;
  best_day_number: number;
  viral_rate: number;
  post_count: number;
}

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const getViralRateColor = (rate: number): string => {
  if (rate >= 0.6) return 'bg-green-500';
  if (rate >= 0.4) return 'bg-yellow-500';
  if (rate >= 0.2) return 'bg-orange-500';
  return 'bg-red-500';
};

const getViralRateColorHex = (rate: number): string => {
  if (rate >= 0.6) return '#22c55e';
  if (rate >= 0.4) return '#eab308';
  if (rate >= 0.2) return '#f97316';
  return '#ef4444';
};

export function BestDay() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoadingFromModel, setIsLoadingFromModel] = useState(true);
  const [analysis, setAnalysis] = useState<BestDayAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlatform, setSelectedPlatform] = useState<string>('all');
  const [selectedContentType, setSelectedContentType] = useState<string>('all');

  // Try to load best days from trained model on mount
  useEffect(() => {
    const loadFromModel = async () => {
      try {
        const result = await getBestDays();
        if (result.success && result.analysis) {
          setAnalysis(result.analysis);
        }
      } catch {
        // No saved analysis available, that's okay
      } finally {
        setIsLoadingFromModel(false);
      }
    };
    loadFromModel();
  }, []);

  const handleRefreshFromModel = async () => {
    setIsLoadingFromModel(true);
    try {
      const result = await getBestDays();
      if (result.success && result.analysis) {
        setAnalysis(result.analysis);
        setError(null);
      } else {
        setError('No analysis available. Train a model first.');
      }
    } catch {
      setError('Failed to load analysis from model.');
    } finally {
      setIsLoadingFromModel(false);
    }
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile?.type === 'text/csv' || droppedFile?.name.endsWith('.csv')) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError('Please drop a CSV file');
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalysis(null);

    try {
      const result = await analyzeBestDay(file);
      if (result.success) {
        setAnalysis(result.analysis);
      } else {
        setError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Filter heatmap data based on selections
  const filteredHeatmapData = analysis?.heatmap_data?.filter((item: HeatmapCell) => {
    if (selectedPlatform !== 'all' && item.platform !== selectedPlatform) return false;
    if (selectedContentType !== 'all' && item.content_type !== selectedContentType) return false;
    return true;
  }) || [];

  // Aggregate data by day for bar chart
  const dayAggregatedData = DAYS.map((day, index) => {
    const dayData = filteredHeatmapData.filter((item: HeatmapCell) => item.day_number === index);
    const avgRate = dayData.length > 0
      ? dayData.reduce((sum: number, item: HeatmapCell) => sum + item.viral_rate, 0) / dayData.length
      : 0;
    const totalPosts = dayData.reduce((sum: number, item: HeatmapCell) => sum + item.post_count, 0);
    return {
      day,
      day_short: day.slice(0, 3),
      viral_rate: Number(avgRate.toFixed(4)),
      post_count: totalPosts,
    };
  });

  const bestDayFromChart = dayAggregatedData.reduce((best, current) =>
    current.viral_rate > best.viral_rate ? current : best, dayAggregatedData[0]
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Best Day to Post Analysis</h2>
          <p className="text-muted-foreground">
            Discover the optimal days for posting to maximize virality
          </p>
        </div>
        {analysis && (
          <Button variant="outline" onClick={handleRefreshFromModel} disabled={isLoadingFromModel}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoadingFromModel ? 'animate-spin' : ''}`} />
            Refresh from Model
          </Button>
        )}
      </div>

      {/* Loading State */}
      {isLoadingFromModel && !analysis && (
        <Card>
          <CardContent className="py-12">
            <div className="flex flex-col items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4" />
              <p className="text-muted-foreground">Loading analysis from trained model...</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* File Upload - Show when no analysis and not loading */}
      {!analysis && !isLoadingFromModel && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Dataset
            </CardTitle>
            <CardDescription>
              Upload your training CSV file to analyze the best posting days
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragging
                  ? 'border-primary bg-primary/5'
                  : 'border-muted-foreground/25 hover:border-primary/50'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-lg font-medium mb-2">
                {file ? file.name : 'Drop your CSV file here'}
              </p>
              <p className="text-sm text-muted-foreground mb-4">
                or click to browse
              </p>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload-bestday"
              />
              <Button asChild variant="outline">
                <label htmlFor="file-upload-bestday" className="cursor-pointer">
                  Select File
                </label>
              </Button>
            </div>

            {file && (
              <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <BarChart3 className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium">{file.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
                <Button onClick={handleAnalyze} disabled={isAnalyzing}>
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    'Analyze'
                  )}
                </Button>
              </div>
            )}

            {error && (
              <div className="p-4 bg-destructive/10 text-destructive rounded-lg">
                {error}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Analysis Results */}
      {analysis && (
        <>
          {/* Summary Cards */}
          <div className="grid md:grid-cols-3 gap-4">
            <Card className="border-primary/30 bg-gradient-to-br from-primary/10 to-transparent">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Overall Best Day
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <Calendar className="h-8 w-8 text-primary" />
                  <span className="text-3xl font-bold">{analysis.overall_best_day}</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Posts Analyzed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-8 w-8 text-green-500" />
                  <span className="text-3xl font-bold">{analysis.total_posts_analyzed.toLocaleString()}</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Combinations Analyzed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-8 w-8 text-blue-500" />
                  <span className="text-3xl font-bold">{analysis.best_days_by_combination?.length || 0}</span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                Filter Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Platform</label>
                  <select
                    className="w-40 px-3 py-2 rounded-md border border-input bg-background"
                    value={selectedPlatform}
                    onChange={(e) => setSelectedPlatform(e.target.value)}
                  >
                    <option value="all">All Platforms</option>
                    {analysis.platforms?.map((platform: string) => (
                      <option key={platform} value={platform}>
                        {platform.charAt(0).toUpperCase() + platform.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Content Type</label>
                  <select
                    className="w-40 px-3 py-2 rounded-md border border-input bg-background"
                    value={selectedContentType}
                    onChange={(e) => setSelectedContentType(e.target.value)}
                  >
                    <option value="all">All Types</option>
                    {analysis.content_types?.map((type: string) => (
                      <option key={type} value={type}>
                        {type.charAt(0).toUpperCase() + type.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex items-end">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setSelectedPlatform('all');
                      setSelectedContentType('all');
                    }}
                  >
                    Reset Filters
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Bar Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Viral Rate by Day of Week</CardTitle>
              <CardDescription>
                {selectedPlatform === 'all' && selectedContentType === 'all'
                  ? 'Average viral rate across all platforms and content types'
                  : `Filtered by ${selectedPlatform !== 'all' ? selectedPlatform : 'all platforms'} ${selectedContentType !== 'all' ? `/ ${selectedContentType}` : ''}`
                }
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dayAggregatedData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="day_short" className="text-muted-foreground" />
                    <YAxis
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      className="text-muted-foreground"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px',
                      }}
                      formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Viral Rate']}
                      labelFormatter={(label) => DAYS.find(d => d.startsWith(label)) || label}
                    />
                    <Bar dataKey="viral_rate" radius={[4, 4, 0, 0]}>
                      {dayAggregatedData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.day === bestDayFromChart.day ? '#22c55e' : 'hsl(var(--primary))'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              {bestDayFromChart && (
                <div className="mt-4 text-center">
                  <Badge variant="success" className="text-sm px-4 py-1">
                    Best Day: {bestDayFromChart.day} ({(bestDayFromChart.viral_rate * 100).toFixed(1)}% viral rate)
                  </Badge>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Best Day by Combination Table */}
          <Card>
            <CardHeader>
              <CardTitle>Best Day by Platform & Content Type</CardTitle>
              <CardDescription>
                Optimal posting day for each platform and content type combination
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-medium">Platform</th>
                      <th className="text-left py-3 px-4 font-medium">Content Type</th>
                      <th className="text-left py-3 px-4 font-medium">Best Day</th>
                      <th className="text-left py-3 px-4 font-medium">Viral Rate</th>
                      <th className="text-left py-3 px-4 font-medium">Posts</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.best_days_by_combination?.map((item: BestDayResult, index: number) => (
                      <tr key={index} className="border-b border-border/50 hover:bg-muted/50">
                        <td className="py-3 px-4">
                          <Badge variant="outline">
                            {item.platform.charAt(0).toUpperCase() + item.platform.slice(1)}
                          </Badge>
                        </td>
                        <td className="py-3 px-4 capitalize">{item.content_type}</td>
                        <td className="py-3 px-4 font-medium">{item.best_day}</td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className={`w-3 h-3 rounded-full ${getViralRateColor(item.viral_rate)}`} />
                            <span>{(item.viral_rate * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-muted-foreground">{item.post_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Heatmap Grid */}
          <Card>
            <CardHeader>
              <CardTitle>Viral Rate Heatmap</CardTitle>
              <CardDescription>
                Visual representation of viral rates across all days and combinations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr>
                      <th className="text-left py-2 px-3 font-medium">Platform / Content</th>
                      {DAYS.map((day) => (
                        <th key={day} className="text-center py-2 px-3 font-medium text-sm">
                          {day.slice(0, 3)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.best_days_by_combination?.map((combo: BestDayResult) => {
                      const comboData = analysis.heatmap_data?.filter(
                        (h: HeatmapCell) => h.platform === combo.platform && h.content_type === combo.content_type
                      ) || [];

                      return (
                        <tr key={`${combo.platform}-${combo.content_type}`} className="border-b border-border/30">
                          <td className="py-2 px-3 font-medium text-sm">
                            {combo.platform} / {combo.content_type}
                          </td>
                          {DAYS.map((_, dayIndex) => {
                            const cellData = comboData.find((h: HeatmapCell) => h.day_number === dayIndex);
                            const rate = cellData?.viral_rate || 0;
                            return (
                              <td key={dayIndex} className="text-center py-2 px-1">
                                <div
                                  className="w-full h-8 rounded flex items-center justify-center text-xs font-medium"
                                  style={{
                                    backgroundColor: cellData
                                      ? getViralRateColorHex(rate)
                                      : 'hsl(var(--muted))',
                                    color: rate >= 0.4 ? 'white' : 'inherit',
                                    opacity: cellData ? 0.8 + rate * 0.2 : 0.3,
                                  }}
                                >
                                  {cellData ? `${(rate * 100).toFixed(0)}%` : '-'}
                                </div>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-center gap-6 mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-red-500" />
                  <span>&lt;20%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-orange-500" />
                  <span>20-40%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-yellow-500" />
                  <span>40-60%</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-green-500" />
                  <span>&gt;60%</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* New Analysis Button */}
          <div className="text-center">
            <Button
              variant="outline"
              onClick={() => {
                setAnalysis(null);
                setFile(null);
                setSelectedPlatform('all');
                setSelectedContentType('all');
              }}
            >
              Analyze New Dataset
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
